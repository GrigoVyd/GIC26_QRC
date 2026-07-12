"""
Quantum Reservoir Computing — gate-based Ising-type reservoir.

Architecture per time step
--------------------------
|0…0>  →  H⊗n  →  [Encode(x) → Ising layer] × n_layers  →  Measure

Reservoir features
------------------
  • <Z_i>      for i = 0…n-1          (n features)
  • <Z_i Z_j>  for i < j              (n*(n-1)/2 features)

The reservoir weights are random and *fixed* after construction.
Only the downstream linear readout is trained.
"""

from __future__ import annotations

import numpy as np
from itertools import combinations
from math import comb
from typing import Literal, Optional

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel


# ---------------------------------------------------------------------------
# Helper: vectorised expectation values from a statevector
# ---------------------------------------------------------------------------

def _observables_from_sv(sv: Statevector, n_qubits: int,
                         max_order: int = 2) -> np.ndarray:
    """
    Compute Z and ZZ expectation values from a Statevector in one pass.

    In Qiskit's convention, state index k has qubit-i bit = (k >> i) & 1.
    Signs: +1 for |0>, -1 for |1>.
    """
    probs = np.abs(sv.data) ** 2                    # (2^n,)
    indices = np.arange(2 ** n_qubits)
    # bits[k, i] = bit of qubit i in basis state k
    bits = ((indices[:, None] >> np.arange(n_qubits)[None, :]) & 1).astype(np.float64)
    signs = 1.0 - 2.0 * bits                        # (2^n, n)

    z_exp = probs @ signs                            # (n,)

    # ZZ: <Z_i Z_j> = sum_k p_k * sign_k_i * sign_k_j
    weighted = signs * probs[:, None]               # (2^n, n)
    zz_matrix = weighted.T @ signs                  # (n, n)

    values = [z_exp]
    if max_order >= 2:
        values.append(np.asarray([
            zz_matrix[i, j] for i in range(n_qubits) for j in range(i + 1, n_qubits)
        ]))
    for order in range(3, min(max_order, n_qubits) + 1):
        values.append(np.asarray([
            float(np.sum(probs * np.prod(signs[:, idx], axis=1)))
            for idx in combinations(range(n_qubits), order)
        ]))
    return np.concatenate(values)


def _observables_from_counts(counts: dict, n_qubits: int, n_shots: int,
                             max_order: int = 2) -> np.ndarray:
    """
    Compute Z and ZZ expectation values from shot-based measurement counts.

    Qiskit bitstrings: bitstr[0] = most-significant qubit (qubit n-1),
                       bitstr[-1] = least-significant qubit (qubit 0).
    """
    z_acc = np.zeros(n_qubits)
    zz_acc = np.zeros((n_qubits, n_qubits))

    for bitstr, count in counts.items():
        prob = count / n_shots
        # Convert to qubit-indexed bits (qubit 0 first)
        bits = np.array([int(b) for b in reversed(bitstr)], dtype=np.float64)
        signs = 1.0 - 2.0 * bits
        z_acc += signs * prob
        zz_acc += np.outer(signs, signs) * prob

    values = [z_acc]
    if max_order >= 2:
        values.append(np.asarray([
            zz_acc[i, j] for i in range(n_qubits) for j in range(i + 1, n_qubits)
        ]))
    if max_order >= 3:
        high = {order: np.zeros(comb(n_qubits, order))
                for order in range(3, min(max_order, n_qubits) + 1)}
        for bitstr, count in counts.items():
            prob = count / n_shots
            bits = np.array([int(b) for b in reversed(bitstr)], dtype=np.float64)
            signs = 1.0 - 2.0 * bits
            for order, acc in high.items():
                for k, idx in enumerate(combinations(range(n_qubits), order)):
                    acc[k] += prob * float(np.prod(signs[list(idx)]))
        values.extend(high[order] for order in sorted(high))
    return np.concatenate(values)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class QuantumReservoir:
    """
    Gate-based quantum reservoir with Ising-type random interactions.

    Parameters
    ----------
    n_qubits      : reservoir size (5, 10, or 15 recommended)
    n_layers      : encoding + interaction repetitions
    connectivity  : qubit graph topology
    seed          : reproducibility seed
    encoding_axis : input rotation axis; ry/rz are nontrivial on initial |+>
    noise_model   : Qiskit Aer NoiseModel (None → exact statevector)
    n_shots       : measurement shots (only used when noise_model is set)
    """

    def __init__(
        self,
        n_qubits: int = 5,
        n_layers: int = 3,
        connectivity: Literal["linear", "grid", "random", "all-to-all"] = "random",
        seed: int = 42,
        noise_model: Optional[NoiseModel] = None,
        n_shots: int = 2048,
        recurrent: bool = False,
        memory_scale: float = 0.5,
        encoding_axis: Literal["rx", "ry", "rz", "ryrz"] = "rx",
        observable_order: int = 2,
    ) -> None:
        """
        recurrent      : if True, the previous sample's Z expectations are fed back as
                         an additional Ry encoding on each qubit. Use
                         ``transform_sequential`` to drive the reservoir with this memory.
        memory_scale   : multiplier on the previous-step Z values when used as feedback
                         angle (Ry(memory_scale * π * z_prev[i])). Tunable [0, 1].
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.connectivity = connectivity
        self.seed = seed
        self.noise_model = noise_model
        self.n_shots = n_shots
        self.recurrent = recurrent
        self.memory_scale = memory_scale
        self.encoding_axis = encoding_axis
        self.observable_order = int(observable_order)

        rng = np.random.RandomState(seed)
        self._edges = self._build_edges(rng)
        # Local rotation angles per layer per qubit  (n_layers, n_qubits, 3)
        self._local_w = rng.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3))
        # ZZ coupling strengths per layer per edge  (n_layers, n_edges)
        self._zz_w = rng.uniform(0, np.pi, (n_layers, len(self._edges) or 1))

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_edges(self, rng: np.random.RandomState) -> list[tuple[int, int]]:
        n = self.n_qubits
        if self.connectivity == "linear":
            return [(i, i + 1) for i in range(n - 1)]
        if self.connectivity == "grid":
            # Near-square row-major lattice. This maps directly to square-lattice
            # superconducting devices such as IQM Garnet/Emerald without logical
            # SWAPs when a matching connected sub-grid is available.
            cols = int(np.ceil(np.sqrt(n)))
            edges = []
            for i in range(n):
                if i % cols != cols - 1 and i + 1 < n:
                    edges.append((i, i + 1))
                if i + cols < n:
                    edges.append((i, i + cols))
            return edges
        if self.connectivity == "all-to-all":
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        # random: ~40 % of all pairs, at least n-1 edges
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        k = max(n - 1, int(len(all_pairs) * 0.4))
        idxs = rng.choice(len(all_pairs), min(k, len(all_pairs)), replace=False)
        return [all_pairs[i] for i in sorted(idxs)]

    # ------------------------------------------------------------------
    # Circuit builder
    # ------------------------------------------------------------------

    def _build_circuit(
        self,
        x: np.ndarray,
        add_measurement: bool = False,
        z_prev: Optional[np.ndarray] = None,
    ) -> QuantumCircuit:
        """
        Build the QRC circuit for input vector x.

        x is assumed to be already normalised to [-1, 1].
        Each qubit i is angle-encoded with x[i % len(x)] using encoding_axis.
        If z_prev is provided (recurrent mode), each qubit i additionally gets an
        Ry(memory_scale * π * z_prev[i]) encoding before the Ising layer — this
        introduces classical-feedback memory across time steps.
        """
        n = self.n_qubits
        qc = QuantumCircuit(n)

        # Initial superposition
        for i in range(n):
            qc.h(i)

        for layer in range(self.n_layers):
            # ---- angle encoding ----
            for i in range(n):
                angle = float(x[i % len(x)]) * np.pi
                if self.encoding_axis == "rx":
                    qc.rx(angle, i)
                elif self.encoding_axis == "ry":
                    qc.ry(angle, i)
                elif self.encoding_axis == "rz":
                    qc.rz(angle, i)
                elif self.encoding_axis == "ryrz":
                    qc.ry(angle, i)
                    qc.rz(0.5 * angle, i)
                else:
                    raise ValueError(f"unknown encoding_axis={self.encoding_axis!r}")
                if z_prev is not None:
                    qc.ry(float(z_prev[i % len(z_prev)]) * self.memory_scale * np.pi, i)

            # ---- ZZ Ising couplings ----
            for k, (qi, qj) in enumerate(self._edges):
                w = float(self._zz_w[layer, k % self._zz_w.shape[1]])
                qc.cx(qi, qj)
                qc.rz(w, qj)
                qc.cx(qi, qj)

            # ---- local reservoir rotations (random, fixed) ----
            for i in range(n):
                qc.rx(float(self._local_w[layer, i, 0]), i)
                qc.ry(float(self._local_w[layer, i, 1]), i)
                qc.rz(float(self._local_w[layer, i, 2]), i)

        if add_measurement:
            qc.measure_all()

        return qc

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _features_noiseless(self, x: np.ndarray, z_prev: Optional[np.ndarray] = None) -> np.ndarray:
        qc = self._build_circuit(x, add_measurement=False, z_prev=z_prev)
        sv = Statevector(qc)
        return _observables_from_sv(sv, self.n_qubits, self.observable_order)

    def _features_noisy(self, x: np.ndarray, backend: AerSimulator, z_prev: Optional[np.ndarray] = None) -> np.ndarray:
        qc = self._build_circuit(x, add_measurement=True, z_prev=z_prev)
        job = backend.run(
            transpile(qc, backend),
            shots=self.n_shots,
            noise_model=self.noise_model,
        )
        counts = job.result().get_counts()
        return _observables_from_counts(
            counts, self.n_qubits, self.n_shots, self.observable_order
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_features(self) -> int:
        """Dimensionality of the reservoir feature vector."""
        n = self.n_qubits
        return sum(comb(n, order)
                   for order in range(1, min(self.observable_order, n) + 1))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Map input matrix X → reservoir feature matrix (stateless / parallel-friendly).

        Parameters
        ----------
        X : (n_samples, n_input_features)  — will be tanh-normalised internally.

        Returns
        -------
        R : (n_samples, n_reservoir_features)
        """
        X_norm = np.tanh(X)

        if self.noise_model is not None:
            backend = AerSimulator()
            return np.array([self._features_noisy(x, backend) for x in X_norm])

        return np.array([self._features_noiseless(x) for x in X_norm])

    def transform_sequential(
        self,
        X: np.ndarray,
        initial_z: Optional[np.ndarray] = None,
        return_final_z: bool = False,
    ):
        """
        Recurrent feature extraction with classical Z-feedback across samples.

        At sample t, the previous step's <Z> expectations are fed back as Ry
        angles into the encoding (see ``_build_circuit``). Useful for time-series
        data where reservoir state should carry information across time steps.

        Parameters
        ----------
        X            : (T, n_input_features) — sequential input, will be tanh-normalised.
        initial_z    : (n_qubits,) — initial Z values; zeros if None.
        return_final_z : if True, also return the final Z vector (use to warm-start test).

        Returns
        -------
        R : (T, n_reservoir_features)
        [optionally] z_final : (n_qubits,)
        """
        X_norm = np.tanh(X)
        z = np.zeros(self.n_qubits) if initial_z is None else initial_z.copy()
        backend = AerSimulator() if self.noise_model is not None else None

        feats = []
        for x in X_norm:
            if backend is not None:
                f = self._features_noisy(x, backend, z_prev=z)
            else:
                f = self._features_noiseless(x, z_prev=z)
            feats.append(f)
            z = f[: self.n_qubits].copy()  # first n entries are <Z_i>

        R = np.array(feats)
        if return_final_z:
            return R, z
        return R

    # sklearn-compatible alias
    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.transform(X)

    def __repr__(self) -> str:
        return (
            f"QuantumReservoir(n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"connectivity='{self.connectivity}', encoding_axis='{self.encoding_axis}', "
            f"n_features={self.n_features}, "
            f"noisy={self.noise_model is not None})"
        )
