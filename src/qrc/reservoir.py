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
from typing import Literal, Optional

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel


# ---------------------------------------------------------------------------
# Helper: vectorised expectation values from a statevector
# ---------------------------------------------------------------------------

def _observables_from_sv(sv: Statevector, n_qubits: int) -> np.ndarray:
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

    zz_vals = [zz_matrix[i, j] for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    return np.concatenate([z_exp, zz_vals])


def _observables_from_counts(counts: dict, n_qubits: int, n_shots: int) -> np.ndarray:
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

    zz_vals = [zz_acc[i, j] for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    return np.concatenate([z_acc, zz_vals])


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
    noise_model   : Qiskit Aer NoiseModel (None → exact statevector)
    n_shots       : measurement shots (only used when noise_model is set)
    """

    def __init__(
        self,
        n_qubits: int = 5,
        n_layers: int = 3,
        connectivity: Literal["linear", "random", "all-to-all"] = "random",
        seed: int = 42,
        noise_model: Optional[NoiseModel] = None,
        n_shots: int = 2048,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.connectivity = connectivity
        self.seed = seed
        self.noise_model = noise_model
        self.n_shots = n_shots

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

    def _build_circuit(self, x: np.ndarray, add_measurement: bool = False) -> QuantumCircuit:
        """
        Build the QRC circuit for input vector x.

        x is assumed to be already normalised to [-1, 1].
        Each qubit i is angle-encoded with x[i % len(x)].
        """
        n = self.n_qubits
        qc = QuantumCircuit(n)

        # Initial superposition
        for i in range(n):
            qc.h(i)

        for layer in range(self.n_layers):
            # ---- angle encoding ----
            for i in range(n):
                qc.rx(float(x[i % len(x)]) * np.pi, i)

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

    def _features_noiseless(self, x: np.ndarray) -> np.ndarray:
        qc = self._build_circuit(x, add_measurement=False)
        sv = Statevector(qc)
        return _observables_from_sv(sv, self.n_qubits)

    def _features_noisy(self, x: np.ndarray, backend: AerSimulator) -> np.ndarray:
        qc = self._build_circuit(x, add_measurement=True)
        job = backend.run(
            transpile(qc, backend),
            shots=self.n_shots,
            noise_model=self.noise_model,
        )
        counts = job.result().get_counts()
        return _observables_from_counts(counts, self.n_qubits, self.n_shots)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_features(self) -> int:
        """Dimensionality of the reservoir feature vector."""
        n = self.n_qubits
        return n + n * (n - 1) // 2

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Map input matrix X → reservoir feature matrix.

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

    # sklearn-compatible alias
    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.transform(X)

    def __repr__(self) -> str:
        return (
            f"QuantumReservoir(n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"connectivity='{self.connectivity}', n_features={self.n_features}, "
            f"noisy={self.noise_model is not None})"
        )
