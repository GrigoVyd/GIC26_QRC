"""
Annealer-style Quantum Reservoir — first-approximation classical simulation.

This is a *simulation* of what a D-Wave-like reservoir would do, intended for
prototyping the architecture and getting empirical numbers before committing
to the engineering effort of a real D-Wave SDK integration. See
``docs/dwave_qrc_exploration.md`` for the broader plan and references.

Substrate model
---------------
A transverse-field Ising Hamiltonian, evaluated at a fixed mid-anneal point:

    H = - sum_i sigma^x_i  +  sum_i h_i sigma^z_i  +  sum_{(i,j) in E} J_ij sigma^z_i sigma^z_j

Initial state: |+>^n (the ground state of -sum sigma^x_i — i.e. the annealer
at s=0). Evolution: U(t) = exp(-i H t), simulated via Trotter

    U(t) ~ [ exp(-i H_Z dt) exp(-i H_X dt) ]^m,    dt = t/m

Readout: <sigma^z_i> and <sigma^z_i sigma^z_j> expectation values on the same
edge set used for couplings (this gives a feature count comparable to the
gate-based reservoir while keeping things tractable for statevector sim).

Mapping to a real D-Wave run
----------------------------
On Advantage hardware:
  * The Trotter steps go away — D-Wave does continuous-time evolution.
  * The transverse field comes from the global A(s) ramp, not per-qubit Rx.
  * Z and ZZ "expectations" are replaced by shot-based estimates from N samples.
  * h_i and J_ij must respect Pegasus connectivity and the [-2, 2] / [-1, 1] ranges.

The qualitative dynamics — random Ising reservoir driven by input biases —
should carry over. This module is the cheapest way to find out whether the
encoding scheme (h_i instead of Rx) is competitive on our task.
"""

from __future__ import annotations

import numpy as np
from typing import Literal

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .reservoir import _observables_from_sv


class AnnealerReservoir:
    """
    Trotterized transverse-field Ising reservoir.

    Parameters
    ----------
    n_qubits        : reservoir size (statevector sim caps at ~14 in practice).
    n_input         : number of qubits whose h_i is set by the input data.
                      The remaining n_qubits - n_input qubits act as "memory"
                      qubits with small fixed random h_i.
    evolution_time  : total evolution time t (dimensionless; arbitrary scale
                      absorbed into h_i and J_ij magnitudes).
    trotter_steps   : Trotter order m. 2-4 is plenty for short t.
    connectivity    : "linear" / "random" / "all-to-all" — graph of J_ij.
    density         : edge fraction used when connectivity="random".
    h_memory_scale  : magnitude of random h_i on memory qubits.
    input_scale     : multiplier applied to data-driven h_i. Probes the
                      dynamical regime: small => transverse-field dominates;
                      large => problem Hamiltonian dominates. The interesting
                      "edge of chaos" lies near input_scale ~ 1 (Martinez-Pena
                      et al. 2021).
    seed            : RNG seed for J_ij and memory-qubit biases.
    """

    def __init__(
        self,
        n_qubits: int = 10,
        n_input: int | None = None,
        evolution_time: float = 1.0,
        trotter_steps: int = 3,
        connectivity: Literal["linear", "random", "all-to-all"] = "random",
        density: float = 0.4,
        h_memory_scale: float = 0.5,
        input_scale: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_input = n_input if n_input is not None else n_qubits
        if not (0 < self.n_input <= n_qubits):
            raise ValueError("n_input must be in (0, n_qubits].")
        self.evolution_time = float(evolution_time)
        self.trotter_steps = int(trotter_steps)
        self.connectivity = connectivity
        self.density = density
        self.h_memory_scale = float(h_memory_scale)
        self.input_scale = float(input_scale)
        self.seed = seed

        rng = np.random.RandomState(seed)
        self._edges = self._build_edges(rng)
        # Fixed random ZZ couplings J_ij in [-1, 1]
        self._J = rng.uniform(-1.0, 1.0, len(self._edges) or 1)
        # Fixed random bias on memory qubits (input qubits get h from data)
        n_mem = n_qubits - self.n_input
        self._h_memory = rng.uniform(-h_memory_scale, h_memory_scale, n_mem) if n_mem > 0 else np.array([])

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def _build_edges(self, rng: np.random.RandomState) -> list[tuple[int, int]]:
        n = self.n_qubits
        if self.connectivity == "linear":
            return [(i, i + 1) for i in range(n - 1)]
        if self.connectivity == "all-to-all":
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        # random
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        k = max(n - 1, int(len(all_pairs) * self.density))
        idxs = rng.choice(len(all_pairs), min(k, len(all_pairs)), replace=False)
        return [all_pairs[i] for i in sorted(idxs)]

    # ------------------------------------------------------------------
    # Circuit (Trotterized exp(-iHt) on |+>^n)
    # ------------------------------------------------------------------

    def _h_vector(self, x: np.ndarray) -> np.ndarray:
        """Combine data-driven h_i (first n_input qubits) and fixed memory biases."""
        h = np.zeros(self.n_qubits, dtype=float)
        for i in range(self.n_input):
            h[i] = self.input_scale * float(x[i % len(x)])
        if self._h_memory.size > 0:
            h[self.n_input:] = self._h_memory
        return h

    def _build_circuit(self, x: np.ndarray) -> QuantumCircuit:
        n = self.n_qubits
        qc = QuantumCircuit(n)

        # |+>^n  (the s=0 ground state of -sum sigma^x_i)
        for i in range(n):
            qc.h(i)

        h_vec = self._h_vector(x)
        dt = self.evolution_time / self.trotter_steps

        # Trotter steps: alternate exp(-i H_X dt) and exp(-i H_Z dt)
        #   H_X = -sum sigma^x_i  ->  exp(-i (-X) dt) per qubit  =  Rx(-2 dt)
        #   H_Z = sum h_i sigma^z + sum J_ij sigma^z sigma^z
        #       exp(-i h_i Z dt)            ->  Rz(2 h_i dt)
        #       exp(-i J_ij Z_i Z_j dt)     ->  CX(i,j) Rz(2 J_ij dt) CX(i,j)
        for _ in range(self.trotter_steps):
            for i in range(n):
                qc.rx(-2.0 * dt, i)
            for i in range(n):
                if h_vec[i] != 0.0:
                    qc.rz(2.0 * h_vec[i] * dt, i)
            for (qi, qj), Jij in zip(self._edges, self._J):
                qc.cx(qi, qj)
                qc.rz(2.0 * Jij * dt, qj)
                qc.cx(qi, qj)

        return qc

    # ------------------------------------------------------------------
    # Public API — mirrors QuantumReservoir
    # ------------------------------------------------------------------

    @property
    def n_features(self) -> int:
        n = self.n_qubits
        return n + n * (n - 1) // 2

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Map input matrix X -> reservoir feature matrix.

        Inputs are tanh-normalised so h_i lands in [-1, 1] (a reasonable proxy
        for D-Wave's native h range of [-2, 2] after our internal scaling).
        """
        X_norm = np.tanh(X)
        feats = []
        for x in X_norm:
            qc = self._build_circuit(x)
            sv = Statevector(qc)
            feats.append(_observables_from_sv(sv, self.n_qubits))
        return np.array(feats)

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.transform(X)

    def __repr__(self) -> str:
        return (
            f"AnnealerReservoir(n_qubits={self.n_qubits}, n_input={self.n_input}, "
            f"t={self.evolution_time}, m={self.trotter_steps}, "
            f"input_scale={self.input_scale}, "
            f"conn='{self.connectivity}', n_features={self.n_features})"
        )
