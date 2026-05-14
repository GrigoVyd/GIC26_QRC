"""
Strategy C — IQP-style encoding reservoir.

Encoding layer (applied before each Ising reservoir block):

  H^⊗n
  → Rz(π·xᵢ²) ∀i                  [quadratic single-qubit terms]
  → [CX – Rz(π·xᵢ·xⱼ) – CX] ∀ edges  [quadratic cross terms]
  → H^⊗n

This is a data re-uploading variant of the IQP (Instantaneous Quantum
Polynomial) circuit, providing a degree-2 polynomial feature map before
the Ising reservoir interacts.  The linear readout can then capture
second-order nonlinearities of the input without increasing reservoir depth.

After the IQP encoding block, the same Ising ZZ reservoir is applied.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from .base import QRCStrategy


class IQPStrategy(QRCStrategy):

    label = "IQP Encoding"

    def _init_weights(self) -> None:
        n, L = self.n_qubits, self.n_layers
        self._edges = self._random_edges()
        self._local_w = self._rng.uniform(0, 2 * np.pi, (L, n, 3))
        ne = len(self._edges) or 1
        self._zz_w = self._rng.uniform(0, np.pi, (L, ne))

    def _random_edges(self) -> list[tuple[int, int]]:
        n = self.n_qubits
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        k = max(n - 1, int(len(all_pairs) * 0.4))
        idxs = self._rng.choice(len(all_pairs), min(k, len(all_pairs)), replace=False)
        return [all_pairs[i] for i in sorted(idxs)]

    def _iqp_encoding(self, qc: QuantumCircuit, x: np.ndarray) -> None:
        """Apply one IQP feature-map block for input x."""
        n = self.n_qubits
        # H layer
        for i in range(n):
            qc.h(i)
        # Single-qubit quadratic terms: Rz(π·xᵢ²)
        for i in range(n):
            xi = float(x[i % len(x)])
            qc.rz(np.pi * xi ** 2, i)
        # Two-qubit cross terms: CX – Rz(π·xᵢ·xⱼ) – CX on encoding edges
        for qi, qj in self._edges:
            xi = float(x[qi % len(x)])
            xj = float(x[qj % len(x)])
            qc.cx(qi, qj)
            qc.rz(np.pi * xi * xj, qj)
            qc.cx(qi, qj)
        # Final H layer (closes the IQP block)
        for i in range(n):
            qc.h(i)

    def build_circuit(self, x: np.ndarray) -> QuantumCircuit:
        n = self.n_qubits
        qc = QuantumCircuit(n)
        # Initial superposition
        for i in range(n):
            qc.h(i)
        for layer in range(self.n_layers):
            # --- IQP degree-2 encoding block ---
            self._iqp_encoding(qc, x)
            # --- Ising ZZ reservoir interactions ---
            for k, (qi, qj) in enumerate(self._edges):
                w = float(self._zz_w[layer, k % self._zz_w.shape[1]])
                qc.cx(qi, qj)
                qc.rz(w, qj)
                qc.cx(qi, qj)
            # --- local reservoir rotations ---
            for i in range(n):
                qc.rx(float(self._local_w[layer, i, 0]), i)
                qc.ry(float(self._local_w[layer, i, 1]), i)
                qc.rz(float(self._local_w[layer, i, 2]), i)
        return qc
