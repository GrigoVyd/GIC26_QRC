"""
Strategy A — Ising ZZ reservoir.

Architecture per layer:
  Rx(π·xᵢ) ∀i  →  [CX – Rz(w_zz) – CX] per edge  →  Rx Ry Rz local rotations

This is the baseline strategy, identical in circuit structure to the existing
QuantumReservoir class. Implemented here as a QRCStrategy for fair tournament
comparison.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from .base import QRCStrategy


class IsingStrategy(QRCStrategy):

    label = "Ising ZZ"

    def _init_weights(self) -> None:
        n, L = self.n_qubits, self.n_layers
        self._edges = self._random_edges()
        # local rotation angles  (L, n, 3)
        self._local_w = self._rng.uniform(0, 2 * np.pi, (L, n, 3))
        # ZZ coupling per layer per edge  (L, n_edges)
        ne = len(self._edges) or 1
        self._zz_w = self._rng.uniform(0, np.pi, (L, ne))

    def _random_edges(self) -> list[tuple[int, int]]:
        n = self.n_qubits
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        k = max(n - 1, int(len(all_pairs) * 0.4))
        idxs = self._rng.choice(len(all_pairs), min(k, len(all_pairs)), replace=False)
        return [all_pairs[i] for i in sorted(idxs)]

    def build_circuit(self, x: np.ndarray) -> QuantumCircuit:
        n = self.n_qubits
        qc = QuantumCircuit(n)
        for i in range(n):
            qc.h(i)
        for layer in range(self.n_layers):
            # --- angle encoding ---
            for i in range(n):
                qc.rx(float(x[i % len(x)]) * np.pi, i)
            # --- ZZ Ising couplings ---
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
