"""
Strategy B — Heisenberg XX+YY+ZZ reservoir.

Effective Hamiltonian per edge:
  H = Jx·XX + Jy·YY + Jz·ZZ

Gate decompositions
-------------------
  ZZ:  CX – Rz(wz) – CX
  XX:  Ry(π/2)⊗Ry(π/2) – CX – Rx(wx) – CX – Ry(-π/2)⊗Ry(-π/2)
  YY:  Rx(-π/2)⊗Rx(-π/2) – CX – Ry(wy) – CX – Rx(π/2)⊗Rx(π/2)

Advantage over Ising: richer entanglement structure, captures XY-plane
correlations that Ising misses.
Trade-off: 3× more two-qubit gates → more noise-sensitive (relevant for
Phase 3 hardware discussion).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from .base import QRCStrategy


class HeisenbergStrategy(QRCStrategy):

    label = "Heisenberg XX+YY+ZZ"

    def _init_weights(self) -> None:
        n, L = self.n_qubits, self.n_layers
        self._edges = self._random_edges()
        self._local_w = self._rng.uniform(0, 2 * np.pi, (L, n, 3))
        ne = len(self._edges) or 1
        # three coupling weights per edge: wx, wy, wz
        self._heis_w = self._rng.uniform(0, np.pi, (L, ne, 3))

    def _random_edges(self) -> list[tuple[int, int]]:
        n = self.n_qubits
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        k = max(n - 1, int(len(all_pairs) * 0.4))
        idxs = self._rng.choice(len(all_pairs), min(k, len(all_pairs)), replace=False)
        return [all_pairs[i] for i in sorted(idxs)]

    @staticmethod
    def _zz_gate(qc: QuantumCircuit, qi: int, qj: int, w: float) -> None:
        qc.cx(qi, qj)
        qc.rz(w, qj)
        qc.cx(qi, qj)

    @staticmethod
    def _xx_gate(qc: QuantumCircuit, qi: int, qj: int, w: float) -> None:
        qc.ry(np.pi / 2, qi)
        qc.ry(np.pi / 2, qj)
        qc.cx(qi, qj)
        qc.rx(w, qj)
        qc.cx(qi, qj)
        qc.ry(-np.pi / 2, qi)
        qc.ry(-np.pi / 2, qj)

    @staticmethod
    def _yy_gate(qc: QuantumCircuit, qi: int, qj: int, w: float) -> None:
        qc.rx(-np.pi / 2, qi)
        qc.rx(-np.pi / 2, qj)
        qc.cx(qi, qj)
        qc.ry(w, qj)
        qc.cx(qi, qj)
        qc.rx(np.pi / 2, qi)
        qc.rx(np.pi / 2, qj)

    def build_circuit(self, x: np.ndarray) -> QuantumCircuit:
        n = self.n_qubits
        qc = QuantumCircuit(n)
        for i in range(n):
            qc.h(i)
        for layer in range(self.n_layers):
            # --- angle encoding ---
            for i in range(n):
                qc.rx(float(x[i % len(x)]) * np.pi, i)
            # --- Heisenberg XX+YY+ZZ interactions ---
            for k, (qi, qj) in enumerate(self._edges):
                eidx = k % self._heis_w.shape[1]
                wx = float(self._heis_w[layer, eidx, 0])
                wy = float(self._heis_w[layer, eidx, 1])
                wz = float(self._heis_w[layer, eidx, 2])
                self._xx_gate(qc, qi, qj, wx)
                self._yy_gate(qc, qi, qj, wy)
                self._zz_gate(qc, qi, qj, wz)
            # --- local reservoir rotations ---
            for i in range(n):
                qc.rx(float(self._local_w[layer, i, 0]), i)
                qc.ry(float(self._local_w[layer, i, 1]), i)
                qc.rz(float(self._local_w[layer, i, 2]), i)
        return qc
