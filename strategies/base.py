"""
QRCStrategy — abstract base class for all reservoir strategies.

Each strategy owns its own circuit-building logic but shares the same
readout pipeline: transform(X) → StandardScaler → Ridge regression.

Subclasses must implement:
  - label : str                  short display name
  - build_circuit(x, n)          returns a QuantumCircuit (no measurements)
  - n_features(n_qubits) → int   reservoir feature dimensionality
"""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit import transpile


def _observables_vectorised(sv: Statevector, n: int) -> np.ndarray:
    """Z and ZZ expectation values from a statevector in one numpy pass."""
    probs = np.abs(sv.data) ** 2
    indices = np.arange(2 ** n)
    bits = ((indices[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
    signs = 1.0 - 2.0 * bits                        # +1 for |0>, -1 for |1>
    z_exp = probs @ signs
    zz = (signs * probs[:, None]).T @ signs
    zz_vals = [zz[i, j] for i in range(n) for j in range(i + 1, n)]
    return np.concatenate([z_exp, zz_vals])


def _observables_from_counts(counts: dict, n: int, n_shots: int) -> np.ndarray:
    z_acc = np.zeros(n)
    zz_acc = np.zeros((n, n))
    for bitstr, count in counts.items():
        p = count / n_shots
        bits = np.array([int(b) for b in reversed(bitstr)], dtype=np.float64)
        signs = 1.0 - 2.0 * bits
        z_acc += signs * p
        zz_acc += np.outer(signs, signs) * p
    zz_vals = [zz_acc[i, j] for i in range(n) for j in range(i + 1, n)]
    return np.concatenate([z_acc, zz_vals])


class QRCStrategy(abc.ABC):
    """
    Abstract base for QRC architecture strategies.

    Parameters
    ----------
    n_qubits    : reservoir size
    n_layers    : encoding + interaction repetitions
    seed        : reproducibility seed
    noise_model : Qiskit Aer NoiseModel (None → exact statevector)
    n_shots     : shots for noisy simulation
    """

    def __init__(
        self,
        n_qubits: int = 5,
        n_layers: int = 2,
        seed: int = 42,
        noise_model: Optional[NoiseModel] = None,
        n_shots: int = 2048,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.seed = seed
        self.noise_model = noise_model
        self.n_shots = n_shots
        self._rng = np.random.RandomState(seed)
        self._init_weights()

    def _init_weights(self) -> None:
        """Subclasses may override to initialise additional random weights."""

    @property
    @abc.abstractmethod
    def label(self) -> str:
        """Short display name, e.g. 'Ising ZZ'."""

    @abc.abstractmethod
    def build_circuit(self, x: np.ndarray) -> QuantumCircuit:
        """
        Build the full QRC circuit for input vector x (already normalised).
        Must NOT include measurement instructions.
        """

    @property
    def reservoir_dim(self) -> int:
        n = self.n_qubits
        return n + n * (n - 1) // 2

    # ------------------------------------------------------------------
    # Feature extraction (shared by all strategies)
    # ------------------------------------------------------------------

    def _features_noiseless(self, x: np.ndarray) -> np.ndarray:
        sv = Statevector(self.build_circuit(x))
        return _observables_vectorised(sv, self.n_qubits)

    def _features_noisy(self, x: np.ndarray, backend: AerSimulator) -> np.ndarray:
        qc = self.build_circuit(x)
        qc.measure_all()
        job = backend.run(
            transpile(qc, backend),
            shots=self.n_shots,
            noise_model=self.noise_model,
        )
        return _observables_from_counts(
            job.result().get_counts(), self.n_qubits, self.n_shots
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map (n_samples, n_input) → (n_samples, reservoir_dim)."""
        X_n = np.tanh(X)
        if self.noise_model is not None:
            backend = AerSimulator()
            return np.array([self._features_noisy(x, backend) for x in X_n])
        return np.array([self._features_noiseless(x) for x in X_n])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(label='{self.label}', "
            f"n_qubits={self.n_qubits}, n_layers={self.n_layers})"
        )
