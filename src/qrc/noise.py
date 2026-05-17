"""
Pre-built Qiskit Aer noise models for benchmarking QRC under realistic noise.

Three regimes are provided:
  - depolarizing_noise  : uniform depolarizing error on all gates
  - amplitude_damping_noise : T1 relaxation (amplitude damping)
  - combined_noise      : depolarizing + amplitude damping (most realistic)
"""

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
)


def depolarizing_noise(
    p_single: float = 0.001,
    p_two: float = 0.01,
) -> NoiseModel:
    """
    Depolarizing noise on single- and two-qubit gates.

    Typical NISQ values: p_single ≈ 0.001, p_two ≈ 0.01.
    """
    nm = NoiseModel()
    err_1q = depolarizing_error(p_single, 1)
    err_2q = depolarizing_error(p_two, 2)
    nm.add_all_qubit_quantum_error(err_1q, ["rx", "ry", "rz", "h"])
    nm.add_all_qubit_quantum_error(err_2q, ["cx"])
    return nm


def amplitude_damping_noise(gamma: float = 0.01) -> NoiseModel:
    """
    Amplitude damping (T1 relaxation) on single-qubit gates.

    gamma ≈ gate_time / T1. For 50-ns gate and 50-µs T1: gamma ≈ 0.001.
    """
    nm = NoiseModel()
    err = amplitude_damping_error(gamma)
    nm.add_all_qubit_quantum_error(err, ["rx", "ry", "rz", "h"])
    return nm


def combined_noise(
    p_depol: float = 0.001,
    gamma: float = 0.005,
) -> NoiseModel:
    """
    Combined depolarizing + amplitude damping — closest to real hardware.

    Two-qubit depolarizing is set to 10× single-qubit rate (typical ratio).
    """
    nm = NoiseModel()
    err_dep_1q = depolarizing_error(p_depol, 1)
    err_dep_2q = depolarizing_error(min(p_depol * 10, 0.999), 2)
    err_amp = amplitude_damping_error(gamma)

    # Compose: depolarizing then amplitude damping on 1-qubit gates
    err_1q = err_dep_1q.compose(err_amp)

    nm.add_all_qubit_quantum_error(err_1q, ["rx", "ry", "rz", "h"])
    nm.add_all_qubit_quantum_error(err_dep_2q, ["cx"])
    return nm


# Convenience table for sweep experiments
NOISE_LEVELS = {
    "none":   None,
    "low":    depolarizing_noise(p_single=0.001,  p_two=0.005),
    "medium": depolarizing_noise(p_single=0.005,  p_two=0.02),
    "high":   depolarizing_noise(p_single=0.01,   p_two=0.05),
    "combined_low":  combined_noise(p_depol=0.001, gamma=0.002),
    "combined_high": combined_noise(p_depol=0.005, gamma=0.01),
}
