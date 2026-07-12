__all__ = [
    "QuantumReservoir",
    "depolarizing_noise",
    "amplitude_damping_noise",
    "combined_noise",
]


def __getattr__(name):
    if name == "QuantumReservoir":
        from .reservoir import QuantumReservoir
        return QuantumReservoir
    if name in {"depolarizing_noise", "amplitude_damping_noise", "combined_noise"}:
        from . import noise
        return getattr(noise, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
