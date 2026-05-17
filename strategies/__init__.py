from .base import QRCStrategy
from .strategy_ising import IsingStrategy
from .strategy_heisenberg import HeisenbergStrategy
from .strategy_iqp import IQPStrategy

STRATEGIES: dict[str, type[QRCStrategy]] = {
    "ising":       IsingStrategy,
    "heisenberg":  HeisenbergStrategy,
    "iqp":         IQPStrategy,
}

__all__ = ["QRCStrategy", "IsingStrategy", "HeisenbergStrategy", "IQPStrategy", "STRATEGIES"]
