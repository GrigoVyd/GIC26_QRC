"""
Financial volatility data loader for GIC 2026 QRC — Track 1.

  load_financial_data  — SPY next-day realized volatility (regression)

Returns (X_train, X_test, y_train, y_test) with chronological split.
Features are delay-embedded log-returns; optionally extended with
absolute and squared returns for richer GARCH-like signal.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import pandas as pd


def load_financial_data(
    ticker: str = "SPY",
    start: str = "2010-01-01",
    end: str = "2024-12-31",
    vol_window: int = 21,
    delay: int = 5,
    test_size: float = 0.2,
    enhanced_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Download SPY daily prices via yfinance and build a volatility prediction task.

    Target  : next-day realized volatility (annualised, vol_window-day rolling std).
    Features: delay-embedded log-returns [r(t-delay+1), …, r(t)].
    If enhanced_features=True, also appends |r| and r² delay-embedded windows,
    giving the model stronger GARCH-like signal (3×delay total features).

    Falls back to a synthetic GARCH(1,1) process if yfinance is unavailable.
    """
    try:
        import yfinance as yf

        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        prices = raw["Close"].squeeze().dropna().values

        if len(prices) < vol_window + delay + 10:
            raise ValueError("Too few data points.")

        log_ret = np.diff(np.log(prices + 1e-10))

        rv = (
            pd.Series(log_ret)
            .rolling(vol_window)
            .std()
            .values[vol_window:] * np.sqrt(252)
        )
        ret = log_ret[vol_window:]

    except Exception as exc:
        warnings.warn(
            f"yfinance download failed ({exc}). Using synthetic GARCH data.", stacklevel=2
        )
        ret, rv = _synthetic_garch(n=3000, vol_window=vol_window, seed=0)

    X, y = _delay_embed(ret, rv, delay)

    if enhanced_features:
        X_abs, _ = _delay_embed(np.abs(ret), rv, delay)
        X_sq, _ = _delay_embed(ret ** 2, rv, delay)
        min_len = min(len(X), len(X_abs), len(X_sq))
        X = np.hstack([X[:min_len], X_abs[:min_len], X_sq[:min_len]])
        y = y[:min_len]

    return _split(X, y, test_size)


def _synthetic_garch(
    n: int = 3000,
    vol_window: int = 21,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """GARCH(1,1) simulation used as financial data fallback."""
    rng = np.random.RandomState(seed)
    omega, alpha, beta = 1e-4, 0.08, 0.90
    h = np.zeros(n + 1)
    r = np.zeros(n + 1)
    h[0] = omega / (1 - alpha - beta)
    for t in range(1, n + 1):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
        r[t] = np.sqrt(h[t]) * rng.randn()

    log_ret = r[1:]
    rv = (
        pd.Series(log_ret)
        .rolling(vol_window)
        .std()
        .values[vol_window:] * np.sqrt(252)
    )
    return log_ret[vol_window:], rv


def _delay_embed(
    u: np.ndarray, target: np.ndarray, delay: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Build delay-embedded input matrix and corresponding targets."""
    X = np.array([u[t - delay: t] for t in range(delay, len(u))])
    y = target[delay:]
    min_len = min(len(X), len(y))
    return X[:min_len], y[:min_len]


def _split(
    X: np.ndarray, y: np.ndarray, test_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronological split (no shuffle) to avoid data leakage in time series."""
    n_test = max(1, int(len(X) * test_size))
    n_train = len(X) - n_test
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]
