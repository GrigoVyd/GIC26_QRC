"""
Data loaders for all GIC 2026 QRC tracks.

  load_mnist_digits    — common benchmark (classification)
  load_financial_data  — track 1: SPY volatility forecasting (regression)
  load_narma           — standard RC benchmark (NARMA-k regression)
  load_lorenz          — synthetic chaotic time series (weather proxy)
  load_noaa_weather    — track 2: real weather data via meteostat (optional)

Each loader returns (X_train, X_test, y_train, y_test) for classification,
or (X, y) for time-series tasks (caller splits as needed).
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# MNIST digits  (common benchmark — classification)
# ---------------------------------------------------------------------------

def load_mnist_digits(
    n_components: int = 8,
    n_samples: int | None = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load sklearn's digits dataset (8×8 images, 10 classes, 1797 samples).

    Pipeline: flatten → StandardScaler → PCA(n_components) → scale to [-1,1].

    Returns
    -------
    X_train, X_test : (n_train, n_components), (n_test, n_components)
    y_train, y_test : integer class labels
    """
    digits = load_digits()
    X, y = digits.data.astype(np.float64), digits.target

    if n_samples is not None:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), min(n_samples, len(X)), replace=False)
        X, y = X[idx], y[idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    pca = PCA(n_components=n_components, random_state=seed)
    X_tr = pca.fit_transform(X_tr)
    X_te = pca.transform(X_te)

    # Normalise to [-1, 1] per feature (based on training set)
    max_abs = np.abs(X_tr).max(axis=0, keepdims=True) + 1e-8
    X_tr /= max_abs
    X_te /= max_abs

    return X_tr, X_te, y_tr, y_te


# ---------------------------------------------------------------------------
# Financial volatility  (track 1 — regression)
# ---------------------------------------------------------------------------

def load_financial_data(
    ticker: str = "SPY",
    start: str = "2010-01-01",
    end: str = "2024-12-31",
    vol_window: int = 21,
    delay: int = 5,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Download SPY daily prices via yfinance and build a volatility prediction task.

    Target  : next-day realized volatility (annualised, 21-day rolling std).
    Features: delay-embedded log-returns [r(t-delay+1), …, r(t)].

    Falls back to a synthetic GARCH-like process if yfinance is unavailable.
    """
    try:
        import yfinance as yf

        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        prices = raw["Close"].squeeze().dropna().values

        if len(prices) < vol_window + delay + 10:
            raise ValueError("Too few data points.")

        log_ret = np.diff(np.log(prices + 1e-10))

        # Realized volatility: rolling std × sqrt(252)
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
    return _split(X, y, test_size)


def _synthetic_garch(
    n: int = 3000,
    vol_window: int = 21,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple GARCH(1,1) simulation as financial data fallback."""
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


# ---------------------------------------------------------------------------
# NARMA  (standard reservoir computing benchmark — regression)
# ---------------------------------------------------------------------------

def load_narma(
    order: int = 5,
    n_samples: int = 2000,
    warmup: int = 100,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a NARMA-k time series and return a regression dataset.

    NARMA-k:
      y(t) = 0.3 y(t-1)
             + 0.05 y(t-1) Σ_{i=1}^{k} y(t-i)
             + 1.5 u(t-k) u(t-1)
             + 0.1

    u ~ Uniform(0, 0.5). Features: delay-embedded inputs u(t-delay:t).
    Target: y(t).
    """
    rng = np.random.RandomState(seed)
    total = n_samples + warmup + order + 1
    u = rng.uniform(0, 0.5, total)
    y = np.zeros(total)

    for t in range(order, total):
        y[t] = (
            0.3 * y[t - 1]
            + 0.05 * y[t - 1] * sum(y[t - i] for i in range(1, order + 1))
            + 1.5 * u[t - order] * u[t - 1]
            + 0.1
        )

    u = u[warmup:]
    y = y[warmup:]

    X, tgt = _delay_embed(u, y, delay=order)
    return _split(X[:n_samples], tgt[:n_samples], test_size)


# ---------------------------------------------------------------------------
# Lorenz attractor  (track 2 weather proxy — regression)
# ---------------------------------------------------------------------------

def load_lorenz(
    n_samples: int = 2000,
    delay: int = 5,
    variable: int = 0,
    noise_level: float = 0.02,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate the Lorenz-63 system and return a one-step-ahead prediction task.

    Used as weather proxy when real NOAA data is unavailable.
    variable : 0=x, 1=y, 2=z.
    """
    from scipy.integrate import solve_ivp

    def lorenz(t, s, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
        x, y, z = s
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    rng = np.random.RandomState(seed)
    ic = rng.randn(3)
    total_t = n_samples + delay + 200          # extra for transient
    sol = solve_ivp(
        lorenz,
        [0, total_t * 0.02],
        ic,
        t_eval=np.linspace(4.0, total_t * 0.02, n_samples + delay + 100),
        max_step=0.01,
        method="RK45",
    )
    series = sol.y[variable]
    series = (series - series.mean()) / (series.std() + 1e-8)
    series += rng.randn(len(series)) * noise_level

    X, y = _delay_embed(series, series, delay)
    # Target is one step ahead
    X = X[:-1]
    y = series[delay + 1: delay + 1 + len(X)]
    return _split(X[:n_samples], y[:n_samples], test_size)


# ---------------------------------------------------------------------------
# NOAA / meteostat  (track 2 — optional real weather)
# ---------------------------------------------------------------------------

def load_noaa_weather(
    lat: float = 40.6413,
    lon: float = -73.7781,
    alt: float = 4.0,
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    delay: int = 5,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Download daily average temperature from meteostat (JFK default).

    Falls back to Lorenz system if meteostat is not installed or unavailable.
    """
    try:
        from meteostat import Point, Daily
        from datetime import datetime

        loc = Point(lat, lon, alt)
        s = datetime.strptime(start, "%Y-%m-%d")
        e = datetime.strptime(end, "%Y-%m-%d")
        data = Daily(loc, s, e).fetch()
        temps = data["tavg"].dropna().values

        if len(temps) < delay + 50:
            raise ValueError("Insufficient weather records.")

        temps = (temps - temps.mean()) / (temps.std() + 1e-8)
        X, y = _delay_embed(temps, temps, delay)
        X = X[:-1]
        y = temps[delay + 1: delay + 1 + len(X)]
        return _split(X, y, test_size)

    except Exception as exc:
        warnings.warn(
            f"meteostat unavailable ({exc}). Using Lorenz system as weather proxy.",
            stacklevel=2,
        )
        return load_lorenz(delay=delay, test_size=test_size)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

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
