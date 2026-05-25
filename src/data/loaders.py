"""
Financial volatility data loader for GIC 2026 QRC — Track 1.

  load_financial_data    — SPY next-day realized volatility (regression)
  load_financial_data_v2 — same task, with HAR-RV features, log-vol target,
                           and prior-RV series for a fair persistence baseline

Returns (X_train, X_test, y_train, y_test) with chronological split.
"""

from __future__ import annotations

import warnings
from typing import Tuple, Dict, Any, Optional

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


# ---------------------------------------------------------------------------
# v2 loader  — HAR-RV features + log-vol target + prior-RV (Persistence)
# ---------------------------------------------------------------------------

def load_financial_data_v2(
    ticker: str = "SPY",
    start: str = "2010-01-01",
    end: str = "2024-12-31",
    vol_window: int = 21,
    delay: int = 5,
    test_size: float = 0.2,
    log_target: bool = True,
    include_har: bool = True,
    include_log_har: bool = False,
    include_return_features: bool = True,
    residual_target: bool = False,
) -> Dict[str, Any]:
    """
    SPY next-day realized-vol forecast with stronger conditioning information.

    Compared to ``load_financial_data``, this returns a dict so we can also
    pass through (a) the prior-RV series (for a fair Persistence baseline) and
    (b) the target-transform metadata so metrics can be reported on the natural
    vol scale even when the model is trained on log-vol.

    Features per sample (causally available at forecast time τ predicting RV[τ]):
      • delay log-returns  : r[τ-δ..τ-1]                   (δ features)
      • delay |returns|    : |r[τ-δ..τ-1]|                 (δ features, if include_return_features)
      • delay r²            : r²[τ-δ..τ-1]                  (δ features, if include_return_features)
      • HAR-RV daily        : RV[τ-1]                       (1 feature, if include_har)
      • HAR-RV weekly       : mean(RV[τ-5..τ-1])            (1 feature, if include_har)
      • HAR-RV monthly      : mean(RV[τ-22..τ-1])           (1 feature, if include_har)

    Returns a dict with:
      X_train, X_test, y_train, y_test    — features and (transformed) targets
      y_train_raw, y_test_raw             — RV on natural (annualised vol) scale
      persistence_train, persistence_test — RV[τ-1] for naive baseline
      target_transform                    — "log" or None
      feature_names                       — list of feature column names
    """
    try:
        import yfinance as yf

        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        prices = raw["Close"].squeeze().dropna().values

        if len(prices) < vol_window + delay + 30:
            raise ValueError("Too few data points.")

        log_ret_full = np.diff(np.log(prices + 1e-10))  # length = T-1

    except Exception as exc:
        warnings.warn(
            f"yfinance download failed ({exc}). Using synthetic GARCH data.", stacklevel=2
        )
        log_ret_full, _ = _synthetic_garch(n=3000, vol_window=vol_window, seed=0)
        # _synthetic_garch already truncated the warmup; we need the raw returns here:
        log_ret_full = _synthetic_garch_raw(n=3000, seed=0)

    # Realized volatility (annualised, vol_window-day rolling std)
    # rv_full[i] uses log_ret_full[i - vol_window + 1 : i + 1]
    rv_full = (
        pd.Series(log_ret_full).rolling(vol_window).std().values * np.sqrt(252)
    )

    # We can only build a sample for index τ if all features are causally available.
    # Need: log_ret[τ-δ..τ-1] (=> τ >= δ)
    #       RV[τ-1] (=> rv_full[τ-1] not NaN => τ-1 >= vol_window - 1 => τ >= vol_window)
    #       HAR monthly mean RV[τ-22..τ-1] (=> τ-22 >= vol_window - 1 => τ >= vol_window + 21)
    #       RV[τ] (target) not NaN => τ >= vol_window - 1
    min_tau = max(delay, vol_window + (21 if include_har else 0))
    max_tau = len(log_ret_full)  # exclusive

    rows = []
    targets = []
    persistence = []
    for tau in range(min_tau, max_tau):
        feats = []

        if include_return_features:
            r_win = log_ret_full[tau - delay : tau]
            feats.append(r_win)
            feats.append(np.abs(r_win))
            feats.append(r_win ** 2)

        if include_har:
            rv_prev = rv_full[tau - 1]
            rv_w5 = np.mean(rv_full[tau - 5 : tau])
            rv_w22 = np.mean(rv_full[tau - 22 : tau])
            feats.append(np.array([rv_prev, rv_w5, rv_w22]))

        if include_log_har:
            log_rv_prev = np.log(rv_full[tau - 1] + 1e-10)
            log_rv_w5 = np.mean(np.log(rv_full[tau - 5 : tau] + 1e-10))
            log_rv_w22 = np.mean(np.log(rv_full[tau - 22 : tau] + 1e-10))
            feats.append(np.array([log_rv_prev, log_rv_w5, log_rv_w22]))

        rows.append(np.concatenate(feats))
        targets.append(rv_full[tau])
        persistence.append(rv_full[tau - 1])

    X = np.array(rows)
    y_raw = np.array(targets)
    pers = np.array(persistence)

    # Build feature names for diagnostics
    feature_names: list[str] = []
    if include_return_features:
        feature_names += [f"r_lag{delay - i}" for i in range(delay)]
        feature_names += [f"abs_r_lag{delay - i}" for i in range(delay)]
        feature_names += [f"r2_lag{delay - i}" for i in range(delay)]
    if include_har:
        feature_names += ["har_rv_d", "har_rv_w", "har_rv_m"]
    if include_log_har:
        feature_names += ["log_har_rv_d", "log_har_rv_w", "log_har_rv_m"]

    # Target choice
    #   residual_target=True  →  y = log(RV[t]) - log(RV[t-1])  (forces log_target=True semantics)
    #   log_target=True       →  y = log(RV[t])
    #   else                  →  y = RV[t]
    log_pers = np.log(pers + 1e-10)
    if residual_target:
        y = np.log(y_raw + 1e-10) - log_pers
        transform_label = "residual_log"
    elif log_target:
        y = np.log(y_raw + 1e-10)
        transform_label = "log"
    else:
        y = y_raw
        transform_label = None

    n_test = max(1, int(len(X) * test_size))
    n_train = len(X) - n_test

    return {
        "X_train": X[:n_train],
        "X_test": X[n_train:],
        "y_train": y[:n_train],
        "y_test": y[n_train:],
        "y_train_raw": y_raw[:n_train],
        "y_test_raw": y_raw[n_train:],
        "persistence_train": pers[:n_train],
        "persistence_test": pers[n_train:],
        "log_persistence_train": log_pers[:n_train],
        "log_persistence_test": log_pers[n_train:],
        "target_transform": transform_label,
        "feature_names": feature_names,
    }


def _synthetic_garch_raw(n: int = 3000, seed: int = 42) -> np.ndarray:
    """GARCH(1,1) raw log-return series (no warmup truncation)."""
    rng = np.random.RandomState(seed)
    omega, alpha, beta = 1e-4, 0.08, 0.90
    h = np.zeros(n + 1)
    r = np.zeros(n + 1)
    h[0] = omega / (1 - alpha - beta)
    for t in range(1, n + 1):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
        r[t] = np.sqrt(h[t]) * rng.randn()
    return r[1:]


def invert_target(
    y_transformed: np.ndarray,
    target_transform: Optional[str],
    log_persistence: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Invert the target transform so metrics can be computed on the natural vol scale.

    Modes:
      • None            — y already on vol scale
      • "log"           — y is log(RV);          vol = exp(y)
      • "residual_log"  — y is log(RV[t]) - log(RV[t-1]);
                          vol = exp(log_persistence + y) = RV[t-1] * exp(y)
                          (requires log_persistence aligned to the same samples)
    """
    if target_transform == "log":
        return np.exp(y_transformed)
    if target_transform == "residual_log":
        if log_persistence is None:
            raise ValueError("residual_log inversion requires log_persistence array.")
        return np.exp(log_persistence + y_transformed)
    return y_transformed
