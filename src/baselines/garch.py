"""
GARCH(1,1) baseline + helper for hybrid feature/target augmentation.

This module exports:
  * GARCHBaseline      — standalone forecaster (used as a baseline)
  * fit_garch_proxy_series — returns the GARCH-based RV proxy for every
    sample in a design matrix, fitted on the training portion only. Used by
    load_financial_data_v2 to inject the proxy as a feature (`garch_proxy_rv`)
    or to redefine the target as a residual against GARCH.

The GARCH(1,1) model of Bollerslev (1986) is the canonical econometric
volatility benchmark. It models conditional variance of daily log returns:

    sigma^2(t) = omega + alpha * r(t-1)^2 + beta * sigma^2(t-1)

GARCH naturally forecasts next-day variance of returns, not next-day
realized-volatility (RV[t+1] in our problem). To compare on the same scale
as our target, we use the standard rolling-window proxy:

    predicted_RV^2[t+1] = (sum of past 20 days returns^2 + sigma_garch^2(t+1)) / 21
    predicted_RV[t+1]   = sqrt(predicted_RV^2[t+1] * 252)   # annualised

This treats GARCH as a one-step variance forecaster whose forecast replaces
the unknown ret^2(t+1) inside the realized-volatility definition. Twenty of
the 21 terms in the RV window are known returns; only one term is the
GARCH forecast. The result is therefore close to Persistence but with the
GARCH-specific structural correction.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd


class GARCHBaseline:
    """
    GARCH(1,1) baseline for next-day realized-volatility prediction.

    Uses the `arch` package's `arch_model` with constant mean, GARCH(1,1)
    variance, Normal residuals (the standard textbook spec).

    Workflow:
      fit(X, y) — fits GARCH on the training log-return series; y is ignored.
                  The log-return series must be passed via set_returns() OR
                  reconstructed from the delay-embedded X (we take the last
                  column of X as ret[t-1]).
      predict(X) — for each test sample with returns ret[t-20..t-1] (taken
                   from the X feature matrix), produces predicted RV[t+1] via
                   the rolling-window proxy described above.

    Parameters
    ----------
    rescale     : multiply returns by this before GARCH fit (the `arch`
                  package warns when daily log returns are very small;
                  rescaling avoids numerical issues). 100.0 is the common
                  choice (percentage returns).
    vol_window  : rolling window used to form the realized-volatility target
                  (must match the data loader's vol_window; default 21).
    """

    def __init__(self, rescale: float = 100.0, vol_window: int = 21) -> None:
        self.rescale = float(rescale)
        self.vol_window = int(vol_window)
        self._train_returns: np.ndarray | None = None
        self._garch_result = None

    def set_returns(self, log_ret_full: np.ndarray, n_train: int) -> "GARCHBaseline":
        """
        Provide the full log-return series. n_train marks the train/test
        boundary in the *post-rolling-window* aligned space used by the loader.
        """
        self._train_returns = np.asarray(log_ret_full[:n_train], dtype=float)
        self._full_returns = np.asarray(log_ret_full, dtype=float)
        self._n_train = int(n_train)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GARCHBaseline":
        from arch import arch_model

        if self._train_returns is None:
            # Fallback: reconstruct returns from delay-embedded X
            # X[:, 0..4] are r_lag5..r_lag1 in the v2 loader
            ret = np.concatenate([X[0, :5], X[1:, 4]])
            self._train_returns = ret

        scaled = self._train_returns * self.rescale
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(scaled, mean="Constant", vol="GARCH", p=1, q=1,
                                dist="normal", rescale=False)
            self._garch_result = model.fit(disp="off", show_warning=False)
        return self

    @staticmethod
    def fit_proxy_series(
        log_ret_full: np.ndarray,
        n_train_returns: int,
        vol_window: int = 21,
        rescale: float = 100.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit GARCH(1,1) on the training portion of ``log_ret_full`` and return,
        for every index t in [vol_window, len(log_ret_full)):
            sigma2_path[t]  — GARCH-forecast next-day variance at index t,
                              using the recursion seeded from the training fit
            proxy_rv[t]     — predicted realized vol = sqrt( ((sum_{i=t-vw+1}^{t-1} r_i^2)
                              + sigma2_path[t]) / vol_window * 252 )

        Only training returns are used to fit GARCH parameters; the recursion
        is then rolled forward through the entire series. No look-ahead leak.

        Parameters
        ----------
        log_ret_full     : (T,) log-return series
        n_train_returns  : number of returns that count as "training" (the
                            first n_train_returns elements of log_ret_full)
        vol_window       : rolling window for the proxy (must match loader)
        rescale          : numerical-stability multiplier for arch_model

        Returns
        -------
        sigma2_path : (T,) per-step next-day variance forecast (log-return scale)
        proxy_rv    : (T,) RV proxy on the original (vol) scale
                       NaN for the first (vol_window-1) entries.
        """
        from arch import arch_model

        train_returns = log_ret_full[:n_train_returns]
        scaled = train_returns * rescale
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(scaled, mean="Constant", vol="GARCH", p=1, q=1,
                              dist="normal", rescale=False).fit(disp="off",
                                                                show_warning=False)
        params = res.params
        omega = float(params["omega"])
        alpha = float(params["alpha[1]"])
        beta = float(params["beta[1]"])

        # Seed sigma2 with the last in-sample conditional variance (scaled units)
        sigma2 = float(res.conditional_volatility[-1] ** 2)
        ret_scaled = log_ret_full * rescale

        T = len(log_ret_full)
        sigma2_path = np.empty(T, dtype=float)
        # Recursion: at index t, sigma2 forecasts variance of return at t,
        # conditioned on return at t-1.
        # For t in [0, n_train_returns), we use the in-sample conditional variance.
        cond_var = np.asarray(res.conditional_volatility) ** 2  # (n_train_returns,)
        sigma2_path[:n_train_returns] = cond_var
        # For t >= n_train_returns, roll forward
        sigma2 = float(cond_var[-1])
        for t in range(n_train_returns, T):
            r_prev = ret_scaled[t - 1]
            sigma2 = omega + alpha * (r_prev ** 2) + beta * sigma2
            sigma2_path[t] = sigma2

        # Un-rescale to log-return variance scale
        sigma2_path = sigma2_path / (rescale ** 2)

        # Build proxy RV using rolling-window decomposition
        proxy_rv = np.full(T, np.nan, dtype=float)
        for t in range(vol_window - 1, T):
            past_start = t - vol_window + 1
            past_end = t  # exclusive; the t-th squared return is replaced by sigma2 forecast
            past_ret2 = log_ret_full[past_start : past_end] ** 2
            avg_var = (np.sum(past_ret2) + sigma2_path[t]) / vol_window
            proxy_rv[t] = np.sqrt(max(avg_var, 1e-12) * 252.0)

        return sigma2_path, proxy_rv

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict next-day RV for each test row in X. The full return series
        (set via set_returns()) is rolled forward through the test horizon
        using the fitted GARCH(1,1) recursion, and each next-day variance
        forecast is plugged into the rolling-window RV proxy.

        n_test must match (len(log_returns_full) - test_first_ret_idx).
        """
        if self._garch_result is None:
            raise RuntimeError("GARCH must be fit() before predict().")

        n_test = len(X)
        params = self._garch_result.params
        omega = float(params["omega"])
        alpha = float(params["alpha[1]"])
        beta = float(params["beta[1]"])

        # Seed sigma2 with the last in-sample conditional variance
        sigma2 = float(self._garch_result.conditional_volatility[-1] ** 2)

        # Full return series in the rescaled space
        ret_scaled = self._full_returns * self.rescale
        # `self._n_train` is the index in log_ret_full of the first test target's
        # next-day return; see test_first_ret_idx in the loader.
        test_first = int(self._n_train)

        # GARCH recursion across the test horizon
        sigma2_path = np.empty(n_test, dtype=float)
        for i in range(n_test):
            # Forecast variance for the return at index (test_first + i).
            # Recursion uses the return at the PREVIOUS index.
            prev_idx = test_first + i - 1
            r_prev = ret_scaled[prev_idx] if prev_idx >= 0 else 0.0
            sigma2 = omega + alpha * (r_prev ** 2) + beta * sigma2
            sigma2_path[i] = sigma2

        # Un-rescale back to original log-return variance scale
        sigma2_path = sigma2_path / (self.rescale ** 2)

        # Rolling-window RV proxy: for each test step i, the target RV uses
        # log_ret_full[(test_first + i) - vol_window + 1 : (test_first + i) + 1].
        # The last element of that window is the "next-day return" we are
        # forecasting; replace it with sigma2_path[i] as its squared-return proxy.
        full_ret = self._full_returns
        rv_pred = np.empty(n_test, dtype=float)
        for i in range(n_test):
            target_idx = test_first + i
            past_start = target_idx - self.vol_window + 1
            past_end = target_idx  # exclusive — last element replaced by GARCH forecast
            past_ret2 = full_ret[max(0, past_start) : past_end] ** 2
            avg_var = (np.sum(past_ret2) + sigma2_path[i]) / self.vol_window
            rv_pred[i] = np.sqrt(max(avg_var, 1e-12) * 252.0)

        return rv_pred
