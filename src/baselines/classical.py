"""
Classical baselines for fair comparison with QRC.

  PersistenceBaseline  — naive y_hat(t) = y(t-1)
  RidgeBaseline        — linear Ridge on raw delay-embedded features
  EchoStateNetwork     — classical reservoir computing (ESN)
  ARBaseline           — autoregressive model via numpy polyfit / Ridge

All models expose fit(X_train, y_train) / predict(X_test).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class PersistenceBaseline:
    """Predict the last observed value: y_hat(t) = last feature = x(t)."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceBaseline":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Last column of X is the most-recent observation in delay embedding
        return X[:, -1].copy()


# ---------------------------------------------------------------------------
# Ridge on raw features
# ---------------------------------------------------------------------------

class RidgeBaseline:
    """Standard Ridge regression / classification on raw (delay-embedded) features."""

    def __init__(self, alpha: float = 1.0, classification: bool = False) -> None:
        self.alpha = alpha
        self.classification = classification
        self._scaler = StandardScaler()
        self._model = RidgeClassifier(alpha=alpha) if classification else Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeBaseline":
        Xs = self._scaler.fit_transform(X)
        self._model.fit(Xs, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._scaler.transform(X))


# ---------------------------------------------------------------------------
# Echo State Network  (classical reservoir computing)
# ---------------------------------------------------------------------------

class EchoStateNetwork:
    """
    Classical Echo State Network (ESN) — the direct classical analogue of QRC.

    Parameters
    ----------
    n_reservoir     : reservoir neurons
    spectral_radius : largest eigenvalue of recurrent weight matrix W
    sparsity        : fraction of non-zero entries in W
    input_scaling   : scale factor on input weights
    leak_rate       : α in s(t) = (1-α)s(t-1) + α·tanh(Ws(t-1) + W_in·x(t))
    ridge_alpha     : regularisation for the readout Ridge
    warmup          : initial steps discarded to let the reservoir wash out
    seed            : random seed
    """

    def __init__(
        self,
        n_reservoir: int = 100,
        spectral_radius: float = 0.95,
        sparsity: float = 0.1,
        input_scaling: float = 1.0,
        leak_rate: float = 1.0,
        ridge_alpha: float = 1e-4,
        warmup: int = 50,
        seed: int = 42,
    ) -> None:
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.ridge_alpha = ridge_alpha
        self.warmup = warmup
        self.seed = seed

        rng = np.random.RandomState(seed)
        self.W = self._build_W(rng)
        self.W_in: np.ndarray | None = None
        self._readout = Ridge(alpha=ridge_alpha)

    def _build_W(self, rng: np.random.RandomState) -> np.ndarray:
        n = self.n_reservoir
        W = rng.randn(n, n)
        W[rng.rand(n, n) > self.sparsity] = 0.0
        eigs = np.linalg.eigvals(W)
        max_eig = np.max(np.abs(eigs))
        if max_eig > 1e-10:
            W *= self.spectral_radius / max_eig
        return W

    def _run(self, X: np.ndarray) -> np.ndarray:
        """Drive reservoir with input sequence X, return states after warmup."""
        n_in = X.shape[1]
        if self.W_in is None:
            rng = np.random.RandomState(self.seed + 1)
            self.W_in = rng.randn(self.n_reservoir, n_in) * self.input_scaling

        s = np.zeros(self.n_reservoir)
        states = []
        for t, x in enumerate(X):
            s_new = np.tanh(self.W @ s + self.W_in @ x)
            s = (1.0 - self.leak_rate) * s + self.leak_rate * s_new
            if t >= self.warmup:
                states.append(s.copy())
        return np.array(states)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EchoStateNetwork":
        """
        X : (T, n_in) -- full training sequence
        y : (T,)      -- targets aligned with X
        """
        states = self._run(X)
        y_trimmed = y[self.warmup: self.warmup + len(states)]
        self._readout.fit(states, y_trimmed)
        # Save reservoir state after all training samples for warm-start on test
        s = np.zeros(self.n_reservoir)
        for x in X:
            s_new = np.tanh(self.W @ s + self.W_in @ x)
            s = (1.0 - self.leak_rate) * s + self.leak_rate * s_new
        self._last_state = s.copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Start from last training state; collect ALL test states (no warmup skip)
        s = getattr(self, "_last_state", np.zeros(self.n_reservoir)).copy()
        states = []
        for x in X:
            s_new = np.tanh(self.W @ s + self.W_in @ x)
            s = (1.0 - self.leak_rate) * s + self.leak_rate * s_new
            states.append(s.copy())
        return self._readout.predict(np.array(states))


# ---------------------------------------------------------------------------
# AR baseline  (autoregressive, purely linear)
# ---------------------------------------------------------------------------

class ARBaseline:
    """
    AutoRegressive model of order p.  Equivalent to Ridge on delay-embedded lags.
    """

    def __init__(self, order: int = 5, alpha: float = 1e-3) -> None:
        self.order = order
        self.alpha = alpha
        self._model = Ridge(alpha=alpha)
        self._scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ARBaseline":
        Xs = self._scaler.fit_transform(X[:, -self.order:])
        self._model.fit(Xs, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(self._scaler.transform(X[:, -self.order:]))


# ---------------------------------------------------------------------------
# Shared metric utilities
# ---------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Return RMSE, MAE, R^2, NMSE, and QLIKE.

    QLIKE (Patton 2011) is the canonical volatility-forecasting loss:
        QLIKE = mean( y_true / y_pred  -  log(y_true / y_pred)  -  1 )
    where y_true, y_pred are positive *variance*-scale (we apply it to vol
    using the standard convention y^2). Asymmetric: penalises under-prediction
    of volatility more than over-prediction, which is what risk managers want.
    Lower is better; 0 corresponds to a perfect forecast.
    """
    residuals = y_true - y_pred
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    var_y = np.var(y_true) + 1e-12
    r2 = 1.0 - mse / var_y
    nmse = mse / var_y

    # QLIKE on variance scale (squared vol); clip to avoid div-by-zero / log(0)
    y_t = np.clip(y_true ** 2, 1e-12, None)
    y_p = np.clip(y_pred ** 2, 1e-12, None)
    ratio = y_t / y_p
    qlike = float(np.mean(ratio - np.log(ratio) - 1.0))

    return {"RMSE": rmse, "MAE": mae, "R2": r2, "NMSE": nmse, "QLIKE": qlike}


def mincer_zarnowitz(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Mincer-Zarnowitz regression: y_true = a + b * y_pred + eps.

    Reports:
      mz_intercept  — a, should be 0 if forecast is unbiased
      mz_slope      — b, should be 1 if forecast is efficient
      mz_r2         — explained variance of the regression
      mz_pvalue     — Wald joint test p-value for (a=0, b=1); higher is better
                       (do NOT reject the null that the forecast is unbiased+efficient)
    """
    from scipy import stats

    x = np.asarray(y_pred, dtype=float)
    y = np.asarray(y_true, dtype=float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])

    # OLS estimates
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(beta[0]), float(beta[1])

    # Residuals and covariance of beta
    resid = y - X @ beta
    sigma2 = float(np.sum(resid ** 2) / max(n - 2, 1))
    XtX_inv = np.linalg.inv(X.T @ X)
    cov_beta = sigma2 * XtX_inv

    # Wald joint test of H0: a=0, b=1
    R = np.eye(2)
    r0 = np.array([0.0, 1.0])
    diff = R @ beta - r0
    mid = R @ cov_beta @ R.T
    try:
        wald = float(diff @ np.linalg.solve(mid, diff))
        pvalue = float(1.0 - stats.chi2.cdf(wald, df=2))
    except np.linalg.LinAlgError:
        pvalue = float("nan")

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    mz_r2 = 1.0 - ss_res / ss_tot

    return {
        "mz_intercept": a,
        "mz_slope": b,
        "mz_r2": mz_r2,
        "mz_pvalue": pvalue,
    }


def print_metrics(name: str, metrics: dict) -> None:
    m = metrics
    extra = ""
    if "QLIKE" in m:
        extra = f"  QLIKE={m['QLIKE']:.5f}"
    print(
        f"  {name:<35}  RMSE={m['RMSE']:.5f}  MAE={m['MAE']:.5f}"
        f"  R2={m['R2']:.4f}  NMSE={m['NMSE']:.5f}{extra}"
    )
