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
        X : (T, n_in) — full training sequence
        y : (T,)      — targets aligned with X
        """
        states = self._run(X)
        y_trimmed = y[self.warmup: self.warmup + len(states)]
        self._readout.fit(states, y_trimmed)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        states = self._run(X)
        return self._readout.predict(states)


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
    """Return RMSE, MAE, R², and NMSE (normalized mean squared error)."""
    residuals = y_true - y_pred
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    var_y = np.var(y_true) + 1e-12
    r2 = 1.0 - mse / var_y
    nmse = mse / var_y
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "NMSE": nmse}


def print_metrics(name: str, metrics: dict) -> None:
    m = metrics
    print(
        f"  {name:<35}  RMSE={m['RMSE']:.5f}  MAE={m['MAE']:.5f}"
        f"  R²={m['R2']:.4f}  NMSE={m['NMSE']:.5f}"
    )
