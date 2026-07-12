"""Training-only input projections for hardware-sized quantum reservoirs."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ReservoirInputProjector:
    """Map the full causal feature vector to one value per reservoir qubit."""

    def __init__(self, n_outputs: int, mode: str = "first", seed: int = 42):
        self.n_outputs = int(n_outputs)
        self.mode = mode
        self.seed = int(seed)
        self.scaler = StandardScaler()
        self.pca = None
        self.matrix = None
        self.indices = None

    def fit(self, X: np.ndarray, feature_names=None):
        X = np.asarray(X, dtype=float)
        if self.mode == "first":
            self.indices = np.arange(min(self.n_outputs, X.shape[1]))
            self.scaler.fit(X[:, self.indices])
        elif self.mode == "selected":
            self.indices = self._selected_indices(X.shape[1], feature_names)
            self.scaler.fit(X[:, self.indices])
        elif self.mode == "pca":
            Xs = self.scaler.fit_transform(X)
            self.pca = PCA(n_components=self.n_outputs, whiten=True, random_state=self.seed)
            self.pca.fit(Xs)
        elif self.mode == "random":
            Xs = self.scaler.fit_transform(X)
            rng = np.random.RandomState(self.seed)
            self.matrix = rng.normal(size=(X.shape[1], self.n_outputs)) / np.sqrt(X.shape[1])
            # Normalize projected columns using training data.
            projected = Xs @ self.matrix
            scale = projected.std(axis=0)
            self.matrix = self.matrix / np.where(scale > 1e-12, scale, 1.0)
        else:
            raise ValueError(f"unknown input projection mode {self.mode!r}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.mode in ("first", "selected"):
            out = self.scaler.transform(X[:, self.indices])
            if out.shape[1] < self.n_outputs:
                out = np.resize(out, (len(out), self.n_outputs))
            return out
        Xs = self.scaler.transform(X)
        if self.mode == "pca":
            return self.pca.transform(Xs)
        return Xs @ self.matrix

    def fit_transform(self, X: np.ndarray, feature_names=None) -> np.ndarray:
        return self.fit(X, feature_names=feature_names).transform(X)

    def _selected_indices(self, n_features: int, feature_names) -> np.ndarray:
        """Prior-guided causal subset: recent returns, HAR state, GARCH anchor."""
        if feature_names:
            names = list(feature_names)
            preferred = []
            groups = [
                [i for i, x in enumerate(names) if x.startswith("ret_lag")][-1:],
                [i for i, x in enumerate(names) if x.startswith("abs_ret")][-1:],
                [i for i, x in enumerate(names) if x.startswith("sq_ret")][-1:],
                [i for i, x in enumerate(names) if x.startswith("har_rv")],
                [i for i, x in enumerate(names) if x.startswith("log_har")],
                [i for i, x in enumerate(names) if "garch_proxy" in x],
            ]
            for group in groups:
                preferred.extend(group)
            preferred = list(dict.fromkeys(preferred))
        else:
            preferred = []
        if len(preferred) < self.n_outputs:
            # Backfill from the newest columns, which contain the slow volatility
            # state and proxy features in load_financial_data_v2.
            preferred.extend(i for i in range(n_features - 1, -1, -1)
                             if i not in preferred)
        return np.asarray(preferred[:self.n_outputs], dtype=int)
