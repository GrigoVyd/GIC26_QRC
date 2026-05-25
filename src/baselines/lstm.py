"""
LSTM baseline for SPY realized-volatility forecasting.

A small single-layer LSTM trained on the same delay-embedded feature set as
the other models in the project. Architecture is deliberately minimal — the
purpose is a competent "neural network baseline" as named by the GIC 2026
Phase 2 rubric, not a state-of-the-art predictor.

Inputs treated as (batch, seq_len=delay, n_features_per_step). For the v2
feature set (5-day delay + HAR), the LSTM sees a 5-step sequence with
delay-aligned channels.

Trained with Adam + MSE on the same target the other models use (log-vol
residual or log-vol or raw RV depending on caller). Deterministic — seed
controls weight init, no dropout.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32, n_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def _to_sequences(X: np.ndarray, delay: int) -> np.ndarray:
    """
    Reshape (n_samples, n_features) to (n_samples, delay, n_features_per_step).

    The v2 loader produces features in groups: returns(5) + |r|(5) + r^2(5)
    + HAR(3) + log-HAR(3). The first 15 columns are delay-aligned (5 lags
    per channel x 3 channels); the trailing HAR columns are scalar per sample.
    We give the LSTM the 3-channel delay-embedded part as a sequence and tile
    the HAR scalars at each timestep so they reach the readout.
    """
    n, p = X.shape
    n_delay_feats = 3 * delay
    if p < n_delay_feats:
        # Fallback: treat the whole vector as a single-step sequence
        return X.reshape(n, 1, p)
    delay_part = X[:, :n_delay_feats].reshape(n, 3, delay).transpose(0, 2, 1)
    static_part = X[:, n_delay_feats:]  # (n, p - 3*delay)
    if static_part.shape[1] == 0:
        return delay_part.astype(np.float32)
    static_tiled = np.broadcast_to(static_part[:, None, :], (n, delay, static_part.shape[1]))
    seq = np.concatenate([delay_part, static_tiled], axis=2)
    return seq.astype(np.float32)


class LSTMBaseline:
    """
    Minimal LSTM regression baseline.

    Parameters
    ----------
    delay        : sequence length (must match the loader's delay parameter).
    hidden       : LSTM hidden size (default 32).
    n_layers     : LSTM layer count (default 1).
    epochs       : training epochs (default 60).
    lr           : Adam learning rate.
    batch_size   : SGD batch size.
    seed         : RNG seed for deterministic init / training.
    device       : "cpu" or "cuda". Defaults to "cpu" (small data).
    """

    def __init__(
        self,
        delay: int = 5,
        hidden: int = 32,
        n_layers: int = 1,
        epochs: int = 60,
        lr: float = 5e-3,
        batch_size: int = 64,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        self.delay = delay
        self.hidden = hidden
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self._net: _LSTMNet | None = None
        self._scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMBaseline":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        Xs = self._scaler.fit_transform(X)
        seq = _to_sequences(Xs, self.delay)  # (n, delay, n_feat_per_step)
        n_feat = seq.shape[2]

        self._net = _LSTMNet(n_feat, hidden=self.hidden, n_layers=self.n_layers).to(self.device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        X_t = torch.from_numpy(seq).to(self.device)
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(self.device)
        n = len(X_t)

        self._net.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(self.seed + epoch))
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                optimizer.zero_grad()
                pred = self._net(X_t[idx])
                loss = loss_fn(pred, y_t[idx])
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("LSTM must be fit() before predict().")
        Xs = self._scaler.transform(X)
        seq = _to_sequences(Xs, self.delay)
        self._net.eval()
        with torch.no_grad():
            preds = self._net(torch.from_numpy(seq).to(self.device)).cpu().numpy()
        return preds.astype(np.float64)
