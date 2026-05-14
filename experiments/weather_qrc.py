"""
Weather / Chaotic Time-Series Forecasting — GIC 2026 Track 2

Uses Lorenz-63 as a deterministic weather proxy (NOAA via meteostat if available).
One-step-ahead prediction benchmark.

Outputs
-------
  results/weather_results.csv
  results/weather_rmse.png
  results/weather_predictions.png
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.qrc.reservoir import QuantumReservoir
from src.data.loaders import load_noaa_weather
from src.baselines.classical import (
    PersistenceBaseline,
    ARBaseline,
    RidgeBaseline,
    EchoStateNetwork,
    regression_metrics,
    print_metrics,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _qrc_regress(
    qrc: QuantumReservoir,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    sc = StandardScaler()
    R_tr = qrc.transform(sc.fit_transform(X_tr))
    R_te = qrc.transform(sc.transform(X_te))
    sc2 = StandardScaler()
    R_tr = sc2.fit_transform(R_tr)
    R_te = sc2.transform(R_te)
    readout = Ridge(alpha=alpha)
    readout.fit(R_tr, y_tr)
    return readout.predict(R_te)


def run(seed: int = 42, delay: int = 5) -> pd.DataFrame:
    print("=" * 65)
    print("  WEATHER / LORENZ FORECAST — QRC vs. Baselines")
    print("=" * 65)

    X_tr, X_te, y_tr, y_te = load_noaa_weather(delay=delay)
    print(f"\nTrain: {len(X_tr)}, Test: {len(X_te)}, Delay: {delay}\n")

    rows = []
    predictions = {}

    # ---- Baselines ----
    print("Classical baselines:")

    pers = PersistenceBaseline()
    pers.fit(X_tr, y_tr)
    y_hat = pers.predict(X_te)
    m = regression_metrics(y_te, y_hat)
    print_metrics("Persistence", m)
    rows.append({"Model": "Persistence", **m})
    predictions["Persistence"] = y_hat

    ar = ARBaseline(order=delay)
    ar.fit(X_tr, y_tr)
    y_hat = ar.predict(X_te)
    m = regression_metrics(y_te, y_hat)
    print_metrics(f"AR({delay})", m)
    rows.append({"Model": f"AR({delay})", **m})
    predictions[f"AR({delay})"] = y_hat

    ridge = RidgeBaseline()
    ridge.fit(X_tr, y_tr)
    y_hat = ridge.predict(X_te)
    m = regression_metrics(y_te, y_hat)
    print_metrics("Ridge (linear)", m)
    rows.append({"Model": "Ridge", **m})
    predictions["Ridge"] = y_hat

    esn = EchoStateNetwork(n_reservoir=200, seed=seed)
    esn.fit(X_tr, y_tr)
    y_hat = esn.predict(X_te)
    m = regression_metrics(y_te, y_hat)
    print_metrics("ESN (200 neurons)", m)
    rows.append({"Model": "ESN", **m})
    predictions["ESN"] = y_hat

    # ---- QRC ----
    print("\nQRC:")
    for n_qubits in [5, 10, 15]:
        qrc = QuantumReservoir(
            n_qubits=n_qubits, n_layers=3, connectivity="random", seed=seed
        )
        t0 = time.time()
        y_hat = _qrc_regress(qrc, X_tr[:, :n_qubits], X_te[:, :n_qubits], y_tr)
        elapsed = time.time() - t0
        m = regression_metrics(y_te, y_hat)
        label = f"QRC {n_qubits}q"
        print_metrics(label, m)
        m["elapsed_s"] = elapsed
        rows.append({"Model": label, **m})
        if n_qubits == 10:
            predictions["QRC 10q"] = y_hat

    df = pd.DataFrame(rows).sort_values("NMSE")
    df.to_csv(os.path.join(RESULTS_DIR, "weather_results.csv"), index=False)
    print(f"\nResults saved → results/weather_results.csv")

    # ---- RMSE bar chart ----
    fig, ax = plt.subplots(figsize=(10, 5))
    colours = ["#4C72B0" if "QRC" in r["Model"] else "#DD8452"
               for _, r in df.iterrows()]
    ax.barh(df["Model"], df["RMSE"], color=colours)
    ax.set_xlabel("RMSE (normalised units)")
    ax.set_title("Weather/Lorenz Forecast — RMSE (lower is better)")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "weather_rmse.png"), dpi=150)

    # ---- Prediction overlay ----
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    n_show = min(200, len(y_te))
    ax2.plot(y_te[:n_show], label="True", lw=1.5, c="black")
    for name, yp in predictions.items():
        ax2.plot(yp[:n_show], label=name, lw=1.0, alpha=0.8)
    ax2.set_xlabel("Test step")
    ax2.set_ylabel("Normalised temperature / Lorenz x")
    ax2.set_title("Weather/Lorenz Forecast — Test Period")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    fig2.savefig(os.path.join(RESULTS_DIR, "weather_predictions.png"), dpi=150)
    print("Charts saved  → results/weather_*.png")

    return df


if __name__ == "__main__":
    df = run()
    print("\n--- Summary ---")
    print(df[["Model", "RMSE", "MAE", "R2", "NMSE"]].to_string(index=False))
