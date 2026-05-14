"""
Financial Volatility Forecasting — GIC 2026 Track 1

Predicts next-day realized volatility of SPY using delay-embedded log-returns
as input. Compares QRC against Persistence, AR, Ridge, and ESN baselines.

Outputs
-------
  results/financial_results.csv
  results/financial_rmse.png
  results/financial_predictions.png
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
from src.data.loaders import load_financial_data
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
    alpha: float = 1.0,
) -> np.ndarray:
    sc_in = StandardScaler()
    R_tr = qrc.transform(sc_in.fit_transform(X_tr))
    R_te = qrc.transform(sc_in.transform(X_te))

    sc_r = StandardScaler()
    R_tr = sc_r.fit_transform(R_tr)
    R_te = sc_r.transform(R_te)

    readout = Ridge(alpha=alpha)
    readout.fit(R_tr, y_tr)
    return readout.predict(R_te)


def run(seed: int = 42, delay: int = 5) -> pd.DataFrame:
    print("=" * 65)
    print("  FINANCIAL VOLATILITY FORECAST — QRC vs. Baselines (SPY)")
    print("=" * 65)

    X_tr, X_te, y_tr, y_te = load_financial_data(delay=delay)
    print(f"\nTrain: {len(X_tr)}, Test: {len(X_te)}, "
          f"Delay embedding: {delay} days\n")

    rows = []
    predictions = {}

    # ---- Baselines ----
    print("Classical baselines:")

    pers = PersistenceBaseline()
    pers.fit(X_tr, y_tr)
    y_hat = pers.predict(X_te)
    m = regression_metrics(y_te, y_hat)
    print_metrics("Persistence (y_hat = last obs.)", m)
    rows.append({"Model": "Persistence", **m})
    predictions["Persistence"] = y_hat

    ar = ARBaseline(order=delay)
    ar.fit(X_tr, y_tr)
    y_hat = ar.predict(X_te)
    m = regression_metrics(y_te, y_hat)
    print_metrics(f"AR({delay})", m)
    rows.append({"Model": f"AR({delay})", **m})
    predictions[f"AR({delay})"] = y_hat

    ridge = RidgeBaseline(alpha=1.0)
    ridge.fit(X_tr, y_tr)
    y_hat = ridge.predict(X_te)
    m = regression_metrics(y_te, y_hat)
    print_metrics("Ridge (linear)", m)
    rows.append({"Model": "Ridge (linear)", **m})
    predictions["Ridge"] = y_hat

    esn = EchoStateNetwork(n_reservoir=200, seed=seed)
    esn.fit(X_tr, y_tr)
    y_hat = esn.predict(X_te)
    m = regression_metrics(y_te, y_hat)
    print_metrics("ESN (n_res=200)", m)
    rows.append({"Model": "ESN", **m})
    predictions["ESN"] = y_hat

    # ---- QRC sweep ----
    print("\nQuantum Reservoir Computing:")
    for n_qubits in [5, 10, 15]:
        for n_layers in [2, 3]:
            qrc = QuantumReservoir(
                n_qubits=n_qubits,
                n_layers=n_layers,
                connectivity="random",
                seed=seed,
            )
            t0 = time.time()
            y_hat = _qrc_regress(qrc, X_tr[:, :n_qubits], X_te[:, :n_qubits], y_tr)
            elapsed = time.time() - t0
            m = regression_metrics(y_te, y_hat)
            label = f"QRC {n_qubits}q L{n_layers}"
            print_metrics(label, m)
            m["elapsed"] = elapsed
            rows.append({"Model": label, **m})
            if n_qubits == 10 and n_layers == 3:
                predictions["QRC 10q L3"] = y_hat

    df = pd.DataFrame(rows).sort_values("NMSE")
    df.to_csv(os.path.join(RESULTS_DIR, "financial_results.csv"), index=False)
    print(f"\nResults saved → results/financial_results.csv")

    # ---- RMSE bar chart ----
    fig, ax = plt.subplots(figsize=(10, 5))
    colours = ["#4C72B0" if "QRC" in r["Model"] else "#DD8452"
               for _, r in df.iterrows()]
    ax.barh(df["Model"], df["RMSE"], color=colours)
    ax.set_xlabel("RMSE (annualised volatility)")
    ax.set_title("Financial Volatility Forecast — RMSE (lower is better)")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "financial_rmse.png"), dpi=150)

    # ---- Time-series prediction overlay ----
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    n_show = min(200, len(y_te))
    ax2.plot(y_te[:n_show], label="True vol.", lw=1.5, c="black")
    for name, yp in predictions.items():
        ax2.plot(yp[:n_show], label=name, lw=1.0, alpha=0.8)
    ax2.set_xlabel("Test day")
    ax2.set_ylabel("Realized volatility")
    ax2.set_title("Volatility Forecast — Test Period")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    fig2.savefig(os.path.join(RESULTS_DIR, "financial_predictions.png"), dpi=150)
    print("Charts saved  → results/financial_*.png")

    return df


if __name__ == "__main__":
    df = run()
    print("\n--- Summary (sorted by NMSE) ---")
    print(df[["Model", "RMSE", "MAE", "R2", "NMSE"]].to_string(index=False))
