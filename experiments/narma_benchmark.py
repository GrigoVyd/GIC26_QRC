"""
NARMA Benchmark — standard reservoir computing evaluation.

NARMA-5 and NARMA-10 are the canonical time-series benchmarks for RC systems.
Primary metric: NMSE (Normalised Mean Squared Error).

Outputs
-------
  results/narma_results.csv
  results/narma_nmse.png
  results/narma_predictions.png
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
from src.data.loaders import load_narma
from src.baselines.classical import (
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
    R_tr = sc.fit_transform(qrc.transform(X_tr))
    R_te = sc.transform(qrc.transform(X_te))
    readout = Ridge(alpha=alpha)
    readout.fit(R_tr, y_tr)
    return readout.predict(R_te)


def _run_order(order: int, seed: int = 42) -> list[dict]:
    print(f"\n{'─'*60}")
    print(f"  NARMA-{order}")
    print(f"{'─'*60}")

    X_tr, X_te, y_tr, y_te = load_narma(order=order, n_samples=2000, seed=seed)
    print(f"  Train: {len(X_tr)}, Test: {len(X_te)}, Input dim: {X_tr.shape[1]}")

    rows = []
    first_qrc_preds = None

    # ---- Baselines ----
    print("\n  Classical baselines:")

    ar = ARBaseline(order=order)
    ar.fit(X_tr, y_tr)
    y_hat = ar.predict(X_te)
    m = regression_metrics(y_te, y_hat)
    print_metrics(f"  AR({order})", m)
    rows.append({"Task": f"NARMA-{order}", "Model": f"AR({order})", **m})

    ridge = RidgeBaseline()
    ridge.fit(X_tr, y_tr)
    y_hat = ridge.predict(X_te)
    m = regression_metrics(y_te, y_hat)
    print_metrics("  Ridge", m)
    rows.append({"Task": f"NARMA-{order}", "Model": "Ridge", **m})

    esn = EchoStateNetwork(n_reservoir=100, seed=seed)
    esn.fit(X_tr, y_tr)
    y_hat = esn.predict(X_te)
    m = regression_metrics(y_te, y_hat)
    print_metrics("  ESN (100 neurons)", m)
    rows.append({"Task": f"NARMA-{order}", "Model": "ESN-100", **m})
    esn_preds = y_hat

    esn_large = EchoStateNetwork(n_reservoir=500, seed=seed)
    esn_large.fit(X_tr, y_tr)
    y_hat = esn_large.predict(X_te)
    m = regression_metrics(y_te, y_hat)
    print_metrics("  ESN (500 neurons)", m)
    rows.append({"Task": f"NARMA-{order}", "Model": "ESN-500", **m})

    # ---- QRC sweep ----
    print("\n  QRC:")
    for n_qubits in [5, 10, 15]:
        qrc = QuantumReservoir(
            n_qubits=n_qubits,
            n_layers=3,
            connectivity="random",
            seed=seed,
        )
        t0 = time.time()
        y_hat = _qrc_regress(qrc, X_tr[:, :n_qubits], X_te[:, :n_qubits], y_tr)
        elapsed = time.time() - t0
        m = regression_metrics(y_te, y_hat)
        label = f"QRC {n_qubits}q"
        print_metrics(f"  {label}", m)
        m["elapsed_s"] = elapsed
        rows.append({"Task": f"NARMA-{order}", "Model": label, **m})
        if n_qubits == 10:
            first_qrc_preds = y_hat

    return rows, y_te, esn_preds, first_qrc_preds


def run(seed: int = 42) -> pd.DataFrame:
    print("=" * 65)
    print("  NARMA BENCHMARK — QRC vs. Classical Reservoir Computing")
    print("=" * 65)

    all_rows = []
    plot_data = {}

    for order in [5, 10]:
        rows, y_te, esn_preds, qrc_preds = _run_order(order, seed)
        all_rows.extend(rows)
        plot_data[order] = (y_te, esn_preds, qrc_preds)

    df = pd.DataFrame(all_rows).sort_values(["Task", "NMSE"])
    df.to_csv(os.path.join(RESULTS_DIR, "narma_results.csv"), index=False)
    print(f"\nResults saved → results/narma_results.csv")

    # ---- NMSE grouped bar chart ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, order in zip(axes, [5, 10]):
        sub = df[df["Task"] == f"NARMA-{order}"]
        colours = ["#4C72B0" if "QRC" in m else "#DD8452" for m in sub["Model"]]
        ax.bar(sub["Model"], sub["NMSE"], color=colours)
        ax.set_title(f"NARMA-{order}")
        ax.set_ylabel("NMSE (lower is better)")
        ax.tick_params(axis="x", rotation=30)
    plt.suptitle("NARMA Benchmark — NMSE", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "narma_nmse.png"), dpi=150)

    # ---- Prediction overlay ----
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 6))
    for ax, order in zip(axes2, [5, 10]):
        y_te, esn_p, qrc_p = plot_data[order]
        n_show = min(150, len(y_te))
        ax.plot(y_te[:n_show], label="True", lw=1.5, c="black")
        ax.plot(esn_p[:n_show], label="ESN-100", lw=1.0, alpha=0.8)
        if qrc_p is not None:
            ax.plot(qrc_p[:n_show], label="QRC 10q", lw=1.0, alpha=0.8)
        ax.set_title(f"NARMA-{order}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig2.savefig(os.path.join(RESULTS_DIR, "narma_predictions.png"), dpi=150)
    print("Charts saved  → results/narma_*.png")

    return df


if __name__ == "__main__":
    df = run()
    print("\n--- Summary (sorted by Task, NMSE) ---")
    print(df[["Task", "Model", "RMSE", "R2", "NMSE"]].to_string(index=False))
