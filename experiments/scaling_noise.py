"""
Qubit Scaling & Noise Analysis — GIC 2026 Required Evaluation

Two sub-experiments:

  1. Qubit scaling  — NARMA-5 NMSE vs. n_qubits ∈ {5, 10, 15}
                      and n_layers ∈ {1, 2, 3}

  2. Noise analysis — NARMA-5 NMSE under depolarizing noise at varied strength,
                      comparing noiseless, low, medium, high regimes.

Outputs
-------
  results/scaling_results.csv
  results/noise_results.csv
  results/scaling_nmse.png       (heatmap: qubits × layers)
  results/noise_degradation.png  (line plot: NMSE vs noise level)
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
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.qrc.reservoir import QuantumReservoir
from src.qrc.noise import depolarizing_noise, NOISE_LEVELS
from src.data.loaders import load_narma
from src.baselines.classical import regression_metrics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _evaluate_qrc(
    n_qubits: int,
    n_layers: int,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    noise_model=None,
    seed: int = 42,
    alpha: float = 0.1,
) -> dict:
    """Fit and evaluate one QRC configuration. Returns metric dict."""
    qrc = QuantumReservoir(
        n_qubits=n_qubits,
        n_layers=n_layers,
        connectivity="random",
        seed=seed,
        noise_model=noise_model,
    )
    n = min(n_qubits, X_tr.shape[1])
    t0 = time.time()

    sc = StandardScaler()
    R_tr = qrc.transform(sc.fit_transform(X_tr[:, :n]))
    R_te = qrc.transform(sc.transform(X_te[:, :n]))

    sc_r = StandardScaler()
    R_tr = sc_r.fit_transform(R_tr)
    R_te = sc_r.transform(R_te)

    readout = Ridge(alpha=alpha)
    readout.fit(R_tr, y_tr)
    y_hat = readout.predict(R_te)
    elapsed = time.time() - t0

    m = regression_metrics(y_te, y_hat)
    m["elapsed_s"] = elapsed
    return m


# ---------------------------------------------------------------------------
# Experiment 1 — Qubit scaling
# ---------------------------------------------------------------------------

def run_scaling(seed: int = 42) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("  QUBIT SCALING ANALYSIS  (NARMA-5, noiseless)")
    print("=" * 65)

    X_tr, X_te, y_tr, y_te = load_narma(order=5, n_samples=800, seed=seed)

    rows = []
    qubit_list = [5, 10, 15]
    layer_list = [1, 2, 3]

    for n_q in qubit_list:
        for n_l in layer_list:
            print(f"  QRC {n_q}q × {n_l} layers ...", end=" ", flush=True)
            m = _evaluate_qrc(n_q, n_l, X_tr, X_te, y_tr, y_te, seed=seed)
            rows.append({"n_qubits": n_q, "n_layers": n_l, **m})
            print(f"NMSE={m['NMSE']:.5f}  ({m['elapsed_s']:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "scaling_results.csv"), index=False)
    print(f"Saved → results/scaling_results.csv")

    # ---- Heatmap ----
    pivot = df.pivot(index="n_qubits", columns="n_layers", values="NMSE")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".4f", cmap="viridis_r", ax=ax,
        cbar_kws={"label": "NMSE (lower = better)"},
    )
    ax.set_title("QRC Qubit Scaling — NARMA-5 NMSE")
    ax.set_xlabel("n_layers")
    ax.set_ylabel("n_qubits")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "scaling_nmse.png"), dpi=150)
    print("Chart saved   → results/scaling_nmse.png")

    return df


# ---------------------------------------------------------------------------
# Experiment 2 — Noise analysis
# ---------------------------------------------------------------------------

def run_noise(n_qubits: int = 10, n_layers: int = 3, seed: int = 42) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print(f"  NOISE ANALYSIS  (NARMA-5, {n_qubits}q × {n_layers} layers)")
    print("=" * 65)

    X_tr, X_te, y_tr, y_te = load_narma(order=5, n_samples=400, seed=seed)

    # Noiseless baseline
    rows = []
    print(f"  noiseless   ...", end=" ", flush=True)
    m = _evaluate_qrc(n_qubits, n_layers, X_tr, X_te, y_tr, y_te,
                      noise_model=None, seed=seed)
    rows.append({"noise": "none", "p_single": 0.0, **m})
    print(f"NMSE={m['NMSE']:.5f}")

    # Depolarizing sweep
    p_singles = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    for p in p_singles:
        nm = depolarizing_noise(p_single=p, p_two=p * 10)
        label = f"depol p={p:.3f}"
        print(f"  {label} ...", end=" ", flush=True)
        m = _evaluate_qrc(n_qubits, n_layers, X_tr, X_te, y_tr, y_te,
                          noise_model=nm, seed=seed)
        rows.append({"noise": label, "p_single": p, **m})
        print(f"NMSE={m['NMSE']:.5f}  ({m['elapsed_s']:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "noise_results.csv"), index=False)
    print(f"Saved → results/noise_results.csv")

    # ---- Degradation curve ----
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(df["NMSE"].iloc[0], ls="--", c="navy", label="noiseless", lw=1.5)
    noisy = df.iloc[1:]
    ax.plot(noisy["p_single"], noisy["NMSE"], "o-", c="#DD8452",
            label=f"QRC {n_qubits}q (depolarizing)")
    ax.set_xscale("log")
    ax.set_xlabel("Single-qubit depolarizing probability")
    ax.set_ylabel("NMSE")
    ax.set_title("QRC Performance Under Depolarizing Noise — NARMA-5")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "noise_degradation.png"), dpi=150)
    print("Chart saved   → results/noise_degradation.png")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_scale = run_scaling(seed=seed)
    df_noise = run_noise(seed=seed)
    return df_scale, df_noise


if __name__ == "__main__":
    df_s, df_n = run()
    print("\n--- Scaling Results ---")
    print(df_s[["n_qubits", "n_layers", "NMSE", "R2"]].to_string(index=False))
    print("\n--- Noise Results ---")
    print(df_n[["noise", "NMSE", "R2"]].to_string(index=False))
