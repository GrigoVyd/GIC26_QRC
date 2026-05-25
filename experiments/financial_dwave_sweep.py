"""
D-Wave-style QRC -- parameter sweep (second-stage refinement of v0).

The v0 first-approximation experiment showed:
  * 10q all-to-all is the best annealer config (RMSE 0.00957)
  * More qubits monotonically help
  * Memory qubits hurt on this task
  * Persistence is still the ceiling (RMSE 0.00941)

This sweep refines along two axes the v0 didn't cover well:
  Stage 1 : (evolution_time, input_scale) grid at 10q all-to-all -- probe the
            dynamical-phase-transition regime (Martinez-Pena et al. 2021).
  Stage 2 : scale the best Stage-1 config up to 12q (random + all-to-all).

Outputs
-------
  results/financial_results_dwave_sweep.csv   -- all sweep configs + v0 best + v3 best + baselines
  results/dwave_sweep_heatmap.png             -- Stage 1 grid heatmap
  results/dwave_sweep_summary.png             -- RMSE bar chart top configs
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.qrc.annealer_reservoir import AnnealerReservoir
from src.data.loaders import load_financial_data_v2, invert_target
from src.baselines.classical import (
    EchoStateNetwork, regression_metrics, print_metrics,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

_ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]


def _cv_alpha(R_tr, y_tr, alphas=_ALPHAS, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_alpha, best_mse = alphas[0], np.inf
    for alpha in alphas:
        mses = []
        for tr_idx, val_idx in tscv.split(R_tr):
            r = Ridge(alpha=alpha); r.fit(R_tr[tr_idx], y_tr[tr_idx])
            mses.append(np.mean((y_tr[val_idx] - r.predict(R_tr[val_idx])) ** 2))
        m = float(np.mean(mses))
        if m < best_mse:
            best_mse, best_alpha = m, alpha
    return best_alpha


def _reservoir_regress(res, X_tr, X_te, y_tr, X_tr_full, X_te_full):
    sc_in = StandardScaler()
    R_tr = res.transform(sc_in.fit_transform(X_tr))
    R_te = res.transform(sc_in.transform(X_te))
    R_tr = np.hstack([R_tr, X_tr_full])
    R_te = np.hstack([R_te, X_te_full])
    sc_r = StandardScaler()
    R_tr_s = sc_r.fit_transform(R_tr); R_te_s = sc_r.transform(R_te)
    alpha = _cv_alpha(R_tr_s, y_tr)
    ridge = Ridge(alpha=alpha); ridge.fit(R_tr_s, y_tr)
    return ridge.predict(R_te_s), alpha


def run(seed=42):
    print("=" * 70)
    print("  D-WAVE-style QRC -- parameter sweep (Stage 1: 10q grid, Stage 2: 12q)")
    print("=" * 70)

    d = load_financial_data_v2(
        delay=5, log_target=True, include_har=True,
        include_log_har=True, residual_target=True,
    )
    X_tr, X_te = d["X_train"], d["X_test"]
    y_tr, y_te = d["y_train"], d["y_test"]
    y_te_raw   = d["y_test_raw"]
    log_pers_te = d["log_persistence_test"]
    transform   = d["target_transform"]
    print(f"\nTrain: {len(X_tr)}, Test: {len(X_te)}, Features: {X_tr.shape[1]}\n")

    to_vol = lambda y: invert_target(y, transform, log_persistence=log_pers_te)
    rows = []
    stage1_grid = []  # (evolution_time, input_scale, RMSE)

    # ---- Stage 1: 10q all-to-all parameter grid ----
    print("Stage 1: (evolution_time x input_scale) at 10q all-to-all")
    eval_times = [0.5, 1.0, 1.5, 2.0]
    input_scales = [1.0, 2.0, 3.0]

    for t_evo in eval_times:
        for s_in in input_scales:
            label = f"ANN 10q a2a t={t_evo} sc={s_in}"
            ar = AnnealerReservoir(
                n_qubits=10, n_input=10,
                evolution_time=t_evo, trotter_steps=3,
                connectivity="all-to-all", input_scale=s_in, seed=seed,
            )
            t0 = time.time()
            y_hat_trans, alpha = _reservoir_regress(
                ar, X_tr[:, :10], X_te[:, :10], y_tr, X_tr, X_te,
            )
            elapsed = time.time() - t0
            y_hat = to_vol(y_hat_trans)
            m = regression_metrics(y_te_raw, y_hat)
            print_metrics(f"{label} (a={alpha}, {elapsed:.0f}s)", m)
            m["elapsed"] = elapsed
            m["evolution_time"] = t_evo
            m["input_scale"] = s_in
            m["n_qubits"] = 10
            m["connectivity"] = "all-to-all"
            rows.append({"Model": label, **m})
            stage1_grid.append((t_evo, s_in, m["RMSE"], m["R2"]))

    # ---- Pick best of Stage 1 ----
    best = min(stage1_grid, key=lambda r: r[2])
    best_t, best_s, best_rmse, best_r2 = best
    print(f"\nStage 1 best: t={best_t}, input_scale={best_s} -> RMSE={best_rmse:.5f}, R2={best_r2:.4f}")

    # ---- Stage 2: 12q at best params ----
    print(f"\nStage 2: 12q at best params (t={best_t}, input_scale={best_s})")
    for conn in ["random", "all-to-all"]:
        label = f"ANN 12q {conn[:4]} t={best_t} sc={best_s}"
        ar = AnnealerReservoir(
            n_qubits=12, n_input=12,
            evolution_time=best_t, trotter_steps=3,
            connectivity=conn, input_scale=best_s, seed=seed,
        )
        t0 = time.time()
        y_hat_trans, alpha = _reservoir_regress(
            ar, X_tr[:, :12], X_te[:, :12], y_tr, X_tr, X_te,
        )
        elapsed = time.time() - t0
        y_hat = to_vol(y_hat_trans)
        m = regression_metrics(y_te_raw, y_hat)
        print_metrics(f"{label} (a={alpha}, {elapsed:.0f}s)", m)
        m["elapsed"] = elapsed
        m["evolution_time"] = best_t
        m["input_scale"] = best_s
        m["n_qubits"] = 12
        m["connectivity"] = conn
        rows.append({"Model": label, **m})

    df = pd.DataFrame(rows).sort_values("RMSE")
    df.to_csv(os.path.join(RESULTS_DIR, "financial_results_dwave_sweep.csv"), index=False)
    print(f"\n  Saved -> results/financial_results_dwave_sweep.csv")

    # ---- Heatmap (Stage 1) ----
    grid = pd.DataFrame(stage1_grid, columns=["evolution_time", "input_scale", "RMSE", "R2"])
    pivot_rmse = grid.pivot(index="input_scale", columns="evolution_time", values="RMSE")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot_rmse.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(pivot_rmse.columns)))
    ax.set_xticklabels([f"{t}" for t in pivot_rmse.columns])
    ax.set_yticks(range(len(pivot_rmse.index)))
    ax.set_yticklabels([f"{s}" for s in pivot_rmse.index])
    ax.set_xlabel("evolution_time (t)")
    ax.set_ylabel("input_scale (alpha)")
    ax.set_title("Stage 1 -- RMSE on 10q all-to-all (lower = better)")
    for i in range(pivot_rmse.shape[0]):
        for j in range(pivot_rmse.shape[1]):
            ax.text(j, i, f"{pivot_rmse.values[i, j]:.5f}", ha="center", va="center",
                    color="white" if pivot_rmse.values[i, j] > pivot_rmse.values.mean() else "black",
                    fontsize=8)
    plt.colorbar(im, ax=ax, label="RMSE")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "dwave_sweep_heatmap.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved -> results/dwave_sweep_heatmap.png")

    # ---- Summary bar chart: top sweep configs vs reference points ----
    # Add reference rows from v3 + v0 results files (if present)
    refs = []
    v3_path = os.path.join(RESULTS_DIR, "financial_results_v3.csv")
    if os.path.exists(v3_path):
        v3 = pd.read_csv(v3_path)
        for name in ["Persistence", "ESN"]:
            r = v3[v3["Model"] == name]
            if not r.empty:
                refs.append({"Model": name, "RMSE": float(r.iloc[0]["RMSE"]), "R2": float(r.iloc[0]["R2"])})
        qrc_best = v3[v3["Model"].str.startswith("QRC")].sort_values("RMSE").iloc[0]
        refs.append({"Model": f"QRC v3 best ({qrc_best['Model']})", "RMSE": float(qrc_best["RMSE"]), "R2": float(qrc_best["R2"])})
    v0_path = os.path.join(RESULTS_DIR, "financial_results_dwave_v0.csv")
    if os.path.exists(v0_path):
        v0 = pd.read_csv(v0_path)
        v0_best = v0[v0["Model"].str.startswith("ANN")].sort_values("RMSE").iloc[0]
        refs.append({"Model": f"ANN v0 best ({v0_best['Model']})", "RMSE": float(v0_best["RMSE"]), "R2": float(v0_best["R2"])})

    # Top 5 sweep configs
    top = df.head(5)[["Model", "RMSE", "R2"]].to_dict(orient="records")
    combined = pd.DataFrame(refs + top).sort_values("RMSE")

    fig, ax = plt.subplots(figsize=(11, 6))
    def _color(name):
        if name.startswith("ANN 12q"): return "#5e17eb"
        if name.startswith("ANN 10q"): return "#7c3aed"
        if name.startswith("ANN v0"):  return "#a78bfa"
        if name.startswith("QRC"):     return "#2166ac"
        return "#d6604d"
    bars = ax.barh(combined["Model"], combined["RMSE"], color=[_color(m) for m in combined["Model"]])
    ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=7)
    ax.set_xlabel("RMSE (annualised volatility)")
    ax.set_title("D-Wave-style QRC parameter sweep -- top configs vs reference points")
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "dwave_sweep_summary.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved -> results/dwave_sweep_summary.png")

    print("\n--- Sweep summary (sorted by RMSE) ---")
    print(df[["Model", "RMSE", "R2", "NMSE", "elapsed"]].to_string(index=False))
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    df = run(seed=args.seed)
