"""
D-Wave-style QRC — first-approximation experiment (classical simulation).

Tests whether the annealer-style reservoir (transverse-field Ising,
h_i encoding, fixed random J_ij, Trotterized evolution from |+>^n) is
competitive with the gate-based reservoir on our SPY-volatility task.

This is a *simulation* of what would run on D-Wave Advantage hardware. It
keeps the same readout and target as v3 (HAR + log-HAR features, residual
log-vol target) so the comparison is apples-to-apples vs the v3 gate-based
QRC results in ``results/financial_results_v3.csv``.

Outputs
-------
  results/financial_results_dwave_v0.csv     — all configs sorted by NMSE
  results/financial_rmse_dwave_v0.png        — RMSE bar chart
  results/financial_predictions_dwave_v0.png — best annealer config + baselines
  results/dwave_vs_v3.csv                    — head-to-head with v3 gate-based
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
import matplotlib.gridspec as gridspec
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.qrc.reservoir import QuantumReservoir
from src.qrc.annealer_reservoir import AnnealerReservoir
from src.data.loaders import load_financial_data_v2, invert_target
from src.baselines.classical import (
    ARBaseline,
    RidgeBaseline,
    EchoStateNetwork,
    regression_metrics,
    print_metrics,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

_ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]


# ---------------------------------------------------------------------------
# Helpers (same as v3)
# ---------------------------------------------------------------------------

def _cv_alpha(R_tr, y_tr, alphas=_ALPHAS, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_alpha, best_mse = alphas[0], np.inf
    for alpha in alphas:
        mses = []
        for tr_idx, val_idx in tscv.split(R_tr):
            ridge = Ridge(alpha=alpha)
            ridge.fit(R_tr[tr_idx], y_tr[tr_idx])
            mses.append(np.mean((y_tr[val_idx] - ridge.predict(R_tr[val_idx])) ** 2))
        m = float(np.mean(mses))
        if m < best_mse:
            best_mse, best_alpha = m, alpha
    return best_alpha


def _reservoir_regress(res, X_tr, X_te, y_tr, X_tr_full=None, X_te_full=None):
    """Generic: Reservoir (with .transform()) + CV-tuned Ridge readout."""
    sc_in = StandardScaler()
    R_tr = res.transform(sc_in.fit_transform(X_tr))
    R_te = res.transform(sc_in.transform(X_te))

    if X_tr_full is not None and X_te_full is not None:
        R_tr = np.hstack([R_tr, X_tr_full])
        R_te = np.hstack([R_te, X_te_full])

    sc_r = StandardScaler()
    R_tr_s = sc_r.fit_transform(R_tr)
    R_te_s = sc_r.transform(R_te)

    alpha = _cv_alpha(R_tr_s, y_tr)
    ridge = Ridge(alpha=alpha)
    ridge.fit(R_tr_s, y_tr)
    return ridge.predict(R_te_s), alpha


def _rolling_rmse(y_true, y_pred, window=30):
    res = (y_true - y_pred) ** 2
    return np.array([
        np.sqrt(np.mean(res[max(0, i - window): i + 1])) for i in range(len(res))
    ])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_rmse_bar(df, save_prefix):
    fig, ax = plt.subplots(figsize=(10, 6))
    def _color(name):
        if name.startswith("ANN"):
            return "#7c3aed"          # purple — annealer
        if name.startswith("QRC"):
            return "#2166ac"          # blue   — gate-based
        return "#d6604d"              # red    — baselines
    colours = [_color(r["Model"]) for _, r in df.iterrows()]
    bars = ax.barh(df["Model"], df["RMSE"], color=colours, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=7)
    ax.set_xlabel("RMSE (annualised volatility)")
    ax.set_title("D-Wave-style QRC (first approximation) vs. gate-based + baselines")
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"{save_prefix}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> results/{save_prefix}.png")


def _plot_predictions(y_te, predictions, best_key, metrics_map, save_prefix):
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("D-Wave-style QRC (first approximation) — Test Period", fontsize=13, y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    ax_top = fig.add_subplot(gs[0, :])
    ax_sc  = fig.add_subplot(gs[1, 0])
    ax_rr  = fig.add_subplot(gs[1, 1])

    days = np.arange(len(y_te))
    ax_top.plot(days, y_te, color="black", lw=2.0, label="True volatility", zorder=5)
    show = [(best_key, "#7c3aed", "-", 1.8),
            ("QRC v3 best", "#2166ac", "--", 1.3),
            ("ESN", "#d6604d", ":", 1.3),
            ("Persistence", "#9e9e9e", "-.", 1.0)]
    for key, c, ls, lw in show:
        if key in predictions:
            m = metrics_map.get(key, {})
            ax_top.plot(days, predictions[key], color=c, ls=ls, lw=lw,
                        label=f"{key}  (R²={m.get('R2', float('nan')):.3f})", alpha=0.9)
    ax_top.set_xlabel("Test day"); ax_top.set_ylabel("Realized volatility")
    ax_top.legend(fontsize=7.5, loc="upper left"); ax_top.grid(True, alpha=0.25)

    if best_key in predictions:
        y_hat = predictions[best_key]
        m = metrics_map.get(best_key, {})
        ax_sc.scatter(y_te, y_hat, s=6, alpha=0.4, color="#7c3aed")
        lo, hi = min(y_te.min(), y_hat.min()), max(y_te.max(), y_hat.max())
        ax_sc.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="Perfect")
        ax_sc.set_xlabel("True"); ax_sc.set_ylabel("Predicted")
        ax_sc.set_title(f"{best_key} — R²={m.get('R2', float('nan')):.3f}", fontsize=9)
        ax_sc.legend(fontsize=8); ax_sc.grid(True, alpha=0.25)

    for key, c, ls in [(best_key, "#7c3aed", "-"),
                       ("QRC v3 best", "#2166ac", "--"),
                       ("ESN", "#d6604d", ":")]:
        if key in predictions:
            ax_rr.plot(days, _rolling_rmse(y_te, predictions[key]),
                       color=c, ls=ls, lw=1.3, label=key)
    ax_rr.set_xlabel("Test day"); ax_rr.set_ylabel("30-day rolling RMSE")
    ax_rr.set_title("Rolling RMSE", fontsize=9); ax_rr.legend(fontsize=8); ax_rr.grid(True, alpha=0.25)

    fig.savefig(os.path.join(RESULTS_DIR, f"{save_prefix}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> results/{save_prefix}.png")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(seed=42, fast=False):
    print("=" * 70)
    print("  D-WAVE-STYLE QRC — first-approximation classical simulation")
    print("=" * 70)

    d = load_financial_data_v2(
        delay=5, log_target=True, include_har=True,
        include_log_har=True, residual_target=True,
    )
    X_tr, X_te = d["X_train"], d["X_test"]
    y_tr, y_te = d["y_train"], d["y_test"]
    y_te_raw   = d["y_test_raw"]
    pers_te    = d["persistence_test"]
    log_pers_te = d["log_persistence_test"]
    transform  = d["target_transform"]
    print(f"\nTrain: {len(X_tr)}, Test: {len(X_te)}, Features: {X_tr.shape[1]}")
    print(f"  Target transform: {transform}\n")

    to_vol = lambda y: invert_target(y, transform, log_persistence=log_pers_te)

    rows, predictions, metrics_map = [], {}, {}

    # ---- Classical baselines (same set as v3) ----
    print("Classical baselines:")

    m = regression_metrics(y_te_raw, pers_te)
    print_metrics("Persistence (RV[t-1])", m)
    rows.append({"Model": "Persistence", **m})
    predictions["Persistence"] = pers_te.copy(); metrics_map["Persistence"] = m

    for name, model in [("AR(5)", ARBaseline(order=5)),
                        ("Ridge", RidgeBaseline(alpha=1.0)),
                        ("ESN",   EchoStateNetwork(n_reservoir=200, seed=seed))]:
        model.fit(X_tr, y_tr)
        y_hat = to_vol(model.predict(X_te))
        m = regression_metrics(y_te_raw, y_hat)
        print_metrics(name, m)
        rows.append({"Model": name, **m})
        predictions[name] = y_hat; metrics_map[name] = m

    # ---- Gate-based v3 best config (single reference point) ----
    print("\nGate-based QRC reference (v3 best config):")
    qrc = QuantumReservoir(n_qubits=5, n_layers=2, connectivity="random", seed=seed)
    y_hat_trans, alpha = _reservoir_regress(
        qrc, X_tr[:, :5], X_te[:, :5], y_tr, X_tr_full=X_tr, X_te_full=X_te,
    )
    y_hat = to_vol(y_hat_trans)
    m = regression_metrics(y_te_raw, y_hat)
    print_metrics(f"QRC v3 best (5q L2 rand, a={alpha})", m)
    rows.append({"Model": "QRC v3 best", **m})
    predictions["QRC v3 best"] = y_hat; metrics_map["QRC v3 best"] = m

    # ---- Annealer-style sweep ----
    print("\nD-Wave-style annealer reservoir (sweep):")
    if fast:
        sweep = [
            dict(n_qubits=6, n_input=6, evolution_time=1.0, trotter_steps=3, connectivity="random"),
        ]
    else:
        sweep = [
            # (n_qubits, n_input, evolution_time, trotter_steps, connectivity)
            dict(n_qubits=6,  n_input=6,  evolution_time=0.5, trotter_steps=3, connectivity="random"),
            dict(n_qubits=6,  n_input=6,  evolution_time=1.0, trotter_steps=3, connectivity="random"),
            dict(n_qubits=6,  n_input=6,  evolution_time=2.0, trotter_steps=4, connectivity="random"),
            dict(n_qubits=8,  n_input=8,  evolution_time=1.0, trotter_steps=3, connectivity="random"),
            dict(n_qubits=8,  n_input=4,  evolution_time=1.0, trotter_steps=3, connectivity="random"),  # 4 input + 4 memory
            dict(n_qubits=10, n_input=10, evolution_time=1.0, trotter_steps=3, connectivity="random"),
            dict(n_qubits=10, n_input=5,  evolution_time=1.0, trotter_steps=3, connectivity="random"),  # 5 input + 5 memory
            dict(n_qubits=10, n_input=10, evolution_time=1.0, trotter_steps=3, connectivity="all-to-all"),
        ]

    best_ann_nmse = np.inf
    best_ann_key  = ""

    for cfg in sweep:
        n_q = cfg["n_qubits"]
        n_in = cfg["n_input"]
        t_evo = cfg["evolution_time"]
        m_trot = cfg["trotter_steps"]
        conn = cfg["connectivity"]
        label = f"ANN {n_q}q/{n_in}in t={t_evo} m={m_trot} {conn[:4]}"
        ar = AnnealerReservoir(
            n_qubits=n_q, n_input=n_in,
            evolution_time=t_evo, trotter_steps=m_trot,
            connectivity=conn, seed=seed,
        )
        t0 = time.time()
        y_hat_trans, alpha = _reservoir_regress(
            ar, X_tr[:, :n_in], X_te[:, :n_in], y_tr,
            X_tr_full=X_tr, X_te_full=X_te,
        )
        elapsed = time.time() - t0
        y_hat = to_vol(y_hat_trans)
        m = regression_metrics(y_te_raw, y_hat)
        print_metrics(f"{label} (a={alpha}, {elapsed:.1f}s)", m)
        m["elapsed"] = elapsed
        rows.append({"Model": label, **m})
        predictions[label] = y_hat; metrics_map[label] = m

        if m["NMSE"] < best_ann_nmse:
            best_ann_nmse = m["NMSE"]; best_ann_key = label

    # ---- Save + plot ----
    df = pd.DataFrame(rows).sort_values("NMSE")
    df.to_csv(os.path.join(RESULTS_DIR, "financial_results_dwave_v0.csv"), index=False)
    print(f"\n  Saved -> results/financial_results_dwave_v0.csv")
    print(f"  Best annealer config: {best_ann_key}  (NMSE={best_ann_nmse:.5f})")

    _plot_rmse_bar(df, "financial_rmse_dwave_v0")
    _plot_predictions(y_te_raw, predictions, best_ann_key, metrics_map,
                      "financial_predictions_dwave_v0")

    # ---- Head-to-head with v3 ----
    v3_path = os.path.join(RESULTS_DIR, "financial_results_v3.csv")
    if os.path.exists(v3_path):
        v3 = pd.read_csv(v3_path)
        v3_best = v3[v3["Model"].str.startswith("QRC")].sort_values("NMSE").iloc[0]
        ann_best = df[df["Model"].str.startswith("ANN")].sort_values("NMSE").iloc[0]
        comparison = pd.DataFrame([
            {"Track": "Persistence",        **dict(v3[v3["Model"] == "Persistence"].iloc[0])},
            {"Track": "ESN (classical)",    **dict(v3[v3["Model"] == "ESN"].iloc[0])},
            {"Track": "Gate-based QRC v3 (best)", **dict(v3_best)},
            {"Track": "D-Wave-style v0 (best)",   **dict(ann_best)},
        ])
        keep = ["Track", "Model", "RMSE", "MAE", "R2", "NMSE"]
        comparison = comparison[[c for c in keep if c in comparison.columns]]
        comparison.to_csv(os.path.join(RESULTS_DIR, "dwave_vs_v3.csv"), index=False)
        print("  Saved -> results/dwave_vs_v3.csv")
        print("\nHead-to-head:")
        print(comparison.to_string(index=False))

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="One config only")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = run(seed=args.seed, fast=args.fast)
    print("\n--- Summary (sorted by NMSE) ---")
    print(df[["Model", "RMSE", "MAE", "R2", "NMSE"]].to_string(index=False))
