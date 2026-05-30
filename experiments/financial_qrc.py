"""
Market Volatility Forecasting -- GIC 2026 Track 1

Single-file experiment for SPY next-day realized volatility:
  * Classical baselines: Persistence, AR(5), Ridge, ESN, GARCH(1,1), LSTM
  * Quantum reservoirs:  gate-based (5q L2), annealer-style (10q a2a)
  * GARCH-hybrid QRC:    annealer QRC predicting GARCH residuals (headline)

Data pipeline (load_financial_data_v2):
  * HAR-RV features (daily, weekly, monthly lagged RV) + log-HAR + return features
  * Log-vol residual target: y = log(RV[t]) - log(RV[t-1])
  * GARCH proxy as feature + GARCH-residual target variant
  * Chronological 80/20 split, no shuffle

Outputs
-------
  results/financial_results.csv
  results/financial_rmse.png
  results/financial_predictions.png   (2-subplot: overlay + scatter + rolling RMSE)
  results/financial_predictions_doc.png  (1200 DPI publication copy)
"""

from __future__ import annotations

import os
import sys
import time
import warnings

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
    mincer_zarnowitz,
    print_metrics,
)
from src.baselines.garch import GARCHBaseline
from src.baselines.lstm import LSTMBaseline

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
warnings.filterwarnings("ignore")

_ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cv_alpha(R_tr, y_tr, alphas=_ALPHAS, n_splits=5):
    """Select Ridge alpha via time-series cross-validation."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_alpha, best_mse = alphas[0], np.inf
    for alpha in alphas:
        mses = []
        for tr_idx, val_idx in tscv.split(R_tr):
            r = Ridge(alpha=alpha)
            r.fit(R_tr[tr_idx], y_tr[tr_idx])
            mses.append(np.mean((y_tr[val_idx] - r.predict(R_tr[val_idx])) ** 2))
        m = float(np.mean(mses))
        if m < best_mse:
            best_mse, best_alpha = m, alpha
    return best_alpha


def _reservoir_regress(res, X_tr, X_te, y_tr, X_tr_full, X_te_full):
    """Transform through reservoir, concatenate raw features, CV-tune readout."""
    sc_in = StandardScaler()
    R_tr = res.transform(sc_in.fit_transform(X_tr))
    R_te = res.transform(sc_in.transform(X_te))
    # Hybrid: reservoir features + original features
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
    r2 = (y_true - y_pred) ** 2
    return np.array([np.sqrt(np.mean(r2[max(0, i - window): i + 1]))
                     for i in range(len(r2))])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_predictions(y_te, predictions, metrics_map, best_qrc_key, save_prefix):
    """2-row figure: overlay + scatter + rolling RMSE."""
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("SPY Realized Volatility Forecast -- Test Period", fontsize=13, y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    ax_top = fig.add_subplot(gs[0, :])
    ax_sc  = fig.add_subplot(gs[1, 0])
    ax_rr  = fig.add_subplot(gs[1, 1])

    days = np.arange(len(y_te))

    # Shade high-vol regime
    high_vol_thresh = np.percentile(y_te, 75)
    high_vol = y_te > high_vol_thresh
    in_regime = False
    for i, hv in enumerate(high_vol):
        if hv and not in_regime:
            start = i; in_regime = True
        elif not hv and in_regime:
            ax_top.axvspan(start, i, color="#ffcccc", alpha=0.4)
            in_regime = False
    if in_regime:
        ax_top.axvspan(start, len(y_te) - 1, color="#ffcccc", alpha=0.4)

    import matplotlib.patches as mpatches
    hv_patch = mpatches.Patch(color="#ffcccc", alpha=0.6, label="High vol (>P75)")

    ax_top.plot(days, y_te, color="black", lw=2.0, label="True volatility", zorder=5)

    # Plot best QRC, GARCH, and Persistence
    series = [
        (best_qrc_key,   "#7c3aed", "-",  1.8),
        ("GARCH(1,1)",   "#2166ac", "--", 1.4),
        ("Persistence",  "#9e9e9e", "-.", 1.0),
    ]
    for key, c, ls, lw in series:
        if key not in predictions:
            continue
        m = metrics_map.get(key, {})
        label = f"{key}  (RMSE={m.get('RMSE', float('nan')):.5f}, R2={m.get('R2', float('nan')):.4f})"
        ax_top.plot(days, predictions[key], color=c, ls=ls, lw=lw, label=label, alpha=0.9)

    ax_top.set_xlabel("Test day")
    ax_top.set_ylabel("Realized volatility (annualised)")
    handles, labels = ax_top.get_legend_handles_labels()
    ax_top.legend([hv_patch] + handles, [hv_patch.get_label()] + labels,
                  fontsize=7.5, loc="upper left")
    ax_top.grid(True, alpha=0.25)

    # Scatter: predicted vs actual
    if best_qrc_key in predictions:
        y_hat = predictions[best_qrc_key]
        ax_sc.scatter(y_te, y_hat, s=6, alpha=0.4, color="#7c3aed", label=best_qrc_key)
        if "GARCH(1,1)" in predictions:
            ax_sc.scatter(y_te, predictions["GARCH(1,1)"], s=4, alpha=0.3,
                          color="#2166ac", label="GARCH(1,1)")
        lo = min(y_te.min(), y_hat.min())
        hi = max(y_te.max(), y_hat.max())
        ax_sc.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="Perfect fit")
        ax_sc.set_xlabel("True volatility")
        ax_sc.set_ylabel("Predicted volatility")
        m_best = metrics_map.get(best_qrc_key, {})
        ax_sc.set_title(f"Predicted vs Actual\nR2={m_best.get('R2', float('nan')):.4f}", fontsize=9)
        ax_sc.legend(fontsize=7)
        ax_sc.grid(True, alpha=0.25)

    # Rolling RMSE
    for key, c, ls in [(best_qrc_key, "#7c3aed", "-"),
                       ("GARCH(1,1)", "#2166ac", "--"),
                       ("Persistence", "#9e9e9e", "-.")]:
        if key in predictions:
            ax_rr.plot(days, _rolling_rmse(y_te, predictions[key]),
                       color=c, ls=ls, lw=1.3, label=key)
    ax_rr.set_xlabel("Test day")
    ax_rr.set_ylabel("30-day rolling RMSE")
    ax_rr.set_title("Rolling RMSE", fontsize=9)
    ax_rr.legend(fontsize=8)
    ax_rr.grid(True, alpha=0.25)

    fig.savefig(os.path.join(RESULTS_DIR, f"{save_prefix}.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS_DIR, f"{save_prefix}_doc.png"), dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> results/{save_prefix}.png  (+ _doc.png at 1200 DPI)")


def _plot_rmse_bar(df):
    fig, ax = plt.subplots(figsize=(11, 6))
    def _color(name):
        if "QRC" in name or "ANN" in name: return "#7c3aed"
        if "GARCH" in name: return "#2166ac"
        return "#d6604d"
    bars = ax.barh(df["Model"], df["RMSE"],
                   color=[_color(m) for m in df["Model"]],
                   edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=7)
    ax.set_xlabel("RMSE (annualised volatility)")
    ax.set_title("SPY Volatility Forecast -- RMSE by Model (lower is better)")
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "financial_rmse.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved -> results/financial_rmse.png")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(seed: int = 42, n_seeds: int = 3, fast: bool = False):
    print("=" * 78)
    print("  SPY MARKET VOLATILITY FORECAST")
    print("  Classical baselines + QRC + GARCH-hybrid QRC")
    print("=" * 78)

    # ---- Load data: HAR-RV features, log-vol residual target ----
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
    print(f"  Target transform: {transform}")
    print(f"  Features: {d['feature_names']}\n")

    to_vol = lambda y: invert_target(y, transform, log_persistence=log_pers_te)

    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    metrics_map: dict[str, dict] = {}

    def record(name, y_pred, seed_val=0, elapsed=0.0, config=""):
        m = regression_metrics(y_te_raw, y_pred)
        mz = mincer_zarnowitz(y_te_raw, y_pred)
        m.update(mz)
        m["elapsed"] = elapsed
        m["seed"] = seed_val
        m["config"] = config
        print_metrics(f"{name:<42} (seed={seed_val})", m)
        rows.append({"Model": name, **m})
        predictions[name] = y_pred
        metrics_map[name] = m
        return m

    # ==================================================================
    # 1. Classical baselines
    # ==================================================================
    print("[1/4] Classical baselines\n" + "-" * 78)

    record("Persistence", pers_te.copy(), config="persistence")

    t0 = time.time()
    ar = ARBaseline(order=5); ar.fit(X_tr, y_tr)
    record("AR(5)", to_vol(ar.predict(X_te)), elapsed=time.time()-t0, config="ar")

    t0 = time.time()
    r = RidgeBaseline(alpha=1.0); r.fit(X_tr, y_tr)
    record("Ridge", to_vol(r.predict(X_te)), elapsed=time.time()-t0, config="ridge")

    t0 = time.time()
    esn = EchoStateNetwork(n_reservoir=200, seed=seed); esn.fit(X_tr, y_tr)
    record("ESN (200 nodes)", to_vol(esn.predict(X_te)), elapsed=time.time()-t0, config="esn")

    t0 = time.time()
    garch = GARCHBaseline(vol_window=d["vol_window"])
    garch.set_returns(d["log_returns_full"], n_train=d["test_first_ret_idx"])
    garch.fit(X_tr, y_tr)
    record("GARCH(1,1)", garch.predict(X_te), elapsed=time.time()-t0, config="garch")

    if not fast:
        t0 = time.time()
        lstm = LSTMBaseline(delay=5, hidden=32, epochs=60, lr=5e-3, seed=seed)
        lstm.fit(X_tr, y_tr)
        record("LSTM (1L, 32h)", to_vol(lstm.predict(X_te)), elapsed=time.time()-t0, config="lstm")

    # ==================================================================
    # 2. Gate-based QRC reference
    # ==================================================================
    print(f"\n[2/4] Gate-based QRC (5q L2 random)\n" + "-" * 78)
    t0 = time.time()
    qrc = QuantumReservoir(n_qubits=5, n_layers=2, connectivity="random", seed=seed)
    y_hat, _ = _reservoir_regress(qrc, X_tr[:, :5], X_te[:, :5], y_tr, X_tr, X_te)
    record("QRC gate (5q L2 rand)", to_vol(y_hat), elapsed=time.time()-t0, config="gate_qrc")

    # ==================================================================
    # 3. Annealer QRC (multi-seed ensemble)
    # ==================================================================
    n_seeds_actual = 1 if fast else n_seeds
    print(f"\n[3/4] Annealer QRC (10q a2a t=2.0) -- {n_seeds_actual} seed(s)\n" + "-" * 78)
    ann_preds = []
    for s_off in range(n_seeds_actual):
        seed_i = seed + s_off
        t0 = time.time()
        ar_qrc = AnnealerReservoir(
            n_qubits=10, n_input=10, evolution_time=2.0, trotter_steps=3,
            connectivity="all-to-all", input_scale=1.0, seed=seed_i,
        )
        y_hat, _ = _reservoir_regress(ar_qrc, X_tr[:, :10], X_te[:, :10], y_tr, X_tr, X_te)
        y_vol = to_vol(y_hat)
        ann_preds.append(y_vol)
        record(f"ANN 10q a2a t=2.0", y_vol, seed_val=seed_i,
               elapsed=time.time()-t0, config="ann_qrc")

    if n_seeds_actual > 1:
        ann_ensemble = np.mean(np.stack(ann_preds, axis=0), axis=0)
        record(f"ANN 10q a2a ({n_seeds_actual}-seed mean)", ann_ensemble, config="ann_ensemble")

    # ==================================================================
    # 4. GARCH-hybrid QRC (best approach: GARCH-residual target)
    # ==================================================================
    print(f"\n[4/4] GARCH-hybrid QRC (residual target) -- {n_seeds_actual} seed(s)\n" + "-" * 78)
    d_garch = load_financial_data_v2(
        delay=5, log_target=True, include_har=True, include_log_har=True,
        include_garch_proxy=True, garch_residual_target=True,
    )
    to_vol_garch = lambda y: invert_target(
        y, d_garch["target_transform"], log_garch_proxy=d_garch["log_garch_proxy_test"]
    )
    Xg_tr, Xg_te = d_garch["X_train"], d_garch["X_test"]
    yg_tr = d_garch["y_train"]
    yg_te_raw = d_garch["y_test_raw"]

    garch_hybrid_preds = []
    for s_off in range(n_seeds_actual):
        seed_i = seed + s_off
        t0 = time.time()
        ar_qrc = AnnealerReservoir(
            n_qubits=10, n_input=10, evolution_time=2.0, trotter_steps=3,
            connectivity="all-to-all", input_scale=1.0, seed=seed_i,
        )
        y_hat, _ = _reservoir_regress(
            ar_qrc, Xg_tr[:, :10], Xg_te[:, :10], yg_tr, Xg_tr, Xg_te,
        )
        y_vol = to_vol_garch(y_hat)
        garch_hybrid_preds.append(y_vol)
        record("QRC+GARCH hybrid", y_vol, seed_val=seed_i,
               elapsed=time.time()-t0, config="garch_hybrid")

    if n_seeds_actual > 1:
        gh_ensemble = np.mean(np.stack(garch_hybrid_preds, axis=0), axis=0)
        record(f"QRC+GARCH hybrid ({n_seeds_actual}-seed)", gh_ensemble, config="garch_hybrid_ens")

    # ==================================================================
    # Save results & plot
    # ==================================================================
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "financial_results.csv"), index=False)
    print(f"\n  Saved -> results/financial_results.csv")

    # Build leaderboard: one row per unique model (use ensemble where available)
    # Only exclude per-seed rows if an ensemble row exists for that group
    has_ann_ens = df["config"].eq("ann_ensemble").any()
    has_gh_ens  = df["config"].eq("garch_hybrid_ens").any()
    exclude_configs = set()
    if has_ann_ens:
        exclude_configs.add("ann_qrc")
    if has_gh_ens:
        exclude_configs.add("garch_hybrid")
    df_lead = df[~df["config"].isin(exclude_configs)].copy()
    df_lead = df_lead.sort_values("RMSE")
    _plot_rmse_bar(df_lead)

    # Determine best QRC key for prediction plot
    qrc_rows = df_lead[df_lead["config"].str.contains("qrc|hybrid|ann|gate", case=False, na=False)]
    best_qrc_key = qrc_rows.iloc[0]["Model"] if not qrc_rows.empty else ""
    print(f"\n  Best QRC config: {best_qrc_key}")

    _plot_predictions(y_te_raw, predictions, metrics_map, best_qrc_key,
                      save_prefix="financial_predictions")

    return df_lead


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="1 seed, skip LSTM")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=3)
    args = parser.parse_args()

    df = run(seed=args.seed, n_seeds=args.n_seeds, fast=args.fast)
    print("\n--- Summary (sorted by RMSE) ---")
    cols = ["Model", "RMSE", "MAE", "R2", "NMSE"]
    if "QLIKE" in df.columns:
        cols.append("QLIKE")
    print(df[cols].to_string(index=False))
