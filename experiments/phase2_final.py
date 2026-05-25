"""
Phase 2 Final Experiment — all baselines, all metrics, multi-seed ensemble,
VIX-bucketed regime analysis.

This is the headline experiment for the GIC 2026 Phase 2 submission. It runs:

  Classical baselines (all required by Track A rubric):
    Persistence, AR(5), Ridge, ESN, GARCH(1,1), LSTM
  Quantum reservoirs:
    Gate-based v3 best (5q L2 random)
    Annealer-style best (10q all-to-all, t=2.0, alpha=1.0) -- 5 seeds for error bars

  Metrics (all required by Track A rubric):
    RMSE, MAE, R^2, NMSE, QLIKE (Patton 2011), Mincer-Zarnowitz regression

  Regime-conditional sub-analysis:
    Test dates bucketed by VIX percentile (calm / normal / turbulent terciles)
    Per-bucket RMSE for the top 5 models

Outputs
-------
  results/phase2_final_summary.csv     -- one row per (model, seed) with all metrics
  results/phase2_final_regime.csv      -- per-bucket RMSE for top models
  results/phase2_final_leaderboard.png -- bar chart of headline RMSE +/- std
  results/phase2_final_predictions.png -- predicted vs true overlay + scatter
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


def _vix_test_series(n_test: int, start_date: str = "2010-01-01",
                     end_date: str = "2024-12-31") -> np.ndarray | None:
    """Fetch VIX close prices and align to the test period via tail alignment."""
    try:
        import yfinance as yf
        raw = yf.download("^VIX", start=start_date, end=end_date, progress=False, auto_adjust=True)
        vix = raw["Close"].squeeze().dropna().values
    except Exception as e:
        print(f"VIX fetch failed ({e}); skipping regime analysis.")
        return None
    if len(vix) < n_test:
        return None
    return vix[-n_test:]  # tail alignment — matches the chronological 80/20 split


def _regime_buckets(vix: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float, str]]]:
    """Split into terciles by VIX value. Returns (bucket_idx, bucket_meta)."""
    q1, q2 = np.percentile(vix, [33.33, 66.67])
    buckets = np.zeros(len(vix), dtype=int)
    buckets[vix > q1] = 1
    buckets[vix > q2] = 2
    meta = [
        (float(vix.min()), float(q1), "calm"),
        (float(q1), float(q2), "normal"),
        (float(q2), float(vix.max()), "turbulent"),
    ]
    return buckets, meta


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(seed: int = 42, n_seeds: int = 5):
    print("=" * 78)
    print("  GIC 2026 PHASE 2 FINAL EXPERIMENT")
    print("  All Track A baselines, all metrics, multi-seed, VIX-bucketed regime")
    print("=" * 78)

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
    n_test = len(X_te)
    print(f"\nTrain: {len(X_tr)}, Test: {n_test}, Features: {X_tr.shape[1]}")
    print(f"  Target transform: {transform}\n")

    to_vol = lambda y: invert_target(y, transform, log_persistence=log_pers_te)
    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}

    def record(name, y_pred, seed_val=0, elapsed=0.0):
        m = regression_metrics(y_te_raw, y_pred)
        mz = mincer_zarnowitz(y_te_raw, y_pred)
        m.update(mz); m["elapsed"] = elapsed; m["seed"] = seed_val
        print_metrics(f"{name:<32} (seed={seed_val})", m)
        rows.append({"Model": name, **m})
        predictions[name] = y_pred
        return m

    # ---- Classical baselines (deterministic — single seed=42 used for ESN) ----
    print("[1/3] Classical baselines\n" + "-" * 78)

    record("Persistence", pers_te.copy())

    t0 = time.time()
    ar = ARBaseline(order=5); ar.fit(X_tr, y_tr)
    record("AR(5)", to_vol(ar.predict(X_te)), elapsed=time.time()-t0)

    t0 = time.time()
    r = RidgeBaseline(alpha=1.0); r.fit(X_tr, y_tr)
    record("Ridge", to_vol(r.predict(X_te)), elapsed=time.time()-t0)

    t0 = time.time()
    esn = EchoStateNetwork(n_reservoir=200, seed=seed); esn.fit(X_tr, y_tr)
    record("ESN (200 nodes)", to_vol(esn.predict(X_te)), elapsed=time.time()-t0)

    t0 = time.time()
    garch = GARCHBaseline(vol_window=d["vol_window"])
    garch.set_returns(d["log_returns_full"], n_train=d["test_first_ret_idx"])
    garch.fit(X_tr, y_tr)
    record("GARCH(1,1)", garch.predict(X_te), elapsed=time.time()-t0)

    t0 = time.time()
    lstm = LSTMBaseline(delay=5, hidden=32, epochs=60, lr=5e-3, seed=seed)
    lstm.fit(X_tr, y_tr)
    record("LSTM (1L, 32h)", to_vol(lstm.predict(X_te)), elapsed=time.time()-t0)

    # ---- Gate-based QRC v3 best config (single seed reference) ----
    print(f"\n[2/3] Gate-based QRC reference (5q L2 random)\n" + "-" * 78)
    t0 = time.time()
    qrc = QuantumReservoir(n_qubits=5, n_layers=2, connectivity="random", seed=seed)
    y_hat, _ = _reservoir_regress(qrc, X_tr[:, :5], X_te[:, :5], y_tr, X_tr, X_te)
    record("QRC v3 gate (5q L2 rand)", to_vol(y_hat), elapsed=time.time()-t0)

    # ---- Annealer-style QRC at the sweep optimum, multi-seed ensemble ----
    print(f"\n[3/3] Annealer QRC (10q a2a t=2.0 alpha=1.0) -- {n_seeds} seeds\n" + "-" * 78)
    ann_preds_per_seed: list[np.ndarray] = []
    for s_off in range(n_seeds):
        seed_i = seed + s_off
        t0 = time.time()
        ar_qrc = AnnealerReservoir(
            n_qubits=10, n_input=10, evolution_time=2.0, trotter_steps=3,
            connectivity="all-to-all", input_scale=1.0, seed=seed_i,
        )
        y_hat, _ = _reservoir_regress(
            ar_qrc, X_tr[:, :10], X_te[:, :10], y_tr, X_tr, X_te,
        )
        y_vol = to_vol(y_hat)
        ann_preds_per_seed.append(y_vol)
        record(f"ANN 10q a2a t=2.0 a=1.0", y_vol, seed_val=seed_i, elapsed=time.time()-t0)

    # Ensemble mean prediction
    ann_mean_pred = np.mean(np.stack(ann_preds_per_seed, axis=0), axis=0)
    record("ANN 10q a2a (5-seed mean)", ann_mean_pred)

    # ---- Build full results table ----
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "phase2_final_summary.csv"), index=False)
    print(f"\n  Saved -> results/phase2_final_summary.csv")

    # Aggregate ANN multi-seed -> mean +/- std row for the headline table
    ann_seeds_df = df[df["Model"] == "ANN 10q a2a t=2.0 a=1.0"]
    ann_summary = {
        "Model": "ANN 10q a2a (mean +/- std, 5 seeds)",
        "RMSE_mean": ann_seeds_df["RMSE"].mean(),
        "RMSE_std":  ann_seeds_df["RMSE"].std(),
        "R2_mean":   ann_seeds_df["R2"].mean(),
        "R2_std":    ann_seeds_df["R2"].std(),
        "QLIKE_mean":ann_seeds_df["QLIKE"].mean(),
        "QLIKE_std": ann_seeds_df["QLIKE"].std(),
    }
    print(f"\n  ANN 10q a2a 5-seed: RMSE = {ann_summary['RMSE_mean']:.5f} +/- {ann_summary['RMSE_std']:.5f}")
    print(f"                       R2   = {ann_summary['R2_mean']:.5f} +/- {ann_summary['R2_std']:.5f}")
    print(f"                       QLIKE= {ann_summary['QLIKE_mean']:.5f} +/- {ann_summary['QLIKE_std']:.5f}")
    pd.DataFrame([ann_summary]).to_csv(
        os.path.join(RESULTS_DIR, "phase2_final_ann_ensemble.csv"), index=False,
    )

    # ---- Regime-conditional analysis (VIX terciles) ----
    print(f"\n[Bonus] VIX-bucketed regime-conditional RMSE\n" + "-" * 78)
    vix = _vix_test_series(n_test)
    if vix is not None:
        buckets, meta = _regime_buckets(vix)
        regime_rows = []
        # Top models by RMSE for per-bucket analysis
        top_models = [
            "Persistence", "GARCH(1,1)", "ESN (200 nodes)", "AR(5)",
            "QRC v3 gate (5q L2 rand)", "ANN 10q a2a (5-seed mean)",
        ]
        for name in top_models:
            if name not in predictions:
                continue
            y_pred = predictions[name]
            for b_idx, (lo, hi, lbl) in enumerate(meta):
                mask = buckets == b_idx
                if mask.sum() == 0:
                    continue
                m = regression_metrics(y_te_raw[mask], y_pred[mask])
                regime_rows.append({
                    "Model": name, "Regime": lbl,
                    "VIX_lo": round(lo, 2), "VIX_hi": round(hi, 2),
                    "n_days": int(mask.sum()),
                    **{k: m[k] for k in ("RMSE", "MAE", "R2", "QLIKE")},
                })
        regime_df = pd.DataFrame(regime_rows)
        regime_df.to_csv(os.path.join(RESULTS_DIR, "phase2_final_regime.csv"), index=False)
        print("  Saved -> results/phase2_final_regime.csv")
        print("\n  RMSE by VIX regime (lower better):")
        pivot = regime_df.pivot(index="Model", columns="Regime", values="RMSE")
        # Reorder columns: calm, normal, turbulent
        col_order = [c for c in ["calm", "normal", "turbulent"] if c in pivot.columns]
        pivot = pivot[col_order]
        print(pivot.round(5).to_string())
    else:
        regime_df = None

    # ---- Plots ----
    _plot_leaderboard(df, ann_summary)
    _plot_predictions(y_te_raw, predictions)
    if regime_df is not None:
        _plot_regime(regime_df)

    return df, regime_df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_leaderboard(df: pd.DataFrame, ann_summary: dict) -> None:
    # Best-per-model RMSE for ranking (use mean for the ANN multi-seed row)
    df_unique = df.copy()
    # Replace the per-seed rows with one ensemble row
    df_unique = df_unique[df_unique["Model"] != "ANN 10q a2a t=2.0 a=1.0"]
    df_unique = df_unique[df_unique["Model"] != "ANN 10q a2a (5-seed mean)"]
    df_unique = pd.concat([df_unique, pd.DataFrame([{
        "Model": "ANN 10q (5-seed)",
        "RMSE": ann_summary["RMSE_mean"],
        "R2":   ann_summary["R2_mean"],
        "QLIKE":ann_summary["QLIKE_mean"],
    }])], ignore_index=True)
    df_unique = df_unique.sort_values("RMSE")

    fig, ax = plt.subplots(figsize=(11, 6))
    def _color(name):
        if name.startswith("ANN"): return "#7c3aed"
        if name.startswith("QRC"): return "#2166ac"
        return "#d6604d"
    bars = ax.barh(df_unique["Model"], df_unique["RMSE"], color=[_color(m) for m in df_unique["Model"]])
    ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=7)
    # Add error bars on the ANN ensemble row
    ann_row = df_unique[df_unique["Model"] == "ANN 10q (5-seed)"]
    if not ann_row.empty:
        idx = df_unique.index.get_loc(ann_row.index[0])
        ax.errorbar(ann_summary["RMSE_mean"], idx,
                    xerr=ann_summary["RMSE_std"], fmt="o", color="black",
                    capsize=4, markersize=4)
    ax.set_xlabel("RMSE (annualised volatility)")
    ax.set_title("GIC 2026 Phase 2 -- Final leaderboard (Track A: SPY realized volatility)")
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "phase2_final_leaderboard.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS_DIR, "phase2_final_leaderboard_doc.png"), dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved -> results/phase2_final_leaderboard.png  (+ _doc.png @ 1200 DPI)")


def _plot_predictions(y_te: np.ndarray, predictions: dict) -> None:
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Phase 2 Final -- SPY Realized Volatility Forecast (test period)", fontsize=13, y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    ax_top = fig.add_subplot(gs[0, :])
    ax_sc  = fig.add_subplot(gs[1, 0])
    ax_rr  = fig.add_subplot(gs[1, 1])

    days = np.arange(len(y_te))
    ax_top.plot(days, y_te, color="black", lw=2.0, label="True volatility", zorder=5)
    series = [
        ("ANN 10q a2a (5-seed mean)", "#7c3aed", "-",  1.8),
        ("GARCH(1,1)",                "#2166ac", "--", 1.4),
        ("Persistence",               "#9e9e9e", "-.", 1.0),
    ]
    for key, c, ls, lw in series:
        if key in predictions:
            ax_top.plot(days, predictions[key], color=c, ls=ls, lw=lw, label=key, alpha=0.9)
    ax_top.set_xlabel("Test day"); ax_top.set_ylabel("Realized vol (annualised)")
    ax_top.legend(fontsize=8, loc="upper left"); ax_top.grid(True, alpha=0.25)

    best_key = "ANN 10q a2a (5-seed mean)"
    if best_key in predictions:
        y_hat = predictions[best_key]
        ax_sc.scatter(y_te, y_hat, s=6, alpha=0.4, color="#7c3aed", label=best_key)
        if "GARCH(1,1)" in predictions:
            ax_sc.scatter(y_te, predictions["GARCH(1,1)"], s=4, alpha=0.3, color="#2166ac", label="GARCH(1,1)")
        lo = min(y_te.min(), y_hat.min())
        hi = max(y_te.max(), y_hat.max())
        ax_sc.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="Perfect")
        ax_sc.set_xlabel("True"); ax_sc.set_ylabel("Predicted")
        ax_sc.set_title("Predicted vs Actual", fontsize=9)
        ax_sc.legend(fontsize=7); ax_sc.grid(True, alpha=0.25)

    # Rolling RMSE
    def _rolling_rmse(y_t, y_p, w=30):
        r2 = (y_t - y_p) ** 2
        return np.array([np.sqrt(np.mean(r2[max(0, i - w): i + 1])) for i in range(len(r2))])

    for key, c, ls in [(best_key, "#7c3aed", "-"),
                       ("GARCH(1,1)", "#2166ac", "--"),
                       ("Persistence", "#9e9e9e", "-.")]:
        if key in predictions:
            ax_rr.plot(days, _rolling_rmse(y_te, predictions[key]),
                       color=c, ls=ls, lw=1.3, label=key)
    ax_rr.set_xlabel("Test day"); ax_rr.set_ylabel("30-day rolling RMSE")
    ax_rr.set_title("Rolling RMSE", fontsize=9); ax_rr.legend(fontsize=8); ax_rr.grid(True, alpha=0.25)

    fig.savefig(os.path.join(RESULTS_DIR, "phase2_final_predictions.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS_DIR, "phase2_final_predictions_doc.png"), dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved -> results/phase2_final_predictions.png  (+ _doc.png @ 1200 DPI)")


def _plot_regime(regime_df: pd.DataFrame) -> None:
    pivot = regime_df.pivot(index="Model", columns="Regime", values="RMSE")
    col_order = [c for c in ["calm", "normal", "turbulent"] if c in pivot.columns]
    pivot = pivot[col_order]
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, color=["#5dc863", "#21918c", "#440154"])
    ax.set_ylabel("RMSE per regime")
    ax.set_xlabel("")
    ax.set_title("Regime-conditional RMSE -- VIX terciles (lower is better)")
    ax.legend(title="VIX regime", loc="upper left")
    ax.grid(True, axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "phase2_final_regime.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS_DIR, "phase2_final_regime_doc.png"), dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved -> results/phase2_final_regime.png  (+ _doc.png @ 1200 DPI)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=5, help="annealer ensemble size")
    args = parser.parse_args()
    df, regime_df = run(seed=args.seed, n_seeds=args.n_seeds)
    print("\n--- Done ---")
