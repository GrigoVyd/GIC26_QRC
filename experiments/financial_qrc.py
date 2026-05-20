"""
Market Volatility Forecasting — GIC 2026 Track 1

Predicts next-day realized volatility of SPY using delay-embedded
log-returns (+ absolute/squared returns) as input features.
Compares QRC against Persistence, AR, Ridge, and ESN baselines.

Optimizations vs. the initial version
--------------------------------------
  • Enhanced features: raw + |r| + r² delay embeddings (3× richer signal)
  • Ridge alpha selected via TimeSeriesSplit cross-validation
  • QRC sweep: 5q/10q × 2L/3L × random/all-to-all connectivity
  • Best QRC config tracked and highlighted in all plots

Outputs
-------
  results/financial_results.csv
  results/financial_rmse.png
  results/financial_predictions.png   (full test window, 2-subplot)
  results/financial_predictions_doc.png  (1200 Dpi publication copy)
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

_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cv_alpha(
    R_tr: np.ndarray,
    y_tr: np.ndarray,
    alphas: list[float] = _ALPHAS,
    n_splits: int = 5,
) -> float:
    """Select Ridge alpha via time-series cross-validation (minimise NMSE)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_alpha, best_nmse = alphas[0], np.inf
    for alpha in alphas:
        nmse_list = []
        for tr_idx, val_idx in tscv.split(R_tr):
            ridge = Ridge(alpha=alpha)
            ridge.fit(R_tr[tr_idx], y_tr[tr_idx])
            y_val = ridge.predict(R_tr[val_idx])
            res = y_tr[val_idx] - y_val
            nmse = np.mean(res ** 2) / (np.var(y_tr[val_idx]) + 1e-12)
            nmse_list.append(nmse)
        mean_nmse = float(np.mean(nmse_list))
        if mean_nmse < best_nmse:
            best_nmse = mean_nmse
            best_alpha = alpha
    return best_alpha


def _qrc_regress(
    qrc: QuantumReservoir,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    X_tr_full: np.ndarray | None = None,
    X_te_full: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Transform inputs through QRC, CV-select alpha, fit readout. Returns (y_hat, alpha).

    If X_tr_full/X_te_full are provided, the readout sees both reservoir features
    and the original (enhanced) features — a hybrid quantum+classical approach that
    lets the Ridge exploit both representations.
    """
    sc_in = StandardScaler()
    R_tr = qrc.transform(sc_in.fit_transform(X_tr))
    R_te = qrc.transform(sc_in.transform(X_te))

    if X_tr_full is not None and X_te_full is not None:
        R_tr = np.hstack([R_tr, X_tr_full])
        R_te = np.hstack([R_te, X_te_full])

    sc_r = StandardScaler()
    R_tr_s = sc_r.fit_transform(R_tr)
    R_te_s = sc_r.transform(R_te)

    alpha = _cv_alpha(R_tr_s, y_tr)
    readout = Ridge(alpha=alpha)
    readout.fit(R_tr_s, y_tr)
    return readout.predict(R_te_s), alpha


def _rolling_rmse(y_true: np.ndarray, y_pred: np.ndarray, window: int = 30) -> np.ndarray:
    residuals = (y_true - y_pred) ** 2
    return np.array([
        np.sqrt(np.mean(residuals[max(0, i - window): i + 1]))
        for i in range(len(residuals))
    ])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_predictions(
    y_te: np.ndarray,
    predictions: dict[str, np.ndarray],
    best_qrc_key: str,
    metrics_map: dict[str, dict],
    save_prefix: str,
) -> None:
    """
    Two-row figure:
      Top    — Full test-period overlay (True + best QRC + ESN + Ridge)
      Bottom — Left: scatter predicted vs actual | Right: rolling 30d RMSE
    """
    high_vol_thresh = np.percentile(y_te, 75)

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("SPY Realized Volatility Forecast — Test Period", fontsize=13, y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    ax_top = fig.add_subplot(gs[0, :])
    ax_sc  = fig.add_subplot(gs[1, 0])
    ax_rr  = fig.add_subplot(gs[1, 1])

    days = np.arange(len(y_te))

    # ---- shade high-volatility regime ----
    high_vol = y_te > high_vol_thresh
    in_regime = False
    for i, hv in enumerate(high_vol):
        if hv and not in_regime:
            start = i
            in_regime = True
        elif not hv and in_regime:
            ax_top.axvspan(start, i, color="#ffcccc", alpha=0.4, label="_hv")
            in_regime = False
    if in_regime:
        ax_top.axvspan(start, len(y_te) - 1, color="#ffcccc", alpha=0.4)

    # add a single legend patch for shading
    import matplotlib.patches as mpatches
    hv_patch = mpatches.Patch(color="#ffcccc", alpha=0.6, label="High vol (>P75)")

    # ---- overlaid time series ----
    ax_top.plot(days, y_te, color="black", lw=2.0, label="True volatility", zorder=5)

    plot_order = [best_qrc_key, "ESN", "Ridge"]
    colors     = ["#2166ac", "#d6604d", "#4dac26"]
    styles     = ["-",        "--",      ":"]
    widths     = [1.8,        1.3,       1.3]

    for key, c, ls, lw in zip(plot_order, colors, styles, widths):
        if key not in predictions:
            continue
        m = metrics_map.get(key, {})
        label = f"{key}  (NMSE={m.get('NMSE', float('nan')):.3f}, R²={m.get('R2', float('nan')):.3f})"
        ax_top.plot(days, predictions[key], color=c, ls=ls, lw=lw, label=label, alpha=0.9)

    ax_top.set_xlabel("Test day")
    ax_top.set_ylabel("Realized volatility (annualised)")
    handles, labels = ax_top.get_legend_handles_labels()
    ax_top.legend([hv_patch] + handles, [hv_patch.get_label()] + labels, fontsize=7.5, loc="upper left")
    ax_top.grid(True, alpha=0.25)

    # ---- scatter: predicted vs actual (best QRC) ----
    if best_qrc_key in predictions:
        y_hat_best = predictions[best_qrc_key]
        m_best = metrics_map.get(best_qrc_key, {})
        ax_sc.scatter(y_te, y_hat_best, s=6, alpha=0.4, color="#2166ac")
        lo, hi = min(y_te.min(), y_hat_best.min()), max(y_te.max(), y_hat_best.max())
        ax_sc.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="Perfect fit")
        ax_sc.set_xlabel("True volatility")
        ax_sc.set_ylabel("Predicted volatility")
        ax_sc.set_title(f"{best_qrc_key} — Predicted vs Actual\nR²={m_best.get('R2', float('nan')):.3f}", fontsize=9)
        ax_sc.legend(fontsize=8)
        ax_sc.grid(True, alpha=0.25)

    # ---- rolling 30-day RMSE ----
    if best_qrc_key in predictions:
        rrmse_qrc = _rolling_rmse(y_te, predictions[best_qrc_key])
        ax_rr.plot(days, rrmse_qrc, color="#2166ac", lw=1.4, label=best_qrc_key)
    if "ESN" in predictions:
        rrmse_esn = _rolling_rmse(y_te, predictions["ESN"])
        ax_rr.plot(days, rrmse_esn, color="#d6604d", lw=1.2, ls="--", label="ESN")
    ax_rr.set_xlabel("Test day")
    ax_rr.set_ylabel("30-day rolling RMSE")
    ax_rr.set_title("Rolling RMSE over Test Period", fontsize=9)
    ax_rr.legend(fontsize=8)
    ax_rr.grid(True, alpha=0.25)

    fig.savefig(os.path.join(RESULTS_DIR, f"{save_prefix}.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS_DIR, f"{save_prefix}_doc.png"), dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> results/{save_prefix}.png  (+ _doc.png at 1200 DPI)")


def _plot_rmse_bar(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    colours = ["#2166ac" if "QRC" in r["Model"] else "#d6604d" for _, r in df.iterrows()]
    bars = ax.barh(df["Model"], df["RMSE"], color=colours, edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=7)
    ax.set_xlabel("RMSE (annualised volatility)")
    ax.set_title("SPY Volatility Forecast — RMSE by Model (lower is better)")
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "financial_rmse.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved -> results/financial_rmse.png")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(
    seed: int = 42,
    delay: int = 5,
    fast: bool = False,
    include_15q: bool = False,
) -> pd.DataFrame:
    print("=" * 70)
    print("  SPY MARKET VOLATILITY FORECAST - QRC vs. Baselines")
    print("=" * 70)

    X_tr, X_te, y_tr, y_te = load_financial_data(
        delay=delay, enhanced_features=True
    )
    n_feat = X_tr.shape[1]
    print(f"\nTrain: {len(X_tr)}, Test: {len(X_te)}, Features per sample: {n_feat}")
    print(f"  (delay={delay} days x 3 feature types: raw, |r|, r^2)\n")

    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    metrics_map: dict[str, dict] = {}

    # ---- Classical baselines ----
    print("Classical baselines:")

    for name, model in [
        ("Persistence",   PersistenceBaseline()),
        (f"AR({delay})",  ARBaseline(order=delay)),
        ("Ridge",         RidgeBaseline(alpha=1.0)),
        ("ESN",           EchoStateNetwork(n_reservoir=200, seed=seed)),
    ]:
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)
        m = regression_metrics(y_te, y_hat)
        print_metrics(name, m)
        rows.append({"Model": name, **m})
        predictions[name] = y_hat
        metrics_map[name] = m

    # ---- QRC sweep ----
    print("\nQuantum Reservoir Computing (CV-tuned Ridge alpha):")

    qubit_configs = [5] if fast else ([5, 10, 15] if include_15q else [5, 10])
    layer_configs = [2] if fast else [2, 3]
    conn_configs  = {
        5:  ["random", "all-to-all"],
        10: ["random"],
        15: ["random"],
    }

    best_qrc_nmse = np.inf
    best_qrc_key  = ""

    for n_qubits in qubit_configs:
        for n_layers in layer_configs:
            for conn in (["random"] if fast else conn_configs.get(n_qubits, ["random"])):
                label = f"QRC {n_qubits}q L{n_layers} {conn[:4]}"
                qrc = QuantumReservoir(
                    n_qubits=n_qubits,
                    n_layers=n_layers,
                    connectivity=conn,
                    seed=seed,
                )
                t0 = time.time()
                y_hat, alpha = _qrc_regress(
                    qrc,
                    X_tr[:, :n_qubits],
                    X_te[:, :n_qubits],
                    y_tr,
                    X_tr_full=X_tr,
                    X_te_full=X_te,
                )
                elapsed = time.time() - t0
                m = regression_metrics(y_te, y_hat)
                print_metrics(f"{label} (a={alpha})", m)
                m["elapsed"] = elapsed
                rows.append({"Model": label, **m})
                predictions[label] = y_hat
                metrics_map[label] = m

                if m["NMSE"] < best_qrc_nmse:
                    best_qrc_nmse = m["NMSE"]
                    best_qrc_key  = label

    # ---- Results CSV ----
    df = pd.DataFrame(rows).sort_values("NMSE")
    df.to_csv(os.path.join(RESULTS_DIR, "financial_results.csv"), index=False)
    print(f"\n  Saved -> results/financial_results.csv")
    print(f"  Best QRC config: {best_qrc_key}  (NMSE={best_qrc_nmse:.5f})")

    # ---- Plots ----
    _plot_rmse_bar(df)
    _plot_predictions(
        y_te,
        {k: predictions[k] for k in ["Persistence", f"AR({delay})", "Ridge", "ESN"] + [best_qrc_key] if k in predictions},
        best_qrc_key,
        metrics_map,
        save_prefix="financial_predictions",
    )

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",    action="store_true", help="5q/2L/random only (quick test)")
    parser.add_argument("--full",    action="store_true", help="Include 15-qubit (slow)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--delay",   type=int, default=5)
    args = parser.parse_args()

    df = run(seed=args.seed, delay=args.delay, fast=args.fast, include_15q=args.full)
    print("\n--- Summary (sorted by NMSE) ---")
    print(df[["Model", "RMSE", "MAE", "R2", "NMSE"]].to_string(index=False))
