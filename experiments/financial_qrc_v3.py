"""
Market Volatility Forecasting — v3 (further improvements over v2)

What's new vs v2
----------------
  1. log-HAR features: log(RV[t-1]), log of 5d/22d means — let the linear readout
     mimic Persistence essentially for free, so QRC's nonlinear features can
     focus on the *residual* signal.
  2. residual target: y = log(RV[t]) - log(RV[t-1]).  Predicting 0 reproduces
     Persistence exactly; any non-zero prediction is a deviation the model
     thinks it can justify from the features.  Removes the regularisation-toward-
     mean disadvantage that hurt v2 vs Persistence.
  3. Recurrent QRC: previous step's <Z> expectations are fed back as Ry angles
     in the next circuit — gives QRC the same kind of temporal-memory advantage
     ESN has had all along.

Outputs
-------
  results/financial_results_v3.csv
  results/financial_rmse_v3.png
  results/financial_predictions_v3.png      (full test window)
  results/financial_predictions_v3_doc.png  (1200 DPI)
  results/improvement_table_v3.csv          (v2 vs v3 side-by-side)
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

_ALPHAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cv_alpha(R_tr: np.ndarray, y_tr: np.ndarray, n_splits: int = 5) -> float:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_alpha, best_mse = _ALPHAS[0], np.inf
    for alpha in _ALPHAS:
        mses = []
        for tr_idx, val_idx in tscv.split(R_tr):
            ridge = Ridge(alpha=alpha).fit(R_tr[tr_idx], y_tr[tr_idx])
            mses.append(np.mean((y_tr[val_idx] - ridge.predict(R_tr[val_idx])) ** 2))
        if (m := float(np.mean(mses))) < best_mse:
            best_mse, best_alpha = m, alpha
    return best_alpha


def _qrc_regress(
    qrc: QuantumReservoir,
    X_tr: np.ndarray, X_te: np.ndarray,
    y_tr: np.ndarray,
    X_tr_full: np.ndarray, X_te_full: np.ndarray,
    sequential: bool = False,
) -> tuple[np.ndarray, float]:
    sc_in = StandardScaler()
    X_tr_s = sc_in.fit_transform(X_tr)
    X_te_s = sc_in.transform(X_te)

    if sequential:
        R_tr, z_final = qrc.transform_sequential(X_tr_s, return_final_z=True)
        R_te = qrc.transform_sequential(X_te_s, initial_z=z_final)
    else:
        R_tr = qrc.transform(X_tr_s)
        R_te = qrc.transform(X_te_s)

    # Hybrid: concatenate the original (already-rich) features so the readout
    # has direct access to log-HAR + returns + their nonlinear quantum encodings.
    R_tr = np.hstack([R_tr, X_tr_full])
    R_te = np.hstack([R_te, X_te_full])

    sc_r = StandardScaler()
    R_tr_s2 = sc_r.fit_transform(R_tr)
    R_te_s2 = sc_r.transform(R_te)

    alpha = _cv_alpha(R_tr_s2, y_tr)
    readout = Ridge(alpha=alpha).fit(R_tr_s2, y_tr)
    return readout.predict(R_te_s2), alpha


def _rolling_rmse(y_true: np.ndarray, y_pred: np.ndarray, window: int = 30) -> np.ndarray:
    res = (y_true - y_pred) ** 2
    return np.array([
        np.sqrt(np.mean(res[max(0, i - window): i + 1]))
        for i in range(len(res))
    ])


# ---------------------------------------------------------------------------
# Plotting (same layout as v2 but updated title)
# ---------------------------------------------------------------------------

def _plot_predictions(
    y_te: np.ndarray,
    predictions: dict[str, np.ndarray],
    best_qrc_key: str,
    metrics_map: dict[str, dict],
    save_prefix: str,
) -> None:
    high_vol_thresh = np.percentile(y_te, 75)
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("SPY Realized Volatility — v3 (log-HAR + residual target + recurrent QRC)", fontsize=12, y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    ax_top = fig.add_subplot(gs[0, :])
    ax_sc  = fig.add_subplot(gs[1, 0])
    ax_rr  = fig.add_subplot(gs[1, 1])

    days = np.arange(len(y_te))
    high_vol = y_te > high_vol_thresh
    in_regime, start = False, 0
    for i, hv in enumerate(high_vol):
        if hv and not in_regime:
            start, in_regime = i, True
        elif not hv and in_regime:
            ax_top.axvspan(start, i, color="#ffcccc", alpha=0.4)
            in_regime = False
    if in_regime:
        ax_top.axvspan(start, len(y_te) - 1, color="#ffcccc", alpha=0.4)

    import matplotlib.patches as mpatches
    hv_patch = mpatches.Patch(color="#ffcccc", alpha=0.6, label="High vol (>P75)")

    ax_top.plot(days, y_te, color="black", lw=2.0, label="True volatility", zorder=5)
    plot_order = [best_qrc_key, "ESN", "Ridge", "Persistence"]
    colors     = ["#2166ac", "#d6604d", "#4dac26", "#9e9e9e"]
    styles     = ["-",        "--",      ":",      "-."]
    widths     = [1.8,        1.3,       1.3,      1.0]

    for key, c, ls, lw in zip(plot_order, colors, styles, widths):
        if key not in predictions:
            continue
        m = metrics_map.get(key, {})
        ax_top.plot(days, predictions[key], color=c, ls=ls, lw=lw,
                    label=f"{key}  (NMSE={m.get('NMSE', float('nan')):.4f}, R²={m.get('R2', float('nan')):.4f})",
                    alpha=0.9)

    ax_top.set_xlabel("Test day"); ax_top.set_ylabel("Realized volatility (annualised)")
    handles, labels = ax_top.get_legend_handles_labels()
    ax_top.legend([hv_patch] + handles, [hv_patch.get_label()] + labels, fontsize=7.5, loc="upper left")
    ax_top.grid(True, alpha=0.25)

    if best_qrc_key in predictions:
        y_hat_best = predictions[best_qrc_key]
        m_best = metrics_map.get(best_qrc_key, {})
        ax_sc.scatter(y_te, y_hat_best, s=6, alpha=0.4, color="#2166ac")
        lo, hi = min(y_te.min(), y_hat_best.min()), max(y_te.max(), y_hat_best.max())
        ax_sc.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="Perfect fit")
        ax_sc.set_xlabel("True volatility"); ax_sc.set_ylabel("Predicted volatility")
        ax_sc.set_title(f"{best_qrc_key} — Predicted vs Actual\nR²={m_best.get('R2', float('nan')):.4f}", fontsize=9)
        ax_sc.legend(fontsize=8); ax_sc.grid(True, alpha=0.25)

    for key, c, ls in [(best_qrc_key, "#2166ac", "-"), ("ESN", "#d6604d", "--"), ("Persistence", "#9e9e9e", "-.")]:
        if key in predictions:
            ax_rr.plot(days, _rolling_rmse(y_te, predictions[key]), color=c, lw=1.3, ls=ls, label=key)
    ax_rr.set_xlabel("Test day"); ax_rr.set_ylabel("30-day rolling RMSE")
    ax_rr.set_title("Rolling RMSE over Test Period", fontsize=9)
    ax_rr.legend(fontsize=8); ax_rr.grid(True, alpha=0.25)

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
    ax.set_title("SPY Volatility Forecast v3 — RMSE (lower is better)")
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "financial_rmse_v3.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved -> results/financial_rmse_v3.png")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(seed: int = 42, delay: int = 5, fast: bool = False, include_15q: bool = False) -> pd.DataFrame:
    print("=" * 78)
    print("  SPY VOLATILITY FORECAST v3 — log-HAR + residual target + recurrent QRC")
    print("=" * 78)

    d = load_financial_data_v2(
        delay=delay,
        log_target=True,
        include_har=True,
        include_log_har=True,
        residual_target=True,
        include_return_features=True,
    )
    X_tr, X_te = d["X_train"], d["X_test"]
    y_tr, y_te = d["y_train"], d["y_test"]
    y_te_raw = d["y_test_raw"]
    log_pers_te = d["log_persistence_test"]
    transform = d["target_transform"]

    n_feat = X_tr.shape[1]
    print(f"\nTrain: {len(X_tr)}, Test: {len(X_te)}, Features per sample: {n_feat}")
    print(f"  Feature breakdown: {d['feature_names']}")
    print(f"  Target transform: {transform}  (predicting 0 == Persistence)\n")

    def _to_vol(y_pred_trans: np.ndarray) -> np.ndarray:
        return invert_target(y_pred_trans, transform, log_pers_te)

    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    metrics_map: dict[str, dict] = {}

    # ---- Persistence (predict residual = 0) ----
    print("Classical baselines:")
    y_hat_pers = _to_vol(np.zeros_like(y_te))
    m = regression_metrics(y_te_raw, y_hat_pers)
    print_metrics("Persistence (residual=0)", m)
    rows.append({"Model": "Persistence", **m})
    predictions["Persistence"] = y_hat_pers
    metrics_map["Persistence"] = m

    # ---- AR / Ridge / ESN ----
    for name, model in [
        (f"AR({delay})", ARBaseline(order=delay)),
        ("Ridge",        RidgeBaseline(alpha=1.0)),
        ("ESN",          EchoStateNetwork(n_reservoir=200, seed=seed)),
    ]:
        model.fit(X_tr, y_tr)
        y_hat = _to_vol(model.predict(X_te))
        m = regression_metrics(y_te_raw, y_hat)
        print_metrics(name, m)
        rows.append({"Model": name, **m})
        predictions[name] = y_hat
        metrics_map[name] = m

    # ---- QRC sweep: stateless and recurrent ----
    print("\nQuantum Reservoir Computing (stateless + recurrent):")
    qubit_configs = [5] if fast else ([5, 10, 15] if include_15q else [5, 10])
    layer_configs = [2] if fast else [2, 3]
    conn_configs  = {5: ["random", "all-to-all"], 10: ["random"], 15: ["random"]}

    best_qrc_nmse = np.inf
    best_qrc_key  = ""

    for n_qubits in qubit_configs:
        for n_layers in layer_configs:
            for conn in (["random"] if fast else conn_configs.get(n_qubits, ["random"])):
                for rec_mode in (False, True):
                    suffix = "rec" if rec_mode else "sta"
                    label = f"QRC {n_qubits}q L{n_layers} {conn[:4]} [{suffix}]"
                    qrc = QuantumReservoir(
                        n_qubits=n_qubits,
                        n_layers=n_layers,
                        connectivity=conn,
                        seed=seed,
                        recurrent=rec_mode,
                        memory_scale=0.5,
                    )
                    t0 = time.time()
                    y_hat_trans, alpha = _qrc_regress(
                        qrc,
                        X_tr[:, :n_qubits], X_te[:, :n_qubits], y_tr,
                        X_tr_full=X_tr, X_te_full=X_te,
                        sequential=rec_mode,
                    )
                    y_hat = _to_vol(y_hat_trans)
                    elapsed = time.time() - t0
                    m = regression_metrics(y_te_raw, y_hat)
                    print_metrics(f"{label} (a={alpha})", m)
                    m["elapsed"] = elapsed
                    rows.append({"Model": label, **m})
                    predictions[label] = y_hat
                    metrics_map[label] = m

                    if m["NMSE"] < best_qrc_nmse:
                        best_qrc_nmse = m["NMSE"]
                        best_qrc_key  = label

    df = pd.DataFrame(rows).sort_values("NMSE")
    df.to_csv(os.path.join(RESULTS_DIR, "financial_results_v3.csv"), index=False)
    print(f"\n  Saved -> results/financial_results_v3.csv")
    print(f"  Best QRC config: {best_qrc_key}  (NMSE={best_qrc_nmse:.5f})")

    _plot_rmse_bar(df)
    _plot_predictions(
        y_te_raw,
        {k: predictions[k] for k in ["Persistence", f"AR({delay})", "Ridge", "ESN"] + [best_qrc_key] if k in predictions},
        best_qrc_key,
        metrics_map,
        save_prefix="financial_predictions_v3",
    )

    # ---- v2 vs v3 comparison ----
    v2_path = os.path.join(RESULTS_DIR, "financial_results_v2.csv")
    if os.path.exists(v2_path):
        v2 = pd.read_csv(v2_path)[["Model", "RMSE", "R2", "NMSE"]].set_index("Model")
        v3 = df[["Model", "RMSE", "R2", "NMSE"]].set_index("Model")
        joined = v2.join(v3, lsuffix="_v2", rsuffix="_v3", how="outer")
        joined["RMSE_delta"] = joined["RMSE_v3"] - joined["RMSE_v2"]
        joined.to_csv(os.path.join(RESULTS_DIR, "improvement_table_v3.csv"))
        print("  Saved -> results/improvement_table_v3.csv")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",  action="store_true")
    parser.add_argument("--full",  action="store_true", help="Include 15q (slow)")
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--delay", type=int, default=5)
    args = parser.parse_args()

    df = run(seed=args.seed, delay=args.delay, fast=args.fast, include_15q=args.full)
    print("\n--- Summary (sorted by NMSE) ---")
    print(df[["Model", "RMSE", "MAE", "R2", "NMSE"]].to_string(index=False))
