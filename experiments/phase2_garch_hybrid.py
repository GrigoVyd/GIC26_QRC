"""
Phase 2 -- GARCH-hybrid QRC experiment.

Two ways to apply the GARCH "rolling-window proxy" idea inside the QRC pipeline:

  A. GARCH proxy as exogenous feature
     Add (garch_proxy_rv, log_garch_proxy_rv) as two extra input columns.
     The QRC's hybrid Ridge head can use them directly; the reservoir
     contributes nonlinear corrections on top.

  B. GARCH-residual target
     Target = log(RV[t]) - log(RV_garch_proxy[t]).
     Predicting 0 corresponds exactly to GARCH. Any non-zero output is the
     QRC's contribution. Strictly comparable to GARCH on the same scale.

Three reference rows for context:
  * GARCH(1,1) standalone
  * Ridge on the feature-augmented X (linear-only test of the augmentation)
  * Annealer QRC, t=2.0 alpha=1.0, single seed (same Stage-1 best as phase2_final)

This is run after phase2_final.py so the leaderboards are directly comparable.

Outputs
-------
  results/phase2_garch_hybrid.csv
  results/phase2_garch_hybrid.png
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.qrc.annealer_reservoir import AnnealerReservoir
from src.data.loaders import load_financial_data_v2, invert_target
from src.baselines.classical import (
    RidgeBaseline, regression_metrics, mincer_zarnowitz, print_metrics,
)
from src.baselines.garch import GARCHBaseline

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
warnings.filterwarnings("ignore")

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


def _qrc_regress(res, X_tr, X_te, y_tr, X_tr_full, X_te_full):
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


def run(seed: int = 42, n_seeds: int = 3) -> pd.DataFrame:
    print("=" * 78)
    print("  PHASE 2 -- GARCH-hybrid QRC experiment")
    print("    Option A: GARCH proxy as feature  |  Option B: GARCH-residual target")
    print("=" * 78)

    rows: list[dict] = []

    def record(name, y_pred, y_true_vol, seed_val=0, elapsed=0.0, config=""):
        m = regression_metrics(y_true_vol, y_pred)
        mz = mincer_zarnowitz(y_true_vol, y_pred)
        m.update(mz)
        m["elapsed"] = elapsed; m["seed"] = seed_val; m["config"] = config
        print_metrics(f"{name:<42} (seed={seed_val})", m)
        rows.append({"Model": name, **m})
        return m

    # ============================================================
    # 1. Reference: GARCH(1,1) standalone (no QRC)
    # ============================================================
    print("\n[Ref] GARCH(1,1) standalone")
    print("-" * 78)
    d_base = load_financial_data_v2(
        delay=5, log_target=True, include_har=True,
        include_log_har=True, residual_target=True,
    )
    g = GARCHBaseline(vol_window=d_base["vol_window"])
    g.set_returns(d_base["log_returns_full"], n_train=d_base["test_first_ret_idx"])
    g.fit(d_base["X_train"], d_base["y_train"])
    record("GARCH(1,1) standalone", g.predict(d_base["X_test"]),
            d_base["y_test_raw"], config="garch_only")

    # ============================================================
    # 2. Option A: GARCH proxy as feature, Persistence-residual target
    # ============================================================
    print("\n[A] GARCH proxy as feature, Persistence-residual target")
    print("-" * 78)
    d_a = load_financial_data_v2(
        delay=5, log_target=True, include_har=True, include_log_har=True,
        residual_target=True, include_garch_proxy=True,
    )
    print(f"  Features: {len(d_a['feature_names'])} = {d_a['feature_names']}")
    to_vol_a = lambda y: invert_target(
        y, d_a["target_transform"], log_persistence=d_a["log_persistence_test"]
    )

    # 2a. Ridge-only on augmented X (linear-only test)
    t0 = time.time()
    r_aug = RidgeBaseline(alpha=1.0)
    r_aug.fit(d_a["X_train"], d_a["y_train"])
    record("Ridge + GARCH-feature (Persistence resid.)",
            to_vol_a(r_aug.predict(d_a["X_test"])),
            d_a["y_test_raw"], elapsed=time.time() - t0, config="A-ridge")

    # 2b. Annealer QRC + GARCH-feature, n_seeds seeds
    qrc_a_preds = []
    for s_off in range(n_seeds):
        seed_i = seed + s_off
        t0 = time.time()
        ar = AnnealerReservoir(n_qubits=10, n_input=10, evolution_time=2.0,
                                trotter_steps=3, connectivity="all-to-all",
                                input_scale=1.0, seed=seed_i)
        # First 10 input cols (the n_qubits-many) used to drive h_i; full X to head
        y_hat_trans, _ = _qrc_regress(ar, d_a["X_train"][:, :10],
                                       d_a["X_test"][:, :10], d_a["y_train"],
                                       d_a["X_train"], d_a["X_test"])
        y_vol = to_vol_a(y_hat_trans)
        qrc_a_preds.append(y_vol)
        record("QRC + GARCH-feature (Persistence resid.)",
                y_vol, d_a["y_test_raw"],
                seed_val=seed_i, elapsed=time.time() - t0, config="A-qrc")

    qrc_a_ens = np.mean(np.stack(qrc_a_preds, axis=0), axis=0)
    record("QRC + GARCH-feature (3-seed ensemble)",
            qrc_a_ens, d_a["y_test_raw"], config="A-qrc-ensemble")

    # ============================================================
    # 3. Option B: GARCH-residual target (predicting what GARCH misses)
    # ============================================================
    print("\n[B] GARCH-residual target, GARCH proxy also in features")
    print("-" * 78)
    d_b = load_financial_data_v2(
        delay=5, log_target=True, include_har=True, include_log_har=True,
        include_garch_proxy=True, garch_residual_target=True,
    )
    print(f"  target_transform: {d_b['target_transform']}  (predicting 0 == GARCH)")
    to_vol_b = lambda y: invert_target(
        y, d_b["target_transform"], log_garch_proxy=d_b["log_garch_proxy_test"]
    )

    # 3a. Ridge-only on GARCH-residual target
    t0 = time.time()
    r_b = RidgeBaseline(alpha=1.0)
    r_b.fit(d_b["X_train"], d_b["y_train"])
    record("Ridge on GARCH-residual target",
            to_vol_b(r_b.predict(d_b["X_test"])),
            d_b["y_test_raw"], elapsed=time.time() - t0, config="B-ridge")

    # 3b. Annealer QRC on GARCH-residual target, n_seeds seeds
    qrc_b_preds = []
    for s_off in range(n_seeds):
        seed_i = seed + s_off
        t0 = time.time()
        ar = AnnealerReservoir(n_qubits=10, n_input=10, evolution_time=2.0,
                                trotter_steps=3, connectivity="all-to-all",
                                input_scale=1.0, seed=seed_i)
        y_hat_trans, _ = _qrc_regress(ar, d_b["X_train"][:, :10],
                                       d_b["X_test"][:, :10], d_b["y_train"],
                                       d_b["X_train"], d_b["X_test"])
        y_vol = to_vol_b(y_hat_trans)
        qrc_b_preds.append(y_vol)
        record("QRC on GARCH-residual target",
                y_vol, d_b["y_test_raw"],
                seed_val=seed_i, elapsed=time.time() - t0, config="B-qrc")

    qrc_b_ens = np.mean(np.stack(qrc_b_preds, axis=0), axis=0)
    record("QRC on GARCH-residual (3-seed ensemble)",
            qrc_b_ens, d_b["y_test_raw"], config="B-qrc-ensemble")

    # ============================================================
    # Save + plot
    # ============================================================
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "phase2_garch_hybrid.csv"), index=False)
    print(f"\n  Saved -> results/phase2_garch_hybrid.csv")

    # Bar chart of distinct models (drop per-seed rows in favour of ensembles)
    keep = df[df["config"].isin(["garch_only", "A-ridge", "A-qrc-ensemble",
                                  "B-ridge", "B-qrc-ensemble"])].copy()
    keep = keep.sort_values("RMSE")

    fig, ax = plt.subplots(figsize=(11, 5))
    def _color(name):
        if "QRC" in name:    return "#7c3aed"
        if "Ridge" in name:  return "#3a7d3a"
        return "#d6604d"
    bars = ax.barh(keep["Model"], keep["RMSE"], color=[_color(m) for m in keep["Model"]])
    ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=8)
    ax.set_xlabel("RMSE (annualised volatility)")
    ax.set_title("GARCH-hybrid QRC -- Option A (feature) vs Option B (residual target)")
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "phase2_garch_hybrid.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS_DIR, "phase2_garch_hybrid_doc.png"), dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved -> results/phase2_garch_hybrid.png")

    print("\n--- Summary (sorted by RMSE) ---")
    print(keep[["Model", "RMSE", "R2", "QLIKE"]].to_string(index=False))
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=3)
    args = parser.parse_args()
    run(seed=args.seed, n_seeds=args.n_seeds)
