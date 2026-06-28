"""
Phase 3 — GARCH-hybrid on neutral-atom hardware (QuEra Aquila / Pasqal Fresnel).

Reproduces the Phase 2 headline (docs/phase2_submission/garch_hybrid_explained.md)
on the real neutral-atom platforms: the reservoir predicts only the GARCH
RESIDUAL (y = log RV - log RV_garch), so "predict 0" == GARCH exactly and any
non-zero output is the reservoir's contribution. GARCH proxy is also a readout
feature. The Ridge-on-residual row is the ablation: it should NOT beat GARCH, so
any win is isolable to the reservoir's nonlinear feature expansion.

Same recipe as phase2_garch_hybrid.py, but with the local AHS / Pulser reservoirs
(and ready for the real QuEra hardware path in quera_aquila_qrc.py).

    python experiments/neutral_atom_garch_hybrid.py --atoms 5 --max-train 600 --max-test 250
    python experiments/neutral_atom_garch_hybrid.py --atoms 5 --reservoir pasqal --recurrent

Scaling note: more atoms => more features AND more encoded inputs, but the local
emulators are CPU-bound (QuEra AHS ~1s/program at 5 atoms, ~15s at 8). Larger atom
counts want a GPU emulator (CUDA-Q) or the native QuEra hardware (256 atoms, no
classical sim cost). See docs/quera_aquila_setup.md.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.preprocessing import StandardScaler

from src.qrc.quera_reservoir import QueraReservoir
from src.qrc.pasqal_reservoir import PasqalReservoir
from src.data.loaders import load_financial_data_v2, invert_target
from src.baselines.classical import RidgeBaseline, regression_metrics, print_metrics
from src.baselines.garch import GARCHBaseline
from experiments.quera_aquila_qrc import fit_readout, apply_readout

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
warnings.filterwarnings("ignore")


def _make_reservoir(kind: str, n: int, seed: int):
    if kind == "quera":
        return QueraReservoir(n_atoms=n, geometry="random2d", seed=seed)
    if kind == "pasqal":
        return PasqalReservoir(n_atoms=n, geometry="random2d", seed=seed)
    raise ValueError(kind)


def _reservoir_features(res, X_in_all, shots, n_jobs, recurrent):
    if recurrent and hasattr(res, "transform_sequential"):
        # sequential: state carries across samples (no parallelism)
        return res.transform_sequential(X_in_all, shots=shots)
    return res.transform(X_in_all, device="local", shots=shots, n_jobs=n_jobs)


def run(args) -> None:
    print("=" * 78)
    print("  GIC 2026 PHASE 3 -- GARCH-hybrid on neutral-atom reservoirs")
    print("  Reservoir predicts the GARCH residual (predict 0 == GARCH)")
    print("=" * 78)

    # GARCH-residual config: proxy as feature AND as the residual baseline.
    d = load_financial_data_v2(
        delay=5, log_target=True, include_har=True, include_log_har=True,
        include_garch_proxy=True, garch_residual_target=True,
    )
    X_tr_full, X_te_full = d["X_train"], d["X_test"]
    y_tr = d["y_train"]
    n = args.atoms
    shots = None if args.noiseless else args.shots   # None => exact/infinite-shot (Pasqal only)

    if args.max_train:
        X_tr_full = X_tr_full[-args.max_train:]; y_tr = y_tr[-args.max_train:]
    n_te = min(args.max_test, len(X_te_full)) if args.max_test else len(X_te_full)
    sl = slice(len(X_te_full) - n_te, len(X_te_full))
    X_te_full = X_te_full[sl]
    y_te_raw = d["y_test_raw"][sl]
    log_gp_te = d["log_garch_proxy_test"][sl]
    to_vol = lambda y: invert_target(y, "residual_log_garch", log_garch_proxy=log_gp_te)

    print(f"\nWindow: {len(X_tr_full)} train / {n_te} test  |  atoms={n}  "
          f"shots={'noiseless' if shots is None else shots}  recurrent={args.recurrent}\n")

    rows = []
    preds = {}

    def record(name, y_pred, extra=None):
        m = regression_metrics(y_te_raw, y_pred)
        print_metrics(f"{name:<38}", m)
        rows.append({"Model": name, **m, **(extra or {})})
        preds[name] = y_pred

    # ---- Reference: GARCH(1,1) standalone (the strong baseline to beat) ----
    g = GARCHBaseline(vol_window=d["vol_window"])
    g.set_returns(d["log_returns_full"], n_train=d["test_first_ret_idx"])
    g.fit(d["X_train"], d["y_train"])
    record("GARCH(1,1) standalone", g.predict(d["X_test"])[sl])

    # ---- Ablation: Ridge on the GARCH residual (should NOT beat GARCH) ----
    r = RidgeBaseline(alpha=1.0); r.fit(d["X_train"], d["y_train"])
    record("Ridge on GARCH-residual (ablation)",
           invert_target(r.predict(X_te_full), "residual_log_garch", log_garch_proxy=log_gp_te))

    # ---- Neutral-atom reservoirs on the GARCH residual ----
    sc_in = StandardScaler().fit(X_tr_full[:, :n])
    n_tr = len(X_tr_full)
    X_in_all = np.vstack([sc_in.transform(X_tr_full[:, :n]),
                          sc_in.transform(X_te_full[:, :n])])

    kinds = ["quera", "pasqal"] if args.reservoir == "both" else [args.reservoir]
    if shots is None and "quera" in kinds:
        print("  [note] --noiseless is Pasqal-only (QuEra AHS sim is shot-based); skipping QuEra.")
        kinds = [k for k in kinds if k != "quera"]
    labels = {"quera": "QuEra Aquila", "pasqal": "Pasqal Fresnel"}
    for kind in kinds:
        base = labels[kind] + (" +recurrent" if args.recurrent else "") + (" [noiseless]" if shots is None else "")
        seed_preds = []
        for s_off in range(args.n_seeds):
            seed_i = args.seed + s_off
            res = _make_reservoir(kind, n, seed_i)
            t0 = time.time()
            R_all = _reservoir_features(res, X_in_all, shots, args.n_jobs, args.recurrent)
            R_tr, R_te = R_all[:n_tr], R_all[n_tr:]
            ridge, sc_r, alpha = fit_readout(R_tr, X_tr_full, y_tr)
            pred = to_vol(apply_readout(ridge, sc_r, R_te, X_te_full))
            seed_preds.append(pred)
            if args.n_seeds > 1:
                print(f"    seed {seed_i}: {time.time()-t0:.1f}s")
        if args.n_seeds > 1:
            record(f"{base} on GARCH-residual ({args.n_seeds}-seed ens)",
                   np.mean(np.stack(seed_preds), axis=0))
        else:
            record(f"{base} on GARCH-residual", seed_preds[0], extra={"alpha": alpha})

    # ---- Save + leaderboard ----
    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "neutral_atom_garch_hybrid.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n  Saved -> {os.path.relpath(out_csv)}")
    garch_rmse = df[df["Model"] == "GARCH(1,1) standalone"]["RMSE"].iloc[0]
    print(f"\n  Leaderboard by RMSE (GARCH standalone = {garch_rmse:.5f}):")
    for _, r in df.sort_values("RMSE").iterrows():
        flag = "  <- beats GARCH" if r["RMSE"] < garch_rmse and "GARCH(1,1)" not in r["Model"] else ""
        print(f"    {r['Model']:<40} RMSE={r['RMSE']:.5f}  R2={r['R2']:.4f}  QLIKE={r['QLIKE']:.5f}{flag}")

    _plot(df, garch_rmse)
    print("\n--- Done ---")


def _plot(df, garch_rmse):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = df.sort_values("RMSE")
    def _c(m):
        if "QuEra" in m: return "#7c3aed"
        if "Pasqal" in m: return "#2166ac"
        if "Ridge" in m: return "#3a7d3a"
        return "#d6604d"
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(order["Model"], order["RMSE"], color=[_c(m) for m in order["Model"]])
    ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=8)
    ax.axvline(garch_rmse, color="black", ls="--", lw=1, label=f"GARCH = {garch_rmse:.5f}")
    ax.set_xlabel("RMSE (annualised vol)"); ax.invert_yaxis()
    ax.set_title("Phase 3 -- GARCH-hybrid on neutral-atom reservoirs (predict GARCH residual)")
    ax.legend(fontsize=8); ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "neutral_atom_garch_hybrid.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".png", "_doc.png"), dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {os.path.relpath(out)}  (+ _doc.png @ 1200 DPI)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reservoir", default="both", choices=["quera", "pasqal", "both"])
    p.add_argument("--atoms", type=int, default=5)
    p.add_argument("--shots", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train", type=int, default=600)
    p.add_argument("--max-test", type=int, default=250)
    p.add_argument("--n-jobs", type=int, default=4, dest="n_jobs")
    p.add_argument("--n-seeds", type=int, default=1, dest="n_seeds",
                   help="ensemble size (Phase 2 headline used 3)")
    p.add_argument("--recurrent", action="store_true", help="use sequential memory feedback")
    p.add_argument("--noiseless", action="store_true",
                   help="exact/infinite-shot features (Pasqal only) -- Phase 2 methodology")
    run(p.parse_args())
