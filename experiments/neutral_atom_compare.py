"""
Neutral-atom platform comparison — QuEra Aquila vs Pasqal Fresnel (vs classical).

Runs the SPY realized-volatility forecast through TWO independent neutral-atom
analog reservoirs, both on their free local emulators, and scores them against
Persistence and ESN on the same window.

  QuEra Aquila  (src/qrc/quera_reservoir.py)  — Braket AHS; per-site LOCAL detuning
                                                 encodes the input (spatial).
  Pasqal Fresnel(src/qrc/pasqal_reservoir.py) — Pulser; GLOBAL detuning waveform
                                                 encodes the input (temporal), because
                                                 the real Fresnel device has no DMM.

Same fixed-random-geometry reservoir, same Rydberg (Z, ZZ) readout, same Ridge.
The point: an analog Rydberg reservoir is platform-agnostic, and stays competitive
with the best classical baselines on BOTH platforms despite their different
encodings. Everything here is local emulation (free); Pasqal Fresnel was OFFLINE
for the account at build time, so this is the available comparison.

    python experiments/neutral_atom_compare.py --max-train 300 --max-test 100
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
from src.baselines.classical import EchoStateNetwork, regression_metrics, print_metrics
from experiments.quera_aquila_qrc import fit_readout, apply_readout

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
warnings.filterwarnings("ignore")


def run(args) -> None:
    print("=" * 78)
    print("  GIC 2026 PHASE 3 -- Neutral-atom comparison: QuEra Aquila vs Pasqal Fresnel")
    print("=" * 78)

    d = load_financial_data_v2(delay=5, log_target=True, include_har=True,
                               include_log_har=True, residual_target=True)
    X_tr_full, X_te_full = d["X_train"], d["X_test"]
    y_tr = d["y_train"]
    transform = d["target_transform"]
    n = args.atoms

    if args.max_train:
        X_tr_full = X_tr_full[-args.max_train:]; y_tr = y_tr[-args.max_train:]
    n_te = min(args.max_test, len(X_te_full)) if args.max_test else len(X_te_full)
    sl = slice(len(X_te_full) - n_te, len(X_te_full))
    X_te_full = X_te_full[sl]
    y_te_raw = d["y_test_raw"][sl]; pers_te = d["persistence_test"][sl]
    log_pers_te = d["log_persistence_test"][sl]
    to_vol = lambda y: invert_target(y, transform, log_persistence=log_pers_te)

    sc_in = StandardScaler().fit(X_tr_full[:, :n])
    Xtr_in = sc_in.transform(X_tr_full[:, :n])
    Xte_in = sc_in.transform(X_te_full[:, :n])
    n_tr = len(Xtr_in)
    X_in_all = np.vstack([Xtr_in, Xte_in])

    print(f"\nWindow: {n_tr} train / {n_te} test  |  atoms={n}  shots={args.shots}  "
          f"n_jobs={args.n_jobs}\n")

    rows = []
    preds = {}

    def record(name, y_pred, extra=None):
        m = regression_metrics(y_te_raw, y_pred)
        print_metrics(f"{name:<30}", m)
        rows.append({"Model": name, **m, **(extra or {})})
        preds[name] = y_pred

    # ---- classical baselines ----
    record("Persistence", pers_te.copy())
    esn = EchoStateNetwork(n_reservoir=200, seed=args.seed); esn.fit(d["X_train"], d["y_train"])
    record("ESN (200 nodes)", invert_target(esn.predict(X_te_full), transform, log_persistence=log_pers_te))

    # ---- neutral-atom reservoirs (local emulators) ----
    reservoirs = [
        ("QuEra Aquila (local AHS)", QueraReservoir(n_atoms=n, geometry="random2d", seed=args.seed)),
        ("Pasqal Fresnel (local Pulser)", PasqalReservoir(n_atoms=n, geometry="random2d", seed=args.seed)),
    ]
    for label, res in reservoirs:
        print(f"\n  [{label}] {res}")
        t0 = time.time()
        R_all = res.transform(X_in_all, device="local", shots=args.shots, n_jobs=args.n_jobs)
        print(f"    features in {time.time()-t0:.1f}s")
        R_tr, R_te = R_all[:n_tr], R_all[n_tr:]
        ridge, sc_r, alpha = fit_readout(R_tr, X_tr_full, y_tr)
        record(label, to_vol(apply_readout(ridge, sc_r, R_te, X_te_full)),
               extra={"alpha": alpha})

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "neutral_atom_compare.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n  Saved -> {os.path.relpath(out_csv)}")
    print("\n  Leaderboard by RMSE (lower is better):")
    for _, r in df.sort_values("RMSE").iterrows():
        print(f"    {r['Model']:<32} RMSE={r['RMSE']:.5f}  R2={r['R2']:.4f}")

    _plot(y_te_raw, preds, df)
    print("\n--- Done ---")


def _plot(y_true, preds, df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5))
    order = df.sort_values("RMSE")
    colors = {"Persistence": "#9e9e9e", "ESN (200 nodes)": "#d6604d",
              "QuEra Aquila (local AHS)": "#7c3aed", "Pasqal Fresnel (local Pulser)": "#2166ac"}
    bars = ax0.barh(order["Model"], order["RMSE"],
                    color=[colors.get(m, "#888") for m in order["Model"]])
    ax0.bar_label(bars, fmt="%.5f", padding=3, fontsize=8)
    ax0.set_xlabel("RMSE (annualised vol)"); ax0.invert_yaxis()
    ax0.set_title("Neutral-atom vs classical (local emulators)")
    ax0.grid(True, axis="x", alpha=0.25)

    days = np.arange(len(y_true))
    ax1.plot(days, y_true, color="black", lw=2.0, label="True", zorder=5)
    for key in ("QuEra Aquila (local AHS)", "Pasqal Fresnel (local Pulser)", "Persistence"):
        if key in preds:
            ax1.plot(days, preds[key], lw=1.3, color=colors.get(key), label=key,
                     ls="-" if "Aquila" in key else ("--" if "Fresnel" in key else "-."))
    ax1.set_xlabel("Test day"); ax1.set_ylabel("Realized vol")
    ax1.set_title("Forecast overlay"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.25)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "neutral_atom_compare.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".png", "_doc.png"), dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {os.path.relpath(out)}  (+ _doc.png @ 1200 DPI)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--atoms", type=int, default=5)
    p.add_argument("--shots", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train", type=int, default=300)
    p.add_argument("--max-test", type=int, default=100)
    p.add_argument("--n-jobs", type=int, default=4, dest="n_jobs")
    run(p.parse_args())
