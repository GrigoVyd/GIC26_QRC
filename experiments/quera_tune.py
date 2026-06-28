"""
QuEra Aquila reservoir tuning sweep (local AHS simulator — free).

Scans atom count / geometry / evolution time / local-detuning strength and ranks
each config by forecast RMSE on a recent window, against Persistence and ESN.
Use the winner for the scaled-up validation and the real Aquila run.

Runs entirely on LocalSimulator("braket_ahs"); no credits. The local solver is
~2-3 s/program, so this sweep takes tens of minutes — run it in the background.

    python experiments/quera_tune.py --max-train 70 --max-test 25 --shots 100
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
from src.data.loaders import load_financial_data_v2, invert_target
from src.baselines.classical import EchoStateNetwork, regression_metrics
from experiments.quera_aquila_qrc import fit_readout, apply_readout

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
warnings.filterwarnings("ignore")

# Reservoir configs to scan (kept <= 8 atoms so the local solver stays tractable).
CONFIGS = [
    dict(name="6 rand2d T4 ld2.5", n_atoms=6, geometry="random2d", total_time=4e-6, local_detuning_max=2.5e7),
    dict(name="6 rand2d T2 ld2.5", n_atoms=6, geometry="random2d", total_time=2e-6, local_detuning_max=2.5e7),
    dict(name="6 chain  T4 ld2.5", n_atoms=6, geometry="chain",    total_time=4e-6, local_detuning_max=2.5e7),
    dict(name="6 ring   T4 ld2.5", n_atoms=6, geometry="ring",     total_time=4e-6, local_detuning_max=2.5e7),
    dict(name="6 rand2d T4 ld1.5", n_atoms=6, geometry="random2d", total_time=4e-6, local_detuning_max=1.5e7),
    dict(name="6 rand2d T4 ld4.0", n_atoms=6, geometry="random2d", total_time=4e-6, local_detuning_max=4.0e7),
    dict(name="5 rand2d T4 ld2.5", n_atoms=5, geometry="random2d", total_time=4e-6, local_detuning_max=2.5e7),
    dict(name="8 rand2d T4 ld2.5", n_atoms=8, geometry="random2d", total_time=4e-6, local_detuning_max=2.5e7),
]


def run(args) -> None:
    print("=" * 78)
    print("  QuEra Aquila reservoir tuning sweep (local AHS sim)")
    print("=" * 78)

    d = load_financial_data_v2(delay=5, log_target=True, include_har=True,
                               include_log_har=True, residual_target=True)
    X_tr_full, X_te_full = d["X_train"], d["X_test"]
    y_tr = d["y_train"]
    transform = d["target_transform"]

    if args.max_train:
        X_tr_full = X_tr_full[-args.max_train:]; y_tr = y_tr[-args.max_train:]
    n_te = min(args.max_test, len(X_te_full))
    sl = slice(len(X_te_full) - n_te, len(X_te_full))
    X_te_full = X_te_full[sl]
    y_te_raw = d["y_test_raw"][sl]; pers_te = d["persistence_test"][sl]
    log_pers_te = d["log_persistence_test"][sl]
    to_vol = lambda y: invert_target(y, transform, log_persistence=log_pers_te)

    print(f"Window: {len(X_tr_full)} train / {n_te} test, shots={args.shots}, "
          f"{len(CONFIGS)} configs\n")

    rows = []

    def score(name, y_pred, extra=None):
        m = regression_metrics(y_te_raw, y_pred)
        rows.append({"Model": name, **{k: m[k] for k in ("RMSE", "MAE", "R2", "QLIKE")}, **(extra or {})})
        print(f"  {name:<28} RMSE={m['RMSE']:.5f}  R2={m['R2']:.4f}")
        return m["RMSE"]

    # Classical references (once).
    score("Persistence", pers_te.copy())
    esn = EchoStateNetwork(n_reservoir=200, seed=args.seed); esn.fit(d["X_train"], d["y_train"])
    score("ESN (200 nodes)", invert_target(esn.predict(X_te_full), transform, log_persistence=log_pers_te))

    print()
    for cfg in CONFIGS:
        n = cfg["n_atoms"]
        t0 = time.time()
        res = QueraReservoir(
            n_atoms=n, geometry=cfg["geometry"], total_time=cfg["total_time"],
            local_detuning_max=cfg["local_detuning_max"], seed=args.seed,
        )
        sc_in = StandardScaler().fit(X_tr_full[:, :n])
        R_tr = res.transform(sc_in.transform(X_tr_full[:, :n]), device="local", shots=args.shots, n_jobs=args.n_jobs)
        R_te = res.transform(sc_in.transform(X_te_full[:, :n]), device="local", shots=args.shots, n_jobs=args.n_jobs)
        ridge, sc_r, alpha = fit_readout(R_tr, X_tr_full, y_tr)
        pred = to_vol(apply_readout(ridge, sc_r, R_te, X_te_full))
        score(f"QuEra {cfg['name']}", pred,
              extra={"config": cfg["name"], "n_atoms": n, "alpha": alpha,
                     "secs": round(time.time() - t0, 1)})
        # incremental save so a long run is never lost
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "quera_tune.csv"), index=False)

    df = pd.DataFrame(rows)
    quera = df[df["Model"].str.startswith("QuEra")].sort_values("RMSE")
    pers_rmse = df[df["Model"] == "Persistence"]["RMSE"].iloc[0]
    esn_rmse = df[df["Model"] == "ESN (200 nodes)"]["RMSE"].iloc[0]
    print("\n" + "=" * 78)
    print(f"  Baselines:  Persistence RMSE={pers_rmse:.5f}   ESN RMSE={esn_rmse:.5f}")
    print("  QuEra configs ranked by RMSE (lower is better):")
    for _, r in quera.iterrows():
        flag = " <- beats both baselines" if r["RMSE"] < min(pers_rmse, esn_rmse) else ""
        print(f"    {r['config']:<22} RMSE={r['RMSE']:.5f}  R2={r['R2']:.4f}  ({r['secs']}s){flag}")
    best = quera.iloc[0]
    print(f"\n  BEST: {best['config']}  (RMSE={best['RMSE']:.5f})")
    print(f"  Saved -> results/quera_tune.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-train", type=int, default=70)
    p.add_argument("--max-test", type=int, default=25)
    p.add_argument("--shots", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=4, dest="n_jobs",
                   help="local-sim parallel workers (1=serial, -1=all cores)")
    run(p.parse_args())
