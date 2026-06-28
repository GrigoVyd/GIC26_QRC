"""
QuEra Aquila QRC experiment — the project's declared PRIMARY hardware path.

Runs the SPY realized-volatility forecast through a native neutral-atom analog
reservoir (src/qrc/quera_reservoir.py) on QuEra Aquila via Amazon Braket AHS.
The reservoir's transverse-field-Ising physics is Aquila's native Hamiltonian, so
this is the real-hardware realization of the Phase-2 headline model (not a gate
Trotterization of it).

Strategy (same economical hybrid as the gate-QPU experiment)
-----------------------------------------------------------
The readout is trained on FREE, exact features from the local AHS simulator
(`LocalSimulator("braket_ahs")`); only the most-recent `--max-test` window is run
on Aquila. The local-sim-vs-Aquila gap is the hardware story.

What it reports
---------------
* Feature fidelity: Aquila Rydberg (Z, ZZ) features vs the local-sim reference.
* Forecast leaderboard: Persistence, ESN, QuEra-sim, QuEra-hardware on the same
  test window. Honest claim: the analog reservoir runs on real neutral-atom
  hardware and stays competitive with the best classical baselines.

Examples
--------
    # 0) Footprint only, no run, no AWS/credits needed:
    python experiments/quera_aquila_qrc.py --dry-run --max-test 40

    # 1) Full validation on the FREE local AHS simulator:
    python experiments/quera_aquila_qrc.py --device local --max-test 60

    # 2) Real QuEra Aquila run (spends credits via qBraid; note the latch).
    #    Aquila has finite availability windows — check status first.
    python experiments/quera_aquila_qrc.py --device aquila --max-test 40 --shots 100 --allow-qpu
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

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.qrc.quera_reservoir import QueraReservoir
from src.data.loaders import load_financial_data_v2, invert_target
from src.baselines.classical import EchoStateNetwork, regression_metrics, print_metrics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
warnings.filterwarnings("ignore")

_ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]

# Approximate Amazon Braket QuEra Aquila pricing (verify current rates / qBraid credits).
_AQUILA_PER_TASK = 0.30
_AQUILA_PER_SHOT = 0.01


# ---- readout (same as the gate-QPU experiment) -----------------------------

def _cv_alpha(R_tr, y_tr, alphas=_ALPHAS, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(R_tr) // 3)))
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


def fit_readout(R_tr, X_tr_full, y_tr):
    R_aug = np.hstack([R_tr, X_tr_full])
    sc = StandardScaler(); R_s = sc.fit_transform(R_aug)
    alpha = _cv_alpha(R_s, y_tr)
    ridge = Ridge(alpha=alpha); ridge.fit(R_s, y_tr)
    return ridge, sc, alpha


def apply_readout(ridge, sc, R_te, X_te_full):
    return ridge.predict(sc.transform(np.hstack([R_te, X_te_full])))


# ---- main ------------------------------------------------------------------

def run(args) -> None:
    print("=" * 78)
    print("  GIC 2026 PHASE 3 -- QRC on QuEra Aquila (neutral-atom analog)")
    print("=" * 78)

    d = load_financial_data_v2(
        delay=5, log_target=True, include_har=True,
        include_log_har=True, residual_target=True,
    )
    X_tr_full, X_te_full = d["X_train"], d["X_test"]
    y_tr = d["y_train"]
    y_te_raw_all, pers_te_all = d["y_test_raw"], d["persistence_test"]
    log_pers_te_all = d["log_persistence_test"]
    transform = d["target_transform"]

    n = args.atoms
    if args.max_train and args.max_train < len(X_tr_full):
        X_tr_full = X_tr_full[-args.max_train:]; y_tr = y_tr[-args.max_train:]

    n_te = len(X_te_full) if not args.max_test else min(args.max_test, len(X_te_full))
    sl = slice(len(X_te_full) - n_te, len(X_te_full))
    X_te_full = X_te_full[sl]; y_te_raw = y_te_raw_all[sl]
    pers_te = pers_te_all[sl]; log_pers_te = log_pers_te_all[sl]
    to_vol = lambda y: invert_target(y, transform, log_persistence=log_pers_te)

    res = QueraReservoir(n_atoms=n, geometry=args.geometry, seed=args.seed,
                         total_time=args.total_time, rabi_max=args.rabi_max)
    print(f"\nReservoir : {res}")
    print(f"Train     : {len(X_tr_full)} samples (readout)  |  Test (hardware): {n_te} (most recent)")
    print(f"Target    : {transform}\n")

    sc_in = StandardScaler().fit(X_tr_full[:, :n])
    Xtr_in = sc_in.transform(X_tr_full[:, :n])
    Xte_in = sc_in.transform(X_te_full[:, :n])

    # ---- Footprint / dry-run ----
    est_cost = n_te * (_AQUILA_PER_TASK + args.shots * _AQUILA_PER_SHOT)
    print(f"[1] Aquila footprint: {n_te} tasks x {args.shots} shots "
          f"(~${est_cost:.2f} at list price; billed via qBraid credits)")
    if args.dry_run:
        prog = res.build_program(Xte_in[0])
        print(f"    Program OK: {n} atoms, geometry='{args.geometry}', T={args.total_time*1e6:.1f}us")
        print("\n[DRY RUN] No tasks submitted. Validate on --device local first (free).")
        return

    # ---- Local AHS reference features (free, exact) ----
    print(f"[2] Local AHS reference features (free, n_jobs={args.n_jobs}) ...")
    t0 = time.time()
    R_tr_local = res.transform(Xtr_in, device="local", shots=args.shots, n_jobs=args.n_jobs)
    R_te_local = res.transform(Xte_in, device="local", shots=args.shots, n_jobs=args.n_jobs)
    print(f"    done in {time.time()-t0:.1f}s  (feature dim = {R_tr_local.shape[1]})")

    # ---- Test features on the chosen device ----
    if args.device == "local":
        R_te_dev = R_te_local
        dev_label = "QuEra local AHS sim"
        device = "local"
        n_tasks = 0
    else:
        if not args.allow_qpu:
            raise PermissionError(
                "Refusing to run on Aquila without --allow-qpu (spends credits: "
                f"{n_te} tasks x {args.shots} shots ~ ${est_cost:.2f})."
            )
        from braket.aws import AwsDevice
        from braket.devices import Devices
        device = AwsDevice(Devices.QuEra.Aquila)
        md = device.properties.service
        print(f"\n[3] Submitting {n_te} tasks to QuEra Aquila ...")
        t0 = time.time()
        R_te_dev = res.transform(Xte_in, device=device, shots=args.shots, verbose=True)
        print(f"    Aquila run done in {time.time()-t0:.0f}s")
        dev_label = "QuEra Aquila (hardware)"
        n_tasks = n_te

    # ---- Feature fidelity vs local sim ----
    corr = float(np.corrcoef(R_te_dev.ravel(), R_te_local.ravel())[0, 1])
    mae_feat = float(np.mean(np.abs(R_te_dev - R_te_local)))
    print(f"\n[4] Feature fidelity ({dev_label} vs local sim): "
          f"corr={corr:.4f}  mean abs err={mae_feat:.4f}")

    # ---- Readout + scoring ----
    print("\n[5] Readout (trained on local-sim features) ...")
    ridge, sc_r, alpha = fit_readout(R_tr_local, X_tr_full, y_tr)
    print(f"    Ridge alpha (CV) = {alpha}")
    pred_sim = to_vol(apply_readout(ridge, sc_r, R_te_local, X_te_full))
    pred_dev = to_vol(apply_readout(ridge, sc_r, R_te_dev, X_te_full))

    rows = []

    def record(name, y_pred, extra=None):
        m = regression_metrics(y_te_raw, y_pred)
        print_metrics(f"{name:<34}", m)
        rows.append({"Model": name, **m, **(extra or {})})

    print("\n[6] Scoring on the test window\n" + "-" * 78)
    record("Persistence", pers_te.copy())
    esn = EchoStateNetwork(n_reservoir=200, seed=args.seed); esn.fit(d["X_train"], d["y_train"])
    record("ESN (200 nodes)", invert_target(esn.predict(X_te_full), transform, log_persistence=log_pers_te))
    record(f"QuEra {n}atom -- local AHS sim", pred_sim)
    record(f"QuEra {n}atom -- {dev_label}", pred_dev,
           extra={"corr_vs_sim": corr, "feat_mae": mae_feat, "n_tasks": n_tasks, "shots": args.shots})

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "quera_aquila_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n  Saved -> {os.path.relpath(out_csv)}")

    _plot(y_te_raw, pred_sim, pred_dev, pers_te, R_te_local, R_te_dev, res, dev_label)
    print("\n--- Done ---")


def _plot(y_true, pred_sim, pred_dev, pers, R_sim, R_dev, res, dev_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(17, 5))
    # atom register
    c = res._coords * 1e6
    ax0.scatter(c[:, 0], c[:, 1], s=120, color="#7c3aed", edgecolor="k", zorder=3)
    for i, (x, y) in enumerate(c):
        ax0.annotate(str(i), (x, y), fontsize=8, ha="center", va="center", color="white")
    ax0.set_xlabel("x (um)"); ax0.set_ylabel("y (um)")
    ax0.set_title(f"Aquila register ({res.n_atoms} atoms, {res.geometry})")
    ax0.set_aspect("equal"); ax0.grid(True, alpha=0.25)

    days = np.arange(len(y_true))
    ax1.plot(days, y_true, color="black", lw=2.0, label="True volatility", zorder=5)
    ax1.plot(days, pred_sim, color="#2166ac", ls="--", lw=1.4, label="QuEra local sim")
    ax1.plot(days, pred_dev, color="#7c3aed", ls="-", lw=1.6, label=dev_label)
    ax1.plot(days, pers, color="#9e9e9e", ls="-.", lw=1.0, label="Persistence")
    ax1.set_xlabel("Test day"); ax1.set_ylabel("Realized vol (annualised)")
    ax1.set_title("SPY volatility forecast"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.25)

    ax2.scatter(R_sim.ravel(), R_dev.ravel(), s=4, alpha=0.3, color="#7c3aed")
    lo, hi = min(R_sim.min(), R_dev.min()), max(R_sim.max(), R_dev.max())
    ax2.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="Perfect agreement")
    ax2.set_xlabel("Local-sim feature"); ax2.set_ylabel("Device feature")
    ax2.set_title("Rydberg (Z, ZZ) feature fidelity"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "quera_aquila_qrc.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".png", "_doc.png"), dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {os.path.relpath(out)}  (+ _doc.png @ 1200 DPI)")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="local", choices=["local", "aquila"],
                   help="local AHS simulator (free) or real QuEra Aquila")
    p.add_argument("--atoms", type=int, default=5)  # tuned winner (quera_tune.py): 5 atoms random2d
    p.add_argument("--geometry", default="random2d", choices=["chain", "ring", "random2d"])
    p.add_argument("--shots", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total-time", type=float, default=4.0e-6, dest="total_time")
    p.add_argument("--rabi-max", type=float, default=1.5e7, dest="rabi_max")
    p.add_argument("--max-test", type=int, default=60, help="recent test days on hardware (0=all)")
    p.add_argument("--max-train", type=int, default=400, help="readout train cap (local-sim cost)")
    p.add_argument("--n-jobs", type=int, default=4, dest="n_jobs",
                   help="local-sim parallel workers (1=serial, -1=all cores; "
                        "auto-retries down + falls back to serial on low memory)")
    p.add_argument("--allow-qpu", action="store_true", help="REQUIRED to run on real Aquila")
    p.add_argument("--dry-run", action="store_true", help="build a program + footprint only")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    if args.max_test == 0:
        args.max_test = None
    run(args)
