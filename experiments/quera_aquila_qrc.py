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
import json
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
from src.qrc.hardware_backend import qbraid_api_key
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


def fit_readout(R_tr, X_tr_full, y_tr, alpha_override=None):
    R_aug = np.hstack([R_tr, X_tr_full])
    sc = StandardScaler(); R_s = sc.fit_transform(R_aug)
    alpha = float(alpha_override) if alpha_override is not None else _cv_alpha(R_s, y_tr)
    ridge = Ridge(alpha=alpha); ridge.fit(R_s, y_tr)
    return ridge, sc, alpha


def apply_readout(ridge, sc, R_te, X_te_full):
    return ridge.predict(sc.transform(np.hstack([R_te, X_te_full])))


# ---- main ------------------------------------------------------------------

def run(args) -> None:
    print("=" * 78)
    print("  GIC 2026 PHASE 3 -- QRC on QuEra Aquila (neutral-atom analog)")
    print("=" * 78)

    use_garch = args.hybrid_target == "garch"
    d = load_financial_data_v2(
        delay=5, log_target=True, include_har=True, include_log_har=True,
        residual_target=not use_garch,
        include_garch_proxy=use_garch,
        garch_residual_target=use_garch,
    )
    X_tr_full, X_te_full = d["X_train"], d["X_test"]
    y_tr = d["y_train"]
    y_te_raw_all, pers_te_all = d["y_test_raw"], d["persistence_test"]
    log_pers_te_all = d["log_persistence_test"]
    log_garch_te_all = d["log_garch_proxy_test"]
    transform = d["target_transform"]

    n = args.atoms
    if args.max_train and args.max_train < len(X_tr_full):
        X_tr_full = X_tr_full[-args.max_train:]; y_tr = y_tr[-args.max_train:]

    n_te = len(X_te_full) if not args.max_test else min(args.max_test, len(X_te_full))
    sl = slice(len(X_te_full) - n_te, len(X_te_full))
    X_te_full = X_te_full[sl]; y_te_raw = y_te_raw_all[sl]
    pers_te = pers_te_all[sl]; log_pers_te = log_pers_te_all[sl]
    log_garch_te = log_garch_te_all[sl]
    if use_garch:
        to_vol_scaled = lambda y, s=1.0: invert_target(
            s * y, transform, log_garch_proxy=log_garch_te
        )
    else:
        to_vol_scaled = lambda y, s=1.0: invert_target(
            s * y, transform, log_persistence=log_pers_te
        )
    to_vol = lambda y: to_vol_scaled(y, 1.0)

    res = QueraReservoir(n_atoms=n, geometry=args.geometry, seed=args.seed,
                         total_time=args.total_time, rabi_max=args.rabi_max)
    print(f"\nReservoir : {res}")
    print(f"Train     : {len(X_tr_full)} samples (readout)  |  Test (hardware): {n_te} (most recent)")
    print(f"Target    : {transform}\n")
    print(f"Data      : {d.get('data_source', 'unknown')}\n")

    sc_in = StandardScaler().fit(X_tr_full[:, :n])
    Xtr_in = sc_in.transform(X_tr_full[:, :n])
    Xte_in = sc_in.transform(X_te_full[:, :n])

    # ---- Footprint / dry-run ----
    est_cost = n_te * (_AQUILA_PER_TASK + args.shots * _AQUILA_PER_SHOT)
    print(f"[1] Aquila footprint: {n_te} tasks x {args.shots} shots "
          f"(~${est_cost:.2f} = ~{est_cost * 100:.0f} qBraid credits at list price)")
    if args.device != "local" and est_cost * 100 > args.credit_budget:
        raise RuntimeError(
            f"Estimated Aquila charge {est_cost * 100:.0f} qBraid credits "
            f"exceeds the {args.credit_budget:.0f}-credit cap. No tasks submitted."
        )
    if args.dry_run:
        prog = res.build_program(Xte_in[0])
        print(f"    Program OK: {n} atoms, geometry='{args.geometry}', T={args.total_time*1e6:.1f}us")
        print("\n[DRY RUN] No tasks submitted. Validate on --device local first (free).")
        return

    # ---- Local AHS reference features (free, exact) ----
    print(f"[2] Local AHS reference features (free, n_jobs={args.n_jobs}) ...")
    t0 = time.time()
    # One pooled call for train+test so the worker pool starts up only once.
    R_all = res.transform(np.vstack([Xtr_in, Xte_in]), device="local",
                          shots=args.shots, n_jobs=args.n_jobs)
    R_tr_local, R_te_local = R_all[:len(Xtr_in)], R_all[len(Xtr_in):]
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
        if args.device == "qbraid_aquila":
            from qbraid.runtime import QbraidProvider
            device = QbraidProvider(api_key=qbraid_api_key()).get_device(
                "aws:quera:qpu:aquila"
            )
            md = device.metadata()
            if str(md.get("status", "")).upper() != "ONLINE":
                raise RuntimeError(f"qBraid Aquila is not online: {md.get('status')}")
            pricing = md.get("pricing", {}) or {}
            per_task = float(pricing.get("perTask", 30.0))
            per_shot = float(pricing.get("perShot", 1.0))
            checkpoint_path = os.path.join(
                RESULTS_DIR, f"quera_aquila_checkpoint_{args.result_tag or 'run'}.json"
            )
            completed = 0
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, encoding="utf-8") as f:
                    completed = int(json.load(f).get("completed_rows", 0))
            remaining = max(0, n_te - completed)
            live_credits = remaining * (per_task + args.shots * per_shot)
            print(f"    qBraid live pricing: {per_task} credits/task + "
                  f"{per_shot} credits/shot; {remaining} remaining tasks = "
                  f"{live_credits:.0f} credits")
            if live_credits > args.credit_budget:
                raise RuntimeError(
                    f"Live charge {live_credits:.0f} exceeds cap "
                    f"{args.credit_budget:.0f}; no tasks submitted."
                )
        else:
            from braket.aws import AwsDevice
            from braket.devices import Devices
            device = AwsDevice(Devices.QuEra.Aquila)
            md = device.properties.service
            checkpoint_path = None
        print(f"\n[3] Submitting {n_te} tasks to QuEra Aquila ...")
        t0 = time.time()
        R_te_dev = res.transform(
            Xte_in, device=device, shots=args.shots, verbose=True,
            checkpoint_path=checkpoint_path,
        )
        print(f"    Aquila run done in {time.time()-t0:.0f}s")
        dev_label = "QuEra Aquila (hardware)"
        n_tasks = n_te

    # ---- Feature fidelity vs local sim + label-free affine transfer ----
    R_te_dev_raw = R_te_dev.copy()
    raw_corr = float(np.corrcoef(R_te_dev_raw.ravel(), R_te_local.ravel())[0, 1])
    raw_mae = float(np.mean(np.abs(R_te_dev_raw - R_te_local)))
    feature_scale, feature_bias = 1.0, 0.0
    k_feat = args.feature_calibration_rows
    if k_feat:
        if not (0 < k_feat < n_te):
            raise ValueError("--feature-calibration-rows must be between 1 and max-test-1")
        A = np.column_stack([R_te_dev_raw[:k_feat].ravel(),
                             np.ones(R_te_dev_raw[:k_feat].size)])
        feature_scale, feature_bias = np.linalg.lstsq(
            A, R_te_local[:k_feat].ravel(), rcond=None
        )[0]
        R_te_dev = feature_scale * R_te_dev_raw + feature_bias
    corr = float(np.corrcoef(R_te_dev.ravel(), R_te_local.ravel())[0, 1])
    mae_feat = float(np.mean(np.abs(R_te_dev - R_te_local)))
    print(f"\n[4] Feature fidelity ({dev_label} vs local sim): "
          f"raw corr={raw_corr:.4f}, raw MAE={raw_mae:.4f}; "
          f"aligned corr={corr:.4f}, aligned MAE={mae_feat:.4f}")
    if k_feat:
        print(f"    Label-free feature transfer on first {k_feat} rows: "
              f"scale={feature_scale:.4f}, bias={feature_bias:.4f}")

    # ---- Readout + scoring ----
    print("\n[5] Readout (trained on local-sim features) ...")
    ridge, sc_r, alpha = fit_readout(
        R_tr_local, X_tr_full, y_tr, alpha_override=args.readout_alpha
    )
    print(f"    Ridge alpha (CV) = {alpha}")
    pred_sim_res = apply_readout(ridge, sc_r, R_te_local, X_te_full)
    pred_dev_res = apply_readout(ridge, sc_r, R_te_dev, X_te_full)
    sim_strength = args.correction_strength
    dev_strength = sim_strength
    k_cal = args.hardware_calibration_rows
    if k_cal:
        if not (0 < k_cal < n_te):
            raise ValueError("--hardware-calibration-rows must be between 1 and max-test-1")
        grid = np.linspace(args.strength_min, args.strength_max, args.strength_steps)
        scores = [np.sqrt(np.mean(
            (to_vol_scaled(pred_dev_res, s)[:k_cal] - y_te_raw[:k_cal]) ** 2
        )) for s in grid]
        dev_strength = float(grid[int(np.argmin(scores))])
        print(f"    Device correction calibration: first {k_cal} rows -> "
              f"strength={dev_strength:.3f} (excluded from evaluation)")
    pred_sim = to_vol_scaled(pred_sim_res, sim_strength)
    pred_dev = to_vol_scaled(pred_dev_res, dev_strength)

    rows = []

    def record(name, y_pred, extra=None):
        m = regression_metrics(y_te_raw, y_pred)
        print_metrics(f"{name:<34}", m)
        rows.append({"Model": name, **m, **(extra or {})})

    k_exclude = max(k_cal, k_feat)
    eval_sl = slice(k_exclude, None)
    y_te_raw_all_scoring = y_te_raw
    y_te_raw = y_te_raw[eval_sl]
    print(f"\n[6] Scoring on {len(y_te_raw)} evaluation rows "
          f"({k_exclude} calibration rows excluded)\n" + "-" * 78)
    record("Persistence", pers_te[eval_sl].copy())
    if use_garch:
        record("GARCH proxy (zero residual)", np.exp(log_garch_te[eval_sl]))
    linear = Ridge(alpha=1.0).fit(d["X_train"], d["y_train"])
    record("Ridge residual ablation", to_vol(linear.predict(X_te_full))[eval_sl])
    esn = EchoStateNetwork(n_reservoir=200, seed=args.seed); esn.fit(d["X_train"], d["y_train"])
    record("ESN (200 nodes)", to_vol(esn.predict(X_te_full))[eval_sl])
    record(f"QuEra {n}atom -- local AHS sim", pred_sim[eval_sl],
           extra={"correction_strength": sim_strength})
    record(f"QuEra {n}atom -- {dev_label}", pred_dev[eval_sl],
           extra={"corr_vs_sim": corr, "feat_mae": mae_feat, "n_tasks": n_tasks,
                  "shots": args.shots, "correction_strength": dev_strength,
                  "hardware_calibration_rows": k_cal,
                  "feature_calibration_rows": k_feat,
                  "feature_corr_raw": raw_corr, "feature_mae_raw": raw_mae,
                  "feature_affine_scale": feature_scale,
                  "feature_affine_bias": feature_bias})

    df = pd.DataFrame(rows)
    suffix = f"_{args.result_tag}" if args.result_tag else ""
    out_csv = os.path.join(RESULTS_DIR, f"quera_aquila_summary{suffix}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n  Saved -> {os.path.relpath(out_csv)}")

    out_npz = os.path.join(RESULTS_DIR, f"quera_aquila_features{suffix}.npz")
    np.savez_compressed(
        out_npz, R_test_local=R_te_local, R_test_device_raw=R_te_dev_raw,
        R_test_device=R_te_dev,
        X_test=X_te_full, y_test_raw=y_te_raw_all_scoring,
        hybrid_target=args.hybrid_target, device=str(device), shots=args.shots,
        pred_residual_local=pred_sim_res, pred_residual_device=pred_dev_res,
        correction_strength_local=sim_strength,
        correction_strength_device=dev_strength,
        hardware_calibration_rows=k_cal,
        feature_calibration_rows=k_feat,
        feature_affine_scale=feature_scale, feature_affine_bias=feature_bias,
    )
    print(f"  Saved -> {os.path.relpath(out_npz)}")
    if args.device != "local":
        task_path = os.path.join(RESULTS_DIR, f"quera_aquila_task_ids{suffix}.json")
        with open(task_path, "w", encoding="utf-8") as f:
            json.dump({
                "device": "aws:quera:qpu:aquila",
                "shots": args.shots,
                "hybrid_target": args.hybrid_target,
                "task_ids": res.submitted_task_ids,
            }, f, indent=2)
        print(f"  Saved -> {os.path.relpath(task_path)}")

    _plot(y_te_raw, pred_sim[eval_sl], pred_dev[eval_sl], pers_te[eval_sl],
          R_te_local, R_te_dev, res, dev_label, suffix)
    print("\n--- Done ---")


def _plot(y_true, pred_sim, pred_dev, pers, R_sim, R_dev, res, dev_label, suffix=""):
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
    out = os.path.join(RESULTS_DIR, f"quera_aquila_qrc{suffix}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".png", "_doc.png"), dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {os.path.relpath(out)}  (+ _doc.png @ 1200 DPI)")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="local",
                   choices=["local", "aquila", "qbraid_aquila"],
                   help="local AHS, direct AWS Aquila, or token-backed qBraid Aquila")
    p.add_argument("--atoms", type=int, default=5)  # tuned winner (quera_tune.py): 5 atoms random2d
    p.add_argument("--geometry", default="random2d", choices=["chain", "ring", "random2d"])
    p.add_argument("--shots", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hybrid-target", default="garch",
                   choices=["garch", "persistence"],
                   help="classical baseline whose log residual the analog QRC predicts")
    p.add_argument("--result-tag", default="",
                   help="suffix for CSV/NPZ/plot outputs")
    p.add_argument("--correction-strength", type=float, default=1.0,
                   help="pre-locked multiplier on the predicted baseline residual")
    p.add_argument("--readout-alpha", type=float, default=None,
                   help="lock Ridge alpha instead of selecting it by training CV")
    p.add_argument("--hardware-calibration-rows", type=int, default=0,
                   help="first measured rows used only to recalibrate correction strength")
    p.add_argument("--feature-calibration-rows", type=int, default=0,
                   help="first measured rows for label-free device-to-local feature alignment")
    p.add_argument("--strength-min", type=float, default=-0.25)
    p.add_argument("--strength-max", type=float, default=1.25)
    p.add_argument("--strength-steps", type=int, default=301)
    p.add_argument("--total-time", type=float, default=4.0e-6, dest="total_time")
    p.add_argument("--rabi-max", type=float, default=1.5e7, dest="rabi_max")
    p.add_argument("--max-test", type=int, default=60, help="recent test days on hardware (0=all)")
    p.add_argument("--max-train", type=int, default=400, help="readout train cap (local-sim cost)")
    p.add_argument("--n-jobs", type=int, default=4, dest="n_jobs",
                   help="local-sim parallel workers (1=serial, -1=all cores; "
                        "auto-retries down + falls back to serial on low memory)")
    p.add_argument("--allow-qpu", action="store_true", help="REQUIRED to run on real Aquila")
    p.add_argument("--credit-budget", type=float, default=100.0,
                   help="hard qBraid-credit cap for Aquila (100 credits = $1)")
    p.add_argument("--dry-run", action="store_true", help="build a program + footprint only")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    if args.max_test == 0:
        args.max_test = None
    run(args)
