"""
qBraid hardware QRC experiment — run the quantum reservoir as real quantum jobs.

This is the Phase 3 "advantage on hardware" experiment. It takes the exact same
gate-based QRC pipeline that wins in simulation (5q, 2 layers, random Ising) and
executes the reservoir circuits on a qBraid backend (cloud simulator or QPU),
then reconstructs the Z/ZZ features from real shot counts and runs the trained
linear readout.

What it demonstrates
--------------------
1. **It runs.** The reservoir executes as genuine quantum jobs on qBraid; job ids
   are saved to results/qbraid_job_ids.json as an audit trail.
2. **Feature fidelity.** Hardware-measured features are compared against the exact
   statevector reference (correlation + mean-abs error). Quantifies how well the
   encoding survives finite shots / device noise.
3. **Downstream competitiveness.** The volatility forecast built on hardware
   features is scored against Persistence and ESN on the same test window — the
   honest claim being "the quantum feature map is hardware-realizable and stays
   competitive with the best classical baselines", not "QRC beats everything".

Economical by design
--------------------
* One circuit is needed per time step, so the *full* 3.7k-sample pipeline is far
  too many jobs for a QPU. By default the readout is **trained on the free, exact
  local statevector features** and only the **test window** is run on the backend
  (cap it with ``--max-test``). This is the cheapest faithful test: it measures
  how an ideal-trained readout holds up on noisy hardware features.
* ``--dry-run`` builds circuits and prints the job/shot/credit footprint WITHOUT
  submitting anything (and without needing qbraid installed). Always dry-run
  first before pointing this at a QPU.

Examples
--------
    # 0) Cost preview, no submission, no qbraid needed:
    python experiments/qbraid_hardware_qrc.py --dry-run --max-test 60

    # 1) Full validation on qBraid's free statevector simulator:
    python experiments/qbraid_hardware_qrc.py --device qbraid:qbraid:sim:qir-sv --max-test 120

    # 2) Small real-QPU proof (spends credits — note the explicit latch).
    #    Pick an ONLINE gate QPU first: python experiments/qbraid_list_devices.py --online --gate
    #    (QuEra/Pasqal are analog and will NOT run these gate circuits.)
    python experiments/qbraid_hardware_qrc.py --device openquantum:ionq:qpu:forte-1 \
        --max-test 40 --shots 1024 --max-batch 20 --allow-qpu
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

from src.qrc.reservoir import QuantumReservoir
from src.qrc.hardware_backend import (
    QbraidExecutor,
    reservoir_circuits,
    counts_to_features,
    statevector_features,
    calibrate_bit_order,
)
from src.data.loaders import load_financial_data_v2, invert_target
from src.baselines.classical import (
    EchoStateNetwork,
    regression_metrics,
    print_metrics,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
warnings.filterwarnings("ignore")

_ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]


# ---------------------------------------------------------------------------
# Readout (mirrors phase2_final._reservoir_regress, but on precomputed features)
# ---------------------------------------------------------------------------

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
    """Train the Ridge readout on [reservoir features | raw features]."""
    R_aug = np.hstack([R_tr, X_tr_full])
    sc = StandardScaler()
    R_s = sc.fit_transform(R_aug)
    alpha = _cv_alpha(R_s, y_tr)
    ridge = Ridge(alpha=alpha); ridge.fit(R_s, y_tr)
    return ridge, sc, alpha


def apply_readout(ridge, sc, R_te, X_te_full):
    R_aug = np.hstack([R_te, X_te_full])
    return ridge.predict(sc.transform(R_aug))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args) -> None:
    print("=" * 78)
    print("  GIC 2026 PHASE 3 -- QRC on qBraid quantum hardware")
    print("=" * 78)

    # ---- Data (same pipeline as the winning Phase 2 config) ----
    d = load_financial_data_v2(
        delay=5, log_target=True, include_har=True,
        include_log_har=True, residual_target=True,
    )
    X_tr_full, X_te_full = d["X_train"], d["X_test"]
    y_tr = d["y_train"]
    y_te_raw_all = d["y_test_raw"]
    pers_te_all = d["persistence_test"]
    log_pers_te_all = d["log_persistence_test"]
    transform = d["target_transform"]

    n = args.qubits

    # Optionally cap the train set used for the readout (purely to speed up the
    # local statevector reference; default uses all train samples — they're free).
    if args.max_train and args.max_train < len(X_tr_full):
        X_tr_full = X_tr_full[-args.max_train:]
        y_tr = y_tr[-args.max_train:]

    # Take the most RECENT max_test samples as the hardware test window.
    n_te = len(X_te_full) if not args.max_test else min(args.max_test, len(X_te_full))
    sl = slice(len(X_te_full) - n_te, len(X_te_full))
    X_te_full = X_te_full[sl]
    y_te_raw = y_te_raw_all[sl]
    pers_te = pers_te_all[sl]
    log_pers_te = log_pers_te_all[sl]

    to_vol = lambda y: invert_target(y, transform, log_persistence=log_pers_te)

    print(f"\nReservoir : QuantumReservoir(n_qubits={n}, n_layers={args.layers}, "
          f"connectivity='{args.connectivity}', seed={args.seed})")
    print(f"Train     : {len(X_tr_full)} samples (readout)  |  "
          f"Test (hardware): {n_te} samples (most recent)")
    print(f"Target    : {transform}\n")

    res = QuantumReservoir(
        n_qubits=n, n_layers=args.layers,
        connectivity=args.connectivity, seed=args.seed,
    )

    # ---- Standardise the reservoir inputs exactly as in Phase 2 ----
    sc_in = StandardScaler().fit(X_tr_full[:, :n])
    Xtr_in = sc_in.transform(X_tr_full[:, :n])
    Xte_in = sc_in.transform(X_te_full[:, :n])

    # ---- Local statevector reference features (exact, free) ----
    print("[1] Local statevector reference features (exact, no cost) ...")
    t0 = time.time()
    R_tr_local = statevector_features(res, Xtr_in)
    R_te_local = statevector_features(res, Xte_in)
    print(f"    done in {time.time()-t0:.1f}s  (feature dim = {R_tr_local.shape[1]})")

    # ---- Build the hardware test circuits ----
    circuits_te = reservoir_circuits(res, Xte_in)
    sample = circuits_te[0]
    print(f"\n[2] Hardware circuits: {len(circuits_te)} circuits, "
          f"{sample.num_qubits} qubits, depth ~{sample.depth()}, "
          f"{sample.size()} gates each")

    ex = QbraidExecutor(
        device_id=args.device, shots=args.shots, api_key=args.api_key,
        max_batch=args.max_batch, cache_dir=RESULTS_DIR,
        allow_qpu=args.allow_qpu, verbose=True,
    )
    est = ex.estimate(len(circuits_te))
    print(f"    Footprint: {est['n_circuits']} circuits x {est['shots_per_circuit']} "
          f"shots = {est['total_shots']:,} shots in {est['n_batches']} batch(es) "
          f"on '{args.device}'")

    if args.dry_run:
        print("\n[DRY RUN] No jobs submitted. Re-run without --dry-run to execute.")
        print("          (Run on qbraid:qbraid:sim:qir-sv first; it is free.)")
        return

    # ---- Submit to qBraid and reconstruct features ----
    print(f"\n[3] Submitting to qBraid device '{args.device}' ...")
    counts_te = ex.run(circuits_te)

    # Calibrate bit order against the local reference on a few circuits.
    k_cal = min(8, len(counts_te))
    reverse = calibrate_bit_order(counts_te[:k_cal], R_te_local[:k_cal], n, ex.shots)
    ex.reverse_bits = reverse
    print(f"    Bit-order calibration: reverse_bits={reverse}")
    R_te_hw = counts_to_features(counts_te, n, ex.shots, reverse_bits=reverse)

    # ---- Feature fidelity: hardware vs exact statevector ----
    corr = float(np.corrcoef(R_te_hw.ravel(), R_te_local.ravel())[0, 1])
    mae_feat = float(np.mean(np.abs(R_te_hw - R_te_local)))
    print(f"\n[4] Feature fidelity (hardware vs statevector):")
    print(f"    Pearson corr = {corr:.4f}   |   mean abs feature error = {mae_feat:.4f}")

    # ---- Train readout on local features, evaluate on sim & hardware test ----
    print("\n[5] Readout (trained on exact local features) ...")
    ridge, sc_r, alpha = fit_readout(R_tr_local, X_tr_full, y_tr)
    print(f"    Ridge alpha (CV) = {alpha}")

    pred_sim = to_vol(apply_readout(ridge, sc_r, R_te_local, X_te_full))
    pred_hw = to_vol(apply_readout(ridge, sc_r, R_te_hw, X_te_full))
    label_sim = f"QRC {n}q L{args.layers} -- statevector (sim)"
    label_hw = f"QRC {n}q L{args.layers} -- qBraid hardware"

    # ---- Classical baselines on the same test window ----
    rows = []

    def record(name, y_pred, extra=None):
        m = regression_metrics(y_te_raw, y_pred)
        print_metrics(f"{name:<34}", m)
        row = {"Model": name, **m}
        if extra:
            row.update(extra)
        rows.append(row)
        return m

    print("\n[6] Scoring on the hardware test window\n" + "-" * 78)
    record("Persistence", pers_te.copy())

    esn = EchoStateNetwork(n_reservoir=200, seed=args.seed)
    esn.fit(d["X_train"], d["y_train"])
    esn_pred = invert_target(esn.predict(X_te_full), transform, log_persistence=log_pers_te)
    record("ESN (200 nodes)", esn_pred)

    record(label_sim, pred_sim)
    record(
        label_hw, pred_hw,
        extra={"device": args.device, "shots": ex.shots,
               "feat_corr": corr, "feat_mae": mae_feat,
               "n_jobs": len(ex.submitted_job_ids)},
    )

    # ---- Save ----
    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "qbraid_hardware_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n  Saved -> {os.path.relpath(out_csv)}")

    _plot(y_te_raw, pred_sim, pred_hw, pers_te, R_te_local, R_te_hw, args)

    print("\n--- Done ---")
    print("  Real quantum job ids: results/qbraid_job_ids.json")
    print("  Keep these with the submission as proof of hardware execution.")


def _plot(y_true, pred_sim, pred_hw, pers, R_sim, R_hw, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    days = np.arange(len(y_true))
    ax1.plot(days, y_true, color="black", lw=2.0, label="True volatility", zorder=5)
    ax1.plot(days, pred_sim, color="#2166ac", ls="--", lw=1.4, label="QRC statevector")
    ax1.plot(days, pred_hw, color="#7c3aed", ls="-", lw=1.6, label="QRC qBraid hardware")
    ax1.plot(days, pers, color="#9e9e9e", ls="-.", lw=1.0, label="Persistence")
    ax1.set_xlabel("Test day"); ax1.set_ylabel("Realized vol (annualised)")
    ax1.set_title(f"SPY volatility forecast — {args.device}")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.25)

    ax2.scatter(R_sim.ravel(), R_hw.ravel(), s=4, alpha=0.3, color="#7c3aed")
    lo = min(R_sim.min(), R_hw.min()); hi = max(R_sim.max(), R_hw.max())
    ax2.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="Perfect agreement")
    ax2.set_xlabel("Statevector feature value"); ax2.set_ylabel("Hardware feature value")
    ax2.set_title("Reservoir feature fidelity (Z, ZZ)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "qbraid_hardware_qrc.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".png", "_doc.png"), dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {os.path.relpath(out)}  (+ _doc.png @ 1200 DPI)")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="qbraid:qbraid:sim:qir-sv",
                   help="qBraid device id (default: free 30q statevector sim)")
    p.add_argument("--shots", type=int, default=1024)
    p.add_argument("--qubits", type=int, default=5)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--connectivity", default="random",
                   choices=["linear", "random", "all-to-all"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-test", type=int, default=120,
                   help="number of most-recent test days to run on hardware (0 = all)")
    p.add_argument("--max-train", type=int, default=0,
                   help="cap readout training samples (0 = all; only affects local sim cost)")
    p.add_argument("--max-batch", type=int, default=50,
                   help="circuits per device.run call (keep small on QPUs)")
    p.add_argument("--api-key", default=None,
                   help="qBraid API key (default: QBRAID_API_KEY env / saved config)")
    p.add_argument("--allow-qpu", action="store_true",
                   help="REQUIRED to submit to a non-simulator device (spends credits)")
    p.add_argument("--dry-run", action="store_true",
                   help="build circuits + print footprint, submit nothing (no qbraid needed)")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    if args.max_test == 0:
        args.max_test = None
    run(args)
