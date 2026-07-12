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
from src.qrc.input_projection import ReservoirInputProjector
from src.qrc.noise import depolarizing_noise
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

# OpenQuantum Public Compute prices in USD equivalent (2026-07-11).
# qBraid metadata currently reports 0/0 for these sponsored routes even though
# OpenQuantum deducts Spark credits, so never infer that openquantum:* is free.
# One Spark credit is $2 of Public Compute.
_OPENQUANTUM_PUBLIC_PRICING = {
    "openquantum:ionq:qpu:forte-1": (0.18, 0.04800),
    "openquantum:ionq:qpu:forte-enterprise": (0.18, 0.04800),
    "openquantum:iqm:qpu:emerald": (0.18, 0.00096),
    "openquantum:iqm:qpu:garnet": (0.18, 0.00087),
    "openquantum:rigetti:qpu:cepheus-1-108q": (0.18, 0.000255),
    "openquantum:aqt:qpu:ibex-q1": (0.18, 0.01410),
}


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

    # ---- Data / hybrid target ----
    use_garch = args.hybrid_target == "garch"
    d = load_financial_data_v2(
        delay=5, log_target=True, include_har=True, include_log_har=True,
        residual_target=not use_garch,
        include_garch_proxy=use_garch,
        garch_residual_target=use_garch,
    )
    X_tr_full, X_te_full = d["X_train"], d["X_test"]
    y_tr = d["y_train"]
    y_te_raw_all = d["y_test_raw"]
    pers_te_all = d["persistence_test"]
    log_pers_te_all = d["log_persistence_test"]
    log_garch_te_all = d["log_garch_proxy_test"]
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

    print(f"\nReservoir : QuantumReservoir(n_qubits={n}, n_layers={args.layers}, "
          f"connectivity='{args.connectivity}', encoding_axis='{args.encoding_axis}', "
          f"seed={args.seed})")
    print(f"Train     : {len(X_tr_full)} samples (readout)  |  "
          f"Test (hardware): {n_te} samples (most recent)")
    print(f"Target    : {transform}\n")
    print(f"Data      : {d.get('data_source', 'unknown')}\n")

    res = QuantumReservoir(
        n_qubits=n, n_layers=args.layers,
        connectivity=args.connectivity, seed=args.seed,
        encoding_axis=args.encoding_axis,
        observable_order=args.observable_order,
    )

    # ---- Standardise the reservoir inputs exactly as in Phase 2 ----
    projector = ReservoirInputProjector(
        n, mode=args.input_projection, seed=args.seed
    ).fit(X_tr_full, feature_names=d["feature_names"])
    Xtr_in = projector.transform(X_tr_full)
    Xte_in = projector.transform(X_te_full)

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

    cache_dir = (os.path.join(RESULTS_DIR, f"hardware_{args.result_tag}")
                 if args.result_tag else RESULTS_DIR)
    ex = QbraidExecutor(
        device_id=args.device, shots=args.shots, api_key=args.api_key,
        max_batch=args.max_batch, cache_dir=cache_dir,
        allow_qpu=args.allow_qpu, verbose=True,
    )
    est = ex.estimate(len(circuits_te))
    print(f"    Footprint: {est['n_circuits']} circuits x {est['shots_per_circuit']} "
          f"shots = {est['total_shots']:,} shots in {est['n_batches']} batch(es) "
          f"on '{args.device}'")

    # OpenQuantum pricing is not exposed correctly in qBraid metadata, so show
    # and enforce the audited Spark estimate even during an offline dry-run.
    if args.device.startswith("openquantum:"):
        oq_price = _OPENQUANTUM_PUBLIC_PRICING.get(args.device)
        if oq_price is None:
            raise RuntimeError(
                "No audited OpenQuantum Spark price is configured for this "
                "device. Refusing submission until pricing is verified."
            )
        oq_task_usd, oq_shot_usd = oq_price
        usd_est = len(circuits_te) * (oq_task_usd + args.shots * oq_shot_usd)
        spark_est = usd_est / 2.0
        print(f"    OpenQuantum Public price: ${oq_task_usd}/task + "
              f"${oq_shot_usd}/shot")
        print(f"    Maximum estimated charge: ${usd_est:,.2f} = "
              f"{spark_est:,.2f} Spark credits "
              f"(hard cap: {args.spark_budget:,.2f})")
        if spark_est > args.spark_budget:
            raise RuntimeError(
                f"Estimated charge {spark_est:,.2f} Spark credits exceeds "
                f"the {args.spark_budget:,.2f}-Spark cap. No jobs submitted."
            )

    # A live qBraid pricing check is mandatory immediately before other QPU submissions.
    # qBraid reports prices in credits (for example, AWS Forte currently reports
    # 30 credits/task + 8 credits/shot). Sponsored OpenQuantum routes may report
    # zero; using the exact device id therefore matters.
    if (not args.dry_run and not ex.is_simulator()
            and not args.device.startswith("openquantum:")):
        md = ex.metadata()
        pricing = md.get("pricing", {}) or {}
        per_task = pricing.get("perTask", pricing.get("per_task"))
        per_shot = pricing.get("perShot", pricing.get("per_shot"))
        if per_task is None or per_shot is None:
            if not args.allow_unknown_pricing:
                raise RuntimeError(
                    "qBraid did not report per-task/per-shot pricing. Refusing "
                    "QPU submission; pass --allow-unknown-pricing only after "
                    "confirming the charge in the qBraid Jobs panel."
                )
            print("    Live pricing: unavailable (explicit override enabled)")
        else:
            credit_est = len(circuits_te) * (
                float(per_task) + args.shots * float(per_shot)
            )
            print(f"    Live pricing: {per_task} credits/task + {per_shot} credits/shot")
            print(f"    Maximum estimated charge: {credit_est:,.2f} credits "
                  f"(hard cap: {args.credit_budget:,.2f})")
            if credit_est > args.credit_budget:
                raise RuntimeError(
                    f"Estimated charge {credit_est:,.2f} exceeds the "
                    f"{args.credit_budget:,.2f}-credit cap. No jobs submitted."
                )

    if args.dry_run:
        print("\n[DRY RUN] No jobs submitted. Re-run without --dry-run to execute.")
        print("          (Run on qbraid:qbraid:sim:qir-sv first; it is free.)")
        return

    # ---- Submit to qBraid and reconstruct features ----
    print(f"\n[3] Submitting to qBraid device '{args.device}' ...")
    counts_te = ex.run(circuits_te)

    # Calibrate bit order against the local reference on a few circuits.
    k_cal = min(8, len(counts_te))
    reverse = calibrate_bit_order(
        counts_te[:k_cal], R_te_local[:k_cal], n, ex.shots,
        max_order=args.observable_order,
    )
    ex.reverse_bits = reverse
    print(f"    Bit-order calibration: reverse_bits={reverse}")
    R_te_hw = counts_to_features(
        counts_te, n, ex.shots, reverse_bits=reverse,
        max_order=args.observable_order,
    )

    # ---- Feature fidelity: hardware vs exact statevector ----
    corr = float(np.corrcoef(R_te_hw.ravel(), R_te_local.ravel())[0, 1])
    mae_feat = float(np.mean(np.abs(R_te_hw - R_te_local)))
    print(f"\n[4] Feature fidelity (hardware vs statevector):")
    print(f"    Pearson corr = {corr:.4f}   |   mean abs feature error = {mae_feat:.4f}")

    # ---- Train readout locally, evaluate on sim & hardware test ----
    # Noise-aware training is still free: only the test circuits are sent to QPU.
    R_tr_readout = R_tr_local
    if args.readout_training == "iqm-noise":
        print("\n[5] Generating local shot/noise-aware training features ...")
        training_res = QuantumReservoir(
            n_qubits=n, n_layers=args.layers, connectivity=args.connectivity,
            seed=args.seed, encoding_axis=args.encoding_axis,
            observable_order=args.observable_order,
            noise_model=depolarizing_noise(args.train_p_single, args.train_p_two),
            n_shots=args.train_shots,
        )
        R_tr_readout = training_res.transform(Xtr_in)
    print(f"\n[5] Readout (training={args.readout_training}) ...")
    ridge, sc_r, alpha = fit_readout(R_tr_readout, X_tr_full, y_tr)
    print(f"    Ridge alpha (CV) = {alpha}")

    pred_sim_res = apply_readout(ridge, sc_r, R_te_local, X_te_full)
    pred_hw_res = apply_readout(ridge, sc_r, R_te_hw, X_te_full)
    sim_strength = args.correction_strength
    hw_strength = sim_strength
    k_hw_cal = args.hardware_calibration_rows
    if k_hw_cal:
        if not (0 < k_hw_cal < n_te):
            raise ValueError("--hardware-calibration-rows must be between 1 and max-test-1")
        grid = np.linspace(args.strength_min, args.strength_max, args.strength_steps)
        scores = [np.sqrt(np.mean(
            (to_vol_scaled(pred_hw_res, s)[:k_hw_cal] - y_te_raw[:k_hw_cal]) ** 2
        )) for s in grid]
        hw_strength = float(grid[int(np.argmin(scores))])
        print(f"    Hardware correction calibration: first {k_hw_cal} rows -> "
              f"strength={hw_strength:.3f} (excluded from reported evaluation)")
    pred_sim = to_vol_scaled(pred_sim_res, sim_strength)
    pred_hw = to_vol_scaled(pred_hw_res, hw_strength)
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

    eval_sl = slice(k_hw_cal, None)
    y_eval = y_te_raw[eval_sl]
    print(f"\n[6] Scoring on {len(y_eval)} evaluation rows "
          f"({k_hw_cal} calibration rows excluded)\n" + "-" * 78)

    # record() closes over y_te_raw; use the evaluation block from this point.
    y_te_raw_scoring = y_te_raw
    y_te_raw = y_eval
    record("Persistence", pers_te[eval_sl].copy())
    if use_garch:
        record("GARCH proxy (zero residual)", np.exp(log_garch_te[eval_sl]))

    # Linear-only residual ablation: same target and raw features, no reservoir.
    linear = Ridge(alpha=1.0).fit(d["X_train"], d["y_train"])
    record("Ridge residual ablation", to_vol(linear.predict(X_te_full))[eval_sl])

    esn = EchoStateNetwork(n_reservoir=200, seed=args.seed)
    esn.fit(d["X_train"], d["y_train"])
    esn_raw = esn.predict(X_te_full)
    esn_pred = to_vol(esn_raw)
    record("ESN (200 nodes)", esn_pred[eval_sl])

    record(label_sim, pred_sim[eval_sl], extra={"correction_strength": sim_strength})
    record(
        label_hw, pred_hw[eval_sl],
        extra={"device": args.device, "shots": ex.shots,
               "feat_corr": corr, "feat_mae": mae_feat,
               "n_jobs": len(ex.submitted_job_ids),
               "correction_strength": hw_strength,
               "hardware_calibration_rows": k_hw_cal},
    )

    # ---- Save ----
    df = pd.DataFrame(rows)
    suffix = f"_{args.result_tag}" if args.result_tag else ""
    out_csv = os.path.join(RESULTS_DIR, f"qbraid_hardware_summary{suffix}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n  Saved -> {os.path.relpath(out_csv)}")

    # Raw reusable reservoir features: multiple classical heads can be evaluated
    # without another QPU run (GARCH residual, persistence residual, raw target).
    out_npz = os.path.join(RESULTS_DIR, f"qbraid_hardware_features{suffix}.npz")
    np.savez_compressed(
        out_npz, R_test_statevector=R_te_local, R_test_device=R_te_hw,
        X_test=X_te_full, y_test_raw=y_te_raw_scoring,
        hybrid_target=args.hybrid_target, device=args.device, shots=args.shots,
        pred_residual_statevector=pred_sim_res, pred_residual_device=pred_hw_res,
        correction_strength_statevector=sim_strength,
        correction_strength_device=hw_strength,
        hardware_calibration_rows=k_hw_cal,
        input_projection=args.input_projection,
        observable_order=args.observable_order,
        readout_training=args.readout_training,
        train_shots=args.train_shots,
        train_p_single=args.train_p_single,
        train_p_two=args.train_p_two,
    )
    print(f"  Saved -> {os.path.relpath(out_npz)}")

    _plot(y_eval, pred_sim[eval_sl], pred_hw[eval_sl], pers_te[eval_sl],
          R_te_local, R_te_hw, args, suffix)

    print("\n--- Done ---")
    if ex.is_simulator():
        print(f"  Simulator job ids: {os.path.join(cache_dir, 'qbraid_job_ids.json')}")
        print("  These validate the cloud path but are not QPU-execution proof.")
    else:
        print(f"  Real quantum job ids: {os.path.join(cache_dir, 'qbraid_job_ids.json')}")
        print("  Keep these with the submission as proof of hardware execution.")


def _plot(y_true, pred_sim, pred_hw, pers, R_sim, R_hw, args, suffix=""):
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
    out = os.path.join(RESULTS_DIR, f"qbraid_hardware_qrc{suffix}.png")
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
                   choices=["linear", "grid", "random", "all-to-all"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--encoding-axis", default="rx", choices=["rx", "ry", "rz", "ryrz"],
                   help="input rotation; ry/rz encode nontrivially on initial |+>")
    p.add_argument("--observable-order", type=int, default=2, choices=[1, 2, 3],
                   help="maximum Z-parity order extracted from the same shots")
    p.add_argument("--input-projection", default="first",
                   choices=["first", "selected", "pca", "random"],
                   help="training-only map from full causal features to qubit inputs")
    p.add_argument("--readout-training", default="exact",
                   choices=["exact", "iqm-noise"],
                   help="train readout on exact or local IQM-like shot/noise features")
    p.add_argument("--train-shots", type=int, default=200,
                   help="local simulator shots per training sample for noise-aware readout")
    p.add_argument("--train-p-single", type=float, default=0.001)
    p.add_argument("--train-p-two", type=float, default=0.013)
    p.add_argument("--hybrid-target", default="garch",
                   choices=["garch", "persistence"],
                   help="classical baseline whose log residual the QRC predicts")
    p.add_argument("--result-tag", default="",
                   help="suffix for CSV/NPZ/plot outputs (recommended for QPU runs)")
    p.add_argument("--correction-strength", type=float, default=1.0,
                   help="pre-locked multiplier on the predicted baseline residual")
    p.add_argument("--hardware-calibration-rows", type=int, default=0,
                   help="first measured rows used only to recalibrate correction strength")
    p.add_argument("--strength-min", type=float, default=-0.25)
    p.add_argument("--strength-max", type=float, default=1.25)
    p.add_argument("--strength-steps", type=int, default=301)
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
    p.add_argument("--credit-budget", type=float, default=100.0,
                   help="hard cap on qBraid's live QPU cost estimate (default: 100 credits)")
    p.add_argument("--spark-budget", type=float, default=25.0,
                   help="hard cap for OpenQuantum Public Compute (default: 25 Spark credits = $50)")
    p.add_argument("--allow-unknown-pricing", action="store_true",
                   help="permit QPU submission when qBraid returns no pricing (use only after manual confirmation)")
    p.add_argument("--dry-run", action="store_true",
                   help="build circuits + print footprint, submit nothing (no qbraid needed)")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    if args.max_test == 0:
        args.max_test = None
    run(args)
