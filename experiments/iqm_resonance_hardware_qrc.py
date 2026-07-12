"""Run the validation-locked QRC on IQM Resonance with explicit native layout."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.baselines.classical import regression_metrics
from src.qrc.input_projection import ReservoirInputProjector
from src.qrc.iqm_resonance_backend import (
    compile_native,
    get_quality_metrics,
    get_resonance_backend,
    mitigate_z_zz,
    select_native_grid,
)
from src.qrc.reservoir import QuantumReservoir, _observables_from_counts

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def _features(counts, shots, n_qubits):
    if isinstance(counts, dict):
        counts = [counts]
    return np.vstack([_observables_from_counts(c, n_qubits, shots, 2) for c in counts])


def _retrieve_counts(backend, job_id):
    job = backend.retrieve_job(job_id)
    return job, job.result().get_counts()


def _affine_transfer(backend, job_ids, rows, sim_reference, n_qubits):
    batches = []
    for job_id in job_ids:
        _, counts = _retrieve_counts(backend, job_id)
        batches.append([counts] if isinstance(counts, dict) else counts)
    if len(batches) == 1:
        hardware = _features(batches[0], 500, n_qubits)
    else:
        combined = [
            dict(sum((Counter(batch[i]) for batch in batches), Counter()))
            for i in range(len(batches[0]))
        ]
        hardware = _features(combined, 500 * len(batches), n_qubits)
    target = sim_reference[np.asarray(rows, dtype=int)]
    hw_flat, target_flat = hardware.ravel(), target.ravel()
    scale = float(np.sum((hw_flat - hw_flat.mean()) * (target_flat - target_flat.mean()))
                  / np.sum((hw_flat - hw_flat.mean()) ** 2))
    bias = float(target_flat.mean() - scale * hw_flat.mean())
    return scale, bias


def main(args):
    feature_path = os.path.join(RESULTS, args.feature_file)
    tuning_path = os.path.join(RESULTS, args.tuning_file)
    cached = np.load(feature_path)
    tuning = pd.read_csv(tuning_path)
    locked = tuning[tuning["selected"].astype(str).str.lower() == "true"].iloc[0]
    X, Xte = cached["X_train"], cached["X_test"]
    y, yte = cached["y_train"], cached["y_test_raw"]
    loggp_te = cached["log_garch_test"]
    Rtrain, Rsim = cached["R_train"], cached["R_test"]

    projector = ReservoirInputProjector(
        args.qubits, mode="first", seed=args.seed
    ).fit(X)
    Xtein = projector.transform(Xte)
    reservoir = QuantumReservoir(
        n_qubits=args.qubits, n_layers=1, connectivity="grid", seed=args.seed,
        encoding_axis="rz", observable_order=2,
    )
    exact = reservoir.transform(Xtein)
    logical = [reservoir._build_circuit(x, add_measurement=True) for x in Xtein]

    backend = get_resonance_backend(args.device)
    selection = select_native_grid(backend, args.grid_rows, args.grid_cols)
    layout = selection["layout"]
    compiled = compile_native(logical, backend, layout, seed=args.seed)
    sample = compiled[0]
    print(f"Device={args.device}; circuits={len(compiled)}; shots={args.shots}")
    print(f"Layout={selection['layout_names']} indices={layout}")
    print(f"CZ mean={selection['edge_fidelity_mean']:.6f}, "
          f"min={selection['edge_fidelity_min']:.6f}")
    print(f"Compiled depth={sample.depth()}, ops={dict(sample.count_ops())}")

    metadata_path = os.path.join(RESULTS, f"iqm_{args.device}_{args.result_tag}_job.json")
    if args.retrieve_job:
        job, counts = _retrieve_counts(backend, args.retrieve_job)
    else:
        if not args.allow_qpu:
            with open(metadata_path.replace("_job.json", "_dry_run.json"),
                      "w", encoding="utf-8") as handle:
                json.dump({
                    "device": args.device, "circuits": len(compiled),
                    "shots": args.shots, "total_shots": len(compiled) * args.shots,
                    "layout_indices": layout, "layout_names": selection["layout_names"],
                    "cz_fidelity_mean": selection["edge_fidelity_mean"],
                    "cz_fidelity_min": selection["edge_fidelity_min"],
                    "compiled_depth": sample.depth(),
                    "compiled_ops": dict(sample.count_ops()),
                    "swap_count": sample.count_ops().get("swap", 0),
                    "submitted": False,
                }, handle, indent=2)
            print("DRY RUN: no Resonance job submitted; pass --allow-qpu to execute")
            return
        job = backend.run(compiled, shots=args.shots)
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump({
                "job_id": job.job_id(), "device": args.device,
                "circuits": len(compiled), "shots": args.shots,
                "layout_indices": layout, "layout_names": selection["layout_names"],
                "cz_fidelity_mean": selection["edge_fidelity_mean"],
                "cz_fidelity_min": selection["edge_fidelity_min"],
                "compiled_depth": sample.depth(),
                "compiled_ops": dict(sample.count_ops()),
            }, handle, indent=2)
        print(f"Submitted job {job.job_id()}; metadata -> {metadata_path}")
        counts = job.result().get_counts()

    raw = _features(counts, args.shots, args.qubits)
    metrics, _ = get_quality_metrics(backend)
    qrem = mitigate_z_zz(raw, layout, metrics)
    scale, bias = _affine_transfer(
        backend, args.calibration_job_ids, args.calibration_rows, Rsim, args.qubits
    )
    scenarios = {
        "hardware_raw": raw,
        "hardware_qrem": qrem,
        "hardware_affine": scale * raw + bias,
        "hardware_qrem_affine": scale * qrem + bias,
        "matched_simulator": Rsim,
        "exact_statevector": exact,
    }

    design_train = np.hstack([Rtrain, X])
    scaler = StandardScaler().fit(design_train)
    model = Ridge(alpha=float(locked["alpha"])).fit(
        scaler.transform(design_train), y
    )
    base = np.exp(loggp_te)
    base_rmse = float(np.sqrt(np.mean((base - yte) ** 2)))
    rows = []
    for name, features in scenarios.items():
        residual = model.predict(scaler.transform(np.hstack([features, Xte])))
        prediction = np.exp(loggp_te + float(locked["strength"]) * residual)
        result = regression_metrics(yte, prediction)
        rows.append({
            "Model": name, "device": args.device, "job_id": job.job_id(),
            "shots": args.shots, "qubits": args.qubits, "compiled_depth": sample.depth(),
            "compiled_cz": sample.count_ops().get("cz", 0), "compiled_swaps": 0,
            "feature_corr_vs_exact": float(np.corrcoef(
                features.ravel(), exact.ravel())[0, 1]),
            "feature_mae_vs_exact": float(np.mean(np.abs(features - exact))),
            "affine_scale": scale, "affine_bias": bias,
            "readout_alpha": float(locked["alpha"]),
            "correction_strength": float(locked["strength"]),
            "GARCH_RMSE": base_rmse,
            "Gain_vs_GARCH_%": 100 * (base_rmse - result["RMSE"]) / base_rmse,
            **result,
        })
    out = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS, f"iqm_{args.device}_{args.result_tag}_summary.csv")
    npz_path = os.path.join(RESULTS, f"iqm_{args.device}_{args.result_tag}_features.npz")
    out.to_csv(csv_path, index=False)
    np.savez_compressed(npz_path, raw=raw, qrem=qrem, exact=exact,
                        matched_simulator=Rsim, X_test=Xte, y_test=yte,
                        log_garch_test=loggp_te)
    print(out[["Model", "feature_corr_vs_exact", "feature_mae_vs_exact",
               "RMSE", "GARCH_RMSE", "Gain_vs_GARCH_%"]].to_string(index=False))
    print(f"Saved -> {csv_path}\nSaved -> {npz_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="emerald", choices=["emerald", "garnet"])
    p.add_argument("--qubits", type=int, default=9)
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--grid-rows", type=int, default=3)
    p.add_argument("--grid-cols", type=int, default=3)
    p.add_argument("--shots", type=int, default=500)
    p.add_argument("--feature-file", default="iqm_matched_noise_features_emerald_500.npz")
    p.add_argument("--tuning-file", default="iqm_matched_noise_readout_tuning_emerald_500.csv")
    p.add_argument("--calibration-job-ids", nargs="+", default=[
        "019f5688-779d-7193-90e8-55d26dcc4bb3",
        "019f568a-05a8-7b81-825c-768e073f786f",
    ])
    p.add_argument("--calibration-rows", nargs="+", type=int,
                   default=[0, 23, 47, 71, 95, 119])
    p.add_argument("--retrieve-job")
    p.add_argument("--result-tag", default="q9_grid_rz_500")
    p.add_argument("--allow-qpu", action="store_true")
    main(p.parse_args())
