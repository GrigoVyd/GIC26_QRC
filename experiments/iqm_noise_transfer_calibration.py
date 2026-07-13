"""Calibrate a local IQM noise proxy against a completed Resonance QRC batch."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.loaders import load_financial_data_v2
from src.qrc.input_projection import ReservoirInputProjector
from src.qrc.iqm_resonance_backend import get_resonance_backend, get_quality_metrics
from src.qrc.noise import iqm_effective_noise
from src.qrc.reservoir import QuantumReservoir, _observables_from_counts

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def _corr_mae(reference, candidate):
    return (
        float(np.corrcoef(reference.ravel(), candidate.ravel())[0, 1]),
        float(np.mean(np.abs(reference - candidate))),
    )


def main(args):
    with open(os.path.join(RESULTS, args.preflight_file), encoding="utf-8") as handle:
        preflight = json.load(handle)
    layout = preflight["physical_layout_indices"]

    backend = get_resonance_backend(args.device)
    job = backend.retrieve_job(args.job_id or preflight["diagnostic_job_id"])
    result = job.result()
    counts = result.get_counts()
    if isinstance(counts, dict):
        counts = [counts]

    d = load_financial_data_v2(
        delay=5, log_target=True, include_har=True, include_log_har=True,
        include_garch_proxy=True, garch_residual_target=True,
    )
    if not str(d["data_source"]).startswith("yfinance:"):
        raise RuntimeError(f"real SPY data required, got {d['data_source']}")
    X = d["X_train"][-args.max_train:]
    Xte = d["X_test"][-120:]
    projector = ReservoirInputProjector(9, mode="first", seed=44).fit(
        X, feature_names=d["feature_names"]
    )
    selected_rows = np.asarray(args.rows, dtype=int)
    Xin = projector.transform(Xte)[selected_rows]

    kwargs = dict(
        n_qubits=9, n_layers=1, connectivity="grid", seed=44,
        encoding_axis="rz", observable_order=2,
    )
    exact = QuantumReservoir(**kwargs).transform(Xin)
    hardware = np.vstack([
        _observables_from_counts(c, 9, args.hardware_shots, 2) for c in counts
    ])
    hardware_corr, hardware_mae = _corr_mae(exact, hardware)

    metrics, _ = get_quality_metrics(backend)
    e01 = float(np.mean([metrics[q]["error_0_to_1"] for q in layout]))
    e10 = float(np.mean([metrics[q]["error_1_to_0"] for q in layout]))

    rows = []
    for shots in args.sim_shots:
        for p_two in args.p_two:
            correlations, maes = [], []
            for _ in range(args.repeats):
                model = iqm_effective_noise(
                    p_single=max(args.p_single_floor, p_two / 10.0),
                    p_two=p_two, error_0_to_1=e01, error_1_to_0=e10,
                )
                simulated = QuantumReservoir(
                    **kwargs, noise_model=model, n_shots=shots,
                ).transform(Xin)
                corr, mae = _corr_mae(exact, simulated)
                correlations.append(corr)
                maes.append(mae)
            corr_mean, mae_mean = np.mean(correlations), np.mean(maes)
            match_score = (
                abs(corr_mean - hardware_corr) / 0.05
                + abs(mae_mean - hardware_mae) / 0.01
            )
            rows.append({
                "device": args.device, "hardware_job_id": job.job_id(),
                "shots": shots, "p_two_effective": p_two,
                "p_single_effective": max(args.p_single_floor, p_two / 10.0),
                "readout_error_0_to_1": e01, "readout_error_1_to_0": e10,
                "hardware_corr": hardware_corr, "hardware_mae": hardware_mae,
                "sim_corr_mean": corr_mean,
                "sim_corr_std": float(np.std(correlations)),
                "sim_mae_mean": mae_mean,
                "sim_mae_std": float(np.std(maes)),
                "match_score": match_score,
                "repeats": args.repeats,
                "data_source": d["data_source"],
            })
    out = pd.DataFrame(rows).sort_values(["match_score", "shots", "p_two_effective"])
    path = os.path.join(RESULTS, f"iqm_{args.device}_noise_transfer_calibration.csv")
    out.to_csv(path, index=False)
    print(f"Hardware feature correlation={hardware_corr:.4f}, MAE={hardware_mae:.4f}")
    print(out.head(12).to_string(index=False))
    print(f"Saved -> {os.path.relpath(path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="emerald", choices=["emerald", "garnet"])
    parser.add_argument("--preflight-file", default="iqm_emerald_preflight.json")
    parser.add_argument("--job-id")
    parser.add_argument("--max-train", type=int, default=1200)
    parser.add_argument("--hardware-shots", type=int, default=200)
    parser.add_argument("--rows", nargs="+", type=int, default=[0, 23, 47, 71, 95, 119])
    parser.add_argument("--sim-shots", nargs="+", type=int, default=[200, 500, 1000])
    parser.add_argument("--p-two", nargs="+", type=float,
                        default=[0.0, 0.005, 0.01, 0.02, 0.03, 0.05])
    parser.add_argument("--p-single-floor", type=float, default=0.001)
    parser.add_argument("--repeats", type=int, default=3)
    main(parser.parse_args())
