"""Shot-noise and IQM-like depolarizing stress test for the locked 9q grid QRC."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from qiskit_aer.noise import NoiseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.neutral_atom_garch_hybrid import fit_readout, apply_readout
from src.baselines.classical import regression_metrics
from src.data.loaders import load_financial_data_v2
from src.qrc.input_projection import ReservoirInputProjector
from src.qrc.noise import depolarizing_noise, iqm_effective_noise
from src.qrc.reservoir import QuantumReservoir

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def _vol(loggp, residual, strength):
    return np.exp(loggp + strength * residual)


def main(args):
    tuning = pd.read_csv(os.path.join(RESULTS, args.tuning_file))
    if args.encoding_axis:
        tuning = tuning[tuning["encoding_axis"] == args.encoding_axis]
        if tuning.empty:
            raise ValueError(f"no tuning rows for encoding axis {args.encoding_axis!r}")
    locked = tuning.sort_values(["validation_rank", "cx_count", "seed"]).iloc[0]
    d = load_financial_data_v2(
        delay=5, log_target=True, include_har=True, include_log_har=True,
        include_garch_proxy=True, garch_residual_target=True,
    )
    X, y = d["X_train"][-args.max_train:], d["y_train"][-args.max_train:]
    Xte = d["X_test"][-args.max_test:]
    yte = d["y_test_raw"][-args.max_test:]
    loggp = d["log_garch_proxy_test"][-args.max_test:]
    n = int(locked["qubits"])
    kwargs = dict(
        n_qubits=n, n_layers=int(locked["layers"]),
        connectivity=str(locked["connectivity"]), seed=int(locked["seed"]),
        encoding_axis=str(locked["encoding_axis"]),
        observable_order=int(locked.get("observable_order", 2)),
    )
    projector = ReservoirInputProjector(
        n_outputs=n, mode=str(locked.get("input_projection", "first")),
        seed=int(locked["seed"]),
    ).fit(X, feature_names=d["feature_names"])
    Xin, Xtein = projector.transform(X), projector.transform(Xte)
    exact = QuantumReservoir(**kwargs)
    R, Rte_exact = exact.transform(Xin), exact.transform(Xtein)
    readout, sc_r, alpha = fit_readout(R, X, y)
    exact_res = apply_readout(readout, sc_r, Rte_exact, Xte)

    ideal = QuantumReservoir(**kwargs, noise_model=NoiseModel(), n_shots=args.shots)
    device_noise = (iqm_effective_noise(
        p_single=args.p_single, p_two=args.p_two,
        error_0_to_1=args.error_0_to_1,
        error_1_to_0=args.error_1_to_0,
    ) if args.effective_iqm else depolarizing_noise(args.p_single, args.p_two))
    noisy = QuantumReservoir(
        **kwargs, noise_model=depolarizing_noise(args.p_single, args.p_two),
        n_shots=args.shots,
    )
    noisy.noise_model = device_noise
    scenarios = [
        ("exact statevector", R, Rte_exact),
        (f"{args.shots}-shot ideal sampling", ideal.transform(Xin), ideal.transform(Xtein)),
        (f"{args.shots}-shot IQM noise proxy", noisy.transform(Xin), noisy.transform(Xtein)),
    ]
    k = args.calibration_rows
    grid = np.linspace(-0.25, 1.25, 301)
    rows = []
    garch_pred = np.exp(loggp[k:])
    garch_rmse = float(np.sqrt(np.mean((garch_pred - yte[k:]) ** 2)))
    for name, Rtrain, Rte in scenarios:
        scenario_readout, scenario_sc_r, scenario_alpha = fit_readout(Rtrain, X, y)
        predictions = [("noise-aware", apply_readout(
            scenario_readout, scenario_sc_r, Rte, Xte
        ), scenario_alpha)]
        if name != "exact statevector":
            predictions.insert(0, ("exact-trained", apply_readout(readout, sc_r, Rte, Xte), alpha))
        for training_policy, pred_res, fitted_alpha in predictions:
            locked_strength = float(locked["strength"])
            strengths = [("locked", locked_strength)]
            if name != "exact statevector" and k:
                cal_scores = [np.sqrt(np.mean(
                    (_vol(loggp[:k], pred_res[:k], s) - yte[:k]) ** 2
                )) for s in grid]
                strengths.append(("wide-calibration", float(grid[int(np.argmin(cal_scores))])))
            for policy, strength in strengths:
                pred = _vol(loggp[k:], pred_res[k:], strength)
                m = regression_metrics(yte[k:], pred)
                rows.append({
                    "Scenario": name, "training_policy": training_policy,
                    "strength_policy": policy,
                    "shots": None if name == "exact statevector" else args.shots,
                    "p_single": 0 if "noise proxy" not in name else args.p_single,
                    "p_two": 0 if "noise proxy" not in name else args.p_two,
                    "feature_corr_vs_exact": 1.0 if name == "exact statevector" else
                        float(np.corrcoef(Rte.ravel(), Rte_exact.ravel())[0, 1]),
                    "correction_strength": strength, "readout_alpha": fitted_alpha,
                    "calibration_rows": k, "evaluation_rows": len(yte) - k,
                    "GARCH_RMSE": garch_rmse,
                    "Gain_vs_GARCH_%": 100 * (garch_rmse - m["RMSE"]) / garch_rmse,
                    **m,
                })
    out = pd.DataFrame(rows)
    suffix = f"_{args.result_tag}" if args.result_tag else ""
    path = os.path.join(RESULTS, f"iqm_q9_grid_noise_stress{suffix}.csv")
    out.to_csv(path, index=False)
    print(out[["Scenario", "training_policy", "strength_policy", "feature_corr_vs_exact", "correction_strength",
               "GARCH_RMSE", "RMSE", "Gain_vs_GARCH_%"]].to_string(index=False))
    print(f"\nSaved -> {os.path.relpath(path)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-train", type=int, default=1200)
    p.add_argument("--max-test", type=int, default=120)
    p.add_argument("--calibration-rows", type=int, default=20)
    p.add_argument("--shots", type=int, default=200)
    p.add_argument("--p-single", type=float, default=0.001)
    p.add_argument("--p-two", type=float, default=0.013)
    p.add_argument("--effective-iqm", action="store_true",
                   help="include asymmetric readout in the effective logical noise model")
    p.add_argument("--error-0-to-1", type=float, default=0.01)
    p.add_argument("--error-1-to-0", type=float, default=0.01)
    p.add_argument("--tuning-file", default=
                   "hardware_ready_gate_tuning_iqm_q9_grid_ryrz_rolling4_worst.csv")
    p.add_argument("--result-tag", default="")
    p.add_argument("--encoding-axis", choices=["rx", "ry", "rz", "ryrz"])
    main(p.parse_args())
