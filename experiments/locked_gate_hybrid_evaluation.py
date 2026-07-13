"""Evaluate the validation-locked gate QRC with a separate hardware calibration block."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.neutral_atom_garch_hybrid import fit_readout, apply_readout
from src.baselines.classical import regression_metrics
from src.data.loaders import load_financial_data_v2
from src.qrc.input_projection import ReservoirInputProjector
from src.qrc.reservoir import QuantumReservoir

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def _vol(log_garch, residual, strength):
    return np.exp(log_garch + strength * residual)


def _best_strength(pred_residual, y_true, log_garch, grid):
    rmse = [np.sqrt(np.mean((_vol(log_garch, pred_residual, s) - y_true) ** 2))
            for s in grid]
    return float(grid[int(np.argmin(rmse))])


def _block_bootstrap_rmse_diff(y, pred, baseline, block=5, draws=5000, seed=42):
    """95% moving-block-bootstrap CI for RMSE(model) - RMSE(baseline)."""
    rng = np.random.RandomState(seed)
    n = len(y)
    diffs = []
    for _ in range(draws):
        idx = []
        while len(idx) < n:
            start = rng.randint(0, n)
            idx.extend((start + np.arange(block)) % n)
        idx = np.asarray(idx[:n], dtype=int)
        model_rmse = np.sqrt(np.mean((pred[idx] - y[idx]) ** 2))
        base_rmse = np.sqrt(np.mean((baseline[idx] - y[idx]) ** 2))
        diffs.append(model_rmse - base_rmse)
    return tuple(float(x) for x in np.percentile(diffs, [2.5, 97.5]))


def main(args) -> None:
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
    X, y = d["X_train"], d["y_train"]
    if args.max_train and args.max_train < len(X):
        X, y = X[-args.max_train:], y[-args.max_train:]
    Xte = d["X_test"][-args.max_test:]
    yte = d["y_test_raw"][-args.max_test:]
    loggp = d["log_garch_proxy_test"][-args.max_test:]

    n = int(locked["qubits"])
    res = QuantumReservoir(
        n_qubits=n, n_layers=int(locked["layers"]),
        connectivity=str(locked["connectivity"]), seed=int(locked["seed"]),
        encoding_axis=str(locked.get("encoding_axis", "rx")),
        observable_order=int(locked.get("observable_order", 2)),
    )
    projection = str(locked.get("input_projection", "first"))
    projector = ReservoirInputProjector(
        n_outputs=n, mode=projection, seed=int(locked["seed"]),
    ).fit(X, feature_names=d["feature_names"])
    R = res.transform(projector.transform(X))
    Rte = res.transform(projector.transform(Xte))
    readout, sc_r, alpha = fit_readout(R, X, y)
    pred_res = apply_readout(readout, sc_r, Rte, Xte)

    # Circuit and readout were locked using pre-test validation. The first block
    # is now used only for amplitude transfer calibration; it is excluded from
    # the reported evaluation block.
    k = args.hardware_calibration
    grid = np.linspace(args.strength_min, args.strength_max, args.strength_steps)
    locked_strength = float(locked["strength"])
    hw_strength = (_best_strength(pred_res[:k], yte[:k], loggp[:k], grid)
                   if k else locked_strength)

    rows = []
    predictions = {}
    for label, strength in (
        ("GARCH", 0.0),
        ("QRC correction (training-validation strength)", locked_strength),
        ("QRC correction (hardware-calibrated strength)", hw_strength),
    ):
        pred = _vol(loggp[k:], pred_res[k:], strength)
        predictions[label] = pred
        rows.append({
            "Model": label, "strength": strength,
            "calibration_rows": k, "evaluation_rows": len(yte) - k,
            **regression_metrics(yte[k:], pred),
        })
    out = pd.DataFrame(rows)
    garch_rmse = float(out.loc[out["Model"] == "GARCH", "RMSE"].iloc[0])
    out["Gain_vs_GARCH_%"] = 100 * (garch_rmse - out["RMSE"]) / garch_rmse
    garch_pred = predictions["GARCH"]
    cis = {
        label: _block_bootstrap_rmse_diff(yte[k:], pred, garch_pred)
        for label, pred in predictions.items()
    }
    out["RMSE_diff_vs_GARCH_CI_low"] = out["Model"].map(lambda m: cis[m][0])
    out["RMSE_diff_vs_GARCH_CI_high"] = out["Model"].map(lambda m: cis[m][1])
    out["locked_layers"] = int(locked["layers"])
    out["locked_connectivity"] = str(locked["connectivity"])
    out["locked_seed"] = int(locked["seed"])
    out["locked_encoding_axis"] = str(locked.get("encoding_axis", "rx"))
    out["locked_input_projection"] = projection
    out["locked_observable_order"] = int(locked.get("observable_order", 2))
    out["alpha"] = alpha
    suffix = f"_{args.result_tag}" if args.result_tag else ""
    path = os.path.join(RESULTS, f"locked_gate_hybrid_evaluation{suffix}.csv")
    out.to_csv(path, index=False)
    np.savez_compressed(
        os.path.join(RESULTS, f"locked_gate_hybrid_features{suffix}.npz"),
        R_test=Rte, pred_residual=pred_res, X_test=Xte,
        y_test_raw=yte, log_garch_test=loggp,
        hardware_calibration_rows=k,
    )
    print("LOCKED CIRCUIT")
    cols = ["qubits", "layers", "connectivity", "encoding_axis",
            "input_projection", "observable_order", "seed",
            "cx_count", "depth", "strength", "validation_gain_vs_GARCH_%"]
    cols = [c for c in cols if c in locked.index]
    print(locked[cols].to_string())
    print(f"\nEvaluation strength={hw_strength:.3f}; hardware calibration rows={k}")
    print("\n" + out[["Model", "strength", "evaluation_rows", "RMSE", "Gain_vs_GARCH_%",
                       "RMSE_diff_vs_GARCH_CI_low", "RMSE_diff_vs_GARCH_CI_high"]]
          .to_string(index=False))
    print(f"\nSaved -> {os.path.relpath(path)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-train", type=int, default=800)
    p.add_argument("--max-test", type=int, default=120)
    p.add_argument("--hardware-calibration", type=int, default=20)
    p.add_argument("--tuning-file", default="hardware_ready_gate_tuning.csv")
    p.add_argument("--result-tag", default="")
    p.add_argument("--encoding-axis", choices=["rx", "ry", "rz", "ryrz"])
    p.add_argument("--strength-min", type=float, default=-0.25)
    p.add_argument("--strength-max", type=float, default=1.25)
    p.add_argument("--strength-steps", type=int, default=301)
    main(p.parse_args())
