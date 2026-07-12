"""Validation-only tuning of a shallow gate-QRC correction to GARCH.

The QRC predicts the log residual relative to GARCH. A scalar correction strength
is selected on the final chronological slice of the training period. Including
zero in the grid lets validation reject an unhelpful reservoir and fall back to
GARCH without inspecting the held-out test labels.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.neutral_atom_garch_hybrid import fit_readout, apply_readout
from src.baselines.classical import regression_metrics
from src.data.loaders import load_financial_data_v2
from src.qrc.reservoir import QuantumReservoir
from src.qrc.input_projection import ReservoirInputProjector

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def _vol(log_garch: np.ndarray, residual: np.ndarray, strength: float = 1.0) -> np.ndarray:
    return np.exp(log_garch + strength * residual)


def _choose_strength(
    pred_residual: np.ndarray,
    y_true: np.ndarray,
    log_garch: np.ndarray,
    grid: np.ndarray,
) -> tuple[float, float]:
    scores = [np.sqrt(np.mean((_vol(log_garch, pred_residual, s) - y_true) ** 2))
              for s in grid]
    i = int(np.argmin(scores))
    return float(grid[i]), float(scores[i])


def run(args) -> pd.DataFrame:
    d = load_financial_data_v2(
        delay=5, log_target=True, include_har=True, include_log_har=True,
        include_garch_proxy=True, garch_residual_target=True,
    )
    X_full, y_full = d["X_train"], d["y_train"]
    y_raw_full = d["y_train_raw"]
    log_gp_full = d["log_garch_proxy_train"]
    if args.max_train and args.max_train < len(X_full):
        X = X_full[-args.max_train:]
        y = y_full[-args.max_train:]
        y_raw = y_raw_full[-args.max_train:]
        log_gp = log_gp_full[-args.max_train:]
    else:
        X, y, y_raw, log_gp = X_full, y_full, y_raw_full, log_gp_full

    n_test = min(args.max_test, len(d["X_test"]))
    X_test = d["X_test"][-n_test:]
    y_test_raw = d["y_test_raw"][-n_test:]
    log_gp_test = d["log_garch_proxy_test"][-n_test:]

    n_cal = (args.calibration_rows if args.calibration_rows > 0
             else max(60, int(len(X) * args.calibration_fraction)))
    n_fit = len(X) - args.rolling_folds * n_cal
    if n_fit < 100:
        raise ValueError("Need at least 100 pre-calibration training rows")
    fold_ranges = [(n_fit + f * n_cal, n_fit + (f + 1) * n_cal)
                   for f in range(args.rolling_folds)]

    strengths = np.linspace(args.strength_min, args.strength_max, args.strength_steps)
    rows: list[dict] = []
    print(f"Data: {d['data_source']} | train={len(X)} (initial_fit={n_fit}, "
          f"validation_folds={args.rolling_folds}x{n_cal}) "
          f"| untouched test={n_test}")

    for connectivity in args.connectivities:
        for layers in args.layers:
          for encoding_axis in args.encoding_axes:
           for projection_mode in args.input_projections:
            for observable_order in args.observable_orders:
             for seed in args.seeds:
                t0 = time.time()
                res = QuantumReservoir(
                    n_qubits=args.qubits, n_layers=layers,
                    connectivity=connectivity, seed=seed,
                    encoding_axis=encoding_axis,
                    observable_order=observable_order,
                )
                projector = ReservoirInputProjector(
                    args.qubits, mode=projection_mode, seed=seed
                ).fit(X, feature_names=d["feature_names"])
                Xin = projector.transform(X)
                Xte_in = projector.transform(X_test)
                R = res.transform(Xin)
                R_test = res.transform(Xte_in)

                # Expanding-window out-of-fold predictions. Configuration and
                # correction strength use training-period labels only.
                pred_blocks, y_blocks, raw_blocks, gp_blocks = [], [], [], []
                for start, end in fold_ranges:
                    readout_cal, sc_cal, _ = fit_readout(
                        R[:start], X[:start], y[:start]
                    )
                    pred_blocks.append(apply_readout(
                        readout_cal, sc_cal, R[start:end], X[start:end]
                    ))
                    y_blocks.append(y[start:end])
                    raw_blocks.append(y_raw[start:end])
                    gp_blocks.append(log_gp[start:end])
                pred_cal = np.concatenate(pred_blocks)
                y_cal_raw = np.concatenate(raw_blocks)
                gp_cal = np.concatenate(gp_blocks)
                strength, val_rmse = _choose_strength(
                    pred_cal, y_cal_raw, gp_cal, strengths
                )
                val_garch_rmse = float(np.sqrt(np.mean(
                    (np.exp(gp_cal) - y_cal_raw) ** 2
                )))
                fold_gains = []
                offset = 0
                for raw_block, gp_block in zip(raw_blocks, gp_blocks):
                    size = len(raw_block)
                    pred_block = pred_cal[offset:offset + size]
                    model_rmse = float(np.sqrt(np.mean(
                        (_vol(gp_block, pred_block, strength) - raw_block) ** 2
                    )))
                    base_rmse = float(np.sqrt(np.mean(
                        (np.exp(gp_block) - raw_block) ** 2
                    )))
                    fold_gains.append(100 * (base_rmse - model_rmse) / base_rmse)
                    offset += size

                # Refit on all pre-test history after locking the strength.
                readout, sc_r, alpha = fit_readout(R, X, y)
                pred_test_res = apply_readout(readout, sc_r, R_test, X_test)
                pred_test = _vol(log_gp_test, pred_test_res, strength)
                metrics = regression_metrics(y_test_raw, pred_test)
                garch_test_rmse = float(np.sqrt(np.mean(
                    (np.exp(log_gp_test) - y_test_raw) ** 2
                )))

                sample = res._build_circuit(Xte_in[0], add_measurement=False)
                cx = int(sample.count_ops().get("cx", 0))
                row = {
                    "qubits": args.qubits, "layers": layers,
                    "connectivity": connectivity, "encoding_axis": encoding_axis,
                    "input_projection": projection_mode,
                    "observable_order": observable_order, "seed": seed,
                    "strength": strength, "alpha": alpha,
                    "validation_RMSE": val_rmse,
                    "validation_folds": args.rolling_folds,
                    "validation_rows_per_fold": n_cal,
                    "validation_GARCH_RMSE": val_garch_rmse,
                    "validation_gain_vs_GARCH_%": 100 * (val_garch_rmse - val_rmse) / val_garch_rmse,
                    "validation_fold_gain_mean_%": float(np.mean(fold_gains)),
                    "validation_fold_gain_std_%": float(np.std(fold_gains)),
                    "validation_fold_gain_min_%": float(np.min(fold_gains)),
                    "test_GARCH_RMSE": garch_test_rmse,
                    "test_gain_vs_GARCH_%": 100 * (garch_test_rmse - metrics["RMSE"]) / garch_test_rmse,
                    "cx_count": cx, "depth": sample.depth(),
                    "elapsed": time.time() - t0,
                    "data_source": d["data_source"],
                    **metrics,
                }
                rows.append(row)
                print(f"L{layers} {connectivity:<6} {encoding_axis:<4} "
                      f"{projection_mode:<8} O{observable_order} seed={seed:<3} "
                      f"lambda={strength:.2f} val_gain={row['validation_gain_vs_GARCH_%']:+.2f}% "
                      f"test_gain={row['test_gain_vs_GARCH_%']:+.2f}% cx={cx}")

    df = pd.DataFrame(rows)
    # Lock by validation performance only. Test columns are never used for rank.
    if args.selection == "aggregate":
        df["selection_score"] = -df["validation_RMSE"]
    elif args.selection == "mean-minus-std":
        df["selection_score"] = (df["validation_fold_gain_mean_%"]
                                 - df["validation_fold_gain_std_%"])
    else:
        df["selection_score"] = df["validation_fold_gain_min_%"]
    df["validation_rank"] = df["selection_score"].rank(
        method="min", ascending=False).astype(int)
    df = df.sort_values(["validation_rank", "cx_count", "encoding_axis",
                         "input_projection", "observable_order", "seed"])
    suffix = f"_{args.result_tag}" if args.result_tag else ""
    out = os.path.join(RESULTS, f"hardware_ready_gate_tuning{suffix}.csv")
    df.to_csv(out, index=False)
    best = df.iloc[0]
    print("\nLOCKED CONFIGURATION (selected only by validation RMSE)")
    print(best[["qubits", "layers", "connectivity", "encoding_axis",
                "input_projection", "observable_order", "seed", "strength",
                "selection_score", "validation_gain_vs_GARCH_%",
                "validation_fold_gain_min_%", "test_gain_vs_GARCH_%",
                "test_GARCH_RMSE", "RMSE", "cx_count", "depth"]].to_string())
    print(f"\nSaved -> {os.path.relpath(out)}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--qubits", type=int, default=5)
    p.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--connectivities", nargs="+", default=["linear", "random"],
                   choices=["linear", "grid", "random", "all-to-all"])
    p.add_argument("--encoding-axes", nargs="+", default=["rx"],
                   choices=["rx", "ry", "rz", "ryrz"])
    p.add_argument("--input-projections", nargs="+", default=["first"],
                   choices=["first", "selected", "pca", "random"])
    p.add_argument("--observable-orders", type=int, nargs="+", default=[2],
                   choices=[1, 2, 3])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(40, 48)))
    p.add_argument("--max-train", type=int, default=800)
    p.add_argument("--max-test", type=int, default=120)
    p.add_argument("--calibration-fraction", type=float, default=0.2)
    p.add_argument("--calibration-rows", type=int, default=0,
                   help="fixed rows per validation fold; overrides calibration-fraction")
    p.add_argument("--rolling-folds", type=int, default=1,
                   help="expanding-window validation folds used for model selection")
    p.add_argument("--selection", default="aggregate",
                   choices=["aggregate", "mean-minus-std", "worst-fold"],
                   help="validation-only configuration ranking rule")
    p.add_argument("--strength-min", type=float, default=-0.25)
    p.add_argument("--strength-max", type=float, default=1.25)
    p.add_argument("--strength-steps", type=int, default=61)
    p.add_argument("--result-tag", default="")
    run(p.parse_args())
