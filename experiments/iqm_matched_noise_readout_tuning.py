"""Validation-only readout tuning under an Emerald-matched effective noise model."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.baselines.classical import regression_metrics
from src.data.loaders import load_financial_data_v2
from src.qrc.input_projection import ReservoirInputProjector
from src.qrc.noise import iqm_effective_noise
from src.qrc.reservoir import QuantumReservoir

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def _design(R, X):
    return np.hstack([R, X])


def _fit_predict(Rtr, Xtr, ytr, Rte, Xte, alpha):
    scaler = StandardScaler().fit(_design(Rtr, Xtr))
    model = Ridge(alpha=alpha).fit(scaler.transform(_design(Rtr, Xtr)), ytr)
    return model.predict(scaler.transform(_design(Rte, Xte))), model, scaler


def _vol(log_garch, residual, strength):
    return np.exp(log_garch + strength * residual)


def _rmse(y, pred):
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def main(args):
    d = load_financial_data_v2(
        delay=5, log_target=True, include_har=True, include_log_har=True,
        include_garch_proxy=True, garch_residual_target=True,
    )
    if not str(d["data_source"]).startswith("yfinance:"):
        raise RuntimeError(f"real SPY data required, got {d['data_source']}")
    X = d["X_train"][-args.max_train:]
    y = d["y_train"][-args.max_train:]
    yraw = d["y_train_raw"][-args.max_train:]
    loggp = d["log_garch_proxy_train"][-args.max_train:]
    Xte = d["X_test"][-args.max_test:]
    yte = d["y_test_raw"][-args.max_test:]
    loggp_te = d["log_garch_proxy_test"][-args.max_test:]

    projector = ReservoirInputProjector(args.qubits, mode="first", seed=args.seed).fit(
        X, feature_names=d["feature_names"]
    )
    Xin, Xtein = projector.transform(X), projector.transform(Xte)
    noise = iqm_effective_noise(
        p_single=args.p_single, p_two=args.p_two,
        error_0_to_1=args.error_0_to_1,
        error_1_to_0=args.error_1_to_0,
    )
    reservoir = QuantumReservoir(
        n_qubits=args.qubits, n_layers=1, connectivity="grid", seed=args.seed,
        encoding_axis="rz", observable_order=2,
        noise_model=noise, n_shots=args.shots,
    )
    print(f"Generating matched-noise features: train={len(X)}, test={len(Xte)}")
    R, Rte = reservoir.transform(Xin), reservoir.transform(Xtein)
    np.savez_compressed(
        os.path.join(RESULTS, f"iqm_matched_noise_features_{args.result_tag}.npz"),
        R_train=R, R_test=Rte, X_train=X, X_test=Xte, y_train=y,
        y_train_raw=yraw, y_test_raw=yte, log_garch_train=loggp,
        log_garch_test=loggp_te,
    )

    initial = len(X) - args.folds * args.fold_rows
    if initial < 300:
        raise ValueError("insufficient initial training history")
    ranges = [(initial + i * args.fold_rows, initial + (i + 1) * args.fold_rows)
              for i in range(args.folds)]
    strength_grid = np.linspace(args.strength_min, args.strength_max, args.strength_steps)
    candidates = []
    for alpha in args.alphas:
        preds, raws, gps = [], [], []
        for start, end in ranges:
            pred, _, _ = _fit_predict(
                R[:start], X[:start], y[:start], R[start:end], X[start:end], alpha
            )
            preds.append(pred)
            raws.append(yraw[start:end])
            gps.append(loggp[start:end])
        pred_all, raw_all, gp_all = map(np.concatenate, (preds, raws, gps))
        rmses = [_rmse(raw_all, _vol(gp_all, pred_all, strength))
                 for strength in strength_grid]
        strength = float(strength_grid[int(np.argmin(rmses))])
        fold_gains = []
        for pred, raw, gp in zip(preds, raws, gps):
            base = _rmse(raw, np.exp(gp))
            model = _rmse(raw, _vol(gp, pred, strength))
            fold_gains.append(100 * (base - model) / base)
        candidates.append({
            "alpha": alpha, "strength": strength,
            "validation_gain_mean_%": float(np.mean(fold_gains)),
            "validation_gain_std_%": float(np.std(fold_gains)),
            "validation_gain_min_%": float(np.min(fold_gains)),
            "selection_score": float(np.mean(fold_gains) - np.std(fold_gains)),
        })

    table = pd.DataFrame(candidates).sort_values("selection_score", ascending=False)
    best = table.iloc[0]
    _, model, scaler = _fit_predict(R, X, y, Rte, Xte, float(best["alpha"]))
    pred_residual = model.predict(scaler.transform(_design(Rte, Xte)))
    prediction = _vol(loggp_te, pred_residual, float(best["strength"]))
    metrics = regression_metrics(yte, prediction)
    base_rmse = _rmse(yte, np.exp(loggp_te))
    table["selected"] = False
    table.loc[table.index[0], "selected"] = True
    table.loc[table.index[0], "test_GARCH_RMSE"] = base_rmse
    table.loc[table.index[0], "test_RMSE"] = metrics["RMSE"]
    table.loc[table.index[0], "test_gain_vs_GARCH_%"] = (
        100 * (base_rmse - metrics["RMSE"]) / base_rmse
    )
    table["shots"] = args.shots
    table["p_single"] = args.p_single
    table["p_two"] = args.p_two
    table["data_source"] = d["data_source"]
    path = os.path.join(RESULTS, f"iqm_matched_noise_readout_tuning_{args.result_tag}.csv")
    table.to_csv(path, index=False)
    print(table.to_string(index=False))
    print("\nLOCKED BY TRAINING VALIDATION")
    print(best.to_string())
    print(f"Test RMSE={metrics['RMSE']:.9f}, GARCH={base_rmse:.9f}, "
          f"gain={100 * (base_rmse - metrics['RMSE']) / base_rmse:+.4f}%")
    print(f"Saved -> {os.path.relpath(path)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-train", type=int, default=1200)
    p.add_argument("--max-test", type=int, default=120)
    p.add_argument("--qubits", type=int, default=9)
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--shots", type=int, default=200)
    p.add_argument("--p-single", type=float, default=0.008)
    p.add_argument("--p-two", type=float, default=0.08)
    p.add_argument("--error-0-to-1", type=float, default=0.00888888888888889)
    p.add_argument("--error-1-to-0", type=float, default=0.01302222222222222)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--fold-rows", type=int, default=160)
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[100, 300, 1000, 3000, 10000, 30000, 100000])
    p.add_argument("--strength-min", type=float, default=-0.05)
    p.add_argument("--strength-max", type=float, default=0.50)
    p.add_argument("--strength-steps", type=int, default=111)
    p.add_argument("--result-tag", default="emerald_200")
    main(p.parse_args())
