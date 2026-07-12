"""Validation-only optimizer-matched readout for already collected Amplify AE features."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.baselines.classical import regression_metrics
from src.qrc.ising_reservoir import IsingReservoir

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def design(R, X, include_x=True):
    return np.hstack([R, X]) if include_x else R


def fit_predict(Rtr, Xtr, ytr, Rte, Xte, alpha, include_x=True):
    A = design(Rtr, Xtr, include_x)
    B = design(Rte, Xte, include_x)
    scaler = StandardScaler().fit(A)
    model = Ridge(alpha=alpha).fit(scaler.transform(A), ytr)
    return model.predict(scaler.transform(B))


def rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def main(args):
    paid = np.load(os.path.join(RESULTS, args.feature_file))
    X = paid["X_train"]
    Xte = paid["X_test"]
    y = paid["y_train"]
    yte = paid["y_test_raw"]
    loggp = X[:, args.garch_log_column]
    loggp_te = Xte[:, args.garch_log_column]
    yraw = np.exp(loggp + y)
    data_source = str(np.asarray(paid["data_source"]).item())
    if len(X) != args.max_train or len(Xte) != args.max_test:
        raise RuntimeError("saved AE feature window does not match requested dimensions")
    if not data_source.startswith("yfinance:"):
        raise RuntimeError(f"real SPY data required, got {data_source}")
    Rae = paid["R_test"]

    input_scaler = StandardScaler().fit(X[:, :args.spins])
    Xin = input_scaler.transform(X[:, :args.spins])
    Xtein = input_scaler.transform(Xte[:, :args.spins])
    reservoir = IsingReservoir(
        n_spins=args.spins, connectivity="all-to-all", input_scale=1.0,
        J_scale=1.0, beta=1.0, seed=args.seed,
    )
    print(f"Generating exact ground-state proxy features: {len(X)} train / {len(Xte)} test")
    R = reservoir.transform(Xin, backend="ground")
    Rproxy_te = reservoir.transform(Xtein, backend="ground")
    if args.proxy_only:
        Rae = Rproxy_te
    feature_corr = float(np.corrcoef(Rae.ravel(), Rproxy_te.ravel())[0, 1])
    feature_match = float(np.mean(np.isclose(Rae, Rproxy_te, atol=1e-7)))

    initial = len(X) - args.folds * args.fold_rows
    if initial < 250:
        raise ValueError("insufficient initial training history")
    ranges = [(initial + i * args.fold_rows, initial + (i + 1) * args.fold_rows)
              for i in range(args.folds)]
    strengths = np.linspace(args.strength_min, args.strength_max, args.strength_steps)
    candidates = []
    for include_x in (False, True):
        for alpha in args.alphas:
            fold_preds, fold_raw, fold_gp = [], [], []
            for start, end in ranges:
                fold_preds.append(fit_predict(
                    R[:start], X[:start], y[:start], R[start:end], X[start:end],
                    alpha, include_x,
                ))
                fold_raw.append(yraw[start:end])
                fold_gp.append(loggp[start:end])
            all_pred = np.concatenate(fold_preds)
            all_raw = np.concatenate(fold_raw)
            all_gp = np.concatenate(fold_gp)
            scores = [rmse(all_raw, np.exp(all_gp + s * all_pred)) for s in strengths]
            strength = float(strengths[int(np.argmin(scores))])
            gains = []
            for pred, raw, gp in zip(fold_preds, fold_raw, fold_gp):
                base = rmse(raw, np.exp(gp))
                model = rmse(raw, np.exp(gp + strength * pred))
                gains.append(100 * (base - model) / base)
            candidates.append({
                "include_classical_x": include_x, "alpha": float(alpha),
                "strength": strength, "validation_gain_mean_%": float(np.mean(gains)),
                "validation_gain_std_%": float(np.std(gains)),
                "validation_gain_min_%": float(np.min(gains)),
                "selection_score": float(np.mean(gains) - np.std(gains)),
            })

    table = pd.DataFrame(candidates).sort_values("selection_score", ascending=False)
    best = table.iloc[0]
    pred_resid = fit_predict(
        R, X, y, Rae, Xte, float(best["alpha"]), bool(best["include_classical_x"])
    )
    pred = np.exp(loggp_te + float(best["strength"]) * pred_resid)
    metrics = regression_metrics(yte, pred)
    garch_rmse = rmse(yte, np.exp(loggp_te))

    # Direct IQM-style locked comparator, evaluated without selecting on test labels.
    iqm_resid = fit_predict(R, X, y, Rae, Xte, 300.0, True)
    iqm_pred = np.exp(loggp_te + 0.105 * iqm_resid)
    iqm_metrics = regression_metrics(yte, iqm_pred)

    table["selected"] = False
    table.loc[table.index[0], "selected"] = True
    table.to_csv(os.path.join(RESULTS, args.result_tag + "_validation.csv"), index=False)
    summary = {
        "data_source": data_source,
        "evaluation_feature_source": "exact_ground_proxy" if args.proxy_only else "amplify_ae",
        "paid_cloud_solves_reused": 0 if args.proxy_only else len(Xte),
        "feature_corr_ground_vs_ae": feature_corr,
        "exact_feature_match_fraction": feature_match,
        "selected_include_classical_x": bool(best["include_classical_x"]),
        "selected_alpha": float(best["alpha"]),
        "selected_strength": float(best["strength"]),
        "validation_gain_mean_percent": float(best["validation_gain_mean_%"]),
        "validation_gain_min_percent": float(best["validation_gain_min_%"]),
        "garch_rmse": garch_rmse, "optimizer_matched_metrics": metrics,
        "optimizer_matched_gain_vs_garch_percent": 100 * (garch_rmse - metrics["RMSE"]) / garch_rmse,
        "iqm_locked_alpha": 300.0, "iqm_locked_strength": 0.105,
        "iqm_locked_metrics": iqm_metrics,
        "iqm_locked_gain_vs_garch_percent": 100 * (garch_rmse - iqm_metrics["RMSE"]) / garch_rmse,
    }
    with open(os.path.join(RESULTS, args.result_tag + "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(table.head(10).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--feature-file", default="hybrid_features_amplify_ae_check_ising_sa_seed42.npz")
    p.add_argument("--result-tag", default="amplify_ae_optimizer_transfer_10")
    p.add_argument("--max-train", type=int, default=400)
    p.add_argument("--max-test", type=int, default=10)
    p.add_argument("--spins", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--garch-log-column", type=int, default=22)
    p.add_argument("--proxy-only", action="store_true")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--fold-rows", type=int, default=20)
    p.add_argument("--alphas", type=float, nargs="+", default=[10, 100, 300, 1000, 3000, 10000, 30000])
    p.add_argument("--strength-min", type=float, default=-0.05)
    p.add_argument("--strength-max", type=float, default=0.50)
    p.add_argument("--strength-steps", type=int, default=111)
    main(p.parse_args())
