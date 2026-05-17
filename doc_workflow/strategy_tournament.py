"""
Strategy Tournament -- GIC 2026 Phase 2 Document Workflow

Runs all 3 QRC strategies x 3 tasks x 2 qubit counts and 4 classical
baselines, producing results/doc/strategy_comparison.csv.

Tasks
-----
  narma5    : NARMA-5 (standard RC benchmark, metric: NMSE)
  financial : SPY realized volatility (metric: NMSE, RMSE, R^2)
  mnist01   : MNIST binary 0-vs-1 (metric: Accuracy)

Design choices (tournament only)
---------------------------------
  n_layers = 2  (fixed -- prevents combinatorial explosion)
  connectivity = "random"
  seed = 42
  n_samples capped at 600 train for tournament speed
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategies import STRATEGIES, QRCStrategy
from src.data.loaders import load_narma, load_financial_data, load_mnist_digits
from src.baselines.classical import (
    PersistenceBaseline, ARBaseline, RidgeBaseline,
    EchoStateNetwork, regression_metrics,
)

DOC_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "doc")
os.makedirs(DOC_DIR, exist_ok=True)

N_LAYERS = 2
SEED = 42
N_QUBITS_LIST = [5, 10]
TOURNAMENT_SAMPLES = 600   # training cap for speed


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _qrc_regress(strategy: QRCStrategy, X_tr, X_te, y_tr, alpha=0.1):
    n = strategy.n_qubits
    sc_in = StandardScaler()
    R_tr = strategy.transform(sc_in.fit_transform(X_tr[:, :n]))
    R_te = strategy.transform(sc_in.transform(X_te[:, :n]))
    sc_r = StandardScaler()
    R_tr = sc_r.fit_transform(R_tr)
    R_te = sc_r.transform(R_te)
    readout = Ridge(alpha=alpha)
    readout.fit(R_tr, y_tr)
    return readout.predict(R_te)


def _qrc_classify(strategy: QRCStrategy, X_tr, X_te, y_tr, y_te, alpha=0.1):
    n = strategy.n_qubits
    sc_in = StandardScaler()
    R_tr = strategy.transform(sc_in.fit_transform(X_tr[:, :n]))
    R_te = strategy.transform(sc_in.transform(X_te[:, :n]))
    sc_r = StandardScaler()
    R_tr = sc_r.fit_transform(R_tr)
    R_te = sc_r.transform(R_te)
    clf = RidgeClassifier(alpha=alpha)
    clf.fit(R_tr, y_tr)
    return accuracy_score(y_te, clf.predict(R_te))


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------

def _run_narma5(rows: list, seed: int = SEED) -> None:
    print("\n  Task: NARMA-5")
    X_tr, X_te, y_tr, y_te = load_narma(order=5, n_samples=800, seed=seed)
    X_tr = X_tr[:TOURNAMENT_SAMPLES]
    y_tr = y_tr[:TOURNAMENT_SAMPLES]

    # Baselines
    for cls, name in [
        (ARBaseline(order=5), "AR(5)"),
        (RidgeBaseline(), "Ridge"),
        (EchoStateNetwork(n_reservoir=100, seed=seed), "ESN-100"),
    ]:
        cls.fit(X_tr, y_tr)
        m = regression_metrics(y_te, cls.predict(X_te))
        rows.append({"Task": "NARMA-5", "Strategy": name, "n_qubits": "-",
                     "primary_metric": m["NMSE"], "metric_name": "NMSE",
                     **m, "elapsed_s": 0})
        print(f"    {name:<20}  NMSE={m['NMSE']:.5f}")

    # QRC strategies
    for sname, Cls in STRATEGIES.items():
        for n_q in N_QUBITS_LIST:
            s = Cls(n_qubits=n_q, n_layers=N_LAYERS, seed=seed)
            t0 = time.time()
            y_hat = _qrc_regress(s, X_tr[:, :n_q], X_te[:, :n_q], y_tr)
            elapsed = time.time() - t0
            m = regression_metrics(y_te, y_hat)
            rows.append({"Task": "NARMA-5", "Strategy": s.label,
                         "n_qubits": n_q, "primary_metric": m["NMSE"],
                         "metric_name": "NMSE", **m, "elapsed_s": elapsed})
            print(f"    {s.label:<25} {n_q}q  NMSE={m['NMSE']:.5f}  ({elapsed:.1f}s)")


def _run_financial(rows: list, seed: int = SEED) -> None:
    print("\n  Task: Financial volatility (SPY)")
    X_tr, X_te, y_tr, y_te = load_financial_data(delay=5)
    X_tr = X_tr[:TOURNAMENT_SAMPLES]
    y_tr = y_tr[:TOURNAMENT_SAMPLES]

    for cls, name in [
        (PersistenceBaseline(), "Persistence"),
        (ARBaseline(order=5), "AR(5)"),
        (RidgeBaseline(), "Ridge"),
        (EchoStateNetwork(n_reservoir=100, seed=seed), "ESN-100"),
    ]:
        cls.fit(X_tr, y_tr)
        m = regression_metrics(y_te, cls.predict(X_te))
        rows.append({"Task": "Financial", "Strategy": name, "n_qubits": "-",
                     "primary_metric": m["NMSE"], "metric_name": "NMSE", **m, "elapsed_s": 0})
        print(f"    {name:<20}  NMSE={m['NMSE']:.5f}  R^2={m['R2']:.4f}")

    for sname, Cls in STRATEGIES.items():
        for n_q in N_QUBITS_LIST:
            s = Cls(n_qubits=n_q, n_layers=N_LAYERS, seed=seed)
            t0 = time.time()
            y_hat = _qrc_regress(s, X_tr[:, :n_q], X_te[:, :n_q], y_tr)
            elapsed = time.time() - t0
            m = regression_metrics(y_te, y_hat)
            rows.append({"Task": "Financial", "Strategy": s.label,
                         "n_qubits": n_q, "primary_metric": m["NMSE"],
                         "metric_name": "NMSE", **m, "elapsed_s": elapsed})
            print(f"    {s.label:<25} {n_q}q  NMSE={m['NMSE']:.5f}  R^2={m['R2']:.4f}  ({elapsed:.1f}s)")


def _run_mnist01(rows: list, seed: int = SEED) -> None:
    """Binary MNIST: digit 0 vs digit 1 for speed."""
    print("\n  Task: MNIST binary (0 vs 1)")
    X_tr, X_te, y_tr, y_te = load_mnist_digits(
        n_components=10, n_samples=None, seed=seed
    )
    # Keep only classes 0 and 1
    mask_tr = y_tr <= 1
    mask_te = y_te <= 1
    X_tr, y_tr = X_tr[mask_tr], y_tr[mask_tr]
    X_te, y_te = X_te[mask_te], y_te[mask_te]
    X_tr = X_tr[:TOURNAMENT_SAMPLES]
    y_tr = y_tr[:TOURNAMENT_SAMPLES]

    # Classical baselines (Ridge only for classification)
    sc = StandardScaler()
    clf = RidgeClassifier(alpha=1.0)
    clf.fit(sc.fit_transform(X_tr[:, :5]), y_tr)
    acc = accuracy_score(y_te, clf.predict(sc.transform(X_te[:, :5])))
    rows.append({"Task": "MNIST-01", "Strategy": "Ridge (5 feat.)", "n_qubits": "-",
                 "primary_metric": acc, "metric_name": "Accuracy",
                 "RMSE": None, "MAE": None, "R2": None, "NMSE": None, "elapsed_s": 0})
    print(f"    {'Ridge (5 feat.)':<20}  Acc={acc:.4f}")

    for sname, Cls in STRATEGIES.items():
        for n_q in N_QUBITS_LIST:
            s = Cls(n_qubits=n_q, n_layers=N_LAYERS, seed=seed)
            t0 = time.time()
            acc = _qrc_classify(s, X_tr[:, :n_q], X_te[:, :n_q], y_tr, y_te)
            elapsed = time.time() - t0
            rows.append({"Task": "MNIST-01", "Strategy": s.label,
                         "n_qubits": n_q, "primary_metric": acc,
                         "metric_name": "Accuracy",
                         "RMSE": None, "MAE": None, "R2": None, "NMSE": None,
                         "elapsed_s": elapsed})
            print(f"    {s.label:<25} {n_q}q  Acc={acc:.4f}  ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(seed: int = SEED) -> pd.DataFrame:
    print("=" * 65)
    print("  STRATEGY TOURNAMENT  (3 strategies x 3 tasks)")
    print("=" * 65)

    rows: list[dict] = []
    _run_narma5(rows, seed)
    _run_financial(rows, seed)
    _run_mnist01(rows, seed)

    df = pd.DataFrame(rows)
    out = os.path.join(DOC_DIR, "strategy_comparison.csv")
    df.to_csv(out, index=False)
    print(f"\n  Saved -> {out}")

    # Quick best-per-task summary
    print("\n  -- Best per task --")
    for task in df["Task"].unique():
        sub = df[df["Task"] == task].copy()
        # For NMSE lower is better, for accuracy higher is better
        if sub["metric_name"].iloc[0] == "NMSE":
            best = sub.loc[sub["primary_metric"].idxmin()]
            print(f"    {task:<12}  {best['Strategy']} {best['n_qubits']}q  "
                  f"NMSE={best['primary_metric']:.5f}")
        else:
            best = sub.loc[sub["primary_metric"].idxmax()]
            print(f"    {task:<12}  {best['Strategy']} {best['n_qubits']}q  "
                  f"Acc={best['primary_metric']:.4f}")

    return df


if __name__ == "__main__":
    run()
