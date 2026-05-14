"""
MNIST Digit Classification — GIC 2026 Common Benchmark

Compares QRC (5, 10, 15 qubits) against classical baselines on the
sklearn digits dataset (8×8 images, 10 classes).

Outputs
-------
  results/mnist_results.csv   — accuracy table
  results/mnist_accuracy.png  — bar chart
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.qrc.reservoir import QuantumReservoir
from src.data.loaders import load_mnist_digits

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _qrc_classify(
    qrc: QuantumReservoir,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Transform with QRC, scale, then classify with RidgeClassifier."""
    R_train = qrc.transform(X_train)
    R_test = qrc.transform(X_test)

    scaler = StandardScaler()
    R_train = scaler.fit_transform(R_train)
    R_test = scaler.transform(R_test)

    clf = RidgeClassifier(alpha=alpha)
    clf.fit(R_train, y_train)
    return accuracy_score(y_test, clf.predict(R_test))


def run(seed: int = 42, n_samples: int = 800) -> pd.DataFrame:
    print("=" * 65)
    print("  MNIST BENCHMARK — Quantum Reservoir Computing")
    print("=" * 65)

    MAX_QUBITS = 15
    X_tr, X_te, y_tr, y_te = load_mnist_digits(
        n_components=MAX_QUBITS, n_samples=n_samples, seed=seed
    )
    print(f"\nDataset: {X_tr.shape[0]} train / {X_te.shape[0]} test samples, "
          f"{MAX_QUBITS} PCA components\n")

    rows = []

    # ---- Classical baselines ----
    print("Classical baselines:")
    for n_feat in [5, 10, 15]:
        Xtr_f = X_tr[:, :n_feat]
        Xte_f = X_te[:, :n_feat]

        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr_f)
        Xte_s = sc.transform(Xte_f)

        # Ridge
        clf = RidgeClassifier(alpha=1.0)
        clf.fit(Xtr_s, y_tr)
        acc = accuracy_score(y_te, clf.predict(Xte_s))
        rows.append({"Model": f"Ridge ({n_feat} feat.)", "Accuracy": acc})
        print(f"  Ridge ({n_feat} feat.): {acc:.4f}")

    # RBF-SVM on 5 features
    sc5 = StandardScaler()
    svm = SVC(kernel="rbf", C=10.0, gamma="scale")
    svm.fit(sc5.fit_transform(X_tr[:, :5]), y_tr)
    acc = accuracy_score(y_te, svm.predict(sc5.transform(X_te[:, :5])))
    rows.append({"Model": "SVM-RBF (5 feat.)", "Accuracy": acc})
    print(f"  SVM-RBF (5 feat.):   {acc:.4f}")

    # ---- QRC sweep ----
    print("\nQuantum Reservoir Computing:")
    for n_qubits in [5, 10, 15]:
        Xtr_q = X_tr[:, :n_qubits]
        Xte_q = X_te[:, :n_qubits]

        for conn in ["linear", "random"]:
            qrc = QuantumReservoir(
                n_qubits=n_qubits,
                n_layers=2,
                connectivity=conn,
                seed=seed,
            )
            t0 = time.time()
            acc = _qrc_classify(qrc, Xtr_q, Xte_q, y_tr, y_te)
            elapsed = time.time() - t0

            label = f"QRC {n_qubits}q [{conn}]"
            rows.append({"Model": label, "Accuracy": acc})
            print(f"  {label:<30}  {acc:.4f}  ({elapsed:.1f}s)")

    # ---- Results table ----
    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
    df.to_csv(os.path.join(RESULTS_DIR, "mnist_results.csv"), index=False)
    print(f"\nResults saved → results/mnist_results.csv")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(10, 5))
    colours = ["#4C72B0" if "QRC" in r["Model"] else "#DD8452" for _, r in df.iterrows()]
    ax.barh(df["Model"], df["Accuracy"], color=colours)
    ax.set_xlabel("Accuracy")
    ax.set_title("MNIST Benchmark — QRC vs. Classical Baselines")
    ax.axvline(0.5, ls="--", c="grey", lw=0.8, label="chance")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "mnist_accuracy.png"), dpi=150)
    print("Chart saved   → results/mnist_accuracy.png")

    return df


if __name__ == "__main__":
    df = run()
    print("\n--- Summary ---")
    print(df.to_string(index=False))
