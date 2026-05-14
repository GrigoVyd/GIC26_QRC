"""
GIC 2026 Quantum Reservoir Computing — Master Benchmark Runner

Runs all experiments sequentially and produces a final summary report.

Usage
-----
    python run_all.py                   # all experiments (may take ~30 min)
    python run_all.py --fast            # small datasets, faster run (~5 min)
    python run_all.py --skip-noise      # skip shot-based noise experiments
    python run_all.py --exp mnist       # single experiment
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _header(title: str) -> None:
    print("\n" + "█" * 65)
    print(f"  {title}")
    print("█" * 65 + "\n")


def _run_experiment(name: str, fn, *args, **kwargs):
    _header(name)
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"\n  ✓  {name} completed in {elapsed:.1f}s")
        return result
    except Exception as exc:
        print(f"\n  ✗  {name} FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="GIC26 QRC Benchmark Suite")
    parser.add_argument("--fast", action="store_true",
                        help="Smaller datasets for quick validation run")
    parser.add_argument("--skip-noise", action="store_true",
                        help="Skip shot-based noise experiments (slower)")
    parser.add_argument("--exp", choices=["mnist", "financial", "narma", "weather",
                                          "scaling", "noise", "all"],
                        default="all", help="Which experiment to run")
    args = parser.parse_args()

    # Lazy imports to keep startup fast
    from experiments.mnist_qrc import run as run_mnist
    from experiments.financial_qrc import run as run_financial
    from experiments.narma_benchmark import run as run_narma
    from experiments.weather_qrc import run as run_weather
    from experiments.scaling_noise import run_scaling, run_noise

    seed = 42
    summary_rows = []

    print("\n" + "═" * 65)
    print("  GIC 2026 — Quantum Reservoir Computing Phase 2 Evaluation")
    print("  Team benchmark suite | May 2026")
    print("═" * 65)

    # ------------------------------------------------------------------ MNIST
    if args.exp in ("all", "mnist"):
        n_samples = 400 if args.fast else 800
        df = _run_experiment("MNIST Classification", run_mnist,
                             seed=seed, n_samples=n_samples)
        if df is not None:
            best = df.sort_values("Accuracy", ascending=False).iloc[0]
            summary_rows.append({
                "Experiment": "MNIST (accuracy ↑)",
                "Best Model": best["Model"],
                "Score": f"{best['Accuracy']:.4f}",
            })

    # -------------------------------------------------------------- Financial
    if args.exp in ("all", "financial"):
        df = _run_experiment("Financial Volatility (SPY)", run_financial, seed=seed)
        if df is not None:
            best = df.sort_values("NMSE").iloc[0]
            summary_rows.append({
                "Experiment": "Financial vol. (NMSE ↓)",
                "Best Model": best["Model"],
                "Score": f"{best['NMSE']:.5f}",
            })

    # --------------------------------------------------------------- NARMA
    if args.exp in ("all", "narma"):
        df = _run_experiment("NARMA Benchmark", run_narma, seed=seed)
        if df is not None:
            for order in [5, 10]:
                sub = df[df["Task"] == f"NARMA-{order}"].sort_values("NMSE")
                best = sub.iloc[0]
                summary_rows.append({
                    "Experiment": f"NARMA-{order} (NMSE ↓)",
                    "Best Model": best["Model"],
                    "Score": f"{best['NMSE']:.5f}",
                })

    # -------------------------------------------------------------- Weather
    if args.exp in ("all", "weather"):
        df = _run_experiment("Weather/Lorenz Forecast", run_weather, seed=seed)
        if df is not None:
            best = df.sort_values("NMSE").iloc[0]
            summary_rows.append({
                "Experiment": "Weather (NMSE ↓)",
                "Best Model": best["Model"],
                "Score": f"{best['NMSE']:.5f}",
            })

    # ------------------------------------------------------------- Scaling
    if args.exp in ("all", "scaling"):
        df = _run_experiment("Qubit Scaling Analysis", run_scaling, seed=seed)
        if df is not None:
            best = df.sort_values("NMSE").iloc[0]
            summary_rows.append({
                "Experiment": "Scaling best (NMSE ↓)",
                "Best Model": f"QRC {int(best['n_qubits'])}q L{int(best['n_layers'])}",
                "Score": f"{best['NMSE']:.5f}",
            })

    # --------------------------------------------------------------- Noise
    if args.exp in ("all", "noise") and not args.skip_noise:
        df = _run_experiment("Noise Analysis", run_noise, seed=seed)
        if df is not None:
            noiseless = df[df["noise"] == "none"]["NMSE"].iloc[0]
            worst = df["NMSE"].max()
            summary_rows.append({
                "Experiment": "Noise degradation (NMSE ratio)",
                "Best Model": "noiseless vs high",
                "Score": f"{noiseless:.5f} → {worst:.5f}",
            })

    # ---------------------------------------------------------- Final summary
    if summary_rows:
        print("\n" + "═" * 65)
        print("  PHASE 2 SUMMARY")
        print("═" * 65)
        df_sum = pd.DataFrame(summary_rows)
        print(df_sum.to_string(index=False))
        df_sum.to_csv(os.path.join(RESULTS_DIR, "phase2_summary.csv"), index=False)
        print(f"\n  Full results in: {RESULTS_DIR}/")
        print("  Plots:  results/*.png")
        print("  Tables: results/*.csv")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
