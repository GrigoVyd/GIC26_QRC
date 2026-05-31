"""
SPY Market Volatility Forecast -- main runner.

Usage
-----
  python run_volatility.py              # full sweep (3-seed ensemble)
  python run_volatility.py --fast       # 1 seed, skip LSTM
  python run_volatility.py --n-seeds 5  # 5-seed ensemble
  python run_volatility.py --seed 0     # different random seed

Outputs saved to results/
  financial_results.csv
  financial_rmse.png
  financial_predictions.png
  financial_predictions_doc.png   (1200 DPI publication copy)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from experiments.financial_qrc import run


def main() -> None:
    parser = argparse.ArgumentParser(description="SPY volatility QRC forecast")
    parser.add_argument("--fast",    action="store_true", help="Quick run: 1 seed, skip LSTM")
    parser.add_argument("--seed",    type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--n-seeds", type=int, default=3,  help="Ensemble seeds (default: 3)")
    args = parser.parse_args()

    t_start = time.time()
    df = run(seed=args.seed, n_seeds=args.n_seeds, fast=args.fast)
    elapsed = time.time() - t_start

    print("\n" + "=" * 78)
    print("  FINAL RESULTS - sorted by RMSE (lower is better)")
    print("=" * 78)
    cols = ["Model", "RMSE", "MAE", "R2", "NMSE"]
    if "QLIKE" in df.columns:
        cols.append("QLIKE")
    print(df[cols].to_string(index=False))
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print("Results saved to: results/")


if __name__ == "__main__":
    main()
