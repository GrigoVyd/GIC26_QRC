"""
SPY Market Volatility Forecast — main runner.

Usage
-----
  python run_volatility.py              # full sweep (5q + 10q)
  python run_volatility.py --fast       # 5q/2L only, quick sanity check
  python run_volatility.py --full       # also include 15q (slow)
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
    parser.add_argument("--fast",  action="store_true", help="Quick run: 5q/2L/random only")
    parser.add_argument("--full",  action="store_true", help="Include 15-qubit configs (slow)")
    parser.add_argument("--seed",  type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--delay", type=int, default=5,  help="Delay-embedding window (default: 5)")
    args = parser.parse_args()

    t_start = time.time()
    df = run(seed=args.seed, delay=args.delay, fast=args.fast, include_15q=args.full)
    elapsed = time.time() - t_start

    print("\n" + "=" * 70)
    print("  FINAL RESULTS - sorted by NMSE (lower is better)")
    print("=" * 70)
    print(df[["Model", "RMSE", "MAE", "R2", "NMSE"]].to_string(index=False))
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print("Results saved to: results/")


if __name__ == "__main__":
    main()
