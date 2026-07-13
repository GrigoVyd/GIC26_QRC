"""
Tiny cloud smoke test for the Ising-machine backends.

Use this before running the financial pipeline. It submits a single small Ising
problem and prints only shape/finite summary information. Tokens are read from
environment variables; never put them in the repo.

Examples
--------
PowerShell:
  $env:TOSHIBA_TOKEN="..."
  # Optional if Toshiba gave you a private endpoint:
  # $env:TOSHIBA_URL="https://..."
  python experiments/ising_cloud_smoke.py --backend toshiba

Bash:
  TOSHIBA_TOKEN=... python experiments/ising_cloud_smoke.py --backend toshiba
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.qrc.ising_reservoir import IsingReservoir, _credential


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backend",
        default="toshiba",
        choices=[
            "amplify", "toshiba", "fujitsu", "hitachi", "nec",
            "dwave_amplify",
        ],
    )
    p.add_argument("--spins", type=int, default=6)
    p.add_argument("--reads", type=int, default=20)
    p.add_argument("--timeout-ms", type=int, default=2000)
    args = p.parse_args()

    token_var = "AMPLIFY_AE_TOKEN" if args.backend == "amplify" else f"{args.backend.upper()}_TOKEN"
    has_token = bool(_credential(token_var) or _credential("AMPLIFY_TOKEN"))
    print(f"backend={args.backend} token_present={has_token}")
    if args.backend == "toshiba" and not os.environ.get("TOSHIBA_URL"):
        print("note: TOSHIBA_URL is not set; using the SDK default endpoint.")

    res = IsingReservoir(n_spins=args.spins, input_scale=1.0, beta=1.0, seed=42)
    X = np.zeros((1, args.spins))
    try:
        R = res.transform(
            X,
            backend=args.backend,
            num_reads=args.reads,
            timeout_ms=args.timeout_ms,
        )
    except RuntimeError as exc:
        print(f"FAILED: {exc}")
        raise SystemExit(2) from exc
    print(
        f"ok: features={R.shape} finite={np.isfinite(R).all()} "
        f"mean={float(R.mean()):.4f}"
    )
    if res.last_sampling_stats:
        stats = res.last_sampling_stats[-1]
        print(
            f"ensemble: returned={int(stats['returned_states'])} "
            f"unique={int(stats['unique_states'])}"
        )


if __name__ == "__main__":
    main()
