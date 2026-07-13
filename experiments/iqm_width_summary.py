"""Summarize validation-locked IQM-feasible width/topology experiments."""

from __future__ import annotations

import os
import pandas as pd

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

FILES = [
    ("6q chain", "hardware_ready_gate_tuning_iqm_q6_linear.csv"),
    ("8q chain", "hardware_ready_gate_tuning_iqm_q8_linear.csv"),
    ("10q chain", "hardware_ready_gate_tuning_iqm_q10_linear.csv"),
    ("12q chain", "hardware_ready_gate_tuning_iqm_q12_linear.csv"),
    ("9q grid, old Rx encoding", "hardware_ready_gate_tuning_iqm_q9_grid.csv"),
    ("9q native grid, corrected Ry/Rz", "hardware_ready_gate_tuning_iqm_q9_grid_ryrz_rolling4_worst.csv"),
]


def main():
    rows = []
    for experiment, filename in FILES:
        df = pd.read_csv(os.path.join(RESULTS, filename))
        locked = df.sort_values(["validation_rank", "cx_count", "seed"]).iloc[0]
        rows.append({
            "Experiment": experiment,
            "Qubits": int(locked["qubits"]),
            "Layers": int(locked["layers"]),
            "Connectivity": locked["connectivity"],
            "Encoding": locked.get("encoding_axis", "rx"),
            "Seed": int(locked["seed"]),
            "CX_count": int(locked["cx_count"]),
            "Correction_strength": float(locked["strength"]),
            "Validation_gain_vs_GARCH_%": float(locked["validation_gain_vs_GARCH_%"]),
            "Test_gain_vs_GARCH_%": float(locked["test_gain_vs_GARCH_%"]),
            "Hybrid_RMSE": float(locked["RMSE"]),
            "GARCH_RMSE": float(locked["test_GARCH_RMSE"]),
        })
    out = pd.DataFrame(rows)
    path = os.path.join(RESULTS, "iqm_hardware_aware_width_summary.csv")
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved -> {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
