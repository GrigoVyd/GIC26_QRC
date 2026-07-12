"""Summarize accessible hybrid candidates after a 20-row calibration block."""

from __future__ import annotations

import os
import pandas as pd

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def _row(path, text):
    df = pd.read_csv(os.path.join(RESULTS, path))
    return df[df["Model"].str.contains(text, regex=False)].iloc[0]


def main():
    gate_df = pd.read_csv(os.path.join(RESULTS, "locked_gate_hybrid_evaluation.csv"))
    gate = gate_df[gate_df["Model"].str.contains("hardware-calibrated", regex=False)].iloc[0]
    gate_g = gate_df[gate_df["Model"] == "GARCH"].iloc[0]
    specs = [
        ("Gate QRC — Garnet candidate", gate, gate_g, "5q L3 random seed47", 0.47),
    ]
    for label, path, needle, config in (
        ("Pasqal analog QRC", "neutral_atom_garch_hybrid_pasqal_garch_calibrated_400_120.csv",
         "Pasqal Fresnel", "5 atoms, global detuning"),
        ("Signed-Ising — Amplify candidate", "neutral_atom_garch_hybrid_ising_sa_garch_calibrated_400_120.csv",
         "Ising machine", "10 spins, all-to-all signed J"),
    ):
        model = _row(path, needle)
        base = _row(path, "GARCH(1,1)")
        specs.append((label, model, base, config, float(model["correction_strength"])))

    rows = []
    for label, model, base, config, strength in specs:
        gap = 100 * (float(model["RMSE"]) / float(base["RMSE"]) - 1)
        rows.append({
            "Candidate": label, "Configuration": config,
            "Calibration_rows": 20, "Evaluation_rows": 100,
            "Correction_strength": strength,
            "GARCH_RMSE": float(base["RMSE"]), "Hybrid_RMSE": float(model["RMSE"]),
            "Gap_vs_GARCH_%": gap,
            "At_GARCH_level_1pct": abs(gap) <= 1.0,
        })
    out = pd.DataFrame(rows).sort_values("Gap_vs_GARCH_%")
    path = os.path.join(RESULTS, "calibrated_hardware_candidates.csv")
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved -> {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
