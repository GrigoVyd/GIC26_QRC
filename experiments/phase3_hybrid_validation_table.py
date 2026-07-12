"""
Phase 3 hybrid validation table.

This table keeps the central claim explicit: the GARCH-residual hybrid is the
advantage candidate, while Amplify/Toshiba are executable classical-Ising checks
that test whether signed couplings alone reproduce the edge.
"""

from __future__ import annotations

import os
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def _row(df: pd.DataFrame, contains: str) -> pd.Series:
    hit = df[df["Model"].str.contains(contains, regex=False)]
    if hit.empty:
        raise ValueError(f"Missing row containing {contains!r}")
    return hit.iloc[0]


def _pct_delta(rmse: float, ref: float) -> float:
    return 100.0 * (rmse - ref) / ref


def main() -> None:
    hybrid = pd.read_csv(os.path.join(RESULTS_DIR, "phase2_garch_hybrid.csv"))
    amp = pd.read_csv(os.path.join(RESULTS_DIR, "neutral_atom_garch_hybrid_amplify.csv"))
    tos = pd.read_csv(os.path.join(RESULTS_DIR, "neutral_atom_garch_hybrid_toshiba.csv"))
    qbraid = pd.read_csv(os.path.join(RESULTS_DIR, "qbraid_hardware_summary_120_qir_sv.csv"))

    garch = _row(hybrid, "GARCH(1,1)")
    ridge = _row(hybrid, "Ridge on GARCH-residual target")
    qrc_hybrid = _row(hybrid, "QRC on GARCH-residual (3-seed ensemble)")

    rows = [
        {
            "Tier": "Benchmark",
            "System": "GARCH(1,1)",
            "Window": "746 test",
            "Physics": "classical econometric prior",
            "RMSE": garch["RMSE"],
            "QLIKE": garch["QLIKE"],
            "Delta_vs_GARCH_%": 0.0,
            "Delta_vs_RidgeResidual_%": _pct_delta(garch["RMSE"], ridge["RMSE"]),
            "Interpretation": "strong baseline",
        },
        {
            "Tier": "Linear ablation",
            "System": "Ridge on GARCH residual",
            "Window": "746 test",
            "Physics": "no reservoir",
            "RMSE": ridge["RMSE"],
            "QLIKE": ridge["QLIKE"],
            "Delta_vs_GARCH_%": _pct_delta(ridge["RMSE"], garch["RMSE"]),
            "Delta_vs_RidgeResidual_%": 0.0,
            "Interpretation": "checks whether residual target alone is enough",
        },
        {
            "Tier": "Hybrid advantage candidate",
            "System": "QRC annealer, 3-seed",
            "Window": "746 test",
            "Physics": "transverse-field Ising + signed J",
            "RMSE": qrc_hybrid["RMSE"],
            "QLIKE": qrc_hybrid["QLIKE"],
            "Delta_vs_GARCH_%": _pct_delta(qrc_hybrid["RMSE"], garch["RMSE"]),
            "Delta_vs_RidgeResidual_%": _pct_delta(qrc_hybrid["RMSE"], ridge["RMSE"]),
            "Interpretation": "only current GARCH-beating reservoir result",
        },
    ]

    for label, df in (("Amplify AE", amp), ("Toshiba SQBM+", tos)):
        local_garch = _row(df, "GARCH(1,1)")
        local_ridge = _row(df, "Ridge on GARCH-residual")
        ising = _row(df, "Ising machine signed-J")
        rows.append(
            {
                "Tier": "Executed cloud Ising check",
                "System": label,
                "Window": "50 train / 10 test",
                "Physics": "classical signed Ising, no transverse field",
                "RMSE": ising["RMSE"],
                "QLIKE": ising["QLIKE"],
                "Delta_vs_GARCH_%": _pct_delta(ising["RMSE"], local_garch["RMSE"]),
                "Delta_vs_RidgeResidual_%": _pct_delta(ising["RMSE"], local_ridge["RMSE"]),
                "Interpretation": "runs end-to-end, but does not beat residual ablation",
            }
        )

    qbraid_qrc = _row(qbraid, "qBraid hardware")
    qbraid_persistence = _row(qbraid, "Persistence")
    rows.append(
        {
            "Tier": "qBraid simulator execution",
            "System": "Gate QRC on qir-sv",
            "Window": "120 test",
            "Physics": "gate reservoir, not GARCH hybrid",
            "RMSE": qbraid_qrc["RMSE"],
            "QLIKE": qbraid_qrc["QLIKE"],
            "Delta_vs_GARCH_%": None,
            "Delta_vs_RidgeResidual_%": None,
            "Interpretation": (
                f"execution/fidelity proof; beats persistence by "
                f"{-_pct_delta(qbraid_qrc['RMSE'], qbraid_persistence['RMSE']):.2f}%"
            ),
        }
    )

    out = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "phase3_hybrid_validation.csv")
    out.to_csv(out_csv, index=False)
    print(out[["Tier", "System", "Window", "RMSE", "Delta_vs_GARCH_%", "Interpretation"]]
          .to_string(index=False))
    print(f"\nSaved -> {os.path.relpath(out_csv)}")


if __name__ == "__main__":
    main()
