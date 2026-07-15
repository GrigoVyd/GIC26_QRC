"""Build the credit-safe Phase 3 qBraid reproducibility notebook.

The generated notebook is intentionally safe under ``Run All``: it reads saved
artifacts and runs a local statevector smoke test, but never submits a cloud or
QPU job unless the user edits an explicit opt-in flag.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "phase3_qbraid_reproducibility.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(True),
    }


CELLS = [
    md(
        r"""
        # Phase 3: hybrid QRC hardware evidence

        **Team Quanties - GIC 2026, Track A: Financial Volatility Prediction**

        This notebook reproduces the headline comparison directly from the compact
        CSV/JSON evidence committed with the project, audits the real-QPU job records,
        and validates the fixed quantum-reservoir code path locally.

        > **Credit-safe `Run All`:** no token is read, no network call is made, and no
        > paid job is submitted. The optional qBraid simulator cell is disabled by
        > default and rejects any device whose metadata is not free.
        """
    ),
    md(
        """
        ## Claim discipline

        - The 9-qubit IQM Emerald hybrid has a **0.0895% RMSE point improvement**
          over its matched GARCH baseline on 120 chronological days.
        - Its moving-block-bootstrap interval crosses zero and QLIKE is slightly
          worse, so this is **hardware-level competitiveness**, not proof of quantum
          advantage.
        - The QuEra Aquila run is a real analog-QPU transfer result. It is close to
          GARCH, but does not beat it. A new paid run is permitted only after a
          validation-locked local analog candidate clears the go/no-go gate below.
        """
    ),
    code(
        """
        from pathlib import Path
        import json
        import sys

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import display

        def find_repo_root(start=Path.cwd()):
            for candidate in [start, *start.parents]:
                if (candidate / "src" / "qrc").is_dir() and (candidate / "results").is_dir():
                    return candidate
            raise FileNotFoundError("Open this notebook from the cloned GIC26_QRC repository.")

        ROOT = find_repo_root()
        RESULTS = ROOT / "results"
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        print(f"Repository: {ROOT}")
        print(f"Python:     {sys.version.split()[0]}")
        """
    ),
    md("## 1. Evidence manifest"),
    code(
        """
        manifest = {
            "IQM 9q summary": "iqm_emerald_q9_grid_rz_500_summary.csv",
            "IQM 9q inference": "iqm_emerald_q9_grid_rz_500_inference.json",
            "IQM 12q summary": "iqm_emerald_q12_grid_rz_500_summary.csv",
            "Aquila summary": "quera_aquila_summary_qbraid_aquila_hw_native_neg_23x50.csv",
            "Aquila checkpoint": "quera_aquila_checkpoint_qbraid_aquila_hw_native_neg_23x50.json",
            "Amplify AE summary": "neutral_atom_garch_hybrid_amplify_ae_iqm_locked_400_120.csv",
            "Toshiba summary": "neutral_atom_garch_hybrid_toshiba_garch_400_120.csv",
            "TFIM simulator summary": "phase2_garch_hybrid.csv",
        }
        evidence = pd.DataFrame(
            [{"artifact": label, "path": str(RESULTS / name), "present": (RESULTS / name).exists()}
             for label, name in manifest.items()]
        )
        display(evidence)
        assert evidence["present"].all(), "One or more compact evidence files are missing."
        """
    ),
    md("## 2. Recompute the headline table"),
    code(
        """
        iqm9 = pd.read_csv(RESULTS / manifest["IQM 9q summary"])
        iqm12 = pd.read_csv(RESULTS / manifest["IQM 12q summary"])
        aquila = pd.read_csv(RESULTS / manifest["Aquila summary"])
        amplify = pd.read_csv(RESULTS / manifest["Amplify AE summary"])
        toshiba = pd.read_csv(RESULTS / manifest["Toshiba summary"])
        tfim = pd.read_csv(RESULTS / manifest["TFIM simulator summary"])

        def only(df, column, text):
            rows = df[df[column].astype(str).str.contains(text, regex=False)]
            assert len(rows) == 1, (text, len(rows))
            return rows.iloc[0]

        r9 = only(iqm9, "Model", "hardware_qrem_affine")
        r12 = only(iqm12, "Model", "hardware_affine")
        raq = only(aquila, "Model", "Aquila (hardware)")
        gaq = only(aquila, "Model", "GARCH proxy")
        aae = only(amplify, "Model", "Ising machine signed-J")
        gae = only(amplify, "Model", "GARCH(1,1)")
        tos = only(toshiba, "Model", "Ising machine signed-J")
        gto = only(toshiba, "Model", "GARCH(1,1)")
        sim = only(tfim, "Model", "QRC on GARCH-residual (3-seed ensemble)")
        gsim = only(tfim, "Model", "GARCH(1,1)")

        rows = [
            ("IQM Emerald 9q", "real gate QPU", r9.RMSE, r9.GARCH_RMSE, 120, "primary"),
            ("IQM Emerald 12q", "real gate QPU", r12.RMSE, r12.GARCH_RMSE, 120, "scaling control"),
            ("QuEra Aquila 5a", "real analog QPU", raq.RMSE, gaq.RMSE, 20, "transfer test"),
            ("Fixstars Amplify AE", "cloud Ising machine", aae.RMSE, gae.RMSE, 120, "substrate control"),
            ("Toshiba SQBM+ V2", "cloud Ising machine", tos.RMSE, gto.RMSE, 120, "substrate control"),
            ("TFIM QRC ensemble", "statevector simulator", sim.RMSE, gsim.RMSE, 120, "mechanism evidence"),
        ]
        comparison = pd.DataFrame(rows, columns=["platform", "execution", "RMSE", "matched_GARCH_RMSE", "scored_rows", "role"])
        comparison["gap_vs_GARCH_%"] = 100 * (comparison.RMSE / comparison.matched_GARCH_RMSE - 1)
        display(comparison.style.format({"RMSE": "{:.8f}", "matched_GARCH_RMSE": "{:.8f}", "gap_vs_GARCH_%": "{:+.3f}"}))

        assert np.isclose(comparison.iloc[0]["gap_vs_GARCH_%"], -0.0895060673, atol=1e-6)
        assert np.isclose(comparison.iloc[2]["gap_vs_GARCH_%"], 0.1822917216, atol=1e-6)
        """
    ),
    code(
        """
        plot_df = comparison.sort_values("gap_vs_GARCH_%")
        colors = ["#159c8c" if value < 0 else "#c85a45" for value in plot_df["gap_vs_GARCH_%"]]
        fig, ax = plt.subplots(figsize=(9, 4.4))
        ax.barh(plot_df.platform, plot_df["gap_vs_GARCH_%"], color=colors)
        ax.axvline(0, color="#15253d", linewidth=1)
        ax.set_xlabel("RMSE gap vs matched GARCH (%) - negative is better")
        ax.set_title("Hybrid QRC evidence across execution substrates")
        ax.grid(axis="x", alpha=0.2)
        for i, value in enumerate(plot_df["gap_vs_GARCH_%"]):
            ax.text(value + (0.025 if value >= 0 else -0.025), i, f"{value:+.3f}%",
                    va="center", ha="left" if value >= 0 else "right", fontsize=9)
        plt.tight_layout()
        plt.show()
        """
    ),
    md("## 3. IQM statistical and native-layout audit"),
    code(
        """
        inference = json.loads((RESULTS / manifest["IQM 9q inference"]).read_text(encoding="utf-8"))
        ci_low, ci_high = inference["moving_block_bootstrap_rmse_diff_ci_95"]
        iqm_audit = pd.Series({
            "job_id": inference["job_id"],
            "physical_layout": ", ".join(inference["physical_layout_names"]),
            "circuits x shots": f'{inference["circuits"]} x {inference["shots_per_circuit"]}',
            "compiled depth": inference["compiled_depth"],
            "CZ per circuit": inference["compiled_cz_per_circuit"],
            "SWAPs": inference["compiled_swaps"],
            "RMSE gain vs GARCH": f'{inference["gain_vs_garch_percent"]:.4f}%',
            "bootstrap 95% CI": f"[{ci_low:+.7f}, {ci_high:+.7f}]",
            "P(model better)": inference["bootstrap_probability_model_better"],
            "MZ joint p-value": inference["hardware_hybrid_mincer_zarnowitz"]["joint_p_value_hac"],
        })
        display(iqm_audit.to_frame("value"))
        assert inference["compiled_swaps"] == 0
        assert ci_low < 0 < ci_high
        print("Interpretation: positive point estimate, but no statistically established advantage.")
        """
    ),
    code(
        """
        scaling = pd.DataFrame([
            {"reservoir": "9q native grid", "feature_corr": r9.feature_corr_vs_exact,
             "gap_vs_GARCH_%": 100 * (r9.RMSE / r9.GARCH_RMSE - 1), "depth": r9.compiled_depth, "CZ": r9.compiled_cz},
            {"reservoir": "12q native grid", "feature_corr": r12.feature_corr_vs_exact,
             "gap_vs_GARCH_%": 100 * (r12.RMSE / r12.GARCH_RMSE - 1), "depth": r12.compiled_depth, "CZ": r12.compiled_cz},
        ])
        display(scaling.style.format({"feature_corr": "{:.3f}", "gap_vs_GARCH_%": "{:+.3f}"}))
        assert scaling.loc[1, "feature_corr"] > scaling.loc[0, "feature_corr"]
        assert scaling.loc[1, "gap_vs_GARCH_%"] > scaling.loc[0, "gap_vs_GARCH_%"]
        print("More qubits improved feature correlation but worsened forecast RMSE: width alone was not the advantage.")
        """
    ),
    md("## 4. Aquila audit and 2,500-credit go/no-go rule"),
    code(
        """
        checkpoint = json.loads((RESULTS / manifest["Aquila checkpoint"]).read_text(encoding="utf-8"))
        assert checkpoint["completed_rows"] == 23
        assert len(checkpoint["features"]) == 23
        assert len(checkpoint["task_ids"]) == 23
        assert int(raq.feature_calibration_rows) == 3

        local_aq = only(aquila, "Model", "local AHS sim")
        local_gap = 100 * (local_aq.RMSE / gaq.RMSE - 1)
        hardware_gap = 100 * (raq.RMSE / gaq.RMSE - 1)
        transfer_gap = 100 * (raq.RMSE / local_aq.RMSE - 1)
        aquila_audit = pd.Series({
            "device": "aws:quera:qpu:aquila",
            "tasks": checkpoint["completed_rows"],
            "shots per task": checkpoint["shots"],
            "scored rows": 20,
            "feature calibration rows": 3,
            "observed qBraid credits": 1840,
            "local AHS gap vs GARCH": f"{local_gap:+.3f}%",
            "hardware gap vs GARCH": f"{hardware_gap:+.3f}%",
            "hardware gap vs local AHS": f"{transfer_gap:+.3f}%",
            "first task ID": checkpoint["task_ids"][0],
        })
        display(aquila_audit.to_frame("value"))
        """
    ),
    code(
        """
        # Pricing is the live metadata observed for the completed campaign.
        # Re-read device metadata before any future submission.
        added_budget = 2500
        credits_per_task = 30
        credits_per_shot = 1
        shots = 50
        max_tasks = added_budget // (credits_per_task + shots * credits_per_shot)
        campaign_cost = max_tasks * (credits_per_task + shots * credits_per_shot)
        calibration_rows = 3
        scored_rows = max_tasks - calibration_rows
        required_local_margin_pct = 0.30
        passes_gate = local_gap <= -required_local_margin_pct

        budget_plan = pd.Series({
            "new credit budget": added_budget,
            "locked tasks at 50 shots": max_tasks,
            "estimated campaign cost": campaign_cost,
            "engineering smoke": "first 4 tasks, then resume same checkpoint",
            "label-free calibration rows": calibration_rows,
            "scored rows if completed": scored_rows,
            "required local-simulator advantage": f">= {required_local_margin_pct:.2f}% vs GARCH",
            "current candidate local gap": f"{local_gap:+.3f}%",
            "current candidate passes": passes_gate,
        })
        display(budget_plan.to_frame("value"))
        assert campaign_cost == 2480 and max_tasks == 31 and scored_rows == 28
        print("GO" if passes_gate else "NO-GO: optimize and lock a better analog candidate locally before spending.")
        """
    ),
    md(
        """
        The existing candidate fails the gate because even its ideal local AHS result is
        worse than GARCH. Increasing shots may reduce variance, but cannot repair that
        structural gap. If a new candidate passes the validation gate, the preferred
        campaign is 31 tasks x 50 shots (2,480 credits): submit four engineering-smoke
        tasks first, then resume the identical checkpoint for 27 more tasks. Hyperparameters
        remain frozen and the first three rows are used only for label-free feature transfer.
        """
    ),
    md("## 5. Local quantum-reservoir code-path validation"),
    code(
        """
        from src.qrc.reservoir import QuantumReservoir

        rng = np.random.RandomState(2026)
        X_smoke = rng.normal(size=(6, 9))
        reservoir = QuantumReservoir(
            n_qubits=9, n_layers=1, connectivity="grid", seed=44,
            encoding_axis="rz", observable_order=2,
        )
        R1 = reservoir.transform(X_smoke)
        R2 = reservoir.transform(X_smoke)
        assert R1.shape == (6, 45)
        assert np.all(np.isfinite(R1)) and np.allclose(R1, R2)
        assert float(np.std(R1)) > 0.01
        print(f"PASS: deterministic 9q native-grid reservoir -> {R1.shape}, std={np.std(R1):.4f}")
        """
    ),
    md("## 6. Optional free qBraid simulator smoke test"),
    code(
        """
        RUN_QBRAID_SIM = False  # Change manually to True inside an authenticated qBraid Lab session.

        if RUN_QBRAID_SIM:
            from qbraid.runtime import QbraidProvider
            from qiskit import QuantumCircuit

            provider = QbraidProvider()
            device = provider.get_device("qbraid:qbraid:sim:qir-sv")
            metadata = device.metadata()
            pricing = metadata.get("pricing", {}) or {}
            assert ":sim:" in str(metadata.get("device_id", "qbraid:qbraid:sim:qir-sv"))
            assert float(pricing.get("perTask", 0)) == 0
            assert float(pricing.get("perShot", 0)) == 0

            circuit = QuantumCircuit(2)
            circuit.h(0); circuit.cx(0, 1); circuit.measure_all()
            job = device.run(circuit, shots=256)
            print("Submitted free simulator smoke:", job.id)
        else:
            print("Skipped (default). No qBraid API call and no credits used.")
        """
    ),
    md(
        """
        ## Reproducibility outcome

        A successful default run establishes that:

        1. all reported headline values are recomputed from compact committed artifacts;
        2. the real IQM and Aquila execution records are internally consistent;
        3. native connectivity, shot counts, calibration exclusions, and uncertainty
           qualifiers are explicit; and
        4. the repository's fixed 9-qubit grid reservoir executes deterministically.

        No paid hardware cell is included. Hardware campaigns are launched only from
        the guarded experiment scripts after reviewing the printed footprint, current
        device metadata, a locked configuration manifest, and an explicit credit cap.
        """
    ),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


def validate_default_cells() -> None:
    """Execute code cells in one namespace, matching a clean default Run All."""
    namespace = {"__name__": "__phase3_notebook__"}
    for index, cell in enumerate(CELLS):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, f"{OUT.name}:cell-{index}", "exec"), namespace)
    print("Default notebook validation passed: all code cells executed.")


if __name__ == "__main__":
    main()
    if "--validate" in sys.argv:
        validate_default_cells()
