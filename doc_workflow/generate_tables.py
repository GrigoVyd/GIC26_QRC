"""
Document Table Generator -- GIC 2026 Phase 2

Reads results/doc/strategy_comparison.csv and produces three Markdown tables
in results/doc/ ready to paste into the Phase 2 document (Word or LaTeX).

  table_strategy_summary.md   -- 3 strategies x 3 tasks (primary metric)
  table_baselines_financial.md -- QRC (best) vs classical baselines on SPY
  table_qubit_scaling.md      -- n_qubits x NMSE for best strategy, all tasks

Requires: results/doc/strategy_comparison.csv (from strategy_tournament.py).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DOC_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "doc")
os.makedirs(DOC_DIR, exist_ok=True)


def _write(name: str, content: str) -> str:
    path = os.path.join(DOC_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Saved -> {path}")
    return path


def _fmt_nmse(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{float(v):.4f}"

def _fmt_acc(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{float(v)*100:.1f}%"


# ---------------------------------------------------------------------------
# Table 1 -- Strategy x task summary
# ---------------------------------------------------------------------------

def table_strategy_summary(df: pd.DataFrame) -> str:
    strategies = ["Ising ZZ", "Heisenberg XX+YY+ZZ", "IQP Encoding"]
    tasks = [
        ("NARMA-5",   "NMSE",     _fmt_nmse, "lower"),
        ("Financial", "NMSE",     _fmt_nmse, "lower"),
        ("MNIST-01",  "Accuracy", _fmt_acc,  "higher"),
    ]

    header_cols = [f"{t} ({better}(down)(up))" for t, _, _, better in tasks]
    header = "| Strategy | " + " | ".join(header_cols) + " |\n"
    sep    = "|---|" + "---|" * len(tasks) + "\n"
    rows   = []

    for strat in strategies:
        row_vals = []
        for task, m_name, fmt, direction in tasks:
            sub = df[(df["Strategy"] == strat) & (df["Task"] == task)]
            if sub.empty:
                row_vals.append("--")
                continue
            if direction == "lower":
                best = sub["primary_metric"].min()
            else:
                best = sub["primary_metric"].max()
            row_vals.append(fmt(best))
        rows.append(f"| **{strat}** | " + " | ".join(row_vals) + " |")

    table = (
        "## Table 1 -- Strategy Comparison (best n_qubits per task)\n\n"
        + header + sep + "\n".join(rows) + "\n\n"
        + "_NMSE = Normalized Mean Squared Error ((down) better). "
        "Accuracy = binary 0-vs-1 MNIST ((up) better). "
        "All QRC: n_layers=2, random connectivity, seed=42._\n"
    )
    return _write("table_strategy_summary.md", table)


# ---------------------------------------------------------------------------
# Table 2 -- QRC vs classical baselines (financial track)
# ---------------------------------------------------------------------------

def table_baselines_financial(df: pd.DataFrame) -> str:
    fin = df[df["Task"] == "Financial"].copy()

    # Best QRC per strategy (lowest NMSE)
    qrc_strategies = ["Ising ZZ", "Heisenberg XX+YY+ZZ", "IQP Encoding"]
    classical = ["Persistence", "AR(5)", "Ridge", "ESN-100"]

    models = []
    for strat in classical:
        sub = fin[fin["Strategy"] == strat]
        if not sub.empty:
            r = sub.iloc[0]
            models.append({
                "Model": strat,
                "Type": "Classical",
                "n_qubits": "--",
                "NMSE": r.get("NMSE", np.nan),
                "RMSE": r.get("RMSE", np.nan),
                "R^2":   r.get("R2",   np.nan),
            })

    for strat in qrc_strategies:
        sub = fin[fin["Strategy"] == strat]
        if sub.empty:
            continue
        best_row = sub.loc[sub["NMSE"].astype(float).idxmin()]
        models.append({
            "Model": f"QRC -- {strat}",
            "Type": "Quantum",
            "n_qubits": int(best_row["n_qubits"]) if best_row["n_qubits"] != "-" else "--",
            "NMSE": best_row.get("NMSE", np.nan),
            "RMSE": best_row.get("RMSE", np.nan),
            "R^2":   best_row.get("R2",   np.nan),
        })

    header = "| Model | Type | n_qubits | NMSE (down) | RMSE (down) | R^2 (up) |\n"
    sep    = "|---|---|---|---|---|---|\n"
    rows_md = []
    for m in models:
        nmse = _fmt_nmse(m["NMSE"])
        rmse = _fmt_nmse(m["RMSE"])
        r2   = _fmt_nmse(m["R^2"])
        nq   = str(m["n_qubits"])
        rows_md.append(
            f"| {m['Model']} | {m['Type']} | {nq} | {nmse} | {rmse} | {r2} |"
        )

    table = (
        "## Table 2 -- Financial Volatility (SPY): QRC vs. Classical Baselines\n\n"
        + header + sep + "\n".join(rows_md) + "\n\n"
        + "_SPY daily log-returns (2010-2024), 21-day realized volatility target. "
        "NMSE = MSE / Var(y). Chronological train/test split (80/20). "
        "QRC: n_layers=2, random connectivity._\n"
    )
    return _write("table_baselines_financial.md", table)


# ---------------------------------------------------------------------------
# Table 3 -- Qubit scaling (best strategy, all tasks)
# ---------------------------------------------------------------------------

def table_qubit_scaling(df: pd.DataFrame) -> str:
    tasks = [
        ("NARMA-5",   "NMSE",     _fmt_nmse),
        ("Financial", "NMSE",     _fmt_nmse),
        ("MNIST-01",  "Accuracy", _fmt_acc),
    ]

    # Pick overall best strategy per task (lowest NMSE / highest acc)
    qrc_df = df[df["n_qubits"] != "-"].copy()
    qrc_df["n_qubits"] = qrc_df["n_qubits"].astype(int)

    header = "| n_qubits | " + " | ".join(t for t, _, _ in tasks) + " |\n"
    sep    = "|---|" + "---|" * len(tasks) + "\n"
    rows_md = []

    for n_q in sorted(qrc_df["n_qubits"].unique()):
        row_vals = []
        for task, m_name, fmt in tasks:
            sub = qrc_df[(qrc_df["Task"] == task) & (qrc_df["n_qubits"] == n_q)]
            if sub.empty:
                row_vals.append("--")
                continue
            if m_name == "NMSE":
                val = sub["primary_metric"].min()
            else:
                val = sub["primary_metric"].max()
            row_vals.append(fmt(val))
        rows_md.append(f"| **{n_q}** | " + " | ".join(row_vals) + " |")

    table = (
        "## Table 3 -- Qubit Scaling (best strategy x task)\n\n"
        + header + sep + "\n".join(rows_md) + "\n\n"
        + "_Best metric across all 3 strategies for each qubit count. "
        "n_layers=2, random connectivity, noiseless simulation._\n"
    )
    return _write("table_qubit_scaling.md", table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run() -> None:
    print("=" * 65)
    print("  GENERATING DOCUMENT TABLES")
    print("=" * 65)

    csv_path = os.path.join(DOC_DIR, "strategy_comparison.csv")
    if not os.path.exists(csv_path):
        print(
            f"  ERROR: {csv_path} not found.\n"
            "  Run 'python doc_workflow/strategy_tournament.py' first."
        )
        return

    df = pd.read_csv(csv_path)

    print("\n[1/3] Strategy summary table ...")
    table_strategy_summary(df)

    print("\n[2/3] Baselines comparison (financial) ...")
    table_baselines_financial(df)

    print("\n[3/3] Qubit scaling table ...")
    table_qubit_scaling(df)

    print(f"\n  All tables -> {DOC_DIR}/")
    print("\n  -- Preview of table_strategy_summary.md --")
    with open(os.path.join(DOC_DIR, "table_strategy_summary.md")) as f:
        print(f.read())


if __name__ == "__main__":
    run()
