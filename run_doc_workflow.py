"""
GIC 2026 QRC -- Phase 2 Document Workflow Master Runner

Produces all raw material needed to write the Phase 2 submission:
  - 8 publication-ready PNG figures  -> results/doc/*.png
  - 3 Markdown tables                -> results/doc/*.md
  - Strategy comparison data         -> results/doc/strategy_comparison.csv

WARNING -- AI DISQUALIFICATION
The competition rules state: "use of AI may be disqualified and voided."
This script produces DATA and FIGURES only. All document text (analysis,
conclusions, narrative) must be written by the team without AI assistance.

Usage
-----
  python run_doc_workflow.py              # full run (~30-40 min)
  python run_doc_workflow.py --fast       # small datasets (~8-12 min)
  python run_doc_workflow.py --skip-noise # skip shot-based noise sweep
  python run_doc_workflow.py --tables-only# re-generate tables from existing CSV
  python run_doc_workflow.py --figs-only  # re-generate figures from existing CSV

Steps
-----
  Step 1  Strategy tournament   -- 3 strategies x 3 tasks x 2 qubit counts
  Step 2  Generate figures      -- circuit diagrams, comparison plots, predictions
  Step 3  Generate tables       -- Markdown tables (paste directly into document)
  Step 4  Asset manifest        -- print which file maps to which document section
"""

from __future__ import annotations

import argparse
import os
import sys
import time

DOC_DIR = os.path.join(os.path.dirname(__file__), "results", "doc")
os.makedirs(DOC_DIR, exist_ok=True)

SEP  = "=" * 65
STEP = "#" * 65


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(step: int, title: str) -> None:
    print(f"\n{STEP}")
    print(f"  Step {step}: {title}")
    print(f"{STEP}\n")


def _run(label: str, fn, *args, **kwargs):
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        print(f"\n  [OK]  {label} -- {time.time()-t0:.0f}s")
        return result
    except Exception as exc:
        import traceback
        print(f"\n  [FAIL]  {label}: {exc}")
        traceback.print_exc()
        return None


def _manifest() -> None:
    """Print a map from each output file to the Phase 2 document section."""
    print(f"\n{SEP}")
    print("  ASSET MANIFEST -- which file goes where in the document")
    print(f"{SEP}")

    sections = [
        ("Cover page",
         ["(Use official GIC_2026 Cover Page.docx template -- not generated here)"]),
        ("Sec.1  Focus area & rationale",
         ["results/doc/pipeline_diagram.png",
          "results/doc/financial_predictions_doc.png"]),
        ("Sec.2  Technical approach",
         ["results/doc/circuit_ising.png",
          "results/doc/circuit_heisenberg.png",
          "results/doc/circuit_iqp.png",
          "results/doc/table_strategy_summary.md"]),
        ("Sec.3  Data modelling",
         ["results/doc/pipeline_diagram.png",
          "results/doc/table_baselines_financial.md"]),
        ("Sec.4  Quantum advantage & comparison",
         ["results/doc/strategy_comparison.png",
          "results/doc/scaling_lines.png",
          "results/doc/table_qubit_scaling.md"]),
        ("Sec.5  Platform justification & Phase 3 scaling",
         ["results/doc/noise_robustness.png",
          "results/doc/table_qubit_scaling.md"]),
    ]

    for section, files in sections:
        print(f"\n  {section}")
        for f in files:
            exists = "[OK]" if os.path.exists(
                os.path.join(os.path.dirname(__file__), f)) else "[missing]"
            print(f"    {exists}  {f}")

    print(f"\n  All output files are in:  {DOC_DIR}/")
    print("\n  WARNING: Write document text yourself --"
          " AI authorship may disqualify the submission.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GIC26 Phase 2 document workflow"
    )
    parser.add_argument("--fast",        action="store_true",
                        help="Smaller datasets for a quick validation pass")
    parser.add_argument("--skip-noise",  action="store_true",
                        help="Skip shot-based noise robustness sweep")
    parser.add_argument("--tables-only", action="store_true",
                        help="Re-generate tables from existing CSV (no QRC runs)")
    parser.add_argument("--figs-only",   action="store_true",
                        help="Re-generate figures from existing CSV (no QRC runs)")
    args = parser.parse_args()

    print(f"\n{SEP}")
    print("  GIC 2026 QRC -- Phase 2 Document Workflow")
    print("  Deadline: 31 May 2026  |  Track: Financial Volatility")
    print(f"{SEP}")

    from doc_workflow.strategy_tournament import run as run_tournament
    from doc_workflow.generate_figures import run as run_figures
    from doc_workflow.generate_tables import run as run_tables

    csv_exists = os.path.exists(os.path.join(DOC_DIR, "strategy_comparison.csv"))

    # Step 1 -- Tournament
    if not args.tables_only and not args.figs_only:
        _header(1, "Strategy Tournament  (3 strategies x 3 tasks)")
        _run("Strategy tournament", run_tournament)
    else:
        if not csv_exists:
            print("\n  ERROR: strategy_comparison.csv missing."
                  " Run without --tables-only / --figs-only first.")
            sys.exit(1)
        print("\n  Step 1 skipped (using existing strategy_comparison.csv)")

    # Step 2 -- Figures
    if not args.tables_only:
        _header(2, "Document Figures  (PNGs)")
        _run("Generate figures", run_figures, skip_noise=args.skip_noise)
    else:
        print("\n  Step 2 skipped (--tables-only)")

    # Step 3 -- Tables
    if not args.figs_only:
        _header(3, "Document Tables  (3 Markdown files)")
        _run("Generate tables", run_tables)
    else:
        print("\n  Step 3 skipped (--figs-only)")

    # Step 4 -- Manifest
    _header(4, "Asset Manifest")
    _manifest()

    print("\nDone.\n")


if __name__ == "__main__":
    main()
