# Phase 2 Submission — Gap Analysis & Execution Plan

Current state vs. Phase 2 rubric, organised by what must ship by **May 31, 2026** (T-6 days from this writing) vs. what is Phase 3 material.

## Status against the 7 evaluation criteria

| # | Criterion | Status | Risk |
|---|---|---|---|
| 1 | **QRC Architecture Design** (Hamiltonian, encoding, readout, feedback, hybrid) | **Strong** — transverse-field Ising spec'd in detail in paper draft §2, with both annealer-native and gate-Trotterized variants in repo | Low |
| 2 | **Theoretical & Analytical Justification** | **Strong** — explicit edge-of-chaos justification, 14-config parameter sweep substantiates, ties Persistence on RMSE | Low |
| 3 | **Data Modeling Strategy** | **Partial** — baselines (Persistence/AR/Ridge/ESN) and metrics (RMSE/MAE/R²/NMSE) present; **GARCH and LSTM baselines missing; QLIKE and Mincer–Zarnowitz metrics missing** | **HIGH** |
| 4 | **Track Selection & Problem Framing** | Strong — Track A with regime-shift framing | Low |
| 5 | **Platform Justification & Resources** | Strong — three platforms (QuEra primary / D-Wave alt / IBM backup), per-platform budget table, hybrid sim-train/HW-test strategy | Low |
| 6 | **Phase 3 Execution Plan** | Strong — 10-week milestone plan with fallbacks | Low |
| 7 | **Clarity of Communication** | Pending — needs architecture diagram, polished prose, page count check | Medium |

## Must-ship before May 31 (in priority order)

### MUST-HAVE (judging-rubric blockers)

1. **GARCH(1,1) baseline** *(~3 hours)*
   - Add `arch` library to requirements
   - `src/baselines/classical.py`: new `GARCHBaseline` class wrapping `arch.arch_model`
   - Re-run v3 + dwave_sweep experiments with GARCH row in results
   - **Why critical:** explicitly named in the challenge description as a Track A baseline. Absence is a credibility hit on Criterion #3.

2. **LSTM baseline** *(~4 hours)*
   - Add PyTorch dependency
   - `src/baselines/lstm.py`: minimal 1-layer LSTM, ~32 hidden units, sequence input, MSE loss on log-vol residual
   - Don't try to make it competitive — a vanilla LSTM as named-baseline-checkbox is sufficient
   - **Why critical:** same as GARCH. Named in the rubric.

3. **QLIKE metric** *(~30 min)*
   - `src/baselines/classical.py::regression_metrics` extension: `QLIKE = ŷ/y − log(ŷ/y) − 1` averaged over test
   - Add to all results CSVs
   - **Why critical:** Patton (2011) is the volatility-forecasting-canonical metric; rubric weight high.

4. **Mincer–Zarnowitz regression** *(~30 min)*
   - Regress y_true on y_pred, report (intercept, slope, joint test p-value)
   - One-line addition to each experiment's output table
   - **Why critical:** named in rubric Criterion #3.

5. **Architecture diagram** *(~2 hours)*
   - One figure for the paper: input vector → bias encoding `h_i` → Trotterized Ising evolution → Z, ZZ readout → linear head → vol prediction
   - matplotlib + boxes, or LaTeX TikZ
   - **Why critical:** "A clear diagram of the architecture is strongly encouraged" — §2 of the challenge description.

6. **Cover page** *(15 min)*
   - Download `GIC_2026 Cover Page.docx` from the aqora portal
   - Fill in team name, track selection (Track A), declared platforms (QuEra Aquila primary, D-Wave alt, IBM backup), team members
   - **Why critical:** "Submissions that do not comply with these requirements may be disqualified."

7. **PDF generation pipeline** *(~1 hour)*
   - pandoc convert of `paper_draft.md` → `.docx` → check 3-page count at 11-pt Times New Roman, single spacing
   - Merge with cover page (Word, prepend pages)
   - Verify file name: `TeamName__Phase2_V1.pdf`

### SHOULD-HAVE (de-risk + raw quality)

8. **Multi-seed ensemble for headline number** *(~2 hours QPU sim time + 30 min stats)*
   - Re-run best config (10q a2a t=2.0 α=1.0) at 5 random seeds
   - Report mean ± std RMSE in the paper instead of single-seed point
   - **Why:** without error bars, "ties Persistence" is unfalsifiable — easy attack surface for judges

9. **Regime-bucketed RMSE table** *(~2 hours)*
   - Compute test-set VIX percentile per day (FRED has VIX history)
   - Bucket into terciles (calm / normal / turbulent)
   - Report RMSE per bucket for best QRC vs Persistence vs ESN
   - **Why:** this is the regime-awareness story the JonesTrading stakeholder framing leans on; a one-table differentiator

### NICE-TO-HAVE (only if time permits)

10. **Reproduction notebook** *(~3 hours)*
    - Single `notebooks/phase2_demo.ipynb` that loads data → runs best config → produces headline table & heatmap → all under 5 minutes
    - **Why:** Phase 3 requires qBraid-executable reproducibility; pre-staging is a quality signal but not Phase 2 mandatory

11. **qBraid Lab compatibility check** *(~2 hours)*
    - Spin up qBraid Lab free tier, clone the repo, run the demo notebook end-to-end
    - Capture any path/dependency issues for Phase 3 cleanup
    - **Why:** "Solutions must be reproducible. Code, data references, and run instructions must be sufficient for a third-party reviewer to verify."

## Explicitly deferred to Phase 3 (the paper says so)

* QuEra Aquila hardware execution
* Full noise study (Mitiq/ZNE on Heron)
* Scaling beyond 12 qubits (statevector limit) via tensor-network sim
* Full agentic reproducibility package
* Intraday-horizon forecast variant
* Mid-circuit-measurement recurrent QRC on Quantinuum H2

## Execution timeline (6 working days, T-6 to T-0)

| Day | Date | Deliverables |
|---|---|---|
| T-6 | May 25 | Paper draft + gap analysis (this doc) + branch `phase2-submission` — **DONE** |
| T-5 | May 26 | QLIKE + MZ metrics (items 3, 4); GARCH baseline (item 1) |
| T-4 | May 27 | LSTM baseline (item 2); re-run sweeps with new baselines/metrics |
| T-3 | May 28 | Multi-seed ensemble (item 8); regime-bucketed table (item 9); architecture diagram (item 5) |
| T-2 | May 29 | Update paper draft with new numbers; download cover page (item 6); generate PDF (item 7); proofread |
| T-1 | May 30 | Reserve day: PDF page-count adjustments, last-mile fixes |
| T-0 | May 31 | Submit via aqora before 23:59 EST |

## What I can do right now without further input

Items 1–4 (GARCH, LSTM, QLIKE, MZ) are all pure code additions on the
existing experiment scripts — no platform access needed. Item 5
(architecture diagram) needs only matplotlib. Item 6 needs the cover-page
file from aqora.

**Recommendation:** start by adding QLIKE and Mincer–Zarnowitz to the
metrics module (cheapest, highest rubric impact), then GARCH baseline,
then LSTM, then the diagram. If the user can paste the cover-page DOCX
contents or upload the file, we can pre-fill it.
