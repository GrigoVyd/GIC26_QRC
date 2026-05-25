# Phase 2 Submission — GIC 2026 (qBraid / MITRE / JonesTrading)

Submission deadline: **May 31, 2026, 23:59 EST**.
Track: **A — Financial Volatility Prediction**.

## Files in this directory

| File | Purpose |
|---|---|
| [`paper_draft.md`](paper_draft.md) | Body of the 3-page paper (markdown source). Convert to PDF via pandoc + prepend the official cover page before submitting. |
| [`gap_analysis.md`](gap_analysis.md) | What's missing for a competitive submission. Day-by-day execution plan (T-6 to T-0). |
| [`demo_plan.md`](demo_plan.md) | Reproduction commands, demo notebook outline, qBraid Lab compatibility notes. |

## The submission story in one paragraph

We design a transverse-field Ising QRC for next-day SPY realized volatility,
encode inputs as longitudinal biases, evolve via Trotterized unitary
dynamics, and read out Z and ZZ correlators into a hybrid Ridge head
trained on the residual log-vol target. At 10 qubits with all-to-all
coupling and an `input_scale × evolution_time` configuration chosen on the
edge-of-chaos prediction of Martínez-Peña et al. (2021), the architecture
ties Persistence (RMSE 0.00943 vs 0.00941) while uniformly beating every
classical reservoir, linear, and autoregressive baseline. The native
implementation target is QuEra Aquila (Rydberg) via qBraid's Bloqade SDK;
D-Wave Advantage and IBM Heron are documented alternatives.

## Open items before submission (from gap_analysis.md)

**Must-have:**
1. GARCH(1,1) baseline (rubric)
2. LSTM baseline (rubric)
3. QLIKE metric (Patton 2011)
4. Mincer–Zarnowitz regression
5. Architecture diagram
6. Cover page (download from aqora portal)
7. PDF generation pipeline

**Should-have:**
8. Multi-seed ensemble for the headline number
9. Regime-conditional (VIX-bucketed) RMSE table

**Nice-to-have:**
10. End-to-end demo notebook
11. qBraid Lab end-to-end dry run

## How to convert paper_draft.md → PDF

```bash
# Option A: pandoc directly
pandoc paper_draft.md -o body.docx \
    --reference-doc=template_times_11pt.docx

# Option B: pandoc → LaTeX → PDF (if you prefer LaTeX styling)
pandoc paper_draft.md -o body.tex \
    -V mainfont="Times New Roman" -V fontsize=11pt -V linestretch=1.0
pdflatex body.tex

# Then prepend the official GIC_2026 Cover Page.docx (Word: Insert → Object)
# Final filename: TeamName__Phase2_V1.pdf
```

The 3-page limit is on the **body only** — cover page and references are
excluded. Current paper_draft.md body is sized to fit; references are a
separate page.

## Reviewing the paper

Key claims to verify in the draft:
- §3 results table — taken from `results/financial_results_dwave_sweep.csv`
- §3 heatmap — `results/dwave_sweep_heatmap.png`
- §5 platform budget — back-of-envelope, mention the doc references for
  detailed derivation (`docs/phase3_hardware_plan.md`,
  `docs/dwave_qrc_exploration.md`)
- §6 disclosure paragraph — required by rules
