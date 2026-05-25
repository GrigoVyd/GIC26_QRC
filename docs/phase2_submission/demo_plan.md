# Phase 2 Demo / Reproducibility Plan

The challenge requires "a fully reproducible workflow on the qBraid platform
that judges can re-execute without modification" (§3 Desired Outcomes, point 4).
Phase 2 itself doesn't ship the agentic reproducibility package — that's
Phase 3 — but the paper *does* need to point at runnable demonstrations.

## What's already in the repo (pre-staging for Phase 3)

| Asset | What it shows | Runtime |
|---|---|---|
| `experiments/financial_qrc.py` | v1 baseline: gate-based 5/10/15-qubit QRC sweep | ~15 min |
| `experiments/financial_qrc_v2.py` | v2: HAR-RV + log-vol target lifts QRC R² from 0.56 → 0.91 | ~15 min |
| `experiments/financial_qrc_v3.py` | v3: residual log-vol target + recurrent QRC; first config that beats ESN | ~25 min |
| `experiments/financial_dwave_first_approx.py` | Annealer-style Ising QRC sweep (8 configs) | ~12 min |
| `experiments/financial_dwave_sweep.py` | Edge-of-chaos parameter sweep (14 configs) | ~70 min |
| `docs/phase3_hardware_plan.md` | IBM Heron justification + 10-min/month budget | — |
| `docs/dwave_qrc_exploration.md` | D-Wave/annealer track with literature review | — |

## Demo notebook to ship with the paper

Goal: **one `notebooks/phase2_demo.ipynb` that produces the headline table
and the edge-of-chaos heatmap in under 5 minutes on a laptop.** Judges can
skim it to verify the claims.

Proposed cells:

1. **Imports + reproducibility seed.**
   ```python
   import sys; sys.path.insert(0, "..")
   import numpy as np
   np.random.seed(42)
   ```
2. **Load data via `load_financial_data_v2`** with HAR + log-HAR + residual
   target. Print shapes, feature names, target transform.
3. **Run all 5 classical baselines** (Persistence, AR, Ridge, ESN, GARCH —
   once GARCH is added) and report metrics.
4. **Run gate-based v3 best config** (`QuantumReservoir 5q L=2 random`).
5. **Run annealer-style best config** (`AnnealerReservoir 10q a2a t=2.0
   α=1.0`). This is the headline number.
6. **Render the combined-leaderboard table** and the edge-of-chaos heatmap
   (load pre-computed sweep CSV from `results/`, render heatmap inline).
7. **Conclude** with a markdown cell explaining what the result shows and
   pointing at the paper PDF.

Total cell count: ~10. Total wall-clock on a modern laptop:
- Cells 1–4: ~1 min
- Cell 5: ~3 min (10-qubit statevector × 750 test samples + 2,984 train)
- Cell 6: instant (reading cached CSV)

If we want to skip the 3-minute QRC step on first run, the notebook can
load `results/financial_results_dwave_sweep.csv` and re-derive the
headline row directly. Document both modes in the notebook.

## qBraid Lab compatibility notes

qBraid Lab pre-installs Qiskit, Cirq, PennyLane, Bloqade, Braket SDK. Our
code requires:

| Dependency | qBraid Lab status | Action if missing |
|---|---|---|
| `qiskit >= 1.1` | ✓ pre-installed | none |
| `qiskit-aer` | ✓ pre-installed | none |
| `numpy`, `scipy`, `pandas`, `scikit-learn` | ✓ pre-installed | none |
| `matplotlib`, `seaborn` | ✓ pre-installed | none |
| `yfinance` | likely missing | `pip install yfinance` in notebook cell |
| `arch` (for GARCH baseline) | likely missing | `pip install arch` |
| `torch` (for LSTM baseline) | ✓ pre-installed in ML kernel | select ML kernel or `pip install torch` |
| `tqdm` | ✓ pre-installed | none |

Action items for Phase 3 qBraid deployment:
- Pin all versions in `requirements.txt` to what qBraid Lab ships.
- Replace yfinance dependency with cached CSV in `data/` so the notebook
  is offline-reproducible (Phase 2 submission file should include a SPY
  daily-close CSV from 2010-01-01 to 2024-12-31, ~250 KB).
- Add a `qbraid_environment.yml` if/when qBraid Skills format is finalised.

## Reproduction commands (cheat sheet for paper §6)

```bash
git clone https://github.com/GrigoVyd/GIC26_QRC.git
cd GIC26_QRC
pip install -r requirements.txt

# Headline result from the paper (3 min on a laptop):
python experiments/financial_dwave_first_approx.py --fast

# Full parameter sweep (70 min on a laptop):
python experiments/financial_dwave_sweep.py

# Phase-1 gate-based v3 (25 min):
python experiments/financial_qrc_v3.py

# View the heatmap:
# open results/dwave_sweep_heatmap.png
```

## Phase 3 deliverables (mentioned in paper but built later)

1. **Executable qBraid workflow** — the demo notebook above, packaged as a
   qBraid Skill (`.json` manifest + cells).
2. **Agentic reproducibility package** — qBraid format specifics to be
   announced to finalists; pre-stage by ensuring every result has an
   end-to-end script that produces it.
3. **5-page Phase 3 paper** — extension of this Phase 2 draft with hardware
   results, multi-seed error bars, regime-conditional tables, QuEra-vs-sim
   gap analysis.
