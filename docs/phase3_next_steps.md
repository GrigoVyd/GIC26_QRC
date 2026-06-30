# Phase 3 — Handoff & Next Steps (for any agent or contributor)

This document is a self-contained handoff. Read it top to bottom before continuing
Phase 3. It records **what is built, what is proven, the dead ends (so you don't
repeat them), and the exact next steps with commands, acceptance criteria, and
cost**.

Companion docs: [`phase3_report.md`](phase3_report.md) (results + narrative),
[`quera_aquila_setup.md`](quera_aquila_setup.md), [`pasqal_fresnel_setup.md`](pasqal_fresnel_setup.md),
[`qbraid_setup.md`](qbraid_setup.md).

---

## 1. TL;DR of current status (2026-06-30)

**Goal:** reproduce the Phase 2 GARCH-hybrid headline (QRC beats GARCH on the
GARCH residual) on **real quantum hardware**, across multiple platforms, and
present a **classical → GPU Ising → neutral-atom QPU → quantum-annealer QPU**
timeline beating classical benchmarks (GARCH, LSTM).

**Proven in simulation (in-session reproduction):**
- QRC beats LSTM/ESN/AR on the raw task.
- **Only the quantum-annealer GARCH-hybrid beats GARCH** (RMSE 0.00784 vs 0.00795,
  R² 0.9861, QLIKE 0.00644); the Ridge ablation does **not** (0.00805) → win
  isolable to the reservoir.

**Key scientific finding (do not re-litigate without new evidence):** the
GARCH-beating edge needs **quantum-annealing dynamics (the transverse field)**, not
signed couplings alone. Evidence:
- Neutral-atom Rydberg (QuEra/Pasqal, positive geometry-fixed couplings): 0.00806 — no win.
- Classical signed-coupling Ising (SA / Fixstars Amplify / Toshiba SBM, no transverse
  field): 0.00814 — no win.
- Quantum annealer (transverse-field Ising, sim): 0.00784 — **wins.**
→ The advantage lives at the **D-Wave** tier. Classical/GPU Ising machines and
neutral-atom QPUs reproduce the *competitive* result but not the *advantage*.

**Shot noise is NOT the limiter** (10-atom Pasqal: noiseless 0.00678 == 600-shot
0.00677). Good for hardware: modest shot budgets suffice.

**No QPU has executed the GARCH-beating result yet** — it is noiseless sim. The
remaining work is real-hardware execution (Section 4).

---

## 2. Repo map (what each piece is)

Reservoirs (`src/qrc/`):
| File | Reservoir | Substrate | Notes |
|---|---|---|---|
| `reservoir.py` | gate-based QRC | gate circuit | + `transform_sequential` recurrent |
| `annealer_reservoir.py` | transverse-field Ising (Trotter, statevector) | **the Phase 2 winner** | signed J_ij, per-qubit h_i; noiseless sim of D-Wave |
| `quera_reservoir.py` | QuEra Aquila AHS | neutral-atom Rydberg | per-site local detuning; `transform_sequential`; noiseless via local sim |
| `pasqal_reservoir.py` | Pasqal Fresnel | neutral-atom Rydberg | global detuning (real Fresnel) or per-site DMM (`encoding="local"`, MockDevice); noiseless path |
| `ising_reservoir.py` | signed-coupling Ising machine | **D-Wave / Amplify / Toshiba / SA** | one model, backends: `sa`,`exact`,`dwave`,`amplify`,`toshiba`,`fujitsu`,`hitachi`,`nec`,`dwave_amplify` |
| `hardware_backend.py` | qBraid gate executor | IonQ/IQM/Rigetti via qBraid | barrier-stripping + bit-order calibration + job-id audit |

Experiments (`experiments/`):
| File | What |
|---|---|
| `phase2_final.py` | classical + raw-task leaderboard (GARCH/LSTM/ESN/AR/Persistence/QRC) |
| `phase2_garch_hybrid.py` | **the headline**: annealer QRC beats GARCH on the residual (3-seed) |
| `neutral_atom_garch_hybrid.py` | multi-platform GARCH-hybrid harness (`--reservoir quera/pasqal/pasqal_local/ising_sa`; `--ising-backend`; `--noiseless --recurrent --n-seeds --atoms --max-train --max-test --shots`) |
| `neutral_atom_compare.py` | QuEra vs Pasqal vs classical (raw task) |
| `quera_aquila_qrc.py` | QuEra Aquila: local AHS sim + real `--device aquila --allow-qpu` |
| `qbraid_hardware_qrc.py` / `qbraid_list_devices.py` | gate-QPU path on qBraid |
| `quera_tune.py` | reservoir config sweep |
| `phase3_timeline_figure.py` | the GPU→QPU summary figure |

Key results (`results/`): `phase2_garch_hybrid.csv` (annealer headline),
`neutral_atom_garch_hybrid_{isingSA,persite}.csv`, `quera_aquila_summary.csv`,
`neutral_atom_compare.csv`, `phase3_timeline.png`.

---

## 3. Environment & tokens

Installed (this machine / qBraid Lab): `amazon-braket-sdk`, `pulser`+`pulser-simulation`
(qutip), `dwave-ocean-sdk` (incl. `dwave.samplers` SA + `dwave.system` Leap),
`amplify` (Fixstars, 1.6.1), `qbraid` (gate path). `requirements.txt` documents all.

Tokens needed for the cloud/hardware runs (set as env vars):
- **D-Wave Leap**: `DWAVE_API_TOKEN` (from cloud.dwavesys.com/leap). Used by `dwave.system.DWaveSampler`.
- **Fixstars Amplify**: `AMPLIFY_TOKEN` (amplify.fixstars.com). Used by FixstarsClient and, by default fallback, the other Amplify clients.
- **Toshiba SBM**: `TOSHIBA_TOKEN` (+ optional `TOSHIBA_URL`) or falls back to `AMPLIFY_TOKEN` if Amplify brokers it.
- **qBraid** (gate path / QuEra via Braket): already saved to `~/.qbraid/qbraidrc`. (The key pasted in chat on 2026-06-27 should be **rotated**.)
- **QuEra Aquila**: runs via Braket through qBraid Lab credits (no separate token in Lab).

---

## 4. NEXT STEPS — prioritized, with commands & acceptance criteria

### STEP 1 (HEADLINE) — Execute the D-Wave Advantage run  ⭐ blocked on Leap token
This is the one that turns the gold "beats GARCH" point from simulation into a real
QPU result. Owner action: obtain `DWAVE_API_TOKEN`.

**Plumbing to add first** (in `src/qrc/ising_reservoir.py::_sample_dwave`, currently
a minimal version):
1. **Reuse one minor-embedding.** The coupling graph J is fixed (all-to-all on
   `--atoms` spins); only h changes per sample. Compute the embedding ONCE with
   `minorminer.find_embedding` onto `DWaveSampler().edgelist`, then use
   `dwave.system.FixedEmbeddingComposite(DWaveSampler(), embedding)` for every
   sample. (The current `EmbeddingComposite` re-embeds each call — too slow/costly.)
2. **Short anneal for reservoir dynamics.** Use `annealing_time≈1–20 µs` (short =
   non-equilibrium = the QRC regime), `num_reads≈100–256`.
3. **Hybrid budget control.** Train the readout on **local SA / annealer-sim**
   features; run only the **test window** on D-Wave (start `--max-test 100`, single
   seed). Persist raw samples + problem ids to `results/hardware/dwave/`.
4. **Cost check.** Per dwave_qrc_exploration.md: ~140 µs/sample → full 746-test ≈
   ~10 s QPU time with reused embedding. Within Leap; verify against your plan.

**Run command (after plumbing + token):**
```bash
export DWAVE_API_TOKEN=...
python experiments/neutral_atom_garch_hybrid.py \
    --reservoir ising_sa --ising-backend dwave --atoms 10 \
    --max-train 2000 --max-test 100 --shots 200 --n-seeds 1
# scale to --max-test 0 (full 746) + --n-seeds 3 once the small run looks right
```
**Acceptance:** D-Wave-residual RMSE < GARCH 0.00795 (target ≈ 0.00784, the sim
value) on the full test window; Ridge ablation stays at ~0.00805. Save job ids.
**If it does NOT beat GARCH on hardware:** that's a real finding — report the gap
vs the noiseless annealer sim as the decoherence/embedding cost (still publishable).

### STEP 2 — Toshiba SBM + Fixstars Amplify (classical-GPU tier contrast)
Cheap, accessible; gives the *same-reservoir* classical-vs-quantum contrast.
```bash
export AMPLIFY_TOKEN=...   # and/or TOSHIBA_TOKEN
python experiments/neutral_atom_garch_hybrid.py --reservoir ising_sa \
    --ising-backend toshiba --atoms 10 --max-test 0 --n-seeds 3   # or amplify / fujitsu
```
**Acceptance:** classical Ising machines reproduce ~the SA tier (≈0.00814, i.e.
competitive but NOT beating GARCH). The D-Wave-vs-Toshiba gap at the GARCH line IS
the quantum signal — the cleanest figure for the report.
**Note:** `_sample_amplify` returns the solver's solution set; if it returns too few
samples for stable ⟨ZZ⟩, raise multi-output (`parameters.outputs.num_outputs` /
Toshiba `multishot`/`maxout`) or loop `solve` with seeds.

### STEP 3 — QuEra Aquila real hardware (neutral-atom tier)
Validates the *competitive-on-raw-task* result on real neutral-atom HW (it will NOT
beat GARCH — that's expected and documented).
```bash
# check availability window first (notebooks/quera_aquila_demo.ipynb cell 4)
python experiments/quera_aquila_qrc.py --device aquila --max-test 40 --shots 100 --allow-qpu
```
**Acceptance:** Aquila features track the local-AHS sim (corr > 0.9); forecast ties
the best classical baselines on the window. Cost ≈ tasks×($0.30+shots×$0.01).

### STEP 4 — Finalize the report
After each hardware run:
1. Drop the executed RMSE into `docs/phase3_report.md` (Sections 4, 6).
2. Update `experiments/phase3_timeline_figure.py` (`TIER_RMSE` / `LANDSCAPE`) with
   the **executed** points and regenerate: `python experiments/phase3_timeline_figure.py`.
3. Mark the platform's status ✅-executed in the report Section 6 table.
4. Keep raw shot/sample data + job ids under `results/hardware/` as proof.

---

## 5. Suggested timeline

| When | Milestone | Depends on |
|---|---|---|
| Day 0 | Obtain D-Wave Leap token; add FixedEmbedding plumbing (Step 1.1–1.3) | token |
| Day 0–1 | Small D-Wave test-window run (`--max-test 100`) → sanity | plumbing |
| Day 1–2 | Full D-Wave run (746, 3-seed) → headline hardware number | small run OK |
| Day 1 | Toshiba + Amplify runs (parallel, cheap) | AMPLIFY/TOSHIBA token |
| Day 2 | QuEra Aquila run (during availability window) | qBraid credits |
| Day 3 | Update report + timeline figure with executed points; finalize submission | runs done |

---

## 6. Dead ends already explored (do NOT repeat)
- **Pasqal Fresnel global-only encoding** → does not beat GARCH (0.00806). Per-site
  DMM (MockDevice) → also no (0.00823). Neutral-atom can't realise signed couplings.
- **Classical signed-coupling Ising (SA, full window)** → 0.00814, no win. Signed
  couplings alone (no transverse field) are insufficient.
- **More shots** → not the limiter (noiseless == 600-shot).
- **Scaling atoms on QuEra in CPU sim** → AHS solver ~15 s/program at 8 atoms,
  infeasible ≥10. Use Pasqal/qutip for CPU scaling, GPU/CUDA-Q or real Aquila for QuEra.
- **Threads for local-sim parallelism** → slower than serial (GIL); use processes
  (`n_jobs`, ~2.7× on this box; better on Linux/fork).

## 7. One-command reproductions (sim, no tokens)
```bash
python experiments/phase2_final.py                         # classical + raw leaderboard
python experiments/phase2_garch_hybrid.py --n-seeds 3      # annealer beats GARCH (headline)
python experiments/neutral_atom_garch_hybrid.py --reservoir ising_sa --atoms 10 --max-test 0 --n-seeds 3
python experiments/phase3_timeline_figure.py               # the summary figure
```
