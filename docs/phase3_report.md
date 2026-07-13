# GIC 2026 — Phase 3 Report
## Quantum Reservoir Computing for SPY Volatility: from GPU Ising machines to QPUs

**Track 1 — Dynamic Systems Forecasting (financial volatility).**
Task: next-day SPY realized volatility (21-day annualised), 2010–2024, chronological
80/20 split, 746-day out-of-sample test. All models share the same features, split,
Ridge-readout selection, and target-inversion pipeline (apples-to-apples).

---

## 1. Executive summary

We evaluate quantum reservoir computing (QRC) for volatility forecasting across a
**platform timeline — classical CPU → GPU Ising machine → neutral-atom QPU →
quantum-annealer QPU** — against strong classical benchmarks (GARCH(1,1), LSTM,
ESN, AR, Persistence).

Two honest headline findings:

1. **On the raw task, QRC beats LSTM/ESN/AR/Persistence-class baselines but not
   GARCH.** GARCH has a *structural* advantage on this target (20 of 21 terms in
   the realized-vol window are known at forecast time).
2. **The way to beat GARCH is the hybrid: predict the GARCH *residual* with the
   reservoir.** And the GARCH-beating edge is **specific to quantum-annealing
   dynamics** (transverse-field Ising): the quantum annealer reservoir beats GARCH
   on every metric (RMSE 0.00784 vs 0.00795), while **classical** signed-coupling
   Ising sampling (GPU/SA, 0.00814) and **neutral-atom** Rydberg reservoirs
   (QuEra/Pasqal, 0.00806) reproduce the classical-competitive result but **not**
   the GARCH-beating edge. The transverse field — not signed couplings alone — is
   the active ingredient.

This gives a clean, falsifiable Phase 3 story: **the simulated advantage over the
best classical baseline emerges only in the transverse-field quantum-annealer
tier, and is absent at the classical/GPU and neutral-atom tiers.** Since D-Wave is
currently treated as inaccessible, the near-term submission should frame this as
a physics diagnosis plus an executable platform timeline: Toshiba/Fixstars for
the classical GPU-Ising tier, QuEra/qBraid for real-QPU execution, and D-Wave as a
future validation path rather than the immediate proof.

---

## 2. Classical benchmarks (raw task, 746-day test)

| Model | RMSE | R² | QLIKE |
|---|---:|---:|---:|
| LSTM (1L, 32h) | 0.01066 | 0.9743 | 0.00995 |
| Ridge | 0.00976 | 0.9784 | 0.00894 |
| ESN (200 nodes) | 0.00970 | 0.9787 | 0.00878 |
| AR(5) | 0.00954 | 0.9794 | 0.00918 |
| Persistence | 0.00941 | 0.9800 | 0.00887 |
| **GARCH(1,1)** | **0.00795** | **0.9857** | **0.00686** |

GARCH is the benchmark to beat. LSTM is the weakest — QRC beats it comfortably.

## 3. Quantum reservoir on the raw task

| Model | RMSE | R² | QLIKE |
|---|---:|---:|---:|
| QRC gate-based (5q, L2) | 0.00967 | 0.9788 | 0.00877 |
| QRC annealer (10q a2a, 5-seed) | 0.00945 | 0.9798 | 0.00868 |

QRC **beats LSTM, Ridge, ESN** and is **competitive with AR/Persistence**, but does
not beat GARCH on the raw target — motivating the hybrid.

## 4. The hybrid: predict the GARCH residual (746-day test)

Target `y = log RV − log RV_GARCH`; "predict 0" == GARCH exactly. The reservoir
only has to find structure GARCH misses; GARCH proxy is also a readout feature.
The **Ridge-on-residual** row is the ablation that isolates the reservoir's
nonlinear contribution.

| Platform tier | Model | RMSE | R² | QLIKE | vs GARCH |
|---|---|---:|---:|---:|:--:|
| Benchmark | GARCH(1,1) | 0.00795 | 0.9857 | 0.00686 | — |
| Linear ablation | Ridge on residual | 0.00805 | 0.9853 | 0.00668 | ✗ |
| Neutral-atom QPU | Pasqal Fresnel (global) | 0.00806 | 0.9853 | 0.00680 | ✗ |
| GPU/classical Ising | Ising-SA (signed couplings) | 0.00814 | 0.9850 | 0.00671 | ✗ |
| Neutral-atom QPU | Per-site local detuning (QuEra-like) | 0.00823 | 0.9847 | 0.00684 | ✗ |
| **Quantum-annealer QPU** | **QRC annealer (transverse-field, 3-seed)** | **0.00784** | **0.9861** | **0.00644** | **✓** |

**Only the quantum-annealing reservoir beats GARCH** — on RMSE (−1.4%), R², and
QLIKE (−6%). The Ridge ablation does not, so the win is isolable to the reservoir.
All neutral-atom and classical-Ising variants land at or above the linear ablation.

## 5. Why — the active ingredient is the transverse field

The winning reservoir is a **transverse-field Ising** model with **signed,
all-to-all couplings** `J_ij ∈ [−1,1]` and per-site input fields `h_i = α·x_i`:

```
H(s) = −A(s) Σ σ^x_i  +  B(s) [ Σ h_i σ^z_i  +  Σ_{i<j} J_ij σ^z_i σ^z_j ]
```

We tested removing each capability on real-hardware-faithful substrates:

- **Remove signed couplings** (neutral-atom Rydberg: couplings are positive,
  geometry-fixed `C6/r⁶`) → no win (Pasqal 0.00806, per-site 0.00823).
- **Remove the transverse field** (classical Ising sampling on GPU/SA: signed
  couplings but no quantum `σ^x` dynamics) → no win (0.00814).
- **Keep both** (quantum annealer) → **beats GARCH** (0.00784).

So the edge requires **quantum annealing dynamics**, whose native hardware is
**D-Wave Advantage** (programmable signed couplers + transverse-field anneal).
Neutral-atom QPUs (QuEra/Pasqal) and classical/GPU Ising machines are competitive
with the *other* baselines but cannot realise this specific advantage.

## 6. Platform timeline & hardware-execution status

| Tier | Platform | SDK / backend | Status |
|---|---|---|---|
| Classical CPU | GARCH/LSTM/ESN/AR | scikit-learn, arch | ✅ done |
| GPU Ising machine | **Fixstars Amplify AE** | `amplify` (AmplifyAEClient) | ✅ reservoir built; cloud run ready (AMPLIFY_TOKEN) |
| Ising machine | **Toshiba SBM (SQBM+)** | `amplify` (ToshibaSQBM2Client) | ✅ same API as D-Wave; run ready (TOSHIBA_TOKEN) |
| Ising machines | Fujitsu DA / Hitachi / NEC | `amplify` (one API) | ✅ reachable via the same backend |
| Local Ising (free) | SimulatedAnnealing | `dwave.samplers` | ✅ run (0.00814) |
| Neutral-atom QPU | **QuEra Aquila** | Braket AHS | ✅ reservoir + local AHS run; real-Aquila run ready (qBraid) |
| Neutral-atom QPU | **Pasqal Fresnel** | Pulser | ✅ reservoir + emulator run; HW pending device availability |
| Gate QPU | IonQ / IQM / Rigetti | qBraid runtime | ✅ pipeline live-verified on qBraid `qir-sv` |
| **Quantum annealer QPU** | **D-Wave Advantage** | `dwave.system` (Leap) | reservoir built; **currently inaccessible / future validation** |

**What has run on real cloud infrastructure:** the gate pipeline was verified
end-to-end on the live qBraid `qir-sv` API (real job ids). The updated 120-sample
qBraid simulator run gives feature fidelity 0.990 and keeps the finite-shot QRC
forecast aligned with the exact statevector forecast. Fixstars Amplify AE and
Toshiba SQBM+ now both execute end-to-end as cloud Ising checks. They are useful
QRC-system evidence, but they are deliberately framed as classical signed-Ising
ablations: they do not contain the transverse-field dynamics needed by the
GARCH-beating hybrid result. The GARCH-beating D-Wave-like annealer result
therefore remains a simulated target, not an executed hardware claim.

### Hybrid validation snapshot

`experiments/phase3_hybrid_validation_table.py` writes
`results/phase3_hybrid_validation.csv`, which separates the central hybrid claim
from the executable checks:

| Tier | System | Window | RMSE | Interpretation |
|---|---|---:|---:|---|
| Benchmark | GARCH(1,1) | 746 test | 0.00795 | strong baseline |
| Linear ablation | Ridge on GARCH residual | 746 test | 0.00805 | residual target alone is not enough |
| Hybrid advantage candidate | QRC annealer, 3-seed | 746 test | 0.00784 | only current GARCH-beating reservoir result |
| Executed cloud Ising check | Amplify AE | 50 train / 10 test | 0.02989 | runs, but trails residual ablation |
| Executed cloud Ising check | Toshiba SQBM+ | 50 train / 10 test | 0.02955 | runs, but trails residual ablation |
| qBraid simulator execution | Gate QRC on qir-sv | 120 test | 0.03898 | execution/fidelity proof; not the hybrid |

### Common-window GARCH-hybrid comparison (2026-07-11)

To remove the window mismatch above, the NISQ-ready reservoirs were rerun on the
same real-SPY window (400 train / 120 held-out test, GARCH-residual target). GARCH
scores 0.008318 RMSE and the linear residual ablation scores 0.008630. Pasqal
analog QRC (0.008426), gate QRC (0.008448), QuEra analog QRC (0.008475), and
signed-Ising sampling (0.008498) do not beat GARCH, but every reservoir improves
on the linear ablation by 1.52%--2.36%. This is the clean evidence that nonlinear
reservoir features add value even when the NISQ-ready physics does not reproduce
the full transverse-field advantage.

Artifacts: `results/hybrid_showcase_common_400_120.{csv,png}`. Final execution
commands and budget guards: `docs/hybrid_final_runbook.md`.

### Hardware-safe calibrated hybrid

A shallow-circuit validation sweep locked a Garnet-feasible 5q L3 sparse QRC
(seed 47, 24 CX) without ranking on test performance. Using the first 20 measured
rows only for correction-amplitude transfer and evaluating the remaining 100,
the simulator gives RMSE 0.008023 versus GARCH 0.008015 (0.11% gap). A moving-
block-bootstrap interval for the RMSE difference spans zero, so this candidate is
at GARCH level while retaining an active QRC correction (strength 0.47). Pasqal
and signed-Ising calibrated hybrids also reach within 0.05% of GARCH, but their
selected corrections are nearly zero and should be described as safe fallback,
not advantage. See `results/calibrated_hardware_candidates.csv`.

An IQM-native follow-up found a stronger candidate. Pure width increases on Rx
chain reservoirs (6q/8q/10q/12q) did not generalise. The reason is architectural:
Rx encoding immediately after preparing |+> is a global phase in the first
layer. Replacing it with Ry and using a native 3x3 neighbor grid produced a 9q,
one-layer, 24-CX reservoir. Four expanding validation folds plus worst-fold
selection locked seed 40 and correction strength 0.175 without test ranking. On
the 120-day test it scores RMSE 0.008315 versus GARCH 0.008321, a 0.078% point
improvement. The bootstrap interval spans zero, so this is GARCH-level evidence,
not yet a statistically significant advantage. See
`results/iqm_hardware_aware_width_summary.csv`.

The locked strength was then stress-tested rather than retuned on the test rows.
At 200 shots, ideal sampling gives a 0.095% point improvement over GARCH; an
IQM-like depolarizing proxy (1.3% two-qubit error) gives 0.187%, with feature
correlation about 0.972 to the exact reservoir. A free 20-row strength refit was
rejected because it overfits shot noise. See `results/iqm_q9_grid_noise_stress.csv`.

An expanded validation-only encoding sweep then found a higher-performance
native Rz candidate (9q, L1, grid, seed 44, order-2 observables). All four
validation folds improve on GARCH (worst +0.709%, mean +1.819%); the untouched
120-day test improves by 0.680% (RMSE 0.0082645 versus 0.0083211). Training the
classical readout on free IQM-like 200-shot simulator features retains a +0.508%
gain under the noise proxy. The exact-trained readout does not survive sampled
features, so noise-aware hybrid training is mandatory for this candidate. See
`results/locked_gate_hybrid_evaluation_iqm_q9_grid_rz_order2.csv` and
`results/iqm_q9_grid_noise_stress_rz_order2_noiseaware.csv`.

## 7. Reproducibility

| Result | Command |
|---|---|
| Classical + raw-task leaderboard | `python experiments/phase2_final.py` |
| GARCH-hybrid (annealer) — beats GARCH | `python experiments/phase2_garch_hybrid.py --n-seeds 3` |
| GARCH-hybrid on neutral atoms / Ising machine | `python experiments/neutral_atom_garch_hybrid.py --reservoir {quera,pasqal,pasqal_local,ising_sa} --atoms 10 --max-test 0 --n-seeds 3` |
| QuEra Aquila (real HW) | `python experiments/quera_aquila_qrc.py --device aquila --allow-qpu` |
| Ising on Fixstars Amplify (GPU) | `AMPLIFY_TOKEN=… python experiments/neutral_atom_garch_hybrid.py --reservoir ising_sa --ising-backend amplify --atoms 10 --max-test 0 --n-seeds 3` |
| Ising on Toshiba SBM | `TOSHIBA_TOKEN=… python experiments/ising_cloud_smoke.py --backend toshiba` first, then `… --ising-backend toshiba` |
| Ising on D-Wave (future) | Leap token, then `… --ising-backend dwave` (currently treated as inaccessible) |

Hybrid validation table: `python experiments/phase3_hybrid_validation_table.py`

Reservoirs: `src/qrc/{reservoir,annealer_reservoir,quera_reservoir,pasqal_reservoir,ising_reservoir}.py`.
Setup docs: `docs/{quera_aquila_setup,pasqal_fresnel_setup,qbraid_setup}.md`.

## 8. Honest limitations

- The GARCH-beating margin is small (−1.4% RMSE) and on a near-saturated residual;
  it is consistent across RMSE/R²/QLIKE and isolable to the reservoir, but modest.
- **No QPU has yet *executed* the GARCH-beating result** — it is demonstrated in
  noiseless quantum-annealer simulation. A real D-Wave run would be the direct
  confirmation, but D-Wave is currently treated as inaccessible. Therefore the
  near-term submission should avoid claiming an executed GARCH-beating QPU result.
- Amplify AE and Toshiba SQBM+ now execute, but they are classical Ising machines:
  they test signed couplings without transverse-field dynamics. They should be
  reported as QRC-system checks and ablations, not as replacements for a quantum
  annealer QPU run.
- GARCH(1,1) is the simplest GARCH; a full econometric panel (GJR/EGARCH/HAR) is
  future work. The residual trick should compose with any of them.
