# Final hybrid-QRC simulation results and execution runbook

## Readiness status (2026-07-11)

- Garnet cost/circuit dry run: passed (`120 x 200`, 21.24 Spark).
- qBraid gate submission/scoring smoke test: passed on `qir-sv`; job IDs,
  GARCH-hybrid scores, and reusable NPZ features saved under the result tag.
- Aquila program/cost dry run: passed (`36 x 100`, 4,680 qBraid credits).
- Aquila local end-to-end GARCH-hybrid smoke test: passed; CSV, NPZ, and plot saved.
- Amplify cloud footprint dry run: passed (`10` spins, `45` couplers, `120`
  planned test-only solves). `AMPLIFY_TOKEN` is not set in the current shell, so
  no GPU solves were submitted.
- No real QPU jobs were submitted during preparation.

## Common-window simulation result

All rows below use the same causal SPY dataset (`2010-01-01` to `2024-12-31`),
the same most-recent 400 training rows and 120 held-out test rows, and the same
GARCH-residual target. Predicting zero therefore reproduces GARCH; the reservoir
is responsible only for the nonlinear correction.

| Model | RMSE | Gap vs GARCH | Gain vs linear residual ablation |
|---|---:|---:|---:|
| GARCH(1,1) | 0.008318 | 0.00% | 3.61% |
| Pasqal analog QRC hybrid | 0.008426 | +1.30% | **2.36%** |
| Gate QRC hybrid (5q L2 sparse) | 0.008448 | +1.56% | **2.10%** |
| QuEra analog QRC hybrid | 0.008475 | +1.88% | **1.79%** |
| Signed-Ising hybrid (local SA) | 0.008498 | +2.16% | **1.52%** |
| Ridge residual ablation | 0.008630 | +3.74% | 0.00% |

Interpretation: none of these NISQ-ready reservoirs beats GARCH on this window,
but all four improve on the identical linear residual model. That isolates a
real nonlinear reservoir contribution. The separate full-window transverse-field
annealer simulation remains the only GARCH-beating result (RMSE 0.007844 versus
0.007948, a 1.31% improvement).

## Final hardware/cloud allocation

### Additional backend: direct IQM Resonance / Emerald

Emerald is an additional gate-QPU backend and does not replace the qBraid,
QuEra, or Amplify evidence tiers. `IQM_TOKEN` must be supplied through the
process environment and must never be written into the repository. The direct
backend in `src/qrc/iqm_resonance_backend.py` reads live calibration, selects a
healthy contiguous 3x3 patch, binds logical qubits explicitly, and rejects any
compiled circuit containing a SWAP.

The first hardware preflight used one 100-shot smoke circuit followed by one
batch of six real-SPY circuits at 200 shots. Both jobs completed. The compiled
QRC used 24 native CZ gates, depth 34, and zero SWAPs. On the six-circuit batch,
raw hardware/statevector feature correlation was 0.748 (MAE 0.0746); asymmetric
readout correction changed it only to 0.750 because average readout error was
about 1.1%. This is below the predefined 0.85 go threshold, so no 120-circuit
Emerald forecast was submitted. Job IDs and metrics are recorded in
`results/iqm_emerald_preflight.json`.

Before a full Emerald run, calibrate the simulator noise/readout model against
training-only diagnostic circuits and validate either a higher shot count or a
more strongly regularized noise-aware readout. Re-select physical qubits from
live calibration immediately before every paid batch; the best patch can drift.

This validation was subsequently completed. A hardware-matched effective-noise
model plus four expanding training-only validation folds selected Ridge alpha
300 and correction strength 0.105. At 500 shots its proxy test gain was +0.410%.
Two independent six-circuit Emerald batches reached 0.905 repeat correlation;
their combined 1000-shot feature estimate reached 0.873 correlation to exact.

The resulting full Emerald job (`019f5691-063b-7a11-8bbc-db5057b79259`) ran 120
circuits at 500 shots on `QB17-19/QB25-27/QB33-35`. Every circuit compiled to
depth 32 with 24 native CZ gates and zero SWAPs. The best pre-specified hybrid
processing path (asymmetric readout correction plus feature-only affine transfer)
scored RMSE `0.00831368` versus GARCH `0.00832112`, a **0.0895% point gain**.
Raw hardware features alone also remained positive at +0.0445%. Full-window
hardware/statevector feature correlation was 0.770 after readout correction.

The 95% moving-block-bootstrap interval for model-minus-GARCH RMSE is
`[-0.0000608, +0.0000262]`; it spans zero. Report this as an executed,
GARCH-competitive QRC with a small positive point result, not as statistically
established quantum advantage. Auditable outputs are in
`results/iqm_emerald_q9_grid_rz_500_*`.

Resonance billing evidence is preserved in
`docs/assets/iqm_resonance_credits_2026-07-12.png`. Seven completed Emerald jobs
used 56.25 credits in total. Each full 120-circuit/500-shot job cost 20.25
credits; the 12q six-circuit diagnostic cost 3.75 credits. This stays below the
100-credit ceiling and leaves 43.75 credits unused.

#### Native-grid width scaling: 9q versus 12q

A gated 12q scaling experiment was completed after the 9q result. The 3x4
layout `QB17-20/QB25-28/QB33-36` compiled at depth 38 with 34 native CZ gates
and zero SWAPs. Exact worst-fold validation selected seed 46; Emerald-matched
500-shot validation then locked alpha 1000 and strength 0.20, with mean fold
gain +1.170%, worst fold +0.715%, and proxy test gain +0.453%.

The six-circuit hardware diagnostic passed strongly at 0.898 feature
correlation, so the full job (`019f57d9-c93e-7861-b506-14508c83d2d8`) was run.
Across all 120 rows, feature correlation improved to 0.818 versus 0.770 for 9q,
but forecasting degraded. Raw 12q hardware lost 0.555% to GARCH; the best
feature-aligned path lost 0.196% (RMSE `0.00833738` versus `0.00832107`). QLIKE
also worsened (`0.016605` versus `0.016476`).

Conclusion: additional qubits improved physical feature fidelity but not useful
financial generalization. The 9q circuit remains the final hardware model, and
16q hardware scaling is rejected. This negative scaling result directly answers
the Phase 3 reservoir-size characterization requirement.

### A. IQM Garnet — primary gate-QRC hybrid forecast

Use OpenQuantum Public Compute, 120 measured rows and 200 shots. The performance
candidate is **9 qubits, one layer, native 3x3 grid, Rz input encoding, seed 44,
second-order Z observables, 24 CX operations**. Every validation fold beats
GARCH (worst +0.709%, mean +1.819%). The conservative worst-fold candidate is
the same layout with Ry encoding, seed 40 (worst fold +1.021%). Both inject
input information nontrivially, unlike the previous Rx-on-|+> design.

```powershell
python experiments/qbraid_hardware_qrc.py `
  --device openquantum:iqm:qpu:garnet `
  --hybrid-target garch --result-tag garnet_q9_grid_rz_garch_hw `
  --qubits 9 --layers 1 --connectivity grid --encoding-axis rz --seed 44 `
  --input-projection first --observable-order 2 `
  --max-test 120 --shots 200 --max-batch 20 `
  --correction-strength 0.25 `
  --readout-training iqm-noise --train-shots 200 `
  --train-p-single 0.001 --train-p-two 0.013 `
  --spark-budget 25 --allow-qpu
```

Expected Public cost: `$42.48 = 21.24 Spark`; reserve: `3.76 Spark = $7.52`.
The 3x3 neighbor graph is intended to map to a connected Garnet sub-grid without
logical routing SWAPs. The run saves raw statevector/device feature
matrices as NPZ so additional classical heads do not require more QPU calls.

Pre-hardware exact result without test-time calibration: hybrid RMSE `0.0082645`
versus GARCH `0.0083211`, a **0.680% improvement** on all 120 test rows. A
moving-block bootstrap interval for model-minus-GARCH RMSE is `[-0.000170,
+0.000019]`; the point gain is promising but not yet statistically significant.

Do **not** freely recalibrate correction strength on 20 hardware rows. A noise
stress test showed that this small calibration can overfit shot noise. Keep the
pre-locked strength 0.25 and train the readout on free local shot/noise features.
On all 120 test rows:

| Scenario | Feature correlation | Gain vs GARCH |
|---|---:|---:|
| Exact statevector | 1.000 | +0.680% |
| 200-shot ideal, noise-aware readout | 0.905 | +0.229% |
| 200-shot IQM-like noise, noise-aware readout | 0.910 | +0.508% |

The exact-trained readout fails under sampled features (-39% to -56%);
noise-aware simulator training is therefore mandatory. These point gains are
not proof of quantum advantage, but show the accessible hybrid can remain above
GARCH under the planned shot budget and first-order IQM noise proxy.

Success targets:

- all 120 circuits and job IDs saved;
- hardware/statevector feature correlation at least 0.85 (target 0.90+);
- hardware RMSE improves on the Ridge residual ablation (`0.00863`);
- stretch: hardware RMSE remains near the ideal gate hybrid (`0.00845`).

### B. QuEra Aquila — executed native analog-QRC hybrid

The token-backed qBraid route was executed with a $25 / 2,500-credit balance.
Live pricing was `30 credits/task + 1 credit/shot`. One successful smoke task
and 23 financial tasks used `1,920` credits total, leaving `580` credits.

The hardware-valid program differs from the early local prototype in three
important ways: coordinates use a seeded irregular rectangular lattice so every
nonzero x/y separation is at least 4 um; local detuning is negative; and qBraid
receives `runtime_options={"experimental_capabilities": "ALL"}`. qBraid SDK
0.12.2 still requires the narrow Decimal-serialization workaround in
`hardware_backend.py`. Two validation failures discovered these constraints and
both cost zero credits.

```powershell
python experiments/quera_aquila_qrc.py `
  --device qbraid_aquila --hybrid-target garch `
  --result-tag qbraid_aquila_hw_native_neg_23x50 `
  --atoms 5 --geometry random2d --seed 42 `
  --max-train 400 --max-test 23 --shots 50 --n-jobs 4 `
  --readout-alpha 300 --correction-strength 0.105 `
  --feature-calibration-rows 3 `
  --credit-budget 2000 --allow-qpu
```

The first three measured inputs were used only for label-free global affine
feature alignment; the remaining 20 were scored. Raw hardware/local feature
correlation was `0.3438`; alignment reduced feature MAE from `0.3255` to
`0.2211` with scale `0.3013` and bias `0.5160`. Despite modest raw fidelity, the
conservative hybrid remained extremely close to its matched local proxy:

| Model (20 held-out rows) | RMSE | Relative to Aquila hardware |
|---|---:|---:|
| GARCH | 0.013728 | Aquila is 0.182% worse |
| Local AHS hybrid | 0.013749 | Aquila is 0.029% worse |
| Real Aquila hybrid | 0.013753 | reference |
| Ridge residual ablation | 0.014641 | Aquila is 6.064% better |
| ESN | 0.014725 | Aquila is 6.601% better |

This is real analog-QPU hybrid competitiveness, not quantum advantage. The
scientific value is that native analog hardware reproduces the matched hybrid
forecast almost exactly after label-free feature transfer, while clearly
beating the linear residual ablation. Per-row checkpoints preserve all features
and task IDs and make the expensive sequential run safely resumable.

### C. Pasqal Fresnel — preferred analog sidecar if access is granted

The simulator result is the best NISQ-ready row (`0.008426`). Fresnel is therefore
scientifically attractive, but real pricing is on request and the device is not
currently online for this account. Do not allocate money until qBraid/Pasqal
confirms a price supporting at least 30–40 sequences. Until then, use the Pulser
emulator result as the Pasqal evidence and Aquila as the executable analog QPU.

### D. Amplify AE and Toshiba SQBM+ — executed signed-Ising ablations

Toshiba supplies a quantum-inspired simulated-bifurcation solver. It is a
cross-substrate ablation, not a QPU and not a quantum-advantage claim. The
available Fixstars `SE/...` credential is for Amplify Scheduling Engine and is
not accepted by the Annealing Engine Ising client. The separate `AE/...`
credential executes the GPU Annealing Engine through `--ising-backend amplify`.

Never place tokens in a notebook, command history, or tracked file. Use the
environment variable `TOSHIBA_TOKEN`, or local file
`.secrets/toshiba_token`. The entire `.secrets/` directory is git-ignored.
For Amplify AE, use `AMPLIFY_AE_TOKEN` or `.secrets/amplify_ae_token`.

Use an economical staged run: one 6-spin solve, then 10 held-out rows, and only
then the 120-row run if the returned ensemble is varied and finite. Training
always uses free local SA features.

```powershell
python experiments/ising_cloud_smoke.py --backend toshiba --spins 6 --reads 20
```

Executed final protocol:

```powershell
python experiments/neutral_atom_garch_hybrid.py `
  --reservoir ising_sa --ising-backend toshiba `
  --atoms 10 --ising-connectivity all-to-all `
  --input-scale 1.0 --j-scale 1.0 --beta 1.0 `
  --max-train 400 --max-test 120 --shots 200 --timeout-ms 1000 `
  --cloud-test-only --save-features --result-tag toshiba_garch_400_120
```

Real SPY result: Toshiba RMSE `0.008353` versus GARCH `0.008318` (-0.421%) and
linear residual Ridge `0.008630` (+3.203% improvement). All 120 cloud solves
returned 200/200 unique states; feature correlation versus local SA was
`0.659729`. Correction strength stayed fixed at `1.0`; no test-label calibration
or post-hoc tuning was used.

Amplify AE required an optimizer-matched protocol because each financial solve
returned a single unique optimum, rather than a thermal ensemble. Ten paid rows
matched the exact local ground-state proxy with feature correlation `1.0`.
The final 120-row run therefore trained on exact ground-state features and used
the IQM-locked readout (`alpha=300`, correction strength `0.105`) without test
calibration:

```powershell
python experiments/neutral_atom_garch_hybrid.py `
  --reservoir ising_sa --ising-backend amplify `
  --atoms 10 --ising-connectivity all-to-all `
  --max-train 400 --max-test 120 --shots 200 --timeout-ms 1000 `
  --cloud-test-only --cloud-train-proxy ground `
  --readout-alpha 300 --correction-strength 0.105 `
  --save-features --result-tag amplify_ae_iqm_locked_400_120
```

Executed result: Amplify AE RMSE `0.008322785` versus GARCH `0.008318205`, only
`0.0551%` worse, and linear residual Ridge `0.008629576`, a `3.555%`
improvement. Feature correlation against the local ground-state proxy was
`1.0`; all 120 AE solves returned one unique state. This is an effectively tied
optimizer-hybrid ablation, not evidence of quantum advantage.

Credential-free footprint check:

```powershell
python experiments/neutral_atom_garch_hybrid.py `
  --reservoir ising_sa --ising-backend toshiba --atoms 10 `
  --max-train 400 --max-test 120 --shots 200 `
  --cloud-test-only --dry-run-cloud
```

## Final claim structure

1. **Advantage candidate:** transverse-field annealer QRC + GARCH beats GARCH by
   1.31% on the full 746-day test in simulation.
2. **Nonlinear hybrid value:** every NISQ-ready reservoir improves on the linear
   GARCH-residual ablation on the common 120-day window.
3. **Hardware realization:** Garnet and Aquila test whether gate and analog
   quantum features survive real-device noise.
4. **Cross-substrate ablation:** Amplify AE and Toshiba SQBM+ test signed
   couplings with two non-QPU optimizers and matched local training proxies.

Never compare RMSE values from different windows as a single leaderboard. Report
the full-window advantage, common-window simulation, and small cloud smoke tests
as separate evidence tiers.

## Accessible-hardware calibration result

The same 20-row calibration / 100-row evaluation protocol was tested before any
paid execution:

| Candidate | Hybrid RMSE | GARCH RMSE | Gap |
|---|---:|---:|---:|
| Pasqal analog QRC | 0.008014 | 0.008011 | +0.04% |
| Signed-Ising / Amplify proxy | 0.008015 | 0.008011 | +0.05% |
| Gate QRC / Garnet candidate | 0.008023 | 0.008015 | +0.11% |

All meet the predefined "GARCH level" tolerance of 1%. The gate correction
remains materially active (`strength=0.47`); Pasqal and signed-Ising calibrate
very close to zero, meaning their safe hybrid mostly falls back to GARCH on this
regime. Do not describe that fallback as quantum advantage.
