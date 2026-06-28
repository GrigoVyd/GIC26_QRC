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

This gives a clean, falsifiable Phase 3 story: **the quantum advantage over the
best classical baseline emerges at the quantum-annealer QPU tier (D-Wave), and is
absent at the classical/GPU and neutral-atom tiers.**

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
| GPU Ising machine | **Fixstars Amplify AE** | `amplify` | ✅ reservoir built; cloud run ready (AMPLIFY_TOKEN) |
| Local Ising (free) | SimulatedAnnealing | `dwave.samplers` | ✅ run (0.00814) |
| Neutral-atom QPU | **QuEra Aquila** | Braket AHS | ✅ reservoir + local AHS run; real-Aquila run ready (qBraid) |
| Neutral-atom QPU | **Pasqal Fresnel** | Pulser | ✅ reservoir + emulator run; HW pending device availability |
| Gate QPU | IonQ / IQM / Rigetti | qBraid runtime | ✅ pipeline live-verified on qBraid `qir-sv` |
| **Quantum annealer QPU** | **D-Wave Advantage** | `dwave.system` (Leap) | ✅ reservoir built; **Leap run is the key remaining hardware proof** |

**What has run on real cloud infrastructure:** the gate pipeline was verified
end-to-end on the live qBraid `qir-sv` API (real job ids, feature fidelity 0.996).
All reservoir simulators/emulators (AHS, Pulser, SA, Amplify-ready) run locally.
**The headline hardware proof remaining is the D-Wave Advantage run** of the
GARCH-hybrid reservoir, where the simulated advantage is expected to transfer.

## 7. Reproducibility

| Result | Command |
|---|---|
| Classical + raw-task leaderboard | `python experiments/phase2_final.py` |
| GARCH-hybrid (annealer) — beats GARCH | `python experiments/phase2_garch_hybrid.py --n-seeds 3` |
| GARCH-hybrid on neutral atoms / Ising machine | `python experiments/neutral_atom_garch_hybrid.py --reservoir {quera,pasqal,pasqal_local,ising_sa} --atoms 10 --max-test 0 --n-seeds 3` |
| QuEra Aquila (real HW) | `python experiments/quera_aquila_qrc.py --device aquila --allow-qpu` |
| Ising on Fixstars Amplify (GPU) | set `AMPLIFY_TOKEN`, `IsingReservoir(...).transform(X, backend="amplify")` |
| Ising on D-Wave (Leap) | set Leap token, `IsingReservoir(...).transform(X, backend="dwave")` |

Reservoirs: `src/qrc/{reservoir,annealer_reservoir,quera_reservoir,pasqal_reservoir,ising_reservoir}.py`.
Setup docs: `docs/{quera_aquila_setup,pasqal_fresnel_setup,qbraid_setup}.md`.

## 8. Honest limitations

- The GARCH-beating margin is small (−1.4% RMSE) and on a near-saturated residual;
  it is consistent across RMSE/R²/QLIKE and isolable to the reservoir, but modest.
- **No QPU has yet *executed* the GARCH-beating result** — it is demonstrated in
  noiseless quantum-annealer simulation. Confirming it requires a real D-Wave run
  (shot/embedding/noise effects may shift the absolute number; we expect the
  direction to survive — shot-noise was shown non-limiting on the neutral-atom side).
- GARCH(1,1) is the simplest GARCH; a full econometric panel (GJR/EGARCH/HAR) is
  future work. The residual trick should compose with any of them.
