# Phase 3 — Hardware Selection and Resource Estimation

This document records the reasoning behind the QPU choice for running the
QRC volatility forecast on real quantum hardware, and translates IBM's Open
Plan quota into a concrete monthly experiment budget.

## 1. What the circuit actually demands

From [`src/qrc/reservoir.py`](../src/qrc/reservoir.py), the per-sample circuit is

```
H⊗n  →  [Rx(x·π) → CX-Rz-CX (ZZ) → Rx/Ry/Rz random] × n_layers  →  measure
```

Two-qubit gate counts (native CX, before SWAP overhead):

| Config | Edges/layer | CX/layer | Total CX (L=2) |
|---|---:|---:|---:|
| 5q L2 random (~4 edges) | 4 | 8 | **16** |
| 5q L2 all-to-all (10 edges) | 10 | 20 | **40** |
| 10q L2 random (~18 edges) | 18 | 36 | **72** |
| 10q L2 all-to-all (45 edges) | 45 | 90 | **180** |

The **headline simulator result** (best on R² and RMSE in `results/financial_results_v3.csv`)
is **5q L2 random** — small *and* sparse, which is the regime where current NISQ
hardware works well.

Sample budget per QRC variant: ~3,000 train + 750 test ≈ **3,750 circuits**, at
~256–1024 shots each.

## 2. Vendor comparison

Decision rule applied in order: **(a) does the device run our instruction set,
(b) does the topology fit our coupling pattern, (c) does fidelity × depth survive,
(d) does shot economics scale to thousands of samples.**

### IBM Quantum Heron (133q, heavy-hex, gate-based)

* 2q fidelity ~99.7%; arbitrary universal circuits; Qiskit-native (zero porting cost).
* Heavy-hex penalty: 2–3× CX inflation on our `random` graph at 10q, ~5× on
  `all-to-all` 10q. Acceptable on 5q configs (our best ones).
* Open Plan: free, 10 min QPU/month. Standard plan ~$1.60/sec.
* **Verdict: primary target.**

### IonQ Forte (36q, trapped ion, all-to-all)

* 2q fidelity ~99.6%; native all-to-all → no SWAP overhead on dense graphs.
* The only vendor where 5q all-to-all and 10q random run at full native depth.
* Cost ~$0.0007/shot via AWS Braket → ~$135 for one config × test set × 256 shots.
* **Verdict: one sidecar run for the all-to-all comparison.**

### Quantinuum H2 (56q, all-to-all within zone, mid-circuit measurement)

* 2q fidelity ~99.85% (best in market).
* Native mid-circuit measurement + classical feedback — would let us replace the
  classical Z-feedback in `transform_sequential` with genuine coherent feedback.
  Strongest "quantum advantage angle" in the project.
* Cost prohibitive ($3K–12K/hour) → only viable as a single showcase run.
* **Verdict: stretch goal, only if access is granted.**

### D-Wave Advantage (5000+ qubits, quantum annealer)

* **Cannot execute our instruction set.** It implements a programmable Ising
  Hamiltonian $H = \sum h_i \sigma_i^z + \sum J_{ij}\sigma_i^z\sigma_j^z$ and
  anneals to (approximately) its ground state. No Hadamard, no per-qubit Rx
  encoding, no random Rx/Ry/Rz, no layered structure.
* "QRC on D-Wave" exists in literature (Kitayama et al., García-Beni et al.) but
  uses a **different reservoir architecture**: inputs encoded into `h_i/J_ij`,
  short reverse-anneal as the evolution, sample histograms as features. Porting
  to D-Wave means redesigning encoding, evolution, and feature extraction — a
  rewrite, not a backend swap.
* **Verdict: out of scope for Phase 3.** Note it as future work; do not engineer.

### Quandela / Perceval (photonic, boson-sampling-style)

* Different reservoir paradigm (linear-optical), not a port target.
* Mention as future work; do not engineer for Phase 3.

## 3. Fidelity × depth sanity check

Approximate end-of-circuit fidelity ≈ $f_{2q}^{N_{CX}}$ (generous: ignores 1q
gates, idle errors, SPAM).

| Config | IBM Heron (with SWAP) | IonQ Forte (no SWAP) |
|---|---:|---:|
| 5q L2 random (16 CX) | 0.997¹⁶ ≈ **95%** ✓ | 0.997¹⁶ ≈ **95%** ✓ |
| 10q L2 random (~150 CX) | 0.997¹⁵⁰ ≈ **64%** marginal | 0.997⁷² ≈ **81%** ✓ |
| 10q L2 all-to-all (~500 CX) | 0.997⁵⁰⁰ ≈ **22%** ✗ | 0.997¹⁸⁰ ≈ **58%** marginal |

Reservoir features (Z, ZZ expectation values) degrade gracefully under noise —
and noise can even improve reservoir richness — but below ~60% circuit fidelity
the signal-to-noise on individual features collapses. **5q configs work on
either vendor; 10q dense configs only on all-to-all hardware.**

## 4. Open Plan budget (10 min QPU/month)

IBM's Open Plan counts **QPU execution time only** — not queue, not classical
post-processing. With Sampler V2 + parametric circuits, many circuits are
submitted in one batch so the per-circuit overhead is small.

For **5q L2 random** (16 CX, ~30 single-qubit gates) on Heron, per-shot wallclock
≈ 5–10 μs (gates + measure + reset). Per-circuit QPU time:

| Shots | Batched | Unbatched |
|---|---|---|
| 256 | ~15 ms | ~200 ms |
| 1024 | ~50 ms | ~300 ms |

Converting 600 seconds of monthly QPU time:

| Scenario | Circuits per month |
|---|---:|
| Batched, 256 shots | ~**12,000** |
| Batched, 1024 shots | ~**4,000** |
| Unbatched (naive), 1024 shots | ~**2,000** |
| Worst case (10q deep circuit, 1024 shots) | ~**1,000** |

**Plan around 3,000–4,000 circuits per month** as the working budget.

## 5. The hybrid strategy

**Train the Ridge readout on noiseless simulator features. Run only the test set
on hardware.**

Justification: Ridge weights are a linear map of reservoir features. Hardware
noise biases the features but doesn't break the readout's structure, *as long as
training and test use a consistent feature source*. So we report two numbers:

* `RMSE_sim` — simulator features at train and test (current results).
* `RMSE_hw` — simulator features at train, hardware features at test.

The gap **is** the hardware story; the comparison is fair because the readout
weights are held fixed.

Test set is 750 circuits. At 1024 shots batched, ~38 seconds of QPU time → about
**16 full test-set runs fit in one month.**

## 6. Concrete monthly plan

| Month | What to run | QPU time | Yields |
|---|---|---:|---|
| 1 | Best config (5q L2 random), 1024 shots, test set | ~40 s | Headline HW row in results table |
| 1 | Same config × 4 shot counts (128 / 256 / 512 / 2048) | ~150 s | Shot-noise scaling curve |
| 1 | Re-run best config × 3 (different days) | ~120 s | Calibration-drift error bars |
| 2 | 5q L2 random vs all-to-all vs 5q L3 random, test set | ~120 s | Architecture-survival plot on HW |
| 2 | 10q L2 random, test set | ~100 s | Qubit-scaling on HW |
| 3 | Stretch: full HW train + test for best config | ~400 s | Apples-to-apples HW-trained readout |

Three months → headline result + shot-noise curve + calibration error bars +
architecture sweep + qubit-scaling + (optionally) HW-trained readout. Sufficient
for Phase 2 PDF and the Phase 3 hardware narrative.

## 7. Implementation checklist (when wiring `QiskitRuntimeService`)

1. **Sampler V2 + parametric circuits.** Build the QRC template once with
   `qiskit.circuit.Parameter` objects, then bind the 750 test inputs in a single
   `Sampler.run()` call. Avoids per-circuit submission overhead.
2. **`optimization_level=3`** in `transpile()` and pass a `coupling_map` from the
   live backend properties so the 5q circuit is laid out on the best-calibrated
   qubits of the day.
3. **Hold a `Session`** across multi-config runs to avoid re-queuing.
4. **Hybrid first.** Do not run training on hardware in Month 1 — it eats the
   whole quota for one config when hybrid gives four configs.
5. **Run at the start of the month.** Calibration is freshest right after the
   weekly maintenance window; if a run fails there is time to retry.
6. **Save raw shot data** to `results/hardware/` so we can re-derive features at
   different shot counts without re-running on the QPU.

## 8. Underlying decision rule

When picking quantum hardware, ask in this order:

1. **Does it run my instruction set?** If no, stop. (D-Wave, photonic fail here
   for our gate-based circuit.)
2. **Does the topology fit my coupling pattern?** If our best configs are sparse,
   heavy-hex is fine; if dense, prefer all-to-all (IonQ, Quantinuum).
3. **Does depth × fidelity survive?** Sanity-check $f_{2q}^{N_{CX}}$ before
   committing.
4. **Does shot economics scale?** Multiply samples × configs × shots × $/shot
   and check against the budget envelope.

For this project, our headline config is **small and sparse**, which means
question 1 dominates and IBM Heron is the right answer. Question 2 motivates the
single IonQ sidecar. Questions 3 and 4 confirm both choices are within budget.
