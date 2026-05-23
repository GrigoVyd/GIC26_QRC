# D-Wave as the Main QPU for QRC — Exploration

This document explores **D-Wave quantum annealers** as the primary execution
platform for quantum reservoir computing on the SPY volatility-forecast task,
as an alternative to the gate-based IBM Heron path documented in
[`phase3_hardware_plan.md`](phase3_hardware_plan.md).

It is a **planning artifact**: literature review, architectural redesign sketch,
resource estimation, and decision criteria. No code changes accompany it.

## 1. Why look at D-Wave at all

The Phase 3 hardware plan rejected D-Wave on the grounds that it cannot execute
our gate-based circuit. That rejection is correct *for the current code*, but it
shouldn't end the conversation, because D-Wave has three properties that no
gate-based device matches:

| Property | Advantage / Advantage2 | IBM Heron / IonQ Forte |
|---|---|---|
| Qubit count | 5,000+ (Advantage) / ~7,000 (Advantage2) | 133 / 36 |
| Native ZZ couplings | Yes, all couplings programmable directly | Synthesised from CX-Rz-CX |
| Cost per sample | ~$0.0002 (Leap pay-as-you-go) | $1.60/sec (IBM Std) or $0.0007/shot (IonQ) |
| Programmable transverse field per qubit | No (global only) | Yes (Rx gate) |
| Universal gate set | No | Yes |

For a *reservoir*, where the requirements are "large random-coupled
system whose dynamics is rich and reproducible," the first three rows are
strong selling points. The fourth and fifth rows are where the rewrite cost
lives.

In January 2026, D-Wave also announced a **dual-platform roadmap** including a
gate-model variant alongside their annealer line ([D-Wave press release,
2026-01-27](https://www.dwavequantum.com/company/newsroom/press-release/d-wave-announces-advancements-in-annealing-and-gate-model-quantum-computing-technologies/)).
That gate-model platform is not yet generally available, so this document
focuses on the existing **Advantage / Advantage2 annealer line**.

## 2. What changes architecturally

Our current circuit (see [`src/qrc/reservoir.py`](../src/qrc/reservoir.py)):

```
H⊗n  →  [Rx(x·π) → CX-Rz-CX (ZZ) → Rx/Ry/Rz random] × n_layers  →  measure-{Z, ZZ}
```

D-Wave Advantage exposes only a programmable Ising Hamiltonian and an annealing
schedule:

$$H(s) = -A(s)\sum_i \sigma_i^x \;+\; B(s)\Big[\sum_i h_i \sigma_i^z + \sum_{ij} J_{ij}\sigma_i^z \sigma_j^z\Big]$$

with `s ∈ [0, 1]` and the schedule `A(s), B(s)` operator-fixed by hardware (you
can pause, reverse, and adjust the global ramp but not per-qubit). What survives
from our circuit:

* ZZ couplings → **native and free** (set `J_ij` directly).
* Random reservoir weights → **fine** (random `h_i, J_ij` are just programmed).
* Layered structure → **gone**. Replaced by a continuous evolution.
* Per-qubit Rx encoding → **gone**. Inputs have to enter through `h_i` (longitudinal
  bias), not transverse rotation.
* Single-qubit random rotations → **gone**. Replaced by random `h_i` biases.
* Hadamard initialisation → **implicit** in the annealer's starting state
  $|+\rangle^{\otimes n}$ at `s=0`.

The "evolution" you control is the **anneal schedule** (`s(t)` profile, pause
duration, reverse-anneal start point). Read-out is shot-based in the computational
basis only.

This is a **redesign of the reservoir**, not a backend swap. But the literature
shows it's a redesign with a known shape.

## 3. Literature review

### 3.1 Foundational QRC theory (substrate-agnostic)

* **Fujii & Nakajima, *Phys. Rev. Applied* 8, 024030 (2017)**
  — "Harnessing disordered ensemble quantum dynamics for machine learning."
  Original QRC paper. Shows that a 5–7-qubit disordered Ising system has the
  expressive power of a 500-node classical RNN. The transverse-field Ising model
  is the canonical reservoir. ([arXiv:1602.08159](https://arxiv.org/abs/1602.08159))

* **Mujal, Martínez-Peña, Nokkala, García-Beni, Giorgi, Soriano, Zambrini,
  *Adv. Quantum Tech.* 4, 2100027 (2021)** — "Opportunities in Quantum Reservoir
  Computing and Extreme Learning Machines." The standard review; surveys
  gate-based, photonic, atomic, and analog implementations.
  ([arXiv:2102.11831](https://arxiv.org/abs/2102.11831))

* **Martínez-Peña, Giorgi, Nokkala, Soriano, Zambrini, *Phys. Rev. Lett.* 127,
  100502 (2021)** — "Dynamical Phase Transitions in Quantum Reservoir Computing."
  Identifies *which dynamical regime* gives best reservoir performance — directly
  relevant when picking a D-Wave anneal schedule.

* **Mujal, Martínez-Peña, Giorgi, Soriano, Zambrini, *npj Quantum Information*
  9, 16 (2023)** — "Time-series quantum reservoir computing with weak and
  projective measurements." Treats measurement back-action and shot statistics
  as part of the reservoir, which is exactly the regime D-Wave operates in
  (projective measurement at the end of each anneal).

### 3.2 D-Wave / annealer dynamics for ML (the load-bearing references)

* **Sakurai, Hayashi, Matsumori, Kaji, Kadowaki, Nemoto, arXiv:2601.09938
  (2026)** — "Beyond Optimization: Harnessing Quantum Annealer Dynamics for
  Machine Learning." Direct precedent. Encodes classical data into an Ising
  Hamiltonian, evolves it on a D-Wave annealer, and uses the resulting sample
  distributions as feature maps for classification. Validates on Digits (D-Wave
  hardware) and MNIST (simulator). Key empirical findings:
  - **Short anneal times** improve classification accuracy.
  - The **participation ratio** of the output distribution correlates with
    generalization performance — gives a principled way to tune the anneal
    schedule.
  This is the closest existing template for what a "D-Wave QRC" looks like in
  practice. ([arXiv:2601.09938](https://arxiv.org/abs/2601.09938))

### 3.3 Volatility forecasting with QRC (the task-specific reference)

* **Li, Mukhopadhyay, Bayat, Habibnia, *Phys. Rev. Research* (2025) / arXiv:2505.13933**
  — "Quantum Reservoir Computing for Realized Volatility Forecasting." A
  fully-connected transverse-field Ising reservoir with separate input and
  memory qubits, applied to *exactly our task*. Reports outperforming standard
  econometric and ML benchmarks. Uses wrapper-based forward feature selection
  and Shapley values to identify the most informative inputs.
  ([arXiv:2505.13933](https://arxiv.org/abs/2505.13933))

  **Implication for us:** the reservoir architecture they use (transverse-field
  Ising) is what D-Wave naturally simulates. Their result is a strong
  *existence proof* that an Ising-type reservoir suffices for SPY-style
  realized-volatility forecasting — which is the regime D-Wave can run.

### 3.4 Other analog QRC implementations (parallel substrates)

* **Bravo, Najafi, Gao, Yelin, *PRX Quantum* 3, 030325 (2022)** — Rydberg-atom
  arrays. Same Ising-flavored substrate idea, different hardware.

* **Govia, Ribeill, Rowlands, Ohki, *Phys. Rev. Research* 3, 013077 (2021)**
  — Single nonlinear oscillator as a QRC. Shows the substrate need not be a
  large array; relevant if D-Wave's coupling graph turns out too restrictive
  and we have to embed a smaller logical reservoir.

These are not D-Wave papers, but they reinforce the broader claim that **the
gate model is not the only viable substrate** for QRC.

## 4. Proposed D-Wave QRC architecture for SPY volatility

Adapting our v3 setup (HAR-RV + log-HAR + residual log-vol target) to a D-Wave
Advantage:

### 4.1 Encoding (replaces Rx)

* For each test sample at time τ, compute the 21-dim feature vector
  `x_τ = [returns, |r|, r², HAR_RV, log-HAR_RV]` (already implemented in
  `load_financial_data_v2`).
* Standardise to roughly unit variance, clip to `[-1, 1]`.
* Map onto **20 input qubits** as biases: `h_i = x_τ[i]` (note Advantage uses
  `h ∈ [-2, 2]` natively).
* Leave the remaining ~80 *memory* qubits with `h = 0` — they exist purely to
  enrich the dynamics. This mirrors the "input + memory qubit" split used in
  arXiv:2505.13933.

### 4.2 Random reservoir (replaces fixed random rotations)

* On a sub-graph of the Pegasus topology, set `J_ij` to fixed random values in
  `[-1, 1]` drawn once at construction (analogous to our current `_zz_w` in
  `QuantumReservoir.__init__`).
* Choose the sub-graph to be a dense Pegasus block: ~100 qubits where every
  qubit has ~12–15 immediate neighbors. No chains, no embedding overhead.

### 4.3 Evolution (replaces layered circuit)

Two natural choices, both supported by Advantage's anneal schedule controls:

* **Quench mode** — short forward anneal (e.g. 2 µs total) so the system stays
  far from equilibrium. arXiv:2601.09938 shows this gives higher feature
  expressiveness for classification.
* **Reverse anneal + pause** — start from a reference state (e.g. all-zero),
  reverse-anneal to `s ≈ 0.5`, pause 5–20 µs, ramp back. Lets us tune the
  effective temperature / chaoticity of the dynamics, which Martínez-Peña et
  al. (2021) identifies as the key control for reservoir quality.

### 4.4 Feature extraction (replaces ⟨Z⟩ + ⟨ZZ⟩)

* Read N shots from the final state (Advantage natively returns 1,000–10,000
  shots per programming cycle).
* Compute per-qubit means `⟨σ_i^z⟩` and pairwise correlations
  `⟨σ_i^z σ_j^z⟩` for `(i, j)` on the same Pegasus sub-graph used for couplings.
* For 100 qubits with ~12 neighbours each: 100 + 600 ≈ 700 features per
  sample — about an order of magnitude richer than our current 5q/10q gate
  reservoir.

### 4.5 Readout

Unchanged: TimeSeriesSplit-CV Ridge on standardised features. Predict residual
log-vol; invert with `RV[τ-1] · exp(ŷ)`.

## 5. Resource estimation on D-Wave Leap

D-Wave's free tier on Leap is **1 min QPU time/month** — strict, but the cost
profile is very different from IBM's. Key timings on Advantage:

| Operation | Time | Notes |
|---|---|---|
| Programming cycle (set `h_i, J_ij`) | ~7–10 ms | One per *distinct* problem |
| Single anneal | 5–2000 µs (default 20 µs) | Adjustable |
| Readout per sample | ~120 µs | Fixed overhead |
| Total per sample | ~140 µs (typical) | Anneal + readout |

For our task each test input τ has a different `x_τ` → different `h_i` → a
separate programming cycle. That's the bottleneck, not the anneal.

### 5.1 Per-config QPU time

For 750 test samples × (10 ms programming + 100 anneals × 140 µs):
* Programming: `750 × 10 ms = 7.5 sec`
* Anneal + readout: `750 × 100 × 140 µs ≈ 10.5 sec`
* **Total ≈ 18 seconds of QPU time per full test-set pass.**

### 5.2 What fits in 1 min/month

* **~3 full test-set passes per month**, OR
* 1 full test-set pass + ~20 ablation studies (anneal time, pause length,
  number of shots) using a 50-sample subset, OR
* Side-by-side comparison of 3 different J-matrix seeds (reservoir robustness).

For pay-as-you-go beyond the free tier, Leap charges roughly $2/sec of QPU
time — so a full ablation pack is ~$30. Cheaper than IBM Standard.

### 5.3 The hidden cost: programming overhead

If all 750 test inputs used the *same* `h_i, J_ij`, programming would happen
once and the cost would drop ~100×. This means a useful optimisation is to
**batch inputs** by mapping them onto disjoint sub-graphs of the same
programming — fitting, e.g., 10 different inputs onto 10 × 100-qubit blocks of
the same 1000-qubit problem. Advantage's 5000+ qubit count makes this realistic.
This is the same "spatial multiplexing" trick used in Nakajima et al. (2019).

## 6. Comparison to IBM Heron path

| Dimension | IBM Heron (current plan) | D-Wave Advantage |
|---|---|---|
| Code reuse | ~100% (Qiskit-native) | ~0% (new reservoir class needed) |
| Engineering cost (Phase 3) | ~1–2 days | ~2–3 weeks |
| Reservoir size | 5–10 qubits practical | 100+ qubits practical |
| Feature count per sample | ~25–55 | ~500–1000 |
| QPU time per test-set pass | ~40 sec | ~18 sec |
| Free-tier monthly budget | 10 min (~16 passes) | 1 min (~3 passes) |
| Match to existing literature | Strong (Mujal 2021, Suzuki 2022) | Strong (Sakurai 2026 for annealer ML; Li 2025 for vol-forecast Ising) |
| Direct task precedent | Indirect | **arXiv:2505.13933 used Ising for exact same task** |
| Competition narrative | Conventional, defensible | Higher-risk, higher-novelty |

The Heron path is **clearly correct for the May 31, 2026 Phase 2 deadline.**
The D-Wave path is a **credible parallel-track or post-competition direction**,
because:

1. The task-specific precedent (Li et al. 2025) used an Ising reservoir — D-Wave
   is the natural execution platform for that architecture.
2. Sakurai et al. (2026) just published a method for using D-Wave annealer
   dynamics as a feature map, which is structurally the same idea as QRC.
3. The "100+ qubit reservoir" regime is *only* accessible on D-Wave today.

## 7. Decision: when to pursue this

Pursue the D-Wave track if and only if:

1. The Heron path is already delivering Phase 2 numbers (i.e. v3 results are in,
   hardware test-set run is done).
2. There is engineering bandwidth for ~2 weeks of new reservoir code (input
   encoding via `h_i`, Pegasus sub-graph selection, anneal-schedule sweep, new
   feature extractor).
3. The competition organizers / GIC reviewers value the "novel substrate"
   angle — worth checking via Phase 1 feedback.

Skip if any of the above is false. The current PR (#4) is correct for Phase 2;
this is Phase 3+ material.

## 8. Concrete next steps if we proceed

In order:

1. **Read Sakurai et al. (2026) carefully** — extract their exact encoding scheme,
   anneal schedule, and readout statistics. Their code may be open-sourced.
2. **Sign up for D-Wave Leap**, claim the free tier minute, and run their
   reference Digits example to verify access.
3. **Spike a `src/qrc/annealer_reservoir.py`** with a minimal `transform()`
   method that takes a feature matrix and returns per-qubit/pair statistics.
   Use D-Wave's `dwave-system` SDK.
4. **Pegasus sub-graph picker** — script that finds a dense ~100-qubit block on
   the current Advantage calibration. (D-Wave provides
   `dwave_networkx.pegasus_graph`.)
5. **Anneal-schedule sweep** — run 50-sample test subsets at anneal times
   `{2, 5, 20, 100} µs` and at reverse-anneal pauses `{0, 5, 20} µs`. Pick the
   best by validation NMSE.
6. **Full-task run** at the best schedule. Compare against `v3` simulator
   results on the same residual log-vol target.

## 9. Honest closing assessment

The D-Wave track is **research-grade interesting** for this task because:

* It's the only path to >100-qubit reservoirs today.
* The substrate matches the Ising reservoir used by the closest published
  competitor (arXiv:2505.13933).
* The dynamical-phase-transition story (Martínez-Peña 2021) gives a principled
  knob to tune the anneal schedule.

It is **not the right choice for the Phase 2 deadline**, because the rewrite
cost (Section 6) doesn't fit the calendar. The right place for this work is a
post-Phase-2 extension, a Phase 3 stretch goal, or a follow-up paper.

## References

1. Fujii, K. & Nakajima, K. *Harnessing disordered ensemble quantum dynamics for
   machine learning.* Phys. Rev. Applied 8, 024030 (2017).
   [arXiv:1602.08159](https://arxiv.org/abs/1602.08159)
2. Mujal, P., Martínez-Peña, R., Nokkala, J., García-Beni, J., Giorgi, G. L.,
   Soriano, M. C. & Zambrini, R. *Opportunities in Quantum Reservoir Computing
   and Extreme Learning Machines.* Adv. Quantum Tech. 4, 2100027 (2021).
   [arXiv:2102.11831](https://arxiv.org/abs/2102.11831)
3. Martínez-Peña, R., Giorgi, G. L., Nokkala, J., Soriano, M. C. & Zambrini, R.
   *Dynamical Phase Transitions in Quantum Reservoir Computing.* Phys. Rev.
   Lett. 127, 100502 (2021).
4. Mujal, P., Martínez-Peña, R., Giorgi, G. L., Soriano, M. C. & Zambrini, R.
   *Time-series quantum reservoir computing with weak and projective
   measurements.* npj Quantum Information 9, 16 (2023).
5. Sakurai, A., Hayashi, A., Matsumori, T., Kaji, D., Kadowaki, T. & Nemoto, K.
   *Beyond Optimization: Harnessing Quantum Annealer Dynamics for Machine
   Learning.* arXiv:2601.09938 (2026).
   [arXiv:2601.09938](https://arxiv.org/abs/2601.09938)
6. Li, Q., Mukhopadhyay, C., Bayat, A. & Habibnia, A. *Quantum Reservoir
   Computing for Realized Volatility Forecasting.* Phys. Rev. Research (2025) /
   arXiv:2505.13933.
   [arXiv:2505.13933](https://arxiv.org/abs/2505.13933)
7. Bravo, R. A., Najafi, K., Gao, X. & Yelin, S. F. *Quantum Reservoir Computing
   using Arrays of Rydberg Atoms.* PRX Quantum 3, 030325 (2022).
8. Govia, L. C. G., Ribeill, G. J., Rowlands, G. E. & Ohki, T. A. *Quantum
   reservoir computing with a single nonlinear oscillator.* Phys. Rev. Research
   3, 013077 (2021).
9. Nakajima, K., Fujii, K., Negoro, M., Mitarai, K. & Kitagawa, M. *Boosting
   computational power through spatial multiplexing in quantum reservoir
   computing.* Phys. Rev. Applied 11, 034021 (2019).
10. D-Wave Quantum Inc. *Press release: Advancements in annealing and gate-model
    quantum computing technologies* (2026-01-27).
    [dwavequantum.com](https://www.dwavequantum.com/company/newsroom/press-release/d-wave-announces-advancements-in-annealing-and-gate-model-quantum-computing-technologies/)
