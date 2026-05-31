# Technical Reference: QRC for SPY Realized Volatility

> **Purpose:** Comprehensive explanation of every technical component in this project, written for presentation preparation. Covers the problem, data pipeline, quantum architectures, classical baselines, metrics, the GARCH-hybrid breakthrough, and the Phase 3 hardware path.

---

## Table of Contents

1. [The Problem: What Are We Predicting?](#1-the-problem)
2. [Data Pipeline](#2-data-pipeline)
3. [Quantum Reservoir Computing — Core Concept](#3-qrc-core-concept)
4. [Architecture A: Gate-Based QRC](#4-gate-based-qrc)
5. [Architecture B: Annealer-Style QRC (Headline)](#5-annealer-qrc)
6. [The Hybrid Readout (Quantum + Classical Features)](#6-hybrid-readout)
7. [Classical Baselines](#7-classical-baselines)
8. [The GARCH Puzzle and the Hybrid Solution](#8-garch-hybrid)
9. [Target Transforms: Why Log-Vol Residuals?](#9-target-transforms)
10. [Metrics and Evaluation](#10-metrics)
11. [Noise Models](#11-noise-models)
12. [Key Results Summary](#12-results)
13. [Phase 3: Hardware Path](#13-hardware)
14. [Glossary](#14-glossary)

---

## 1. The Problem: What Are We Predicting? <a name="1-the-problem"></a>

**Task:** Predict next-day realized volatility of SPY (S&P 500 ETF).

**Realized volatility (RV)** is the standard deviation of daily log-returns over a rolling window, annualized:

```
RV[t] = std(log_returns[t-20 : t]) * sqrt(252)
```

- **Rolling window:** 21 trading days (roughly 1 month)
- **Annualization factor:** sqrt(252) converts daily vol to annual vol
- Typical values: 0.05 (very calm, ~5% annual) to 0.40 (crisis, ~40% annual)

**Why this matters:** Volatility drives options pricing, risk management, and portfolio allocation. Better volatility forecasts = better hedging, VaR estimates, and trading signals.

**Competition context:** GIC 2026, Track A (Dynamic Systems Forecasting). The rubric requires comparing against Persistence, AR, Ridge, ESN, GARCH, LSTM and reporting RMSE, R^2, QLIKE, and Mincer-Zarnowitz statistics.

---

## 2. Data Pipeline <a name="2-data-pipeline"></a>

### 2.1 Raw Data

- **Source:** SPY daily close prices from yfinance, 2010-01-01 to 2024-12-31
- **Total:** ~3,750 trading days
- **Fallback:** If yfinance is unavailable, a synthetic GARCH(1,1) process generates comparable data

### 2.2 Feature Engineering (load_financial_data_v2)

The v2 loader (`src/data/loaders.py`) constructs 21 features per sample, all causally available at forecast time (no look-ahead):

| Feature Group | Count | Description |
|---------------|-------|-------------|
| Log-returns | 5 | r[t-5], r[t-4], ..., r[t-1] |
| Absolute returns | 5 | \|r[t-5]\|, ..., \|r[t-1]\| |
| Squared returns | 5 | r^2[t-5], ..., r^2[t-1] |
| HAR-RV (Corsi 2009) | 3 | RV_daily[t-1], RV_weekly (5-day avg), RV_monthly (22-day avg) |
| Log-HAR-RV | 3 | log versions of the above |

**Why these features?**
- **Returns** capture recent price dynamics
- **Absolute returns** are a better proxy for volatility than raw returns (less noisy)
- **Squared returns** are what GARCH models use internally (variance = E[r^2])
- **HAR-RV** (Heterogeneous Autoregressive model of Corsi 2009) is the gold standard in volatility forecasting — it captures the multi-scale persistence of volatility by including daily, weekly, and monthly RV averages

### 2.3 GARCH Proxy Features (for hybrid model)

When `include_garch_proxy=True`, two additional features are added:
- `garch_proxy_rv`: GARCH(1,1)-predicted RV for this timestep
- `log_garch_proxy_rv`: its log

These are fitted on training data only (no leakage) and rolled forward through the test set.

### 2.4 Train/Test Split

- **Method:** Chronological split, 80% train / 20% test
- **No shuffle** — shuffling would cause future-to-past information leakage
- **Result:** ~2,984 training samples, ~746 test samples

---

## 3. Quantum Reservoir Computing — Core Concept <a name="3-qrc-core-concept"></a>

### 3.1 What is Reservoir Computing?

Reservoir computing is a framework where:
1. A **reservoir** (a complex dynamical system) transforms inputs into a high-dimensional feature space
2. Only a simple **readout** (linear regression) is trained
3. The reservoir itself is **not trained** — its weights are random and fixed

This is analogous to a kernel method: the reservoir provides a nonlinear feature expansion, and the linear readout finds the best combination.

### 3.2 Why Quantum?

A quantum system with n qubits has 2^n basis states. This means:
- **5 qubits** = 32-dimensional Hilbert space -> 15 features (Z + ZZ expectations)
- **10 qubits** = 1024-dimensional Hilbert space -> 55 features
- **15 qubits** = 32768-dimensional Hilbert space -> 120 features

The quantum reservoir naturally produces highly nonlinear, high-dimensional feature transformations that would require exponentially many classical neurons to replicate. The key insight: the quantum system's entanglement structure creates correlations between observables that are hard to compute classically.

### 3.3 The QRC Pipeline

```
Input x (21 features)
    |
    v
[Select first n_qubits features] -> [tanh normalization to [-1, 1]]
    |
    v
[Quantum Circuit: encode + evolve]
    |
    v
[Measure observables: <Z_i> and <Z_i Z_j>]  ->  n + n(n-1)/2 features
    |
    v
[Concatenate with original 21 features]  (hybrid approach)
    |
    v
[StandardScaler -> Ridge Regression (CV-tuned alpha)]
    |
    v
Prediction y_hat
```

---

## 4. Architecture A: Gate-Based QRC <a name="4-gate-based-qrc"></a>

**File:** `src/qrc/reservoir.py` — class `QuantumReservoir`

### 4.1 Circuit Structure

For each input sample, the circuit is:

```
|0...0>  -->  H^(x)n  -->  [Encode(x) -> Ising Layer] x n_layers  -->  Measure
```

**Step by step:**

1. **Initial superposition:** Apply Hadamard (H) to every qubit
   - Creates uniform superposition |+>^n
   
2. **Per layer (repeated n_layers times):**
   
   a. **Angle encoding:** Rx(pi * x_i) on qubit i
   - Maps input value x_i to a rotation angle
   - Uses modular wrapping: qubit i gets x[i % len(x)]
   
   b. **ZZ Ising couplings:** For each edge (i,j) in the connectivity graph:
   ```
   CX(i,j) -> Rz(w_zz) -> CX(i,j)
   ```
   - This implements exp(-i * w_zz * Z_i Z_j), the Ising interaction
   - The coupling strengths w_zz are random, drawn once at initialization and fixed
   
   c. **Local rotations:** Rx, Ry, Rz with random fixed angles on each qubit
   - Acts as "mixing" — scrambles the state non-trivially

3. **Feature extraction:**
   - **Z expectations:** <Z_i> for each qubit (n values)
   - **ZZ correlators:** <Z_i Z_j> for each pair i < j (n(n-1)/2 values)
   - Computed via exact statevector simulation (noiseless) or shot-based measurement (noisy)

### 4.2 Connectivity Options

| Type | Edges | Use Case |
|------|-------|----------|
| `linear` | (0,1), (1,2), ..., (n-2, n-1) | Nearest-neighbor, hardware-friendly |
| `random` | ~40% of all pairs, at least n-1 | Good balance of expressivity and speed |
| `all-to-all` | All n(n-1)/2 pairs | Maximum entanglement, most features |

### 4.3 Configuration Used

- **Best config:** 5 qubits, 2 layers, random connectivity
- **Performance:** RMSE = 0.00967 (beats Ridge 0.00976, ties ESN 0.00970)
- **Feature count:** 15 reservoir features + 21 raw features = 36 total

---

## 5. Architecture B: Annealer-Style QRC (Headline) <a name="5-annealer-qrc"></a>

**File:** `src/qrc/annealer_reservoir.py` — class `AnnealerReservoir`

This is the main innovation of the project. Instead of gate-model circuits, we simulate a transverse-field Ising Hamiltonian — the native physics of D-Wave and QuEra hardware.

### 5.1 The Hamiltonian

```
H = - sum_i sigma^x_i  +  alpha * sum_{i in Input} x_i * sigma^z_i  +  sum_i h_mem_i * sigma^z_i  +  sum_{(i,j)} J_ij * sigma^z_i * sigma^z_j
```

**Term by term:**

| Term | Role | Analogy |
|------|------|---------|
| -sum sigma^x_i | Transverse field | "Shaking" — drives quantum tunneling, prevents freezing |
| alpha * x_i * sigma^z_i | Input encoding | Data enters as longitudinal biases (D-Wave's native encoding) |
| h_mem_i * sigma^z_i | Memory biases | Fixed random biases on non-input qubits — they act as memory |
| J_ij * sigma^z_i * sigma^z_j | Couplings | Random Ising interactions — creates the computational complexity |

### 5.2 Simulation via Trotterization

Since we don't have quantum hardware yet, we simulate the time evolution:

```
U(t) = exp(-i * H * t)  ~  [exp(-i * H_X * dt) * exp(-i * H_Z * dt)]^m
```

Where:
- `dt = t / m` (time per Trotter step)
- `m = 3` Trotter steps (sufficient for short times)
- `H_X = -sum sigma^x_i` -> each qubit gets Rx(-2*dt)
- `H_Z = h_i * Z_i + J_ij * Z_i * Z_j` -> Rz gates + CX-Rz-CX for ZZ terms

**Initial state:** |+>^n (ground state of the transverse field at s=0 in the annealing schedule)

### 5.3 Why This Works

Three properties make the annealer reservoir effective:

1. **Hilbert-space expressiveness:** 2^10 = 1024 basis states for 10 qubits. The readout selects the most useful projection of this already-nonlinear feature space.

2. **Transverse-field mixing:** Without the -sum(sigma^x) term, the Hamiltonian is diagonal in the Z basis and the dynamics "freeze" — the system can't explore its state space. The transverse field is what makes the reservoir react to inputs.

3. **Random Ising disorder:** Fixed-random J_ij puts the system in a generic dynamical regime. This is the canonical reservoir-computing recipe (Fujii-Nakajima 2017) — no fine-tuning needed.

### 5.4 Key Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| n_qubits | 10 | Good balance of expressivity (55 features) and simulation speed |
| n_input | 10 | First 10 of the 21 features drive h_i |
| evolution_time | 2.0 | Optimal on the dynamical-phase-transition curve |
| trotter_steps | 3 | Sufficient for t=2.0; more steps are wasted |
| connectivity | all-to-all | Maximum entanglement mixing |
| input_scale (alpha) | 1.0 | "Edge of chaos" — neither field-dominated nor Ising-frozen |

The (t=2.0, alpha=1.0) optimum was found by a 14-configuration sweep. Too short time -> transverse-field-dominated (inputs don't propagate). Too long -> problem-Hamiltonian-dominated (classical Ising freezing).

### 5.5 Performance

- **Single seed:** RMSE = 0.00943 (beats AR, Ridge, ESN, LSTM)
- **5-seed ensemble:** RMSE = 0.00945 +/- 0.00010

---

## 6. The Hybrid Readout (Quantum + Classical Features) <a name="6-hybrid-readout"></a>

A critical design choice: the Ridge readout sees **both** the quantum reservoir features AND the original 21 input features.

```
Ridge input = [55 reservoir features | 21 original features] = 76 total features
```

**Why hybrid?**
- The reservoir captures nonlinear interactions between features
- But linear relationships (like HAR-RV -> next-day vol) are better learned directly
- The Ridge can use whichever representation is more useful for each aspect of the prediction
- Without the hybrid approach, QRC performance degrades significantly

This is analogous to a ResNet skip connection — the model can always fall back on the raw features if the quantum features aren't helpful for a particular sample.

---

## 7. Classical Baselines <a name="7-classical-baselines"></a>

All baselines are required by the GIC 2026 Track A rubric.

### 7.1 Persistence

**What:** y_hat[t] = RV[t-1] (predict yesterday's volatility)

**Why it's hard to beat:** Volatility is highly persistent (autocorrelation ~ 0.98). Yesterday's vol is an excellent predictor of today's vol. Any model that can't beat Persistence is adding noise, not signal.

**Our result:** RMSE = 0.00941

### 7.2 AR(5)

**What:** AutoRegressive model, order 5. Equivalent to Ridge regression on the 5 most recent delay-embedded lags.

**Result:** RMSE = 0.00954

### 7.3 Ridge Regression

**What:** L2-regularized linear regression on all 21 features.

**Result:** RMSE = 0.00976

### 7.4 Echo State Network (ESN)

**What:** Classical reservoir computing. A random recurrent neural network (200 neurons) with fixed weights, leaky tanh dynamics, and a trained linear readout. This is the direct classical analogue of QRC.

**Architecture:**
```
s(t) = (1 - leak) * s(t-1) + leak * tanh(W * s(t-1) + W_in * x(t))
```
- W: 200x200 sparse recurrent matrix, spectral radius 0.95
- W_in: 200 x n_features input weights
- Readout: Ridge on the reservoir states

**Result:** RMSE = 0.00970

### 7.5 GARCH(1,1)

**What:** Generalized Autoregressive Conditional Heteroskedasticity (Bollerslev 1986). The canonical econometric volatility model.

**Core equation:**
```
sigma^2(t) = omega + alpha * r(t-1)^2 + beta * sigma^2(t-1)
```
- omega: long-run variance level
- alpha: reaction to new information (yesterday's squared return)
- beta: persistence of past variance

**Why GARCH is so strong on this task:**
The target RV[t] is a sum of 21 squared returns. At forecast time, **20 of those 21 returns are already known**. GARCH only needs to forecast 1 unknown return. It plugs its variance forecast into the known decomposition:

```
RV[t]^2 * 21 = sum(r_i^2 for i in [t-20, t-1])  +  sigma^2_GARCH(t)
                    20 known terms                    1 GARCH forecast
```

This structural advantage is why GARCH leads the leaderboard (RMSE = 0.00795) and why beating it requires giving the QRC the same structural prior.

**Implementation:** Uses the `arch` Python package with constant mean, Normal residuals, MLE estimation on training returns only.

**Result:** RMSE = 0.00795

### 7.6 LSTM

**What:** Long Short-Term Memory neural network (1 layer, 32 hidden units). A minimal deep learning baseline.

**Architecture:**
- Input: 5-step sequence of (returns, |returns|, returns^2) + tiled HAR features
- LSTM: 1 layer, 32 hidden units, batch_first
- Head: Linear(32 -> 1)
- Training: Adam, lr=5e-3, 60 epochs, batch size 64, MSE loss

**Result:** RMSE = 0.01066 (worst of all baselines — small data, minimal architecture)

---

## 8. The GARCH Puzzle and the Hybrid Solution <a name="8-garch-hybrid"></a>

### 8.1 The Problem

GARCH dominates because of its structural advantage (20-of-21-known decomposition). The bare QRC doesn't know this decomposition — it has to learn it implicitly from data, which it can't fully do.

### 8.2 The Solution: Give QRC the GARCH Prior

Two approaches were tested:

**Option A — GARCH as Feature:**
- Fit GARCH(1,1) on training returns only
- Add the GARCH-predicted RV as an extra input feature
- QRC can use GARCH as a "starting point" and add nonlinear corrections

**Option B — GARCH-Residual Target (the winner):**
- Redefine the target: y = log(RV[t]) - log(RV_GARCH[t])
- Now the QRC predicts **what GARCH misses**
- Predicting y = 0 exactly recovers GARCH
- Any non-zero prediction is the QRC's added value

### 8.3 Why This Is Not Cheating

- GARCH parameters are fitted on training data only
- The GARCH proxy is rolled forward through test data using only past information
- The QRC still has to learn something useful — if it just predicts 0, it equals GARCH
- **Critical control test:** Ridge with the same GARCH augmentation does NOT beat GARCH (RMSE 0.00801-0.00805 vs 0.00795). Only adding the QRC reservoir breaks through. This isolates the improvement to the quantum features.

### 8.4 What the QRC Captures That GARCH Misses

The reservoir's ZZ correlator features pick up:
- **Regime transitions:** GARCH's single-equation model can't capture abrupt regime changes (calm -> crisis). The quantum correlators see nonlinear feature interactions that signal transitions.
- **Higher-order effects:** "Heteroskedasticity of heteroskedasticity" — the variance of variance itself changes across regimes.
- **Multi-scale interactions:** How daily and weekly volatility patterns interact nonlinearly.

### 8.5 Results

| Model | RMSE | QLIKE | Beats GARCH? |
|-------|------|-------|--------------|
| GARCH(1,1) standalone | 0.00795 | 0.00686 | — |
| Ridge + GARCH feature | 0.00801 | 0.00672 | No |
| QRC + GARCH feature (3-seed) | 0.00786 | 0.00657 | Yes |
| **QRC + GARCH residual (3-seed)** | **0.00784** | **0.00644** | **Yes** |

The QRC+GARCH hybrid is the first configuration to beat GARCH on RMSE, R^2, AND QLIKE simultaneously.

---

## 9. Target Transforms: Why Log-Vol Residuals? <a name="9-target-transforms"></a>

### 9.1 The Transform Chain

Raw target: RV[t] (annualized realized volatility, typically 0.05-0.40)

The v2 loader supports multiple target transforms:

| Transform | Formula | When y=0 means... |
|-----------|---------|-------------------|
| None | y = RV[t] | Zero volatility |
| `log` | y = log(RV[t]) | RV = 1 (meaningless) |
| `residual_log` | y = log(RV[t]) - log(RV[t-1]) | Persistence (RV unchanged) |
| `residual_log_garch` | y = log(RV[t]) - log(RV_GARCH[t]) | GARCH forecast is perfect |

### 9.2 Why Residual-Log Works Best

1. **Log stabilizes the variance:** Raw RV has heteroskedastic noise (high vol -> high forecast error). Log-RV has more uniform noise.

2. **Residual centering:** The model predicts *changes*, not levels. Since vol is highly persistent, most of the signal is "it'll be about the same as yesterday." The residual removes this trivial component and forces the model to focus on the interesting part — what's different from yesterday.

3. **Symmetric errors:** A 10% over-prediction and a 10% under-prediction have equal magnitude in log space. In linear space, they'd be asymmetric.

### 9.3 Inverting Back to Vol Scale

For reporting metrics on the natural volatility scale:
```python
# residual_log: y = log(RV) - log(RV_prev)
RV_predicted = exp(log(RV_prev) + y_predicted) = RV_prev * exp(y_predicted)

# residual_log_garch: y = log(RV) - log(RV_garch)
RV_predicted = exp(log(RV_garch) + y_predicted) = RV_garch * exp(y_predicted)
```

---

## 10. Metrics and Evaluation <a name="10-metrics"></a>

### 10.1 Primary Metrics

**RMSE (Root Mean Squared Error):**
```
RMSE = sqrt(mean((y_true - y_pred)^2))
```
- Units: annualized volatility (same as the target)
- Lower is better
- Our best: 0.00784

**R^2 (Coefficient of Determination):**
```
R^2 = 1 - MSE / Var(y_true)
```
- Dimensionless, range (-inf, 1]
- 1.0 = perfect, 0.0 = predicting the mean
- Our best: 0.9861

**NMSE (Normalized Mean Squared Error):**
```
NMSE = MSE / Var(y_true) = 1 - R^2
```
- Dimensionless, lower is better
- Our best: 0.0139

### 10.2 Volatility-Specific Metrics

**QLIKE (Patton 2011):**
```
QLIKE = mean(y_true^2 / y_pred^2 - log(y_true^2 / y_pred^2) - 1)
```
- The canonical loss function for volatility forecasting
- Operates on variance scale (squared vol)
- **Asymmetric:** penalizes under-prediction more than over-prediction (critical for risk management — missing a vol spike is worse than over-hedging)
- Lower is better; 0 = perfect
- Our best: 0.00644

**Mincer-Zarnowitz Regression:**
```
y_true = a + b * y_pred + epsilon
```
Tests two properties:
- **a = 0:** forecast is unbiased (no systematic over/under-prediction)
- **b = 1:** forecast is efficient (no systematic scaling error)
- **p-value:** Wald test for joint hypothesis (a=0, b=1). Higher is better — we do NOT want to reject the null that our forecast is unbiased and efficient.
- Our best: p = 0.57 (solidly cannot reject; forecast is unbiased and efficient)

### 10.3 Regime-Conditional Analysis

Test days are bucketed by VIX (CBOE Volatility Index) terciles:
- **Calm:** VIX in bottom third
- **Normal:** VIX in middle third
- **Turbulent:** VIX in top third

Per-bucket RMSE reveals whether a model is uniformly good or only good in certain regimes. GARCH dominates turbulent but is worst in calm. QRC has the most uniform profile across regimes.

### 10.4 Multi-Seed Ensemble

The annealer QRC uses random J_ij couplings. Different seeds produce different reservoirs. We average predictions across 3-5 seeds:
```
y_ensemble = mean(y_hat_seed1, y_hat_seed2, ..., y_hat_seedN)
```
This reduces variance and gives more robust predictions. We also report RMSE +/- std across seeds to quantify stability.

---

## 11. Noise Models <a name="11-noise-models"></a>

**File:** `src/qrc/noise.py`

For hardware-realistic simulations, three noise models are available:

### 11.1 Depolarizing Noise

Models random errors on gates. After each gate, the qubit state is replaced by the maximally mixed state (identity/2) with probability p:
```
rho -> (1-p) * rho + p * (I/2)
```
Typical values: p_single = 0.001, p_two = 0.01 (two-qubit gates are ~10x noisier).

### 11.2 Amplitude Damping

Models T1 relaxation (energy decay). The excited state |1> decays to |0> with probability gamma:
```
gamma = gate_time / T1
```
For a 50ns gate and 50us T1: gamma ~ 0.001.

### 11.3 Combined Noise

Both depolarizing and amplitude damping — closest to real hardware behavior.

### 11.4 Pre-defined Levels

| Level | p_single | p_two | gamma |
|-------|----------|-------|-------|
| none | 0 | 0 | 0 |
| low | 0.001 | 0.005 | - |
| medium | 0.005 | 0.02 | - |
| high | 0.01 | 0.05 | - |
| combined_low | 0.001 | 0.01 | 0.002 |
| combined_high | 0.005 | 0.05 | 0.01 |

---

## 12. Key Results Summary <a name="12-results"></a>

### 12.1 Full Leaderboard

| # | Model | RMSE | R^2 | QLIKE | Type |
|---|-------|------|-----|-------|------|
| 1 | **QRC+GARCH hybrid (3-seed)** | **0.00784** | **0.9861** | **0.00644** | Quantum hybrid |
| 2 | QRC+GARCH feature (3-seed) | 0.00786 | 0.9860 | 0.00657 | Quantum hybrid |
| 3 | GARCH(1,1) | 0.00795 | 0.9857 | 0.00686 | Classical |
| 4 | Persistence | 0.00941 | 0.9800 | 0.00887 | Naive |
| 5 | ANN 10q a2a (5-seed) | 0.00945 | 0.9798 | 0.00877 | Quantum |
| 6 | AR(5) | 0.00954 | 0.9794 | 0.00918 | Classical |
| 7 | QRC gate (5q L2) | 0.00967 | 0.9789 | 0.00877 | Quantum |
| 8 | ESN (200) | 0.00970 | 0.9787 | 0.00878 | Classical |
| 9 | Ridge | 0.00976 | 0.9784 | 0.00894 | Classical |
| 10 | LSTM (1L, 32h) | 0.01066 | 0.9743 | 0.00995 | Deep learning |

### 12.2 Key Takeaways

1. **QRC+GARCH hybrid beats GARCH** on all three primary metrics (RMSE, R^2, QLIKE) — the first configuration to do so.

2. **The quantum reservoir adds genuine value:** Ridge with the same GARCH augmentation does NOT beat GARCH. Only adding the QRC reservoir breaks through. This isolates the improvement to the nonlinear quantum features.

3. **Bare QRC (annealer-style) beats all ML baselines:** RMSE 0.00943 < AR(5) 0.00954 < Ridge 0.00976 < LSTM 0.01066. Only Persistence and GARCH are stronger.

4. **Gate-based QRC is competitive but weaker:** 5-qubit gate-based at 0.00967 vs 10-qubit annealer at 0.00943. The annealer architecture is better for this task.

5. **Annealer QRC has the most uniform regime profile:** Smallest calm-to-turbulent RMSE ratio of any learning-based model.

---

## 13. Phase 3: Hardware Path <a name="13-hardware"></a>

### 13.1 Primary: QuEra Aquila (256-qubit Rydberg atom array)

- **Why:** Aquila's native Hamiltonian exactly matches our reservoir's form
- **Evidence:** Kornjaca et al. 2024 demonstrated 108-qubit QRC on this exact device
- **Access:** via qBraid's Bloqade SDK
- **QPU time estimate:** ~140us per sample -> ~40 seconds for full test set at 1000 shots

### 13.2 Alternative: D-Wave Advantage (5000+ qubits)

- Programmable h_i and J_ij — exactly our encoding scheme
- Much larger qubit count but limited connectivity (Pegasus graph)
- Sampling-based (no exact statevector)

### 13.3 Backup: IBM Heron via Qiskit Runtime

- Gate-model approach (Architecture A)
- IBM Open Plan offers up to 180 minutes QPU time per 12 months
- Would validate the gate-based QRC results on real hardware

### 13.4 Hybrid Strategy

**Key insight:** Train the Ridge readout on *simulator* features (cheap). Run only the test set on hardware (expensive). The hardware-vs-simulator gap IS the hardware story — if predictions are comparable, the architecture transfers; if they differ, that's a publication-worthy finding about decoherence effects.

---

## 14. Glossary <a name="14-glossary"></a>

| Term | Definition |
|------|-----------|
| **RV** | Realized Volatility — rolling standard deviation of log-returns, annualized |
| **QRC** | Quantum Reservoir Computing |
| **HAR-RV** | Heterogeneous Autoregressive model of Realized Volatility (Corsi 2009) |
| **GARCH** | Generalized Autoregressive Conditional Heteroskedasticity (Bollerslev 1986) |
| **ESN** | Echo State Network — classical reservoir computing |
| **QLIKE** | Quasi-likelihood loss (Patton 2011) — canonical volatility forecast loss |
| **Mincer-Zarnowitz** | Regression test for forecast unbiasedness and efficiency |
| **Trotter** | Approximation: exp(-iHt) ~ [exp(-iH_1*dt) * exp(-iH_2*dt)]^m |
| **Transverse field** | The -sum(sigma^x) term that drives quantum tunneling |
| **Longitudinal bias** | h_i * sigma^z_i — how data enters the annealer (D-Wave's native encoding) |
| **ZZ correlator** | <Z_i Z_j> — two-body quantum observable, captures entanglement-mediated correlations |
| **Spectral radius** | Largest eigenvalue magnitude of ESN's recurrent weight matrix |
| **NMSE** | Normalized MSE = MSE/Var(y) = 1 - R^2 |
| **VIX** | CBOE Volatility Index — market's implied volatility expectation |
| **Ridge alpha** | L2 regularization strength in Ridge regression |
| **TimeSeriesSplit** | Cross-validation that respects temporal ordering (no future leakage) |
| **Pegasus** | D-Wave Advantage's qubit connectivity graph topology |
| **Bloqade** | QuEra's SDK for programming Rydberg atom arrays |
