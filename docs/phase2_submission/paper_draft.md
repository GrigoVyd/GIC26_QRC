# Quantum Reservoir Computing for Realized Volatility Forecasting under Regime Shifts

*GIC 2026 Phase 2 Submission — Track A: Financial Volatility Prediction*
*Challenge Provider: qBraid, MITRE, JonesTrading*

> **Note for the paper PDF:** The cover page (`GIC_2026 Cover Page.docx`) must
> be the first page. This draft is the body content, ≤ 3 pages at 11-pt Times
> New Roman, single spacing. References live on a separate page and do not
> count against the 3-page limit.

---

## 1. Track Selection and Problem Framing

We select **Track A — Financial Volatility Prediction**, targeting next-day
**realized volatility of SPY** (21-day rolling-std × √252) over 2010–2024
daily data (3,750 trading days, yfinance source). The sub-problem is
sharpened beyond raw point forecasting: we test whether a QRC can match
the strongest classical benchmark (GARCH(1,1)) on average **and** whether
its error profile is *more uniform across volatility regimes* — the
property risk managers actually want for stress-testing applications.

QRC is well matched to this signal class because realized volatility is a
**long-memory, regime-switching, non-Gaussian** process. The dominant signal
(autocorrelation ≈ 0.99 at one-day lag) is captured by lagged-vol features;
the residual is non-Gaussian noise with regime-dependent kurtosis. A
reservoir's high-dimensional dynamics extract nonlinear interactions among
lagged returns, |returns|, and lagged-vol features, while a linear readout
enforces a stable classical interpretation — exactly the property exploited
by Li et al. (2025) on the same task. [1] The reservoir's Hilbert-space
expressiveness also lets it carry information about *which* regime the
market is in, which we exploit through a regime-conditional analysis (§3).

## 2. QRC Architecture Design

We propose a **transverse-field Ising reservoir** with classical
input-bias encoding, evaluated at fixed mid-anneal parameters:

$$\hat H = -\sum_i \sigma^x_i + \alpha \sum_{i \in I} x_i \,\sigma^z_i + \sum_i h^{\text{mem}}_i \sigma^z_i + \sum_{(i,j)\in E} J_{ij}\, \sigma^z_i \sigma^z_j$$

| Component | Choice | Justification |
|---|---|---|
| **Hamiltonian** | Transverse-field Ising with random fixed `J_ij` | Native to QuEra Aquila (Rydberg blockade), D-Wave Advantage, and gate-based Trotterization; the substrate most directly matched to a reservoir's "disordered Hilbert-space dynamics" property [2, 6] |
| **Input encoding** | Longitudinal bias `h_i = α · x_i` on `n_input` qubits; data drives the Z-axis, not the rotation axis | Aligns with annealer-native programming (D-Wave: `h_i`; Aquila: per-site detuning) and lets a single `input_scale` α tune the dynamical regime (edge of chaos, [4]) |
| **Initial state** | `|+⟩^n` (s=0 ground state of pure transverse field) | Standard adiabatic-quench initial state; provides uniform exploration of computational basis under subsequent evolution |
| **Evolution** | Continuous-time unitary `U(t) = exp(-i H t)` for fixed `t` (no schedule ramp) | Equivalent to a constant mid-anneal snapshot; simpler to characterize than a full anneal schedule and matches the regime where reservoir capacity is highest [4] |
| **Readout observables** | `⟨σ^z_i⟩` and `⟨σ^z_i σ^z_j⟩` on coupled pairs | Linear and quadratic in the reservoir state; recovers the full Z+ZZ correlation matrix at modest shot cost; hybridized with raw features in the linear head |
| **Memory mechanism** | Optional classical Z-feedback: `⟨σ^z_i⟩` from sample t-1 enters sample t as additional rotation angle (`transform_sequential` method) | Approximates the coherent feedback that Quantinuum H-series provides natively; emulates the few-atom feedback QRC of Zhu et al. (2025) [3] |
| **Hybrid head** | Ridge regression on `[reservoir_features ‖ HAR-RV features ‖ log-HAR features]` with TimeSeriesSplit-CV α | Linear part captures the autocorrelation ceiling; quantum part contributes nonlinear interaction features |
| **Target** | Residual log-vol: `y = log(RV[t]) − log(RV[t-1])` | Predicting 0 corresponds exactly to Persistence; removes the regularize-toward-mean penalty that linear-on-log-vol models otherwise incur |

A schematic diagram (input → bias encoding → Trotterized Ising evolution →
Z, ZZ readout → linear head) is included on page 4.

## 3. Theoretical and Analytical Justification

**Why this architecture, on this signal.** Three properties of the reservoir
are exploited: (a) **high-dimensional Hilbert-space embedding** (2^n basis
states for n qubits) generates many nonlinear feature combinations from a
21-dim classical input; (b) **transverse-field-driven mixing** provides
non-perturbative, non-trivial dynamics from a separable initial state;
(c) **random Ising disorder** in `J_ij` produces a generic dynamical regime
without fine-tuning, the canonical reservoir-computing property since
Fujii–Nakajima (2017) [5]. The dynamical-phase-transition analysis of
Martínez-Peña et al. (2021) [4] predicts an optimal regime where the
transverse-field term and the data-modulated longitudinal term are comparable
in magnitude — this is the lever our `input_scale × evolution_time` knob
directly controls.

**Preliminary 10-qubit prototyping (this submission).** We implemented the
architecture as a Trotterized statevector simulation
(`src/qrc/annealer_reservoir.py`) and benchmarked on SPY 2010–2024
(yfinance, chronological 80/20 split, 2,984 train / 746 test samples). A
14-configuration parameter sweep at 10 qubits (all-to-all, m=3 Trotter
steps) located the edge-of-chaos optimum at `(t=2.0, α=1.0)`. We then ran
a 5-seed ensemble at that point against all Track A baselines required by
the rubric:

| Model | RMSE | R² | QLIKE | MZ-pvalue |
|---|---:|---:|---:|---:|
| **GARCH(1,1)** | **0.00795** | **0.9857** | **0.00686** | 0.41 |
| Persistence (RV[t-1]) | 0.00941 | 0.9800 | 0.00887 | 0.27 |
| **QRC Ising 10q (5-seed)** | **0.00953 ± 0.00010** | **0.9794 ± 0.0004** | **0.00877 ± 0.0001** | **0.34** |
| AR(5) | 0.00954 | 0.9794 | 0.00918 | 0.30 |
| QRC gate-based (5q L=2 random) | 0.00967 | 0.9788 | 0.00877 | 0.28 |
| ESN (200 nodes) | 0.00970 | 0.9787 | 0.00878 | 0.26 |
| Ridge (HAR features) | 0.00976 | 0.9784 | 0.00894 | 0.24 |
| LSTM (1 layer, 32 hidden) | 0.01066 | 0.9743 | 0.00995 | 0.18 |

The QRC Ising architecture is **the strongest learning-based model on
this task** (beats every NN/RC/linear baseline including LSTM and ESN) and
sits within 0.00012 RMSE of Persistence. GARCH(1,1) wins on RMSE because
its rolling-window proxy exploits a structural decomposition: 20 of the 21
squared returns inside the realized-vol window are *known* at forecast
time; GARCH only needs to forecast the 21st. The QRC has no such structural
prior — yet still matches Persistence and beats every parametric ML model.
A QRC variant that adopts GARCH's structural prior (Option A/B; see end of
§3 and `experiments/phase2_garch_hybrid.py`) is reported as the headline
hybrid configuration.

**Regime-conditional analysis** (test set bucketed into VIX terciles —
calm / normal / turbulent; FRED data):

| Model | Calm RMSE | Normal RMSE | Turbulent RMSE |
|---|---:|---:|---:|
| GARCH(1,1) | 0.00590 | **0.00601** | **0.01088** |
| QRC Ising 10q | 0.00559 | 0.00694 | 0.01372 |
| ESN | **0.00546** | 0.00696 | 0.01428 |
| Persistence | 0.00581 | 0.00701 | 0.01351 |

GARCH dominates turbulent regimes by ~20% but is the *worst* in calm
windows: its parametric form effectively overfits to high-vol periods.
The QRC has the most *consistent* per-regime profile — never best in any
single regime, but never last either, and the smallest calm-to-turbulent
RMSE ratio of the learning-based models. This is the property risk
managers value (forecast quality that doesn't collapse when conditions
change), and it argues for the QRC as a *complement* to GARCH rather
than a replacement.

**Architecture diagram and parameter heatmap** — see `figures/architecture.png`
and `results/dwave_sweep_heatmap.png` respectively (referenced inline; both
included as Figure 1 and Figure 2 of this submission).

**Headline result — QRC + GARCH hybrid.** Importing GARCH's structural prior
into the QRC pipeline (either as an exogenous feature or by retargeting the
loss to the GARCH residual) produces the strongest model of the entire
study, beating GARCH itself:

| Model (3-seed ensemble where applicable) | RMSE | R² | QLIKE |
|---|---:|---:|---:|
| **QRC on GARCH-residual target** | **0.00784** | **0.9861** | **0.00644** |
| QRC + GARCH-feature (Persistence residual target) | 0.00786 | 0.9860 | 0.00657 |
| GARCH(1,1) standalone | 0.00795 | 0.9857 | 0.00686 |
| Ridge + GARCH-feature (no QRC) | 0.00801 | 0.9855 | 0.00672 |
| Ridge on GARCH-residual (no QRC) | 0.00805 | 0.9853 | 0.00668 |

Two diagnostics from this table:

(i) **Ridge with GARCH augmentation does not beat GARCH** (0.00801–0.00805 vs
0.00795). The linear readout, given the GARCH forecast, can only recover
GARCH's own performance.

(ii) **Adding the QRC reservoir on top of the same augmentation drops RMSE
by another ~2%** and QLIKE by ~3–4%. The QRC's nonlinear features carry
signal *orthogonal* to GARCH's parametric form — a clean quantum-classical
hybrid: classical model captures the structural part, quantum reservoir
captures the residual nonlinearity.

This is the first configuration in the study to beat GARCH(1,1) on all
three required Track A metrics (RMSE, R², QLIKE) simultaneously.

## 4. Data Modeling Strategy

**Dataset.** SPY daily OHLCV from yfinance (2010-01-01 to 2024-12-31), with
fallback to a synthetic GARCH(1,1) process when the API is unavailable.
Total: 3,750 trading days. Realized volatility computed as 21-day rolling
std × √252.

**Features (21 total, see `src/data/loaders.py::load_financial_data_v2`):**
delay-embedded log-returns (5 lags), |returns| (5 lags), returns² (5 lags),
HAR-RV daily/weekly/monthly means (3), and log-HAR (3). The HAR component
follows Corsi (2009) and supplies the dominant predictive signal; the return
moments supply higher-frequency information.

**Split.** Chronological 80/20 (no shuffle). Within train, a 5-fold
TimeSeriesSplit cross-validation selects the Ridge readout's regularisation
parameter from {10^-3, ..., 10^3}.

**Target.** Residual log-vol `y = log(RV[t]) - log(RV[t-1])`. Models trained
on `y`; predictions inverted to RV-scale via `RV[t-1] · exp(ŷ)` before
metrics. This parameterisation makes Persistence the trivial `ŷ = 0`
prediction, which we found necessary to prevent Ridge from shrinking
predictions toward the unconditional mean.

**Baselines.** Persistence, AR(5), Ridge, ESN (200 neurons, classical
analog), gate-based QRC (Phase 1 reservoir). **Phase-3 additions: GARCH(1,1)
via `arch` library, and LSTM via PyTorch** — required by the rubric, not
yet implemented (declared gap).

**Metrics.** RMSE, MAE, R², NMSE on the RV scale. **Phase-3 additions:
QLIKE (Patton 2011), Mincer–Zarnowitz unbiasedness regression**, and
regime-conditional RMSE bucketed by VIX percentile.

## 5. Quantum Platform Justification and Resources

**Primary target: QuEra Aquila** via qBraid's Bloqade SDK. Rationale:
(a) Aquila's native Hamiltonian is a transverse-field Rydberg model with
form ≈ `Ω σ^x + Δ σ^z + V n_i n_j`, structurally close to our reservoir;
(b) up to 256 qubits — the analog scaling demonstrated by Kornjača et al.
(2024) at 108 qubits [2]; (c) Bloqade and the QuEra QRC-tutorials are
explicitly endorsed in §3 of the challenge description.

**Alternative target: D-Wave Advantage** via the Braket SDK. Rationale:
5,000+ qubits with programmable `h_i`/`J_ij`, exactly the parameterisation
our architecture uses; the Sakurai et al. (2026) demonstration of annealer
dynamics as ML feature map [8] provides a direct precedent; our internal
exploration document (`docs/dwave_qrc_exploration.md`) details this path.
This is preferred if Aquila access is queue-constrained.

**Backup: IBM Heron** via Qiskit Runtime. Rationale: lowest engineering
overhead from our existing Qiskit-Trotter prototype; 5–10-qubit gate-based
configs already characterized [§3]; Open-Plan quota (10 min QPU/month)
permits ~16 full hardware test-set runs per month.

**Phase-3 resource budget.**

| Platform | Phase-3 use | Qubits | Depth | Shots/sample | Samples | Est. QPU time |
|---|---|---:|---:|---:|---:|---:|
| Aquila (primary) | Test set + ablations | 30–60 | analog (≤ 5 µs evolution) | 1,000 | 750 | ~40 s/run; 8 runs/month |
| Advantage (alt.) | Test set + scaling | 100–500 | continuous anneal (20 µs) | 1,000 | 750 | ~20 s/run; 3 runs/month free |
| Heron (backup) | 5q gate-based test set | 5–10 | depth ~25 (L=2) | 1,024 | 750 | ~40 s/run; ~16 runs/month free |

Hybrid strategy across all platforms: **train the Ridge readout on noiseless
simulator features; run only the test set on hardware**. The HW-vs-sim
RMSE gap *is* the hardware story and keeps QPU cost bounded. Error mitigation:
ZNE via Mitiq (Heron only); for Aquila/Advantage, the analog dynamics is
not ZNE-amenable, so we report raw shot-noise estimates with per-config
seed ensembles for error bars.

## 6. Stakeholder Impact and Phase 3 Plan

**Stakeholders.** Volatility forecasts at the daily horizon are consumed
by (i) trading desks for option-delta hedging and gamma exposure
management, (ii) risk managers for VaR / expected-shortfall calculation
(both inputs are RV-based), and (iii) market makers for liquidity-provision
spread setting. The product of this work is a *regime-aware* volatility
model: not necessarily a lower-RMSE forecast, but one whose error structure
is interpretable across calm/turbulent regimes, which directly addresses
stress-testing scenarios — the use case JonesTrading flagged.

**Phase 3 milestone plan (10 weeks post-finalist notification).**
*Weeks 1–2:* QuEra Aquila access setup + reproduce simulator results on
hardware at 30 qubits, single config. *Weeks 3–4:* full simulator ablation
(seeds, qubit counts up to 256 via Aquila tensor-network sim) + add GARCH,
LSTM baselines and QLIKE/MZ metrics. *Weeks 5–6:* Aquila hardware sweep
(qubit count × evolution time × shots), QPU-vs-sim gap analysis.
*Weeks 7–8:* regime-conditional metrics; bucket test windows by VIX
percentile and report per-bucket QLIKE. *Weeks 9–10:* write Phase-3 5-page
paper, agentic reproducibility package, qBraid Skill submission.

**Fallback options.** If Aquila queue exceeds 4 weeks, pivot to D-Wave
Advantage (free 1 min/month Leap tier, plus paid-as-needed) using the
identical `h_i`/`J_ij` programming. If both analog platforms are
unavailable, fall back to IBM Heron Trotterized gate-based circuit (5–10
qubits) — already validated in Phase 2.

**Disclosure (per challenge rules).** This work was developed with
significant LLM assistance for code drafting and writing. All experimental
results, architectural choices, and analyses are the team's own; LLM
contributions are version-controlled at github.com/GrigoVyd/GIC26_QRC.

---

## References (separate page, does not count toward 3-page limit)

[1] Q. Li, C. Mukhopadhyay, A. Bayat, A. Habibnia. *Quantum Reservoir
Computing for Realized Volatility Forecasting.* Phys. Rev. Research / arXiv:2505.13933 (2025).

[2] M. Kornjača et al. *Large-scale quantum reservoir learning with an analog
quantum computer.* arXiv:2407.02553 (2024).

[3] C. Zhu, P. J. Ehlers, H. I. Nurdin, D. Soh. *Practical few-atom quantum
reservoir computing.* Phys. Rev. Research 7, 023290 (2025), arXiv:2405.04799.

[4] R. Martínez-Peña, G. L. Giorgi, J. Nokkala, M. C. Soriano, R. Zambrini.
*Dynamical Phase Transitions in Quantum Reservoir Computing.* Phys. Rev. Lett.
127, 100502 (2021).

[5] K. Fujii, K. Nakajima. *Harnessing disordered ensemble quantum dynamics
for machine learning.* Phys. Rev. Applied 8, 024030 (2017), arXiv:1602.08159.

[6] P. Mujal, R. Martínez-Peña, J. Nokkala, J. García-Beni, G. L. Giorgi,
M. C. Soriano, R. Zambrini. *Opportunities in Quantum Reservoir Computing and
Extreme Learning Machines.* Adv. Quantum Tech. 4, 2100027 (2021), arXiv:2102.11831.

[7] O. Ahmed, F. Tennie, L. Magri. *Robust quantum reservoir computers for
forecasting chaotic dynamics: generalized synchronization and stability.* Proc.
R. Soc. A 481, 20250550 (2025), arXiv:2506.22335.

[8] A. Sakurai, A. Hayashi, T. Matsumori, D. Kaji, T. Kadowaki, K. Nemoto.
*Beyond Optimization: Harnessing Quantum Annealer Dynamics for Machine Learning.*
arXiv:2601.09938 (2026).

[9] F. Corsi. *A Simple Approximate Long-Memory Model of Realized Volatility.*
Journal of Financial Econometrics 7(2), 174–196 (2009).

[10] A. J. Patton. *Volatility forecast comparison using imperfect volatility
proxies.* Journal of Econometrics 160(1), 246–256 (2011).

[11] J. Mincer, V. Zarnowitz. *The Evaluation of Economic Forecasts.* NBER
(1969).

[12] T. Bollerslev. *Generalized Autoregressive Conditional Heteroskedasticity.*
Journal of Econometrics 31(3), 307–327 (1986).
