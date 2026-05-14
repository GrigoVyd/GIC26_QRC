# GIC 2026 Quantum Reservoir Computing — Phase 2 Evaluation

**Competition:** [qBraid, MITRE & JonesTrading: GIC 2026](https://aqora.io/competitions/gic-2026-qBraid-MITRE-JonesTrading)  
**Track focus:** Dynamic Systems Forecasting — financial volatility & weather time series  
**Common benchmark:** MNIST digit classification  

---

## Quick start

```bash
# Windows
setup.bat
call venv\Scripts\activate

# Linux / macOS
bash setup.sh
source venv/bin/activate

# Run everything
python run_all.py

# Fast validation pass (~5 min)
python run_all.py --fast

# Single experiment
python experiments/mnist_qrc.py
python experiments/financial_qrc.py
python experiments/narma_benchmark.py
python experiments/weather_qrc.py
python experiments/scaling_noise.py
```

Results land in `results/` as CSV tables + PNG charts.

---

## Architecture

```
GIC26_QRC/
├── src/
│   ├── qrc/
│   │   ├── reservoir.py   # QuantumReservoir — core QRC class
│   │   └── noise.py       # depolarizing / amplitude-damping noise models
│   ├── data/
│   │   └── loaders.py     # MNIST, SPY volatility, NARMA, Lorenz, NOAA
│   └── baselines/
│       └── classical.py   # ESN, Ridge, AR, Persistence
├── experiments/
│   ├── mnist_qrc.py       # classification benchmark
│   ├── financial_qrc.py   # volatility forecasting (Track 1)
│   ├── weather_qrc.py     # chaotic forecasting (Track 2 / Lorenz)
│   ├── narma_benchmark.py # NARMA-5 / NARMA-10 standard RC benchmark
│   └── scaling_noise.py   # qubit scaling (5,10,15 q) + noise sweep
├── results/               # auto-created on first run
├── run_all.py             # master runner
├── requirements.txt
├── setup.bat  (Windows)
└── setup.sh   (Linux/Mac)
```

---

## QRC Design

### Reservoir circuit

```
|0…0>  →  H⊗n
          ┌──────────────────────────────────┐ × n_layers
          │  Angle encoding: Rx(πxᵢ) ∀ i    │
          │  ZZ Ising couplings (random graph)│
          │  Local rotations Rx Ry Rz (fixed) │
          └──────────────────────────────────┘
          Measure → <Zᵢ> and <ZᵢZⱼ>
```

**Reservoir weights are random and fixed** — only the linear readout is trained.

### Feature vector

| Observables | Count |
|-------------|-------|
| `<Zᵢ>`      | `n` |
| `<ZᵢZⱼ>`   | `n(n-1)/2` |
| **Total**   | **n + n(n-1)/2** |

For n=5 → 15 features, n=10 → 55, n=15 → 120.

### Input encoding

Delay embedding: input at time t = `[x(t-k+1), …, x(t)]` → angle-encoded across qubits.  
This gives the reservoir implicit nonlinear memory of order k without maintaining quantum state between steps.

### Readout

`Ridge(alpha)` regression (or `RidgeClassifier` for MNIST).  
Alpha cross-validated on the training split.

---

## Benchmarks evaluated

| Experiment | Metric | Baselines |
|------------|--------|-----------|
| MNIST classification | Accuracy | Ridge, SVM-RBF |
| Financial volatility (SPY) | RMSE, NMSE, R² | Persistence, AR, Ridge, ESN |
| NARMA-5 / NARMA-10 | NMSE | AR, Ridge, ESN-100, ESN-500 |
| Weather / Lorenz | RMSE, NMSE | Persistence, AR, Ridge, ESN |
| Qubit scaling (5,10,15) | NMSE vs n_qubits | — |
| Noise analysis | NMSE vs p_depol | — |

---

## Qubit configurations tested

| n_qubits | n_layers | Connectivity | Reservoir features |
|----------|----------|--------------|--------------------|
| 5  | 1,2,3 | random | 15  |
| 10 | 1,2,3 | random | 55  |
| 15 | 1,2,3 | random | 120 |

Noise models: `depolarizing_noise(p_single, p_two)`, `amplitude_damping_noise(gamma)`, `combined_noise(p_depol, gamma)`.

---

## Platform notes

- **Simulation backend:** Qiskit Aer `AerSimulator` (statevector exact mode for noiseless; density-matrix / shot mode for noise experiments).
- **Cloud:** Upload to qBraid as-is (venv is self-contained). IBM Quantum free plan supported by adding a `QiskitRuntimeService` wrapper around `_evaluate_qrc`.
- **Weather data:** Requires `meteostat` (`pip install meteostat`). Falls back to Lorenz-63 automatically.

---

## Document workflow (Phase 2 PDF preparation)

> ⚠️ **AI disqualification warning** — the competition rules prohibit AI authorship.  
> This workflow produces **data, figures, and tables only**. All document text must be written by the team.

```bash
# Full run — produces 8 PNGs + 3 Markdown tables (~30–40 min)
python run_doc_workflow.py

# Fast validation pass (~8–12 min, smaller datasets)
python run_doc_workflow.py --fast

# Skip shot-based noise sweep (saves ~10 min)
python run_doc_workflow.py --skip-noise

# Re-generate tables only from an existing strategy_comparison.csv
python run_doc_workflow.py --tables-only
```

All outputs land in `results/doc/`.

### Three QRC strategies compared

| Strategy | Hamiltonian / Encoding | Key property |
|----------|----------------------|--------------|
| **A — Ising ZZ** | ZZ couplings only | Baseline; fewest gates |
| **B — Heisenberg XX+YY+ZZ** | Full isotropic exchange | Richer entanglement; 3× more 2q gates |
| **C — IQP Encoding** | Degree-2 polynomial feature map → Ising reservoir | Captures input cross-terms with linear readout |

### Output assets and document sections

| File | Document section |
|------|-----------------|
| `results/doc/circuit_ising/heisenberg/iqp.png` | §2 Technical approach |
| `results/doc/pipeline_diagram.png` | §1 Focus area / §3 Data modelling |
| `results/doc/strategy_comparison.png` | §4 Quantum advantage |
| `results/doc/scaling_lines.png` | §4 Quantum advantage |
| `results/doc/noise_robustness.png` | §5 Platform justification |
| `results/doc/financial_predictions_doc.png` | §1 Focus area |
| `results/doc/table_strategy_summary.md` | §2 Technical approach |
| `results/doc/table_baselines_financial.md` | §3 Data modelling |
| `results/doc/table_qubit_scaling.md` | §5 Phase 3 scaling |

---

## Further data / next steps for Phase 3

| What is needed | Where to get it |
|----------------|-----------------|
| Higher-resolution volatility (tick data) | Bloomberg / Refinitiv |
| Regime-labelled equity data (bull/bear) | NBER, FRED |
| NOAA GHCND daily station data | `meteostat` or NOAA bulk download |
| Real IBM hardware runs | IBM Quantum Cloud (free tier, 127q Eagle) |
| Larger photonic QRC (>5 modes) | Quandela Cloud / Perceval |
| Mid-circuit measurement feedback | Qiskit 1.x `IfElseOp` |
