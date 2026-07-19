# GIC 2026 — Quantum Reservoir Computing: Market Volatility Forecasting

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_black.png" width="150" alt="Launch on qBraid">](https://account.qbraid.com?gitHubUrl=https%3A%2F%2Fgithub.com%2FGrigoVyd%2FGIC26_QRC.git&redirectUrl=notebooks%2Fphase3_qbraid_reproducibility.ipynb)

Quantum Reservoir Computing (QRC) applied to next-day realized volatility
prediction for SPY (S&P 500 ETF). This repo is focused exclusively on
**Track 1: Dynamic Systems Forecasting — Financial Volatility**.

## Phase 3 judge reproducibility

Open `notebooks/phase3_qbraid_reproducibility.ipynb` or use the Launch on
qBraid button above. A default **Run All** is credit-safe: it reads the compact
saved hardware artifacts, rebuilds the comparison table and plot, audits the
IQM/Aquila execution records, and runs a local 9-qubit statevector check. It
does not read tokens, contact a cloud provider, or submit a paid job.

The optional qBraid simulator smoke is disabled by default and contains a
zero-pricing assertion. Real-QPU reruns remain in guarded experiment scripts
with explicit opt-in and credit caps.

## Task

- **Target:** next-day annualised realized volatility (21-day rolling std × √252)
- **Features:** 5-day delay-embedded log-returns, absolute returns, and squared returns (15 features total)
- **Data:** SPY daily closes 2010–2024 via yfinance (falls back to synthetic GARCH if unavailable)
- **Split:** chronological 80/20 (no shuffle — avoids look-ahead bias)

## Models

| Model | Type |
|-------|------|
| Persistence | Baseline: y_hat = last observed |
| AR(5) | Autoregressive, Ridge readout |
| Ridge | Linear on delay-embedded features |
| ESN (200 neurons) | Classical reservoir computing |
| QRC 5q/10q × L2/L3 × random/all-to-all | **Quantum** reservoir computing |

QRC uses an Ising-type gate-based reservoir (Hadamard → Rx encoding → ZZ couplings → local rotations) with Z and ZZ observables as features. Ridge readout alpha is selected via TimeSeriesSplit cross-validation.

## Quick Start

```bash
# Setup
python -m venv venv
venv\Scripts\activate        # Windows
# or: source venv/bin/activate

pip install -r requirements.txt

# Run full sweep (5q + 10q, ~10–20 min)
python run_volatility.py

# Quick sanity check (5q/2L only, ~2 min)
python run_volatility.py --fast

# Include 15-qubit configs (slow)
python run_volatility.py --full
```

## Outputs

```
results/
├── financial_results.csv         # All models × metrics (RMSE, MAE, R², NMSE)
├── financial_rmse.png            # Horizontal RMSE bar chart
├── financial_predictions.png     # Full test-period overlay + scatter + rolling RMSE
└── financial_predictions_doc.png # Same at 1200 DPI for publication
```

## Project Structure

```
GIC26_QRC/
├── run_volatility.py          # Main entry point
├── experiments/
│   └── financial_qrc.py       # Full experiment: baselines + QRC sweep + plots
├── src/
│   ├── qrc/
│   │   ├── reservoir.py       # QuantumReservoir class
│   │   └── noise.py           # Depolarizing / amplitude-damping noise models
│   ├── data/
│   │   └── loaders.py         # load_financial_data() with enhanced features
│   └── baselines/
│       └── classical.py       # Persistence, AR, Ridge, ESN + metrics
└── requirements.txt
```

## Key Metrics

- **NMSE** (Normalized MSE = MSE / Var(y)) — primary; lower is better
- **RMSE** — annualised volatility units
- **R²** — coefficient of determination; higher is better
- **MAE** — mean absolute error

## Dependencies

- qiskit >= 1.1.0, qiskit-aer >= 0.14.0
- numpy, scipy, scikit-learn, pandas, matplotlib, seaborn
- yfinance >= 0.2.40
