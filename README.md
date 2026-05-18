# GIC 2026 — Quantum Reservoir Computing: Market Volatility Forecasting

Quantum Reservoir Computing (QRC) applied to next-day realized volatility
prediction for SPY (S&P 500 ETF). This repo is focused exclusively on
**Track 1: Dynamic Systems Forecasting — Financial Volatility**.

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
