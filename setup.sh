#!/usr/bin/env bash
set -e

echo "=== GIC26 Quantum Reservoir Computing Setup ==="
echo

# Create virtual environment
python3 -m venv venv
echo "[1/3] Virtual environment created."

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet
echo "[2/3] pip upgraded."

# Install dependencies
pip install -r requirements.txt
echo "[3/3] Dependencies installed."

echo
echo "============================================================"
echo " Setup complete!"
echo " Activate with:  source venv/bin/activate"
echo " Run benchmarks: python run_all.py"
echo " Run single:     python experiments/mnist_qrc.py"
echo "============================================================"
