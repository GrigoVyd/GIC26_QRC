@echo off
echo === GIC26 Quantum Reservoir Computing Setup ===
echo.

REM Create virtual environment
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create venv. Ensure Python 3.10+ is installed.
    exit /b 1
)

echo [1/3] Virtual environment created.

REM Activate and upgrade pip
call venv\Scripts\activate
python -m pip install --upgrade pip --quiet

echo [2/3] pip upgraded.

REM Install dependencies
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    exit /b 1
)

echo [3/3] Dependencies installed.
echo.
echo ============================================================
echo  Setup complete!
echo  Activate with:  call venv\Scripts\activate
echo  Run benchmarks: python run_all.py
echo  Run single:     python experiments\mnist_qrc.py
echo ============================================================
