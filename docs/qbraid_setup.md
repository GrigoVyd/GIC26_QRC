# Running the QRC on qBraid quantum hardware (Phase 3)

This guide sets up real quantum-job execution of the quantum reservoir through
[qBraid](https://qbraid.com). It covers both ways to run — **qBraid Lab** (cloud,
recommended) and **local** — and how to go from a free simulator check to a real
QPU run without wasting credits.

## What gets demonstrated

The reservoir circuits that win in simulation (5-qubit, 2-layer, random Ising)
are submitted to a qBraid backend, the Z/ZZ features are rebuilt from shot
counts, and the volatility forecast is scored against the classical baselines.

The deliverables of a run:

| Artifact | Meaning |
|---|---|
| `results/qbraid_job_ids.json` | Real qBraid job ids — **proof of hardware execution**. Keep with the submission. |
| `results/qbraid_hardware_summary.csv` | RMSE / MAE / R² / QLIKE for Persistence, ESN, QRC-sim, QRC-hardware on the test window. |
| `results/qbraid_hardware_qrc.png` | Forecast overlay + reservoir feature fidelity (hardware vs statevector). |

**Honest framing.** In simulation QRC ≈ ESN ≈ Persistence (Persistence ~0.001
RMSE ahead). The hardware result is therefore stated as: *the quantum reservoir
runs as real jobs, its features survive device noise (high correlation with the
exact statevector), and the forecast stays competitive with the best classical
baselines.* Do not claim QRC beats every baseline.

## Files

- `src/qrc/hardware_backend.py` — qBraid executor + counts→features + bit-order calibration. `qbraid` is imported lazily, so the offline path works without it.
- `experiments/qbraid_hardware_qrc.py` — the experiment (CLI).
- `notebooks/qbraid_hardware_demo.ipynb` — guided run for qBraid Lab.

## Option A — qBraid Lab (recommended)

The SDK and credentials are preconfigured in Lab.

1. Sign in at [account.qbraid.com](https://account.qbraid.com) → **Lab**.
2. Open a terminal in Lab and clone this repo:
   ```bash
   git clone <your-repo-url> GIC26_QRC && cd GIC26_QRC
   pip install -r requirements.txt
   ```
3. Open `notebooks/qbraid_hardware_demo.ipynb` and run top to bottom. It does the
   Bell-state smoke test, the dry-run, then the free-simulator run.

## Option B — local machine

1. Get an API key: [account.qbraid.com](https://account.qbraid.com) → **Account → API Keys**.
2. Install + authenticate:
   ```bash
   pip install -r requirements.txt
   pip install "qbraid>=0.12.0"        # not pulled in by the base requirements
   export QBRAID_API_KEY=...           # Windows PowerShell: $env:QBRAID_API_KEY="..."
   ```
   Or save it once from Python:
   ```python
   from qbraid.runtime import QbraidProvider
   QbraidProvider(api_key="...").save_config()
   ```

## Running the experiment

Always dry-run first (no jobs, no qBraid needed):

```bash
python experiments/qbraid_hardware_qrc.py --dry-run --max-test 120
```

Then the **free** statevector simulator (validates the whole cloud path):

```bash
python experiments/qbraid_hardware_qrc.py --device qbraid:qbraid:sim:qir-sv --max-test 120
```

Then, only when ready, a **small** real-QPU run (spends credits — note `--allow-qpu`).
Use an online gate QPU (check first with `qbraid_list_devices.py --online --gate`):

```bash
python experiments/qbraid_hardware_qrc.py --device openquantum:ionq:qpu:forte-1 \
    --max-test 40 --shots 1024 --max-batch 20 --allow-qpu
```

### Key flags

| Flag | Default | Notes |
|---|---|---|
| `--device` | `qbraid:qbraid:sim:qir-sv` | qBraid device id. |
| `--max-test` | `120` | Most-recent test days run on hardware = **number of jobs**. `0` = all (~740). |
| `--shots` | `1024` | Shots per circuit. |
| `--max-batch` | `50` | Circuits per `device.run` call; keep ~20 on QPUs. |
| `--allow-qpu` | off | **Required** to submit to any non-simulator device. |
| `--dry-run` | off | Build + estimate only; submits nothing. |
| `--qubits` / `--layers` / `--connectivity` / `--seed` | `5` / `2` / `random` / `42` | Reservoir config (default = the Phase 2 winner). |

## Cost model (read before using a QPU)

- **Jobs = `--max-test`.** One circuit per forecast day. The readout is trained on
  free, exact local statevector features, so only the test window hits the backend.
- **Total shots = `max-test × shots`.** e.g. `40 × 1024 ≈ 41k` shots.
- QPU billing is per-task + per-shot and provider-specific. The free
  `qbraid:qbraid:sim:qir-sv` simulator costs nothing — use it to confirm the full
  path, then move a small `--max-test` to a QPU.
- Circuit size is NISQ-friendly: 5 qubits, depth ~25, ~74 gates.

## Picking a device

List what your account can reach (status + queue), no secrets stored:

```bash
python experiments/qbraid_list_devices.py --online        # everything online
python experiments/qbraid_list_devices.py --online --gate # gate QPUs only
```

### IMPORTANT: QuEra and Pasqal are analog — they will NOT run this reservoir

`aws:quera:qpu:aquila` (256 atoms) and `azure:pasqal:qpu:fresnel` are
**neutral-atom analog** devices. They run *Analog Hamiltonian Simulation* (you
program a time-dependent Rydberg Hamiltonian), **not gate circuits**. The
gate-based reservoir here (H, Rx, Rz, CX) cannot be submitted to them. Using
QuEra requires a separate analog reservoir (Braket AHS / Bloqade) — see
"QuEra / analog path" below.

### Gate QPUs that run this reservoir (observed online for this account)

| Device id | Qubits | Tech | Notes |
|---|---|---|---|
| `openquantum:ionq:qpu:forte-1` | 36 | trapped ion | **Recommended** — all-to-all connectivity matches the reservoir graph, low error |
| `openquantum:ionq:qpu:forte-enterprise` | 36 | trapped ion | as above |
| `openquantum:iqm:qpu:garnet` | 20 | superconducting | |
| `openquantum:iqm:qpu:emerald` | 54 | superconducting | |
| `openquantum:rigetti:qpu:cepheus-1-108q` | 107 | superconducting | |
| `openquantum:aqt:qpu:ibex-q1` | 12 | trapped ion | enough for the 5q reservoir |

> The same hardware appears under several provider prefixes (`aws:`, `azure:`,
> `openquantum:`, `ionq:`, `rigetti:`). For this account the `openquantum:*` and
> `rigetti:*` IonQ/IQM/Rigetti QPUs were ONLINE while the `aws:ionq:*` / `azure:*`
> mirrors were OFFLINE — always check `qbraid_list_devices.py --online` first.

### Simulators (free / cheap, good for validation and noise studies)

| Device id | Qubits | Notes |
|---|---|---|
| `qbraid:qbraid:sim:qir-sv` | 30 | free statevector — default |
| `aws:aws:sim:sv1` | 34 | statevector |
| `aws:aws:sim:dm1` | 17 | **density-matrix (noise) sim** — use for a fidelity-vs-noise figure |
| `aws:aws:sim:tn1` | 50 | tensor network |
| `ionq:ionq:sim:simulator` | 29 | IonQ ideal/noisy sim |

## Interpreting the output

- **Feature fidelity** — Pearson corr and mean|Δ| between hardware and statevector
  features. On a good simulator/QPU run, corr should be > 0.95.
- **Leaderboard** — RMSE on the test window for Persistence, ESN, QRC-sim,
  QRC-hardware. QRC-hardware close to QRC-sim ⇒ the encoding survived the device.

## Troubleshooting

- `ImportError: qbraid is not installed` — only the submission path needs it;
  `--dry-run` and the local statevector reference don't. `pip install qbraid` (or
  run in Lab).
- `PermissionError: Refusing to submit to non-simulator device` — add `--allow-qpu`.
- `Program contains barrier operations which are not supported by the QIR simulator`
  / `Counts data is not available` — caused by the barrier that Qiskit's
  `measure_all()` inserts; the `qir-sv` simulator rejects it, the job fails, and no
  counts return. `QbraidExecutor.run` now strips barriers (`RemoveBarriers`) from
  every circuit before submission, so this is handled automatically.
- Bitstring endianness is auto-handled: `calibrate_bit_order` compares both
  orderings against the statevector reference and picks the matching one.
