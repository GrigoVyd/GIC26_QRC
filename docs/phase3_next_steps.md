# Phase 3 Handoff & Next Steps

This is the cold-start handoff for any future agent or contributor. Read this
before running more jobs. It records what is built, what is proven, what failed,
and what should happen next now that **D-Wave is considered inaccessible**.

Companion docs:
- `docs/phase3_report.md` - results and submission narrative.
- `docs/quera_aquila_setup.md` - QuEra Aquila AHS path.
- `docs/pasqal_fresnel_setup.md` - Pasqal/Pulser path.
- `docs/qbraid_setup.md` - qBraid gate-QPU path.

---

## 1. Current Status (2026-07-01)

Original goal: show a progression from classical baselines to GPU/Ising machines
to QPU execution for quantum-reservoir volatility forecasting.

Updated constraint: **treat D-Wave Leap as unavailable**. The D-Wave-like
transverse-field annealer remains the *scientific explanation* for why the
GARCH-hybrid wins in simulation, but it is no longer the near-term hardware plan.

### Proven in simulation

- Raw-task QRC beats LSTM/ESN/AR-class baselines, but not GARCH.
- GARCH is the strongest classical benchmark because the realized-volatility
  target contains a rolling-window structural prior.
- The Phase 2 GARCH-hybrid result reproduces exactly:
  - QRC annealer on GARCH-residual, 3-seed ensemble: **RMSE 0.007844**.
  - GARCH(1,1): **RMSE 0.007948**.
  - Ridge residual ablation: **RMSE 0.008051**.
  - The win is real in noiseless transverse-field Ising simulation.

### Proven negative results

- Neutral-atom Rydberg reservoirs do **not** reproduce the GARCH-beating result:
  - Pasqal global encoding: ~0.00806.
  - Per-site DMM/local-detuning surrogate: ~0.00823.
- Classical signed-coupling Ising sampling does **not** reproduce it:
  - Local SA / classical Ising tier: ~0.00814.
- Shot noise is **not** the limiter:
  - Pasqal 10-atom noiseless ~= 600-shot.

Interpretation: the GARCH-beating edge appears tied to **transverse-field
quantum-annealing dynamics**, not merely signed couplings or finite-shot effects.
Because D-Wave is unavailable, the submission should not promise an executed
GARCH-beating QPU result. The honest submission story is:

1. The hybrid simulation beats GARCH and identifies the needed physics.
2. Toshiba/Fixstars/SA test the classical Ising tier.
3. QuEra/qBraid provide executable QPU paths for the competitive reservoir story.
4. D-Wave remains a future hardware validation path, explicitly blocked.

---

## 2. Repo Map

### Reservoirs

| File | Reservoir | Substrate | Status |
|---|---|---|---|
| `src/qrc/reservoir.py` | Gate QRC | gate circuit | qBraid path live-verified on `qir-sv` |
| `src/qrc/annealer_reservoir.py` | Transverse-field Ising | noiseless Trotter/statevector | Phase 2 GARCH-beating winner |
| `src/qrc/quera_reservoir.py` | QuEra Aquila AHS | Rydberg neutral atom | local AHS sim + real Aquila path |
| `src/qrc/pasqal_reservoir.py` | Pasqal/Pulser | Rydberg neutral atom | global + DMM/per-site tests |
| `src/qrc/ising_reservoir.py` | Signed Ising machine | SA / Toshiba / Amplify / Fujitsu / etc. | cloud backends scaffolded |
| `src/qrc/hardware_backend.py` | qBraid executor | gate QPUs | strips barriers, calibrates bit order |

### Experiments

| File | Purpose |
|---|---|
| `experiments/phase2_final.py` | classical + raw-task leaderboard |
| `experiments/phase2_garch_hybrid.py` | reproduced GARCH-hybrid headline |
| `experiments/neutral_atom_garch_hybrid.py` | shared harness for QuEra/Pasqal/Ising |
| `experiments/neutral_atom_compare.py` | QuEra vs Pasqal raw-task comparison |
| `experiments/quera_aquila_qrc.py` | QuEra local sim + real Aquila run |
| `experiments/qbraid_hardware_qrc.py` | gate-QPU path |
| `experiments/ising_cloud_smoke.py` | tiny cloud credential smoke test |
| `experiments/phase3_timeline_figure.py` | report timeline figure |

---

## 3. Tokens And Current Credential State

Never commit tokens. Use environment variables only.

| Platform | Env var | Status |
|---|---|---|
| qBraid / QuEra Braket | saved in `~/.qbraid/qbraidrc` | works, but rotate the pasted key later |
| Toshiba SQBM+ | `TOSHIBA_TOKEN`, optional `TOSHIBA_URL` | token was tested and returned **401 Unauthorized** on SDK default endpoint |
| Fixstars Amplify AE | `AMPLIFY_TOKEN` | not tested with a valid token |
| D-Wave Leap | `DWAVE_API_TOKEN` | **treat unavailable** |

The Toshiba key provided in chat was tested with:

```powershell
$env:TOSHIBA_TOKEN="..."
python experiments/ising_cloud_smoke.py --backend toshiba
```

Result:

```text
FAILED: Amplify backend 'toshiba' rejected the credential (401 Unauthorized).
Set a valid TOSHIBA_TOKEN; for Toshiba/Fujitsu/private endpoints also set
TOSHIBA_URL if the token is tied to a non-default service URL.
```

Action: ask the provider/account page for the matching Toshiba endpoint URL, or
use a new token. The code path itself reaches the cloud service.

---

## 4. Near-Term Plan Without D-Wave

### Step 1 - Make Toshiba/Fixstars The Executable GPU-Ising Tier

Purpose: execute the same signed-coupling Ising reservoir on an accessible cloud
Ising machine. This does **not** replace D-Wave scientifically, because it has no
transverse field, but it gives a real GPU/Ising result for the timeline.

Credential smoke test:

```powershell
$env:TOSHIBA_TOKEN="..."
# If Toshiba gave a service endpoint:
$env:TOSHIBA_URL="https://..."
python experiments/ising_cloud_smoke.py --backend toshiba
```

Small finance run after smoke test passes:

```powershell
python experiments/neutral_atom_garch_hybrid.py `
  --reservoir ising_sa --ising-backend toshiba `
  --atoms 10 --max-train 400 --max-test 120 --shots 100 --n-seeds 1
```

Full run:

```powershell
python experiments/neutral_atom_garch_hybrid.py `
  --reservoir ising_sa --ising-backend toshiba `
  --atoms 10 --max-train 2000 --max-test 0 --shots 300 --n-seeds 3
```

Expected result: near the local SA/classical Ising tier (~0.0081 RMSE), likely
not beating GARCH. That is not failure; it supports the timeline separation
between classical Ising and the simulated quantum-annealer tier.

Acceptance:
- Cloud solve completes.
- Raw result CSV and job/client metadata are saved under `results/hardware/toshiba/`
  if possible.
- Report marks Toshiba as executed and explains the 401/token requirement if not.

### Step 2 - Run Real QuEra Aquila For The QPU-Executed Reservoir Story

Purpose: show real QPU execution on qBraid/Braket. This is not expected to beat
GARCH on the GARCH residual; use it for the raw-task competitive story.

```powershell
python experiments/quera_aquila_qrc.py --dry-run --max-test 40
python experiments/quera_aquila_qrc.py --device aquila --max-test 40 --shots 100 --allow-qpu
```

Acceptance:
- Aquila task IDs saved.
- Hardware features correlate with local AHS features (target >0.9 if noise is
  mild).
- Forecast remains competitive with Persistence/ESN on the same window.

### Step 3 - Optional Gate-QPU Run Through qBraid

Purpose: a second real-QPU proof using the gate reservoir.

```powershell
python experiments/qbraid_list_devices.py --online --gate
python experiments/qbraid_hardware_qrc.py `
  --device openquantum:ionq:qpu:forte-1 `
  --max-test 40 --shots 1024 --max-batch 20 --allow-qpu
```

Known gotchas are already fixed:
- Qiskit barriers are stripped before qBraid submission.
- qBraid `qir-sv` bit order differs from local Aer; calibration is automatic.

### Step 4 - Update Report And Timeline

After each executed cloud/hardware run:

1. Add the executed row to `docs/phase3_report.md`.
2. Update `experiments/phase3_timeline_figure.py`.
3. Regenerate:

```powershell
python experiments/phase3_timeline_figure.py
```

4. Commit result CSVs, figures, and any job-id JSON.

---

## 5. Revised Submission Timeline

| Day | Work | Depends on |
|---|---|---|
| 0 | Fix/confirm Toshiba credentials (`TOSHIBA_URL` if needed) | provider/account |
| 0 | Toshiba smoke test with `ising_cloud_smoke.py` | valid token |
| 0-1 | Small Toshiba finance run | smoke passes |
| 1 | Full Toshiba/Fixstars run | small run passes |
| 1-2 | Real QuEra Aquila run during availability window | qBraid credits |
| 2 | Optional gate-QPU qBraid run | credits/device queue |
| 2-3 | Regenerate report + timeline with executed points | run outputs |

---

## 6. Do Not Repeat These Detours

- Pasqal global/per-site on the GARCH residual: tested, no GARCH beat.
- More shots: tested, not the limiter.
- QuEra local AHS scaling beyond ~8 atoms on CPU: too slow.
- Threads for local simulation: slower than process workers.
- Classical signed Ising alone as a route to GARCH beat: local SA does not beat
  GARCH on the full window. Use Toshiba/Fixstars as the executable GPU tier, not
  as the claimed advantage tier.
- D-Wave as near-term plan: currently considered inaccessible.

---

## 7. No-Token Reproductions

```powershell
python experiments/phase2_final.py
python experiments/phase2_garch_hybrid.py --n-seeds 3
python experiments/neutral_atom_garch_hybrid.py --reservoir ising_sa --atoms 10 --max-test 0 --n-seeds 3
python experiments/phase3_timeline_figure.py
```

