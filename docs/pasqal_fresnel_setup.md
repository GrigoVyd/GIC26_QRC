# Pasqal Fresnel reservoir + QuEra-vs-Pasqal comparison (Phase 3)

Pasqal Fresnel is a second **neutral-atom Rydberg analog** platform alongside QuEra
Aquila. Running the same analog-reservoir idea on two independent platforms is a
platform-robustness result for the submission.

> **Availability:** on this account `azure:pasqal:qpu:fresnel` and the
> `azure:pasqal:sim:emu-tn` cloud emulator were **OFFLINE** at build time (QuEra
> Aquila was online). So the comparison below is on **local emulators** (free);
> the Pasqal hardware path is ready for when Fresnel comes online.

## The key hardware difference (and why it matters)

The real Pasqal `AnalogDevice` (Fresnel) exposes **only a global rydberg channel** —
global Rabi + global detuning, **no per-site addressing / DMM**. QuEra Aquila *does*
have per-site local detuning. That forces different input encodings:

| | QuEra Aquila | Pasqal Fresnel |
|---|---|---|
| SDK | Amazon Braket AHS | Pulser |
| Local sim | `LocalSimulator("braket_ahs")` | `QutipEmulator` (qutip) |
| Couplings | fixed atom geometry (van der Waals) | same |
| Transverse field | global Rabi drive | global Rabi drive |
| **Input encoding** | per-site **local detuning** (spatial) | **global detuning waveform** (temporal) |
| Readout | Rydberg ⟨Z⟩, ⟨ZZ⟩ | same |

Both still produce strongly input-sensitive features; the geometry breaks the
symmetry so atoms respond differently even under Pasqal's global-only drive.

## Files

- [src/qrc/pasqal_reservoir.py](../src/qrc/pasqal_reservoir.py) — `PasqalReservoir`
  (Pulser `AnalogDevice`, global-detuning input encoding, local emulator, `n_jobs`).
- [experiments/neutral_atom_compare.py](../experiments/neutral_atom_compare.py) —
  QuEra vs Pasqal vs Persistence/ESN on the same window.

## Run the comparison (free, local emulators)

```bash
pip install -r requirements.txt          # includes pulser + pulser-simulation (qutip)
python experiments/neutral_atom_compare.py --max-train 400 --max-test 120 --n-jobs 4
```

Output: `results/neutral_atom_compare.csv` and `.png` (RMSE leaderboard + forecast
overlay). On the validation windows, both neutral-atom reservoirs were competitive
with — and often beat — Persistence and ESN.

## Hardware path (when Fresnel is online)

Fresnel runs through **Pulser → pasqal-cloud** (or Azure Quantum), both reachable
via qBraid Lab credits:

```python
from pulser_pasqal import PasqalCloud   # pip install pulser-pasqal
# build the sequence with PasqalReservoir.build_sequence(x), then submit a batch
# of sequences to the Fresnel device through PasqalCloud (one job per forecast day).
```

Same economical hybrid as QuEra/gate paths: train the Ridge readout on free local
emulator features, run only the test window on hardware. Note Fresnel requires
atoms on a **calibrated layout** (register validated against the device) for real
submission — the local emulator accepts free coordinates.

## Honest framing

Same as the rest of Phase 3: the claim is that an analog Rydberg reservoir is
**platform-agnostic and stays competitive with the best classical baselines on two
independent neutral-atom platforms**, each using the encoding its hardware allows —
not that it beats everything everywhere.
