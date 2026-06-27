# Running the reservoir on QuEra Aquila (Phase 3 — primary platform)

QuEra Aquila is the project's **declared primary platform** (TECHNICAL_REFERENCE
§13.1, paper_draft.md). Aquila is a 256-atom **neutral-atom analog** machine: it
runs a time-dependent Rydberg Hamiltonian, *not* gate circuits. The Phase-2
headline reservoir is a transverse-field Ising model — Aquila's *native* physics —
so this is the most faithful hardware realization of the project's core idea.

## What was built

- [src/qrc/quera_reservoir.py](../src/qrc/quera_reservoir.py) — `QueraReservoir`,
  a native Braket **Analog Hamiltonian Simulation (AHS)** reservoir.
- [experiments/quera_aquila_qrc.py](../experiments/quera_aquila_qrc.py) — the
  volatility experiment (local AHS sim → Aquila hardware), CLI.

## How the Ising reservoir maps to Aquila

| Ising reservoir term | Aquila realization |
|---|---|
| Fixed random couplings `J_ij` | Fixed **atom geometry**; van der Waals `C6/r⁶` set by positions |
| Transverse field (mixing) | Global **Rabi drive** `Ω(t)` |
| Input encoding | Per-site **local detuning** weights `w_i = (tanh xᵢ+1)/2 ∈ [0,1]` |
| Readout `⟨Z⟩, ⟨ZZ⟩` | Rydberg occupation `n_i∈{0,1}` → `Z_i = 1−2n_i`, `Z_iZ_j` |

**Honest caveats (this is an *adaptation*, not a 1:1 port):** Aquila couplings are
geometry-determined and all repulsive (no signed random `J_ij`); the transverse
field is global, not per-qubit; local detuning weights are non-negative and share
one time profile. The disorder comes from random atom *positions*. These are real
Phase-3 design decisions, not bugs.

## Hardware limits enforced in the code

Rabi ≤ 1.58×10⁷ rad/s · detuning ≤ 1.25×10⁸ rad/s · duration ≤ 4 µs · atom
spacing ≥ 4 µm · register ≤ 75×76 µm · positions on a 10 nm grid. The reservoir
constructor raises if you exceed any of these, and `program.discretize(device)`
snaps values to Aquila's grid before submission.

## Validate for free first (local AHS simulator)

No AWS, no credits, no qBraid needed — `amazon-braket-sdk` ships a local Rydberg
simulator:

```bash
pip install -r requirements.txt          # includes amazon-braket-sdk

# footprint only (instant):
python experiments/quera_aquila_qrc.py --dry-run --max-test 40

# full pipeline on the FREE local AHS simulator:
python experiments/quera_aquila_qrc.py --device local --max-train 150 --max-test 30
```

The local simulator does exact Schrödinger evolution (~2–3 s per program for 6
atoms), so keep `--max-train` modest. Output: feature fidelity, a leaderboard
(Persistence / ESN / QuEra-sim), and `results/quera_aquila_qrc.png` (register +
forecast + feature scatter).

## Run on real QuEra Aquila

Aquila runs through **Amazon Braket**, billed to your **qBraid credits** inside
qBraid Lab (no personal AWS account needed there). Outside Lab you need AWS
credentials with Braket access.

```bash
# check Aquila is in an availability window first (see note below), then:
python experiments/quera_aquila_qrc.py --device aquila --max-test 40 --shots 100 --allow-qpu
```

- `--allow-qpu` is the required safety latch.
- **Tasks = `--max-test`** (one AHS task per forecast day). The readout is trained
  on free local-sim features, so only the test window hits hardware.
- **Cost (list price):** ≈ `tasks × ($0.30 + shots × $0.01)`. e.g. `40 × (0.30 +
  100×0.01) = $52`. Verify current rates / your qBraid credit balance.
- **Availability:** Aquila is only online during scheduled windows (typically a
  few hours on set days, UTC). Submit during a window or the task will queue. In
  qBraid Lab, `AwsDevice(Devices.QuEra.Aquila).is_available` and
  `.properties.service.executionWindows` tell you when.

## Interpreting the result

The headline is the **local-sim vs Aquila** comparison:
- **Feature fidelity** — corr / mean-abs-error between Aquila Rydberg features and
  the exact local-sim features. Tells you how much real decoherence + finite shots
  perturb the encoding.
- **Forecast leaderboard** — if QuEra-Aquila tracks QuEra-sim and stays at/below
  Persistence & ESN RMSE on the window, the analog reservoir is hardware-realizable
  and predictive. That is the defensible advantage claim.

Keep the Braket task ARNs (printed/returned by Braket) as proof of real
neutral-atom hardware execution for the submission.
