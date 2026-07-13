# 100-credit QPU trial

## Final two-platform allocation

Use both grants, because they are separate and cannot be pooled:

| Platform/model | Trial | Cost | Balance left |
|---|---|---:|---:|
| QuEra Aquila analog QRC | 1 held-out input, 50 shots | 80 qBraid credits ($0.80) | 20 qBraid credits |
| IQM Garnet gate QRC hybrid | 120 held-out inputs, 100 shots | 16.02 Spark ($32.04) | 8.98 Spark ($17.96) |

This gives two honest deliverables. Aquila is the **native analog-QRC hardware
fingerprint**: feature fidelity for one input plus a real task ARN. It is not a
forecast-performance claim because one task is too small. Garnet is the
**forecast showcase**: a meaningful 120-day window for the hybrid
quantum-reservoir/classical-readout model.

On 2026-07-11 both `aws:quera:qpu:aquila` and
`openquantum:iqm:qpu:garnet` were online with queue zero.

The guarded Aquila smoke command is:

```powershell
python experiments/quera_aquila_qrc.py `
  --device aquila --atoms 5 --geometry random2d `
  --max-train 150 --max-test 1 --shots 50 `
  --credit-budget 100 --allow-qpu
```

Run Aquila only during an execution window. Its script now refuses any footprint
above 100 qBraid credits.

## Decision

Use **IQM Garnet through OpenQuantum Public Compute**:

```text
openquantum:iqm:qpu:garnet
```

qBraid credits and OpenQuantum Spark credits are separate balances. On
2026-07-11, qBraid metadata incorrectly made the OpenQuantum routes look free
(`0 + 0`), but OpenQuantum Public Compute deducts Spark credits. The experiment
therefore uses an audited OpenQuantum price table and a hard 25-Spark cap instead
of trusting those zero metadata fields.

Forte is the highest-fidelity topology match, but at the Public Compute price of
`$0.18/task + $0.048/shot`, 25 Spark credits ($50) buy only ten 100-shot
circuits. That is too small for a meaningful forecast window. Garnet costs
`$0.18/task + $0.00087/shot`, so the full validated 120-day showcase costs
`$32.04 = 16.02 Spark credits` and leaves 8.98 Spark for retries.

Use the already validated **5-qubit, 2-layer, sparse random gate reservoir**.
For seed 42 its edges `(0,1), (0,2), (1,3), (2,4)` form a simple five-node path,
so it can be laid out on a connected Garnet path without SWAPs. Do not use the
10-qubit dense annealer circuit for this first NISQ demonstration.

## What the demo tests

The workflow is hybrid in two ways:

1. the nonlinear reservoir features are quantum; the trained Ridge readout is
   classical;
2. training features are computed exactly and free, while only the held-out
   40-day test window is measured on the QPU.

The test reports:

- hardware-vs-statevector feature correlation and mean absolute feature error;
- QPU forecast RMSE versus the identical statevector forecast;
- QPU forecast RMSE versus Persistence and ESN on the same window;
- qBraid job ids as execution evidence.

The locked 100-shot simulator validation completed on 2026-07-11 with feature
correlation `0.9452`. It shows statevector QRC RMSE `0.03644`, finite-shot QRC
RMSE `0.03653`, and Persistence RMSE `0.03667`. The trial therefore passed its
pre-QPU check: the small simulated edge survives 100-shot sampling. The QPU run
asks whether it also survives device noise. This is a showcase/feasibility
result, not a statistically conclusive quantum-advantage claim.

## Locked trial configuration

| Setting | Value |
|---|---:|
| Device | `openquantum:iqm:qpu:garnet` |
| Qubits / layers | 5 / 2 |
| Connectivity / seed | random / 42 |
| Held-out test points | 120 most recent |
| Shots per circuit | 100 |
| Circuit footprint | 5 qubits, depth about 25, 74 operations |
| Total measurements | 12,000 |
| Expected Public Compute cost | $32.04 = 16.02 Spark |
| Spark hard cap | 25 |

The 100-shot setting keeps the test economical while providing enough samples
for useful Z and ZZ expectation estimates.

## Runbook

Use Python 3.10 or newer with Qiskit and qBraid installed.

```powershell
# 1. Confirm the exact device is online and inspect live pricing.
python experiments/qbraid_list_devices.py --online --gate --pricing

# 2. Offline footprint check. This never submits a job.
python experiments/qbraid_hardware_qrc.py `
  --device openquantum:iqm:qpu:garnet `
  --max-test 120 --shots 100 --max-batch 20 `
  --spark-budget 25 --dry-run

# 3. Free qBraid simulator validation.
python experiments/qbraid_hardware_qrc.py `
  --device qbraid:qbraid:sim:qir-sv `
  --max-test 120 --shots 100 --max-batch 20

# 4. Real QPU submission. The live cost guard runs before device.run().
python experiments/qbraid_hardware_qrc.py `
  --device openquantum:iqm:qpu:garnet `
  --max-test 120 --shots 100 --max-batch 20 `
  --spark-budget 25 --allow-qpu
```

Never add `--allow-unknown-pricing` for this trial. If the OpenQuantum route no
longer reports explicit pricing, stop and verify the charge in qBraid before
submitting.

## Go/no-go criteria

Proceed with the QPU run only when all are true:

- exact `openquantum:` device id is `ONLINE`;
- queue is acceptably short;
- reported maximum charge is at most 100 credits;
- simulator run completes and produces 40 result rows/job results;
- no barrier, counts, or bit-order errors occur.

Call the trial successful when:

- all 120 QPU circuits complete and job ids are saved;
- feature correlation with statevector is at least 0.90 (0.95 is the target);
- hardware RMSE remains within 10% of statevector RMSE;
- hardware QRC is competitive with Persistence on the same 40-day window.

If Garnet is unavailable, use Rigetti Cepheus for the same 120-point test
(`$24.66 = 12.33 Spark`) after a transpilation check, not QuEra Aquila. Aquila
runs a different analog reservoir;
previous experiments found that adaptation competitive but it did not reproduce
the hybrid advantage mechanism.
