"""
qBraid hardware execution backend for the quantum reservoir.

This module routes the *same* QRC circuits used in the local experiments to a
real quantum backend through the qBraid runtime, then rebuilds the Z / ZZ
reservoir features from the returned shot counts. It is the bridge between the
statevector prototype and "real quantum jobs" for the GIC 2026 Phase 3 proof.

Design goals
------------
* **No qBraid required for the offline path.** ``qbraid`` is imported lazily,
  *only* when you actually submit jobs. Building circuits, estimating cost
  (``--dry-run``) and computing the local statevector reference all work with a
  plain Qiskit install — so you can validate everything before spending credits.
* **Reuse, not reimplementation.** Feature extraction goes through the existing
  ``_observables_from_counts`` so hardware features are computed identically to
  the noisy-simulator path in ``reservoir.py``.
* **Auditability.** Every submitted job id is written to disk. Those ids are the
  evidence that the reservoir really ran on quantum hardware — keep them with the
  submission.

Typical use
-----------
    from src.qrc.hardware_backend import QbraidExecutor, reservoir_circuits, counts_to_features
    from src.qrc.reservoir import QuantumReservoir

    res = QuantumReservoir(n_qubits=5, n_layers=2, connectivity="random", seed=42)
    circuits = reservoir_circuits(res, X_test)          # list[QuantumCircuit] w/ measure_all

    ex = QbraidExecutor(device_id="qbraid:qbraid:sim:qir-sv", shots=1024)
    counts = ex.run(circuits)                            # list[dict]  (one per circuit)
    R_hw = counts_to_features(counts, res.n_qubits, ex.shots, reverse_bits=ex.reverse_bits)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.passes import RemoveBarriers

from .reservoir import _observables_from_counts, _observables_from_sv

# Qiskit's measure_all() inserts a barrier, which the qBraid QIR simulator (and
# some other backends) reject ("barrier operations ... not supported"). Strip
# barriers from every circuit before submission — they are only optimisation
# hints and are irrelevant to these shallow reservoir circuits.
_REMOVE_BARRIERS = RemoveBarriers()


# ---------------------------------------------------------------------------
# Circuit construction (reuses the reservoir's own builder)
# ---------------------------------------------------------------------------

def reservoir_circuits(reservoir, X: np.ndarray) -> list[QuantumCircuit]:
    """
    Build one measured circuit per input row, using the reservoir's own builder.

    Works for both ``QuantumReservoir`` (gate-based) and ``AnnealerReservoir``
    (Trotterised Ising). Inputs are tanh-normalised exactly as in the reservoirs'
    ``transform`` methods, so circuits are bit-for-bit identical to the local
    pipeline — only the *execution* backend changes.
    """
    X_norm = np.tanh(np.asarray(X, dtype=float))
    circuits: list[QuantumCircuit] = []
    for x in X_norm:
        # QuantumReservoir._build_circuit takes add_measurement; AnnealerReservoir
        # does not (it has no measurement at all). Handle both.
        try:
            qc = reservoir._build_circuit(x, add_measurement=True)  # gate-based
        except TypeError:
            qc = reservoir._build_circuit(x)                        # annealer
            if qc.num_clbits == 0:
                qc = qc.copy()
                qc.measure_all()
        circuits.append(qc)
    return circuits


def counts_to_features(
    counts_list: Sequence[dict],
    n_qubits: int,
    n_shots: int,
    reverse_bits: bool = False,
    max_order: int = 2,
) -> np.ndarray:
    """
    Rebuild the (Z, ZZ) reservoir feature matrix from a list of count dicts.

    ``reverse_bits`` flips the bitstring endianness before parsing. Different
    backends report measurement bitstrings in different qubit orders; the helper
    ``calibrate_bit_order`` figures out which one matches the statevector
    reference, and the result is passed in here.
    """
    rows = []
    for counts in counts_list:
        shots = sum(counts.values()) or n_shots
        if reverse_bits:
            counts = {k[::-1]: v for k, v in counts.items()}
        rows.append(_observables_from_counts(counts, n_qubits, shots, max_order))
    return np.asarray(rows)


def statevector_features(reservoir, X: np.ndarray) -> np.ndarray:
    """Exact, free, local reference features (no measurement, no shots).

    Identical maths to ``reservoir.transform`` in noiseless mode, but routed
    through the shared observable helper so it lines up with the hardware path.
    """
    from qiskit.quantum_info import Statevector

    X_norm = np.tanh(np.asarray(X, dtype=float))
    rows = []
    for x in X_norm:
        try:
            qc = reservoir._build_circuit(x, add_measurement=False)
        except TypeError:
            qc = reservoir._build_circuit(x)
        rows.append(_observables_from_sv(
            Statevector(qc), reservoir.n_qubits,
            getattr(reservoir, "observable_order", 2),
        ))
    return np.asarray(rows)


# ---------------------------------------------------------------------------
# Bit-order calibration
# ---------------------------------------------------------------------------

def calibrate_bit_order(
    counts_list: Sequence[dict],
    reference_features: np.ndarray,
    n_qubits: int,
    n_shots: int,
    max_order: int = 2,
) -> bool:
    """
    Decide whether returned bitstrings need reversing to match the reference.

    Computes features both ways for the supplied (typically small) calibration
    batch and returns the ``reverse_bits`` value that best matches
    ``reference_features`` (lower mean-abs error). Eliminates a whole class of
    silent endianness bugs when moving between backends.
    """
    feats_fwd = counts_to_features(
        counts_list, n_qubits, n_shots, reverse_bits=False, max_order=max_order)
    feats_rev = counts_to_features(
        counts_list, n_qubits, n_shots, reverse_bits=True, max_order=max_order)
    err_fwd = float(np.mean(np.abs(feats_fwd - reference_features)))
    err_rev = float(np.mean(np.abs(feats_rev - reference_features)))
    return err_rev < err_fwd


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

@dataclass
class QbraidExecutor:
    """
    Submit a list of circuits to a qBraid device and collect shot counts.

    Parameters
    ----------
    device_id    : qBraid device id, e.g. ``"qbraid:qbraid:sim:qir-sv"`` (free
                   30-qubit statevector sim) or a QPU id like
                   ``"aws:ionq:qpu:aria-1"``.
    shots        : measurement shots per circuit.
    api_key      : qBraid API key. If None, uses the ``QBRAID_API_KEY`` env var
                   or the saved qBraid config (the default inside qBraid Lab).
    max_batch    : circuits submitted per ``device.run`` call. Keep modest on
                   QPUs; large is fine on simulators.
    cache_dir    : where to persist job ids / metadata (audit trail + resume).
    reverse_bits : endianness for count parsing; usually set by calibration.
    allow_qpu    : safety latch. Submitting to a non-simulator device raises
                   unless this is True (guards against accidental credit burn).
    poll_seconds : delay between status polls while waiting for jobs.
    verbose      : print progress.
    """

    device_id: str = "qbraid:qbraid:sim:qir-sv"
    shots: int = 1024
    api_key: Optional[str] = None
    max_batch: int = 50
    cache_dir: Optional[str] = None
    reverse_bits: bool = False
    allow_qpu: bool = False
    poll_seconds: float = 2.0
    verbose: bool = True

    _provider: object = field(default=None, repr=False, init=False)
    _device: object = field(default=None, repr=False, init=False)
    submitted_job_ids: list[str] = field(default_factory=list, repr=False, init=False)

    # -- lazy qBraid handles ------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    @property
    def provider(self):
        if self._provider is None:
            try:
                from qbraid.runtime import QbraidProvider
            except ImportError as e:  # pragma: no cover - depends on env
                raise ImportError(
                    "qbraid is not installed. Run `pip install qbraid` (already "
                    "available inside qBraid Lab), or use --dry-run / the local "
                    "statevector path which does not need it."
                ) from e
            self._provider = (
                QbraidProvider(api_key=self.api_key or qbraid_api_key())
            )
        return self._provider

    @property
    def device(self):
        if self._device is None:
            self._device = self.provider.get_device(self.device_id)
        return self._device

    def metadata(self) -> dict:
        """Device metadata (num_qubits, type, status, queue_depth, ...)."""
        md = self.device.metadata()
        return dict(md) if not isinstance(md, dict) else md

    def is_simulator(self) -> bool:
        # The qBraid device-id convention is the reliable signal: simulators carry
        # ":sim:" and QPUs ":qpu:" (device.metadata() does not include a type key
        # on all backends). Fall back to metadata only if the id is ambiguous.
        if ":sim:" in self.device_id:
            return True
        if ":qpu:" in self.device_id:
            return False
        try:
            dtype = str(self.metadata().get("device_type", "")).upper()
            return "SIM" in dtype
        except Exception:
            return False

    # -- estimation ---------------------------------------------------------

    def estimate(self, n_circuits: int) -> dict:
        """Rough job/shot footprint for a run. Cheap, no submission."""
        return {
            "device_id": self.device_id,
            "n_circuits": n_circuits,
            "shots_per_circuit": self.shots,
            "total_shots": n_circuits * self.shots,
            "n_batches": int(np.ceil(n_circuits / self.max_batch)),
        }

    # -- execution ----------------------------------------------------------

    def run(self, circuits: Sequence[QuantumCircuit]) -> list[dict]:
        """
        Submit all circuits (in batches) and return counts in input order.

        Raises if the target is a QPU and ``allow_qpu`` is False.
        """
        circuits = [_REMOVE_BARRIERS(c) for c in circuits]
        if not circuits:
            return []

        if not self.is_simulator() and not self.allow_qpu:
            raise PermissionError(
                f"Refusing to submit to non-simulator device '{self.device_id}' "
                f"without allow_qpu=True. This would consume QPU credits "
                f"({len(circuits)} circuits x {self.shots} shots). Set "
                f"allow_qpu=True (or pass --allow-qpu) to proceed intentionally."
            )

        est = self.estimate(len(circuits))
        self._log(
            f"  Submitting {est['n_circuits']} circuits in {est['n_batches']} "
            f"batch(es) of <= {self.max_batch}  ({est['total_shots']:,} shots total) "
            f"-> {self.device_id}"
        )

        all_counts: list[dict] = []
        t0 = time.time()
        for b_start in range(0, len(circuits), self.max_batch):
            batch = circuits[b_start : b_start + self.max_batch]
            counts = self._run_batch(batch)
            all_counts.extend(counts)
            self._log(
                f"    batch {b_start // self.max_batch + 1}/{est['n_batches']} done "
                f"({len(all_counts)}/{len(circuits)} circuits, "
                f"{time.time() - t0:.0f}s elapsed)"
            )
        self._persist_job_ids()
        return all_counts

    def _run_batch(self, batch: list[QuantumCircuit]) -> list[dict]:
        # device.run accepts a single circuit or a list; normalise the return to
        # a list of jobs in either case.
        jobs = self.device.run(batch, shots=self.shots)
        if not isinstance(jobs, (list, tuple)):
            jobs = [jobs]

        for job in jobs:
            jid = getattr(job, "id", None) or getattr(job, "job_id", None)
            if jid is not None:
                self.submitted_job_ids.append(str(jid))
                self._log(f"    submitted job id: {jid}")
        self._persist_job_ids()

        counts: list[dict] = []
        for job in jobs:
            # job.result() blocks until the job reaches a final state in the
            # qBraid runtime; wrap with a best-effort wait for older SDKs.
            wait = getattr(job, "wait_for_final_state", None)
            if callable(wait):
                try:
                    wait()
                except Exception:
                    pass
            result = job.result()
            counts.append(self._extract_counts(result, self.shots))
        return counts

    @staticmethod
    def _extract_counts(result, shots: int) -> dict:
        """Pull a {bitstring: count} dict out of a qBraid Result object."""
        data = getattr(result, "data", result)
        for attr in ("get_counts", "measurement_counts"):
            fn = getattr(data, attr, None)
            if callable(fn):
                try:
                    c = fn()
                    return dict(c)
                except ValueError:
                    pass
            if fn is not None:  # measurement_counts may be a property/dict
                return dict(fn)
        probs_fn = getattr(data, "get_probabilities", None)
        if callable(probs_fn):
            try:
                probs = dict(probs_fn())
                return {str(k): int(round(float(v) * shots)) for k, v in probs.items()}
            except ValueError:
                pass
        probs = getattr(data, "probabilities", None)
        if probs is not None:
            return {str(k): int(round(float(v) * shots)) for k, v in dict(probs).items()}
        measurements = getattr(data, "measurements", None)
        if measurements is not None:
            out: dict[str, int] = {}
            for row in measurements:
                bitstring = "".join(str(int(b)) for b in row)
                out[bitstring] = out.get(bitstring, 0) + 1
            return out
        raise AttributeError(
            "Could not extract counts from result; inspect result.data attributes "
            f"(have: {dir(data)})."
        )

    # -- audit trail --------------------------------------------------------

    def _persist_job_ids(self) -> None:
        if not self.cache_dir or not self.submitted_job_ids:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        path = os.path.join(self.cache_dir, "qbraid_job_ids.json")
        payload = {
            "device_id": self.device_id,
            "shots": self.shots,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "job_ids": self.submitted_job_ids,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        self._log(f"  Job ids ({len(self.submitted_job_ids)}) -> {path}")
def qbraid_api_key() -> str:
    """Read qBraid auth from environment or a git-ignored local secret file."""
    value = os.environ.get("QBRAID_API_KEY", "").strip()
    if value:
        return value
    root = Path(__file__).resolve().parents[2] / ".secrets"
    for name in ("qbraid_token", "qBraid_testQPU_token"):
        try:
            value = (root / name).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        if value:
            return value
    return ""


def patch_qbraid_ahs_decimal_encoder() -> None:
    """Work around qBraid <=0.12.2 failing to JSON-encode Braket AHS Decimals."""
    from decimal import Decimal
    from qbraid.programs.analog._model import AnalogHamiltonianEncoder

    if getattr(AnalogHamiltonianEncoder, "_qrc_decimal_patch", False):
        return
    original = AnalogHamiltonianEncoder.default

    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return original(self, obj)

    AnalogHamiltonianEncoder.default = default
    AnalogHamiltonianEncoder._qrc_decimal_patch = True
