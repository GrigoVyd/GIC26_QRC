"""
QuEra Aquila reservoir — native neutral-atom Analog Hamiltonian Simulation (AHS).

This is the *declared primary platform* for the project (see
docs/TECHNICAL_REFERENCE.md 13.1 and docs/phase2_submission/paper_draft.md). The
Phase-2 headline reservoir is a transverse-field Ising model; Aquila implements
exactly that physics natively, so this module expresses the reservoir as a real
Aquila program instead of a Trotterised gate-circuit *simulation* of it.

Mapping (Ising reservoir -> Aquila Rydberg Hamiltonian)
-------------------------------------------------------
    H(t)/hbar =  (Omega(t)/2) * sum_i (|g><r| + |r><g|)_i      # global drive  (transverse field)
               -  Delta_global(t) * sum_i n_i                  # global detuning
               -  sum_i  w_i * Delta_local(t) * n_i            # LOCAL detuning  (input encoding)
               +  sum_{i<j} (C6 / r_ij^6) n_i n_j              # van der Waals  (FIXED couplings)

  * Fixed random couplings  -> fixed (seeded) atom **geometry**; the van der Waals
    1/r^6 interaction is set by where the atoms sit, not by a programmable J_ij.
  * Transverse field        -> the **global Rabi drive** Omega(t) (same for all atoms).
  * Input encoding          -> **per-site local detuning** weights w_i = f(x_i),
    weights in [0, 1] (Aquila constraint), one spatial pattern per program.
  * Readout                 -> Rydberg occupation n_i in {0,1} per shot; we report
    Z_i = 1 - 2 n_i and Z_i Z_j, i.e. the SAME (Z, ZZ) feature layout the gate
    reservoir and the Ridge readout already use.

Honest caveats vs the simulated Ising model
-------------------------------------------
  * Couplings are geometry-determined and **all repulsive** (n_i n_j >= 0); you
    cannot set arbitrary signed random J_ij as in ``AnnealerReservoir``. The
    disorder comes from random atom positions instead.
  * The transverse field is **global**, not per-qubit.
  * Local detuning weights are **non-negative** and share one time profile.
  So this is an *adaptation* of the Ising reservoir to Aquila's real constraints,
  not a 1:1 port. That adaptation IS the Phase-3 contribution.

Validate for free on the local AHS simulator before spending QuEra credits:
``LocalSimulator("braket_ahs")``. Hardware is a one-line device swap to
``AwsDevice(Devices.QuEra.Aquila)`` (qBraid Lab bills this to your credits).
"""

from __future__ import annotations

import os
import numpy as np
from typing import Literal, Optional, Sequence

from braket.ahs import (
    AnalogHamiltonianSimulation,
    AtomArrangement,
    DrivingField,
    LocalDetuning,
    Pattern,
)
from braket.ahs.field import Field
from braket.timings.time_series import TimeSeries


# ---- Aquila hardware limits (SI units: rad/s, seconds, meters) -------------
AQUILA_RABI_MAX = 1.58e7          # max Rabi frequency [rad/s]
AQUILA_DETUNING_MAX = 1.25e8      # |detuning| max [rad/s]
AQUILA_TIME_MAX = 4.0e-6          # max total duration [s]
AQUILA_MIN_SPACING = 4.0e-6       # min atom spacing [m]
AQUILA_AREA_W = 7.5e-5            # max register width [m]
AQUILA_AREA_H = 7.6e-5            # max register height [m]
AQUILA_MIN_TIME_STEP = 5.0e-8     # min time step [s]


def _ts(times: Sequence[float], values: Sequence[float]) -> TimeSeries:
    """Piecewise-linear TimeSeries from (times, values)."""
    ts = TimeSeries()
    for t, v in zip(times, values):
        ts.put(float(t), float(v))
    return ts


# ---- Local-sim parallelism (the AHS programs are embarrassingly parallel) ----
# Each worker process holds one reservoir + one LocalSimulator and processes a
# slice of the inputs. This is the main free speed-up on a multi-core box; GPU
# only helps at much larger atom counts (the 5-8 atom Hilbert space is tiny).
_WORKER: dict = {}


def _init_worker(reservoir, shots: int) -> None:  # pragma: no cover - subprocess
    from braket.devices import LocalSimulator
    _WORKER["res"] = reservoir
    _WORKER["shots"] = shots
    _WORKER["dev"] = LocalSimulator("braket_ahs")


def _worker_features(x):  # pragma: no cover - subprocess
    res, dev, shots = _WORKER["res"], _WORKER["dev"], _WORKER["shots"]
    result = dev.run(res.build_program(x), shots=shots).result()
    return res.features_from_measurements(result.measurements)


class QueraReservoir:
    """
    Neutral-atom analog reservoir for QuEra Aquila (Braket AHS).

    Parameters
    ----------
    n_atoms          : reservoir size (5-10 recommended; local sim handles ~<=14).
    geometry         : "chain" | "ring" | "random2d" — fixed atom layout (the
                       disordered couplings). "random2d" is the richest reservoir.
    spacing          : base atom spacing [m]. Default 6 um (< blockade radius at
                       Omega_max ~ 8-9 um, so neighbours interact).
    total_time       : total evolution time [s] (<= 4 us).
    ramp_time        : Rabi/local-detuning ramp up == ramp down [s].
    rabi_max         : peak global Rabi drive [rad/s] (<= 1.58e7).
    global_detuning  : constant global detuning during the hold [rad/s].
    local_detuning_max: peak local-detuning magnitude [rad/s]; multiplied by the
                       per-site weights w_i in [0,1] that carry the input.
    seed             : RNG seed for the (fixed) random geometry.
    """

    def __init__(
        self,
        n_atoms: int = 6,
        geometry: Literal["chain", "ring", "random2d"] = "random2d",
        spacing: float = 6.0e-6,
        total_time: float = 4.0e-6,
        ramp_time: float = 1.0e-7,
        rabi_max: float = 1.5e7,
        global_detuning: float = 0.0,
        local_detuning_max: float = 2.5e7,
        seed: int = 42,
    ) -> None:
        if rabi_max > AQUILA_RABI_MAX:
            raise ValueError(f"rabi_max {rabi_max} exceeds Aquila max {AQUILA_RABI_MAX}")
        if total_time > AQUILA_TIME_MAX + 1e-12:
            raise ValueError(f"total_time {total_time} exceeds Aquila max {AQUILA_TIME_MAX}")
        if abs(global_detuning) > AQUILA_DETUNING_MAX or local_detuning_max > AQUILA_DETUNING_MAX:
            raise ValueError("detuning exceeds Aquila max")
        if 2 * ramp_time >= total_time:
            raise ValueError("2*ramp_time must be < total_time")

        self.n_atoms = n_atoms
        self.geometry = geometry
        self.spacing = spacing
        self.total_time = total_time
        self.ramp_time = ramp_time
        self.rabi_max = rabi_max
        self.global_detuning = global_detuning
        self.local_detuning_max = local_detuning_max
        self.seed = seed

        self._coords = self._build_coords(np.random.RandomState(seed))

    # ------------------------------------------------------------------
    # Geometry (the fixed, disordered couplings)
    # ------------------------------------------------------------------

    def _build_coords(self, rng: np.random.RandomState) -> np.ndarray:
        n, s = self.n_atoms, self.spacing
        if self.geometry == "chain":
            coords = np.array([[i * s, 0.0] for i in range(n)])
        elif self.geometry == "ring":
            R = s / (2 * np.sin(np.pi / n))
            coords = np.array([[R * np.cos(2 * np.pi * i / n),
                                R * np.sin(2 * np.pi * i / n)] for i in range(n)])
        else:  # random2d — jittered grid that respects the min-spacing constraint
            cols = int(np.ceil(np.sqrt(n)))
            pts = []
            # Jitter capped at 0.1*s so even adjacent atoms stay > 4um apart:
            # min neighbour distance >= s - 2*0.1*s = 0.8*s (= 4.8um at s=6um).
            for i in range(n):
                r, c = divmod(i, cols)
                jitter = rng.uniform(-0.1, 0.1, size=2) * s
                pts.append([c * s + jitter[0], r * s + jitter[1]])
            coords = np.array(pts)

        coords = coords - coords.min(axis=0)             # shift into the first quadrant
        coords = np.round(coords / 1e-8) * 1e-8           # snap to 10 nm position grid
        self._check_geometry(coords)
        return coords

    def _check_geometry(self, coords: np.ndarray) -> None:
        if np.ptp(coords[:, 0]) > AQUILA_AREA_W or np.ptp(coords[:, 1]) > AQUILA_AREA_H:
            raise ValueError("register exceeds Aquila field of view; reduce spacing/n_atoms")
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = np.hypot(*(coords[i] - coords[j]))
                if d < AQUILA_MIN_SPACING - 1e-12:
                    raise ValueError(
                        f"atoms {i},{j} spaced {d*1e6:.2f}um < {AQUILA_MIN_SPACING*1e6:.0f}um min"
                    )

    def register(self) -> AtomArrangement:
        reg = AtomArrangement()
        for (x, y) in self._coords:
            reg.add((float(x), float(y)))
        return reg

    # ------------------------------------------------------------------
    # Program construction
    # ------------------------------------------------------------------

    def _drive(self) -> DrivingField:
        """Global trapezoidal Rabi pulse (starts/ends at 0) + constant detuning."""
        t0, tr, T = 0.0, self.ramp_time, self.total_time
        amp = _ts([t0, tr, T - tr, T], [0.0, self.rabi_max, self.rabi_max, 0.0])
        det = _ts([t0, tr, T - tr, T],
                  [self.global_detuning] * 4) if self.global_detuning else _ts([t0, T], [0.0, 0.0])
        pha = _ts([t0, T], [0.0, 0.0])
        return DrivingField(amplitude=amp, detuning=det, phase=pha)

    def _local_detuning(self, weights: np.ndarray) -> LocalDetuning:
        """Per-site detuning carrying the input: w_i in [0,1] x Delta_local(t)."""
        t0, tr, T = 0.0, self.ramp_time, self.total_time
        mag = _ts([t0, tr, T - tr, T],
                  [0.0, self.local_detuning_max, self.local_detuning_max, 0.0])
        return LocalDetuning(magnitude=Field(time_series=mag, pattern=Pattern(list(weights))))

    @staticmethod
    def _encode(x: np.ndarray, n: int) -> np.ndarray:
        """Map input -> per-site weights in [0,1] (tanh squashing, then [-1,1]->[0,1])."""
        xn = np.tanh(np.asarray(x, dtype=float))
        w = (xn + 1.0) / 2.0
        # tile/trim to one weight per atom
        return np.array([w[i % len(w)] for i in range(n)])

    def build_program(self, x: np.ndarray) -> AnalogHamiltonianSimulation:
        weights = self._encode(x, self.n_atoms)
        H = self._drive() + self._local_detuning(weights)
        return AnalogHamiltonianSimulation(register=self.register(), hamiltonian=H)

    def build_programs(self, X: np.ndarray) -> list[AnalogHamiltonianSimulation]:
        return [self.build_program(x) for x in np.asarray(X, dtype=float)]

    # ------------------------------------------------------------------
    # Feature extraction (Z, ZZ) — matches the gate reservoir layout
    # ------------------------------------------------------------------

    @property
    def n_features(self) -> int:
        n = self.n_atoms
        return n + n * (n - 1) // 2

    def features_from_measurements(self, measurements) -> np.ndarray:
        """
        Z_i = <1 - 2 n_i>, Z_iZ_j = <(1-2n_i)(1-2n_j)>, from AHS shot results.

        Rydberg occupation n_i = 1 when an atom that was successfully prepared
        (pre_sequence==1) ends in the Rydberg state (post_sequence==0). Shots where
        an atom failed to load are excluded from that atom's averages (hardware
        defect handling; on the local simulator every atom loads).
        """
        n = self.n_atoms
        pre = np.array([list(s.pre_sequence) for s in measurements], dtype=float)   # (shots, n)
        post = np.array([list(s.post_sequence) for s in measurements], dtype=float)  # 1=ground,0=Rydberg
        z = 2.0 * post - 1.0                       # ground -> +1, Rydberg -> -1
        present = pre > 0.5

        z_exp = np.array([z[present[:, i], i].mean() if present[:, i].any() else 0.0
                          for i in range(n)])
        zz_vals = []
        for i in range(n):
            for j in range(i + 1, n):
                mask = present[:, i] & present[:, j]
                zz_vals.append((z[mask, i] * z[mask, j]).mean() if mask.any() else 0.0)
        return np.concatenate([z_exp, zz_vals])

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray, device=None, shots: int = 100,
                  verbose: bool = False, n_jobs: int = 1) -> np.ndarray:
        """
        Run every sample's AHS program and return the (Z, ZZ) feature matrix.

        device : a Braket device. None or "local" -> LocalSimulator("braket_ahs")
                 (free, exact). Pass AwsDevice(Devices.QuEra.Aquila) for hardware
                 (discretised to the device grid automatically).
        shots  : measurement shots per program.
        n_jobs : LOCAL-SIM ONLY. Parallel worker processes (the AHS programs are
                 independent). 1 = serial; -1 = all CPU cores. Ignored on hardware
                 (tasks are submitted serially to the QPU).
        """
        X = np.asarray(X, dtype=float)
        local = device is None or device == "local"

        # ---- parallel local-sim path ----
        if local and n_jobs != 1 and len(X) > 1:
            # Default cap: leave headroom. On Windows each worker re-imports the
            # whole numpy/scipy/braket stack, so too many workers can exhaust the
            # paging file; on Linux (e.g. qBraid Lab) fork is far cheaper and this
            # scales much better. _parallel_features retries with fewer workers and
            # finally falls back to serial, so it never hard-crashes.
            workers = os.cpu_count() if n_jobs in (-1, 0) else n_jobs
            workers = max(1, min(workers, len(X), 8))
            feats = self._parallel_features(list(X), shots, workers, verbose)
            if feats is not None:
                return feats
            # else: fall through to the serial path below

        # ---- serial path (local or hardware) ----
        if local:
            from braket.devices import LocalSimulator
            dev = LocalSimulator("braket_ahs")
        else:
            dev = device

        feats = []
        programs = self.build_programs(X)
        for k, prog in enumerate(programs):
            if not local:
                prog = prog.discretize(dev)         # snap to Aquila's value grid
            result = dev.run(prog, shots=shots).result()
            feats.append(self.features_from_measurements(result.measurements))
            if verbose:
                print(f"    sample {k+1}/{len(programs)} done", flush=True)
        return np.asarray(feats)

    def _parallel_features(self, X_list, shots, workers, verbose):
        """Local-sim feature extraction across worker processes.

        Returns the feature matrix, or None if the pool could not run even at 2
        workers (caller then falls back to serial). On a BrokenProcessPool — most
        often low memory / small Windows paging file — it retries with half the
        workers before giving up.
        """
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool
        try:
            if verbose:
                print(f"    local AHS sim on {workers} worker process(es) ...", flush=True)
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                     initargs=(self, shots)) as pool:
                return np.asarray(list(pool.map(_worker_features, X_list, chunksize=1)))
        except BrokenProcessPool:
            if workers > 2:
                half = workers // 2
                print(f"    [warn] worker pool broke at {workers} workers "
                      f"(low memory?); retrying with {half}", flush=True)
                return self._parallel_features(X_list, shots, half, verbose)
            print("    [warn] parallel pool failed; falling back to serial", flush=True)
            return None

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.transform(X)

    def __repr__(self) -> str:
        return (f"QueraReservoir(n_atoms={self.n_atoms}, geometry='{self.geometry}', "
                f"spacing={self.spacing*1e6:.1f}um, T={self.total_time*1e6:.1f}us, "
                f"rabi_max={self.rabi_max:.2e}, n_features={self.n_features})")
