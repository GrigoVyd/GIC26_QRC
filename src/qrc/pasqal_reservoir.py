"""
Pasqal Fresnel reservoir — neutral-atom analog reservoir via Pulser.

A second neutral-atom platform alongside QuEra Aquila (src/qrc/quera_reservoir.py).
Both are Rydberg-atom analog machines, so the same reservoir idea applies — but
the *real* Pasqal ``AnalogDevice`` (Fresnel) exposes only a GLOBAL rydberg channel
(global Rabi + global detuning, no per-site addressing / DMM). QuEra Aquila, by
contrast, has per-site local detuning. That hardware difference forces a different
input encoding, which is itself a useful cross-platform finding:

    QuEra Aquila   : input -> per-site LOCAL detuning weights w_i           (spatial)
    Pasqal Fresnel : input -> GLOBAL detuning waveform values over time     (temporal)

Both keep a fixed random atom geometry as the disordered reservoir coupling
(van der Waals C6/r^6), a global Rabi drive as the transverse field, and read out
Rydberg occupation as Z_i = 1 - 2 n_i and Z_i Z_j — the SAME feature layout the
Ridge readout already consumes.

Validate for free on Pulser's local emulator (QutipEmulator, via qutip). The
hardware path is Pasqal Fresnel through Pulser/pasqal-cloud or Azure Quantum (both
reachable via qBraid); note Fresnel was OFFLINE for this account at build time.
"""

from __future__ import annotations

import os
import numpy as np
from typing import Literal, Sequence as Seq

from pulser import Register, Sequence, Pulse
from pulser.waveforms import InterpolatedWaveform
from pulser.devices import AnalogDevice


# ---- Pasqal AnalogDevice (Fresnel) limits, read from the device spec ----
_RGB = AnalogDevice.channels["rydberg_global"]
FRESNEL_MAX_AMP = float(_RGB.max_amp)               # rad/us  (~12.57 = 2pi*2MHz)
FRESNEL_MAX_DETUNING = float(_RGB.max_abs_detuning)  # rad/us  (~125.7 = 2pi*20MHz)
FRESNEL_MIN_SPACING = float(AnalogDevice.min_atom_distance)   # um (5)
FRESNEL_MAX_RADIAL = float(AnalogDevice.max_radial_distance)  # um (38)
FRESNEL_MAX_ATOMS = int(AnalogDevice.max_atom_num)            # 80


# ---- local-emulator parallelism (mirrors quera_reservoir) ------------------
_WORKER: dict = {}


def _init_worker(reservoir, shots: int) -> None:  # pragma: no cover - subprocess
    _WORKER["res"] = reservoir
    _WORKER["shots"] = shots


def _worker_features(x):  # pragma: no cover - subprocess
    res, shots = _WORKER["res"], _WORKER["shots"]
    return res._features_local(x, shots)


class PasqalReservoir:
    """
    Neutral-atom analog reservoir for Pasqal Fresnel (Pulser AnalogDevice).

    Parameters
    ----------
    n_atoms        : reservoir size (emulator handles <=~12 comfortably).
    geometry       : "chain" | "ring" | "random2d" — fixed atom layout.
    spacing        : base atom spacing [um] (>= 5 um Fresnel minimum).
    total_time     : pulse duration [ns] (multiple of 4 ns).
    rabi_frac      : peak global Rabi as a fraction of the device max.
    detuning_frac  : input detuning swing as a fraction of the device max.
    seed           : RNG seed for the (fixed) random geometry.
    """

    def __init__(
        self,
        n_atoms: int = 5,
        geometry: Literal["chain", "ring", "random2d"] = "random2d",
        spacing: float = 7.0,
        total_time: int = 2000,
        rabi_frac: float = 0.8,
        detuning_frac: float = 0.6,
        seed: int = 42,
    ) -> None:
        if n_atoms > FRESNEL_MAX_ATOMS:
            raise ValueError(f"n_atoms {n_atoms} > Fresnel max {FRESNEL_MAX_ATOMS}")
        if spacing < FRESNEL_MIN_SPACING:
            raise ValueError(f"spacing {spacing}um < Fresnel min {FRESNEL_MIN_SPACING}um")
        self.n_atoms = n_atoms
        self.geometry = geometry
        self.spacing = spacing
        self.total_time = int(total_time) - (int(total_time) % 4)  # clock period
        self.rabi = rabi_frac * FRESNEL_MAX_AMP
        self.det_scale = detuning_frac * FRESNEL_MAX_DETUNING
        self.seed = seed
        self._coords = self._build_coords(np.random.RandomState(seed))

    # ---- geometry (fixed disordered couplings) ----
    def _build_coords(self, rng) -> np.ndarray:
        n, s = self.n_atoms, self.spacing
        if self.geometry == "chain":
            coords = np.array([[i * s, 0.0] for i in range(n)])
        elif self.geometry == "ring":
            R = s / (2 * np.sin(np.pi / n))
            coords = np.array([[R * np.cos(2 * np.pi * i / n),
                                R * np.sin(2 * np.pi * i / n)] for i in range(n)])
        else:  # random2d — jittered grid, jitter capped so spacing stays >= min
            cols = int(np.ceil(np.sqrt(n)))
            pts = []
            for i in range(n):
                r, c = divmod(i, cols)
                jit = rng.uniform(-0.1, 0.1, size=2) * s
                pts.append([c * s + jit[0], r * s + jit[1]])
            coords = np.array(pts)
        coords = coords - coords.mean(axis=0)          # centre (Pulser uses centred reg)
        self._check_geometry(coords)
        return coords

    def _check_geometry(self, coords: np.ndarray) -> None:
        if np.max(np.linalg.norm(coords, axis=1)) > FRESNEL_MAX_RADIAL:
            raise ValueError("register exceeds Fresnel max radial distance; reduce spacing")
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = float(np.hypot(*(coords[i] - coords[j])))
                if d < FRESNEL_MIN_SPACING - 1e-9:
                    raise ValueError(f"atoms {i},{j} spaced {d:.2f}um < {FRESNEL_MIN_SPACING}um")

    def register(self) -> Register:
        return Register.from_coordinates(self._coords, prefix="q")

    # ---- program construction ----
    def build_sequence(self, x: np.ndarray) -> Sequence:
        """Global Rabi pulse + input-encoding global detuning waveform."""
        seq = Sequence(self.register(), AnalogDevice)
        seq.declare_channel("ryd", "rydberg_global")
        T = self.total_time
        amp = InterpolatedWaveform(T, [0.0, self.rabi, self.rabi, 0.0])
        dvals = np.clip(np.tanh(np.asarray(x, dtype=float)), -1, 1) * self.det_scale
        det = InterpolatedWaveform(T, [0.0, *dvals, 0.0])
        seq.add(Pulse(amp, det, 0.0), "ryd")
        return seq

    # ---- feature extraction (Z, ZZ) from sampled bitstrings ----
    @property
    def n_features(self) -> int:
        n = self.n_atoms
        return n + n * (n - 1) // 2

    def features_from_counts(self, counts: dict) -> np.ndarray:
        """Z_i = <1-2 n_i>, Z_iZ_j from a {bitstring: count} sample ('1'=Rydberg)."""
        n = self.n_atoms
        z = np.zeros(n)
        zz = np.zeros((n, n))
        tot = 0
        for bs, c in counts.items():
            b = np.array([int(ch) for ch in bs], dtype=float)
            s = 1.0 - 2.0 * b
            z += s * c
            zz += np.outer(s, s) * c
            tot += c
        tot = tot or 1
        z /= tot; zz /= tot
        zz_vals = [zz[i, j] for i in range(n) for j in range(i + 1, n)]
        return np.concatenate([z, zz_vals])

    def _features_local(self, x: np.ndarray, shots: int) -> np.ndarray:
        from pulser_simulation import QutipEmulator
        seq = self.build_sequence(x)
        res = QutipEmulator.from_sequence(seq).run()
        counts = res.sample_final_state(N_samples=shots)
        return self.features_from_counts(counts)

    # ---- execution ----
    def transform(self, X: np.ndarray, device: str = "local", shots: int = 200,
                  n_jobs: int = 1, verbose: bool = False) -> np.ndarray:
        """
        Run every sample on Pulser's local emulator -> (Z, ZZ) feature matrix.

        device : "local" only here (hardware submission to Fresnel goes through
                 pulser-pasqal / Azure and is handled in the experiment script).
        n_jobs : local parallel workers (1=serial, -1=all cores), with the same
                 retry/serial-fallback as QueraReservoir.
        """
        if device != "local":
            raise ValueError("PasqalReservoir.transform runs the local emulator; "
                             "use the experiment's hardware path for Fresnel.")
        X = np.asarray(X, dtype=float)
        if n_jobs != 1 and len(X) > 1:
            feats = self._parallel(list(X), shots, n_jobs)
            if feats is not None:
                return feats
        feats = []
        for k, x in enumerate(X):
            feats.append(self._features_local(x, shots))
            if verbose:
                print(f"    sample {k+1}/{len(X)} done", flush=True)
        return np.asarray(feats)

    def _parallel(self, X_list, shots, n_jobs):
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool
        workers = os.cpu_count() if n_jobs in (-1, 0) else n_jobs
        workers = max(1, min(workers, len(X_list), 8))
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                     initargs=(self, shots)) as pool:
                return np.asarray(list(pool.map(_worker_features, X_list, chunksize=1)))
        except BrokenProcessPool:
            if workers > 2:
                return self._parallel(X_list, shots, workers // 2)
            print("    [warn] parallel pool failed; falling back to serial", flush=True)
            return None

    def transform_sequential(self, X: np.ndarray, shots: int = 200,
                             feedback_scale: float = 0.5) -> np.ndarray:
        """
        Recurrent feature extraction with temporal Z feedback (memory across days).

        Fresnel is global-only, so per-site feedback is impossible; instead the
        previous step's <Z_i> are *replayed* as leading points of the global
        detuning waveform before the new input — injecting memory temporally.
        Sequential by construction -> not parallel.
        """
        from pulser_simulation import QutipEmulator
        X = np.asarray(X, dtype=float)
        z = np.zeros(self.n_atoms)
        feats = []
        for x in X:
            seq = self._build_sequence_recurrent(x, z, feedback_scale)
            counts = QutipEmulator.from_sequence(seq).run().sample_final_state(N_samples=shots)
            f = self.features_from_counts(counts)
            feats.append(f)
            z = f[: self.n_atoms].copy()
        return np.asarray(feats)

    def _build_sequence_recurrent(self, x, z_prev, fb) -> Sequence:
        seq = Sequence(self.register(), AnalogDevice)
        seq.declare_channel("ryd", "rydberg_global")
        T = self.total_time
        amp = InterpolatedWaveform(T, [0.0, self.rabi, self.rabi, 0.0])
        xn = np.clip(np.tanh(np.asarray(x, dtype=float)), -1, 1) * self.det_scale
        zf = np.clip(fb * np.asarray(z_prev), -1, 1) * self.det_scale
        det = InterpolatedWaveform(T, [0.0, *zf, *xn, 0.0])  # replay memory, then input
        seq.add(Pulse(amp, det, 0.0), "ryd")
        return seq

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.transform(X)

    def __repr__(self) -> str:
        return (f"PasqalReservoir(n_atoms={self.n_atoms}, geometry='{self.geometry}', "
                f"spacing={self.spacing}um, T={self.total_time}ns, "
                f"rabi={self.rabi:.2f}rad/us, n_features={self.n_features})")
