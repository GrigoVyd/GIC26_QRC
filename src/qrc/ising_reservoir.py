"""
Ising-machine reservoir — the signed-coupling substrate where the GARCH-hybrid
headline lives.

The Phase-2 win came from a transverse-field Ising reservoir with **signed,
all-to-all couplings** J_ij in [-1, 1] (a D-Wave-style model). Neutral-atom
hardware (QuEra/Pasqal) can't realise signed couplings — but Ising machines can.
This reservoir maps the input into local fields h_i, keeps fixed random signed
J_ij, samples the Ising model, and reads <s_i>, <s_i s_j> as the (Z, ZZ) features
the Ridge readout already consumes.

One reservoir, three backends (multiple strategies):
  * "sa"      — dwave.samplers.SimulatedAnnealingSampler. Local, free, classical
                finite-temperature sampling. Fast (no ODE), good for validation.
  * "amplify" — Fixstars Amplify Annealing Engine (GPU Ising machine, cloud). Fast,
                accessible (free tier); classical signed-coupling.
  * "dwave"   — D-Wave Advantage (real quantum annealer, Leap). The faithful
                quantum-dynamics platform closest to the Phase-2 annealer sim.
  * "exact"   — dimod.ExactSolver Boltzmann expectations (n<=~12), the noiseless
                finite-T reference.

Physics note: SA/Amplify sample the *classical* Boltzmann distribution (no
transverse field); D-Wave performs *quantum* annealing (transverse field). The
Phase-2 reservoir that beat GARCH was quantum-dynamics, so D-Wave is the faithful
hardware; SA/Amplify test whether signed couplings alone (classically sampled)
already suffice.
"""

from __future__ import annotations

import os
import numpy as np
from typing import Literal, Optional


class IsingReservoir:
    """
    Signed-coupling Ising reservoir with a pluggable sampler backend.

    Parameters
    ----------
    n_spins      : reservoir size.
    connectivity : "all-to-all" | "random" | "linear" — graph of J_ij.
    density      : edge fraction when connectivity="random".
    input_scale  : multiplier on the data-driven local fields h_i = scale*tanh(x_i).
                   Probes the dynamical regime (edge of chaos near ~1).
    J_scale      : multiplier on the fixed random signed couplings.
    beta         : final inverse temperature for sampling (finite-T richness).
    seed         : RNG seed for the fixed J_ij.
    """

    def __init__(
        self,
        n_spins: int = 10,
        connectivity: Literal["all-to-all", "random", "linear"] = "all-to-all",
        density: float = 0.5,
        input_scale: float = 1.0,
        J_scale: float = 1.0,
        beta: float = 2.0,
        seed: int = 42,
    ) -> None:
        self.n_spins = n_spins
        self.connectivity = connectivity
        self.density = density
        self.input_scale = input_scale
        self.J_scale = J_scale
        self.beta = beta
        self.seed = seed
        rng = np.random.RandomState(seed)
        self._edges = self._build_edges(rng)
        self._J = rng.uniform(-1.0, 1.0, len(self._edges) or 1) * J_scale  # signed!

    def _build_edges(self, rng) -> list[tuple[int, int]]:
        n = self.n_spins
        if self.connectivity == "linear":
            return [(i, i + 1) for i in range(n - 1)]
        if self.connectivity == "all-to-all":
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        k = max(n - 1, int(len(all_pairs) * self.density))
        idx = rng.choice(len(all_pairs), min(k, len(all_pairs)), replace=False)
        return [all_pairs[i] for i in sorted(idx)]

    @property
    def n_features(self) -> int:
        n = self.n_spins
        return n + n * (n - 1) // 2

    # ---- Ising model for an input ----
    def _ising(self, x: np.ndarray) -> tuple[dict, dict]:
        xn = np.tanh(np.asarray(x, dtype=float))
        h = {i: float(self.input_scale * xn[i % len(xn)]) for i in range(self.n_spins)}
        J = {e: float(self._J[k]) for k, e in enumerate(self._edges)}
        return h, J

    def _features(self, sample: np.ndarray, occ: np.ndarray) -> np.ndarray:
        """<s_i>, <s_i s_j> from a (num_reads, n) +/-1 sample matrix."""
        w = occ / occ.sum()
        z = (sample * w[:, None]).sum(0)                       # (n,)
        zz_mat = (sample * w[:, None]).T @ sample             # weighted <s_i s_j>
        n = self.n_spins
        zz = [zz_mat[i, j] for i in range(n) for j in range(i + 1, n)]
        return np.concatenate([z, zz])

    # ---- backends ----
    def _sample_sa(self, h, J, num_reads, num_sweeps=100):
        from dwave.samplers import SimulatedAnnealingSampler
        ss = SimulatedAnnealingSampler().sample_ising(
            h, J, num_reads=num_reads, num_sweeps=num_sweeps,
            beta_range=[0.1, self.beta], seed=self.seed)
        return ss.record.sample.astype(float), ss.record.num_occurrences.astype(float)

    def _sample_exact(self, h, J):
        import dimod
        ss = dimod.ExactSolver().sample_ising(h, J)
        E = ss.record.energy
        p = np.exp(-self.beta * (E - E.min())); p /= p.sum()    # Boltzmann at beta
        return ss.record.sample.astype(float), p

    def _sample_dwave(self, h, J, num_reads, anneal_time=20.0):
        from dwave.system import DWaveSampler, EmbeddingComposite
        sampler = EmbeddingComposite(DWaveSampler())            # needs Leap token
        ss = sampler.sample_ising(h, J, num_reads=num_reads, annealing_time=anneal_time)
        return ss.record.sample.astype(float), ss.record.num_occurrences.astype(float)

    # Fixstars Amplify exposes many Ising machines behind ONE API (solve(model,
    # client)). "amplify" -> Fixstars AE (GPU); "toshiba" -> Toshiba SQBM2 (classical
    # SBM, GPU/FPGA); "fujitsu"/"hitachi"/"nec"/"dwave_amplify" also available. All
    # classical (SBM/AE) machines are the same *tier* as SA -- no transverse field,
    # so they reproduce the classical-competitive result but not the GARCH-beating
    # edge (that needs D-Wave quantum annealing via the "dwave" backend).
    _AMPLIFY_CLIENTS = {
        "amplify": "FixstarsClient", "toshiba": "ToshibaSQBM2Client",
        "fujitsu": "FujitsuDA4Client", "hitachi": "HitachiClient",
        "nec": "NECVA2Client", "dwave_amplify": "DWaveSamplerClient",
    }

    def _make_amplify_client(self, kind: str, timeout_ms: int):
        import amplify
        client = getattr(amplify, self._AMPLIFY_CLIENTS[kind])()
        # token: a kind-specific env var (e.g. TOSHIBA_TOKEN) or the Amplify token
        token = os.environ.get(f"{kind.upper()}_TOKEN") or os.environ.get("AMPLIFY_TOKEN", "")
        if token:
            client.token = token
        url = os.environ.get(f"{kind.upper()}_URL")
        if url:
            client.url = url
        # timeout / multi-sample knobs differ per client; set what exists
        p = client.parameters
        for attr, val in (("timeout", timeout_ms), ("num_outputs", 0),
                          ("multishot", True), ("maxout", 100)):
            try:
                if hasattr(p, attr):
                    setattr(p, attr, val)
            except Exception:
                pass
        return client

    def _sample_amplify(self, h, J, num_reads, kind="amplify", timeout_ms=1000, client=None):
        import amplify
        gen = amplify.VariableGenerator()
        s = gen.array("Ising", self.n_spins)
        obj = sum(h[i] * s[i] for i in h)
        for (i, j), Jij in zip(self._edges, self._J):
            obj = obj + float(Jij) * s[i] * s[j]
        if client is None:
            client = self._make_amplify_client(kind, timeout_ms)
        result = amplify.solve(obj, client)
        rows = []
        for sol in result:
            try:
                rows.append(np.asarray(s.evaluate(sol.values), dtype=float))
            except Exception:
                rows.append(np.array([float(sol.values[s[i]]) for i in range(self.n_spins)]))
        if not rows:                                   # only the best solution returned
            rows.append(np.asarray(s.evaluate(result.best.values), dtype=float))
        sample = np.asarray(rows, dtype=float)
        return sample, np.ones(len(sample))

    def _sample(self, x, backend, num_reads, **kw):
        h, J = self._ising(x)
        if backend == "sa":
            return self._sample_sa(h, J, num_reads, **kw)
        if backend == "exact":
            return self._sample_exact(h, J)
        if backend == "dwave":
            return self._sample_dwave(h, J, num_reads, **kw)
        if backend in self._AMPLIFY_CLIENTS:           # amplify / toshiba / fujitsu / ...
            return self._sample_amplify(h, J, num_reads, kind=backend, **kw)
        raise ValueError(f"unknown backend {backend}")

    # ---- public API ----
    def transform(self, X: np.ndarray, backend: str = "sa", num_reads: int = 200,
                  verbose: bool = False, **kw) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        feats = []
        for k, x in enumerate(X):
            sample, occ = self._sample(x, backend, num_reads, **kw)
            feats.append(self._features(sample, occ))
            if verbose and (k + 1) % 200 == 0:
                print(f"    {k+1}/{len(X)}", flush=True)
        return np.asarray(feats)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def __repr__(self) -> str:
        return (f"IsingReservoir(n_spins={self.n_spins}, conn='{self.connectivity}', "
                f"input_scale={self.input_scale}, beta={self.beta}, "
                f"n_features={self.n_features})")
