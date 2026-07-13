"""IQM Resonance connection, native-grid selection, and Z/ZZ readout correction."""

from __future__ import annotations

import os
from itertools import combinations

import numpy as np
import rustworkx as rx
from qiskit import transpile


RESONANCE_URL = "https://resonance.meetiqm.com"


def get_resonance_backend(device: str = "emerald"):
    """Return an authenticated IQM backend without persisting the token."""
    if not os.environ.get("IQM_TOKEN"):
        raise RuntimeError("IQM_TOKEN is not available in the process environment")
    from iqm.qiskit_iqm import IQMProvider

    return IQMProvider(RESONANCE_URL, quantum_computer=device).get_backend()


def get_quality_metrics(backend):
    """Extract per-qubit readout/1q/coherence and per-edge CZ fidelity."""
    qms = backend.client.get_quality_metric_set()
    name_to_idx = {f"QB{i + 1}": i for i in range(backend.num_qubits)}
    qubits = {i: {} for i in range(backend.num_qubits)}
    pairs = {}
    for observation in qms.observations:
        field, value = observation.dut_field, observation.value
        parts = field.split(".")
        if field.startswith("metrics.ssro.measure.constant."):
            idx = name_to_idx.get(parts[4])
            if idx is None:
                continue
            metric = ".".join(parts[5:])
            mapping = {
                "fidelity": "readout_fidelity",
                "error_0_to_1": "error_0_to_1",
                "error_1_to_0": "error_1_to_0",
            }
            if metric in mapping:
                qubits[idx][mapping[metric]] = value
        elif field.startswith("metrics.rb.prx.drag_crf_sx."):
            idx = name_to_idx.get(parts[4])
            if idx is not None and parts[5].startswith("fidelity"):
                qubits[idx]["one_q_fidelity"] = value
        elif field.startswith("metrics.rb.clifford.uz_cz."):
            if "__" not in parts[4] or not parts[5].startswith("fidelity"):
                continue
            qa, qb = parts[4].split("__")
            ia, ib = name_to_idx.get(qa), name_to_idx.get(qb)
            if ia is not None and ib is not None:
                pairs[tuple(sorted((ia, ib)))] = value
        elif field.startswith("characterization.model."):
            idx = name_to_idx.get(parts[2])
            if idx is None:
                continue
            if parts[3] == "t1_time":
                qubits[idx]["t1"] = value
            elif parts[3] == "t2_echo_time":
                qubits[idx]["t2"] = value
    return qubits, pairs


def _grid_edges(rows: int, cols: int):
    edges = []
    for row in range(rows):
        for col in range(cols):
            q = row * cols + col
            if col + 1 < cols:
                edges.append((q, q + 1))
            if row + 1 < rows:
                edges.append((q, q + cols))
    return edges


def _enumerate_subgrids(backend, rows: int, cols: int, allowed: set[int]):
    target = rx.PyGraph()
    target.add_nodes_from(range(rows * cols))
    for a, b in _grid_edges(rows, cols):
        target.add_edge(a, b, None)

    device = rx.PyGraph()
    device.add_nodes_from(range(backend.num_qubits))
    seen = set()
    for a, b in backend.coupling_map:
        edge = tuple(sorted((a, b)))
        if a in allowed and b in allowed and edge not in seen:
            device.add_edge(a, b, None)
            seen.add(edge)

    layouts, unique = [], set()
    for mapping in rx.vf2_mapping(device, target, subgraph=True, induced=False):
        inverse = {logical: physical for physical, logical in mapping.items()}
        layout = tuple(inverse[i] for i in range(rows * cols))
        if layout not in unique:
            layouts.append(list(layout))
            unique.add(layout)
    return layouts


def select_native_grid(
    backend,
    rows: int = 3,
    cols: int = 3,
    min_readout: float = 0.90,
    min_one_q: float = 0.99,
    min_t1: float = 10e-6,
    min_t2: float = 5e-6,
    min_cz: float = 0.90,
):
    """Select the lowest-error rectangular patch from live calibration data."""
    metrics, cz = get_quality_metrics(backend)
    good = {
        q for q, m in metrics.items()
        if m.get("readout_fidelity", 0) >= min_readout
        and m.get("one_q_fidelity", 0) >= min_one_q
        and m.get("t1", 0) >= min_t1
        and m.get("t2", 0) >= min_t2
    }
    good = {
        q for q in good
        if any(q in edge and fidelity >= min_cz for edge, fidelity in cz.items())
    }
    layouts = _enumerate_subgrids(backend, rows, cols, good)
    if not layouts:
        raise RuntimeError(f"no healthy native {rows}x{cols} grid is available")

    logical_edges = _grid_edges(rows, cols)

    def score(layout):
        edge_cost = sum(
            1.0 - cz.get(tuple(sorted((layout[a], layout[b]))), 0.0)
            for a, b in logical_edges
        )
        readout_cost = 0.10 * sum(
            1.0 - metrics[q].get("readout_fidelity", 0.0) for q in layout
        )
        return edge_cost + readout_cost

    layout = min(layouts, key=score)
    selected_fidelities = [
        cz[tuple(sorted((layout[a], layout[b])))] for a, b in logical_edges
    ]
    return {
        "layout": layout,
        "layout_names": [backend.index_to_qubit_name(q) for q in layout],
        "score": float(score(layout)),
        "candidate_count": len(layouts),
        "edge_fidelity_min": float(np.min(selected_fidelities)),
        "edge_fidelity_mean": float(np.mean(selected_fidelities)),
        "metrics": metrics,
        "cz_fidelities": cz,
    }


def compile_native(circuits, backend, layout, seed: int = 44):
    """Compile logical circuits onto an explicit layout and reject any SWAP."""
    compiled = [
        transpile(
            circuit, backend=backend, initial_layout=layout,
            optimization_level=3, seed_transpiler=seed,
        )
        for circuit in circuits
    ]
    swaps = [c.count_ops().get("swap", 0) for c in compiled]
    if any(swaps):
        raise RuntimeError(f"native-layout invariant failed; SWAP counts={swaps}")
    return compiled


def mitigate_z_zz(features, layout, metrics):
    """Independent asymmetric readout correction for Z and ZZ expectations."""
    features = np.asarray(features, dtype=float)
    one_row = features.ndim == 1
    rows = features[None, :] if one_row else features.copy()
    n = len(layout)
    e01 = np.asarray([metrics[q]["error_0_to_1"] for q in layout])
    e10 = np.asarray([metrics[q]["error_1_to_0"] for q in layout])
    scale = 1.0 - e01 - e10
    bias = e10 - e01
    pairs = list(combinations(range(n), 2))
    corrected = []
    for row in rows:
        z = (row[:n] - bias) / scale
        zz = []
        for k, (a, b) in enumerate(pairs):
            numerator = (
                row[n + k] - scale[a] * bias[b] * z[a]
                - bias[a] * scale[b] * z[b] - bias[a] * bias[b]
            )
            zz.append(numerator / (scale[a] * scale[b]))
        corrected.append(np.clip(np.r_[z, zz], -1.0, 1.0))
    out = np.asarray(corrected)
    return out[0] if one_row else out
