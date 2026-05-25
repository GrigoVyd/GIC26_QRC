"""
Architecture diagram for the GIC 2026 Phase 2 paper.

Produces figures/architecture.png at 1200 DPI. The diagram shows the data
flow: input feature vector -> standardisation -> longitudinal-bias encoding
on n_qubits -> Trotterized transverse-field Ising evolution -> Z + ZZ
expectation readout -> hybrid Ridge head -> next-day RV prediction.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _box(ax, x, y, w, h, text, fc="#e8edf3", ec="#1f3a5f", lw=1.2, fs=8):
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.04",
                          facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color="#0d1d33")


def _arrow(ax, x0, y0, x1, y1, color="#1f3a5f", lw=1.4, label=None):
    arr = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                          mutation_scale=12, color=color, linewidth=lw,
                          shrinkA=2, shrinkB=2)
    ax.add_patch(arr)
    if label is not None:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.04, label,
                ha="center", va="bottom", fontsize=7, color="#5a6877",
                style="italic")


def main():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_aspect("equal")

    # ---- (1) Input feature vector ----
    _box(ax, 0.2, 2.2, 1.6, 1.6,
         "Input features x\n(21-dim)\n\nreturns / |r| / r²\n+ HAR-RV / log-HAR",
         fc="#fff4cc", ec="#9a7d00", fs=7)

    # ---- (2) Standardise + scale ----
    _box(ax, 2.3, 2.6, 1.4, 0.8,
         "Standardise\n(zero mean, unit std)",
         fc="#dfe9d4", ec="#3a7d3a", fs=7)

    # ---- (3) Longitudinal-bias encoding ----
    _box(ax, 4.2, 2.2, 1.7, 1.6,
         "Bias encoding\n\nh_i = α · x_i\non n_input qubits\n(α = input_scale)",
         fc="#fde2e2", ec="#a8312c", fs=7)

    # ---- (4) Quantum reservoir block (the Ising evolution) ----
    res_x, res_y, res_w, res_h = 6.4, 1.4, 3.0, 3.2
    res_box = FancyBboxPatch((res_x, res_y), res_w, res_h,
                              boxstyle="round,pad=0.05,rounding_size=0.06",
                              facecolor="#e6ddff", edgecolor="#5e35b1", linewidth=1.6)
    ax.add_patch(res_box)
    ax.text(res_x + res_w / 2, res_y + res_h - 0.25,
            "Transverse-field Ising reservoir",
            ha="center", va="center", fontsize=9, color="#311b6b", weight="bold")
    ax.text(res_x + res_w / 2, res_y + res_h - 0.65,
            r"$H = -\sum \sigma^x_i + \sum h_i \sigma^z_i + \sum J_{ij} \sigma^z_i \sigma^z_j$",
            ha="center", va="center", fontsize=9, color="#311b6b")
    ax.text(res_x + res_w / 2, res_y + res_h - 1.05,
            r"$U(t) = \exp(-iHt)$  via Trotter (m=3 steps)",
            ha="center", va="center", fontsize=8, color="#311b6b")

    # qubit dots inside the reservoir block (visual schematic)
    n_q_viz = 10
    cy_top = res_y + 1.55
    for i in range(n_q_viz):
        cx = res_x + 0.3 + (res_w - 0.6) * i / (n_q_viz - 1)
        c = Circle((cx, cy_top), 0.10, facecolor="#5e35b1",
                   edgecolor="#311b6b", linewidth=0.8)
        ax.add_patch(c)
    # Edges between non-adjacent qubits to suggest all-to-all
    edges = [(0, 4), (1, 5), (2, 8), (3, 7), (6, 9), (1, 9), (0, 7)]
    for a, b in edges:
        ca = res_x + 0.3 + (res_w - 0.6) * a / (n_q_viz - 1)
        cb = res_x + 0.3 + (res_w - 0.6) * b / (n_q_viz - 1)
        ax.plot([ca, cb], [cy_top, cy_top], color="#5e35b1", lw=0.6, alpha=0.45,
                solid_capstyle="round")
    ax.text(res_x + res_w / 2, cy_top - 0.42,
            "n=10 qubits, all-to-all $J_{ij}$ random fixed",
            ha="center", va="center", fontsize=7, color="#5a6877")

    # initial-state marker
    ax.text(res_x + 0.25, res_y + 0.35,
            r"$|+\rangle^{\otimes n}$", ha="left", va="center",
            fontsize=9, color="#311b6b")

    # ---- (5) Readout: Z and ZZ observables ----
    _box(ax, 9.7, 2.2, 1.6, 1.6,
         "Readout\n\n$\\langle Z_i \\rangle$ (10)\n$\\langle Z_i Z_j \\rangle$ (45)\n= 55 features",
         fc="#cce5ff", ec="#1f3a5f", fs=7)

    # ---- (6) Hybrid Ridge head ----
    _box(ax, 11.5, 2.2, 1.4, 1.6,
         "Hybrid head\n\n[reservoir +\nraw features]\n→ Ridge\n(CV-tuned α)",
         fc="#dfe9d4", ec="#3a7d3a", fs=7)

    # ---- (7) Output ----
    _box(ax, 11.7, 0.5, 1.0, 0.8,
         "$\\hat{y}$ → vol\n(invert log-residual)",
         fc="#fff4cc", ec="#9a7d00", fs=7)

    # ---- Arrows ----
    _arrow(ax, 1.8, 3.0, 2.3, 3.0)
    _arrow(ax, 3.7, 3.0, 4.2, 3.0)
    _arrow(ax, 5.9, 3.0, 6.4, 3.0)
    _arrow(ax, 9.4, 3.0, 9.7, 3.0)
    _arrow(ax, 11.3, 3.0, 11.5, 3.0)
    _arrow(ax, 12.2, 2.2, 12.2, 1.3)

    # Hybrid shortcut: raw features bypass to head
    _arrow(ax, 1.0, 2.2, 1.0, 1.4, color="#9a7d00", lw=1.2)
    ax.plot([1.0, 12.2], [1.4, 1.4], color="#9a7d00", linewidth=1.2, alpha=0.6,
            linestyle=(0, (4, 2)))
    ax.text(6.5, 1.5, "raw features (hybrid shortcut to Ridge)",
            ha="center", va="bottom", fontsize=7, color="#9a7d00", style="italic")

    # ---- Title ----
    fig.text(0.5, 0.93,
             "QRC Architecture — Transverse-field Ising reservoir with hybrid readout",
             ha="center", va="top", fontsize=12, weight="bold", color="#0d1d33")
    fig.text(0.5, 0.05,
             "Target = residual log-vol y = log(RV[t]) − log(RV[t−1]); predicted RV = RV[t−1] · exp(ŷ)",
             ha="center", va="bottom", fontsize=8, style="italic", color="#5a6877")

    out1 = os.path.join(FIG_DIR, "architecture.png")
    out2 = os.path.join(FIG_DIR, "architecture_doc.png")
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    plt.savefig(out2, dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")
    print(f"Saved: {out2} (1200 DPI publication copy)")


if __name__ == "__main__":
    main()
