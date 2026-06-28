"""
Phase 3 summary figure: the GPU -> QPU timeline vs classical benchmarks.

Panel A: full benchmark landscape (RMSE on annualised-vol scale, 746-day test).
Panel B: the GPU -> neutral-atom QPU -> quantum-annealer QPU progression on the
         GARCH-residual hybrid, zoomed on the GARCH benchmark crossing.

Numbers are the headline results from this Phase 3 study (see docs/phase3_report.md).
Produces results/phase3_timeline.png (+ _doc.png @ 1200 DPI).
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# --- headline numbers (RMSE, annualised vol; 746-day test) ---
GARCH = 0.00795
LSTM = 0.01066

# Full landscape: (label, rmse, kind)
LANDSCAPE = [
    ("LSTM (1L,32h)",            0.01066, "classical"),
    ("Ridge (raw)",             0.00976, "classical"),
    ("ESN (200)",               0.00970, "classical"),
    ("QRC gate 5q (raw)",       0.00967, "quantum"),
    ("AR(5)",                   0.00954, "classical"),
    ("Persistence",             0.00941, "classical"),
    ("Per-site local (Rydberg, hybrid)", 0.00823, "neutral"),
    ("Ising-SA signed (GPU, hybrid)",    0.00814, "ising"),
    ("Pasqal neutral-atom (hybrid)",     0.00806, "neutral"),
    ("Ridge residual (ablation)",        0.00805, "classical"),
    ("GARCH(1,1)",                       0.00795, "garch"),
    ("QRC quantum annealer (hybrid)",    0.00784, "quantum_adv"),
]

# Timeline: best GARCH-residual RMSE per platform tier (the progression)
TIERS = ["GPU Ising\n(Amplify / SA)", "Neutral-atom QPU\n(QuEra / Pasqal)",
         "Quantum-annealer QPU\n(D-Wave)"]
TIER_RMSE = [0.00814, 0.00806, 0.00784]

COLORS = {"classical": "#9e9e9e", "quantum": "#2166ac", "neutral": "#7c3aed",
          "ising": "#1b9e77", "garch": "#000000", "quantum_adv": "#d4a017"}


def main() -> None:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # ---- Panel A: full landscape ----
    rows = sorted(LANDSCAPE, key=lambda r: r[1], reverse=True)
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [COLORS[r[2]] for r in rows]
    bars = axA.barh(labels, vals, color=cols)
    axA.bar_label(bars, fmt="%.5f", padding=3, fontsize=7)
    axA.axvline(GARCH, color="black", ls="--", lw=1.2, alpha=0.7)
    axA.set_xlabel("RMSE (annualised volatility) — lower is better")
    axA.set_title("A. Benchmark landscape (746-day test)")
    axA.set_xlim(0.0072, 0.0112)
    axA.grid(True, axis="x", alpha=0.25)
    # legend
    from matplotlib.patches import Patch
    leg = [Patch(color=COLORS["classical"], label="classical"),
           Patch(color=COLORS["quantum"], label="QRC (sim)"),
           Patch(color=COLORS["neutral"], label="neutral-atom QPU"),
           Patch(color=COLORS["ising"], label="Ising machine (GPU)"),
           Patch(color=COLORS["quantum_adv"], label="quantum annealer (beats GARCH)")]
    axA.legend(handles=leg, fontsize=7, loc="lower right")

    # ---- Panel B: GPU -> QPU timeline ----
    x = list(range(len(TIERS)))
    axB.axhspan(0.0070, GARCH, color="#2ca02c", alpha=0.08)
    axB.text(0.05, GARCH - 0.00004, "quantum-advantage zone (beats GARCH)",
             fontsize=8, color="#2ca02c")
    axB.axhline(GARCH, color="black", ls="--", lw=1.3, label=f"GARCH(1,1) = {GARCH:.5f}")
    axB.axhline(LSTM, color="#d6604d", ls=":", lw=1.3, label=f"LSTM = {LSTM:.5f}")
    axB.plot(x, TIER_RMSE, "-o", color="#444", lw=1.5, markersize=8, zorder=5)
    # highlight the advantage point
    axB.plot([x[-1]], [TIER_RMSE[-1]], "o", color=COLORS["quantum_adv"],
             markersize=14, zorder=6, markeredgecolor="black")
    for xi, v in zip(x, TIER_RMSE):
        axB.annotate(f"{v:.5f}", (xi, v), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9)
    axB.set_xticks(x); axB.set_xticklabels(TIERS, fontsize=8)
    axB.set_ylabel("RMSE (GARCH-residual hybrid)")
    axB.set_title("B. GPU → QPU timeline (predict GARCH residual)")
    axB.set_ylim(0.00770, 0.00830)
    axB.legend(fontsize=8, loc="upper right")
    axB.grid(True, axis="y", alpha=0.25)

    fig.suptitle("GIC 2026 Phase 3 — QRC vs classical benchmarks: from GPU Ising machines to QPUs",
                 fontsize=13, y=1.00)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "phase3_timeline.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".png", "_doc.png"), dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {os.path.relpath(out)} (+ _doc.png @ 1200 DPI)")


if __name__ == "__main__":
    main()
