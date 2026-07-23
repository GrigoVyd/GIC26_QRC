"""Build publication figures for the GIC 2026 Phase 3 report draft.

The figures are generated only from committed/saved result tables.  No quantum
jobs, network calls, or model fitting are performed here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "report" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#17365D"
BLUE = "#2F75B5"
TEAL = "#2A9D8F"
GOLD = "#D69E2E"
RED = "#C44E52"
GRAY = "#687386"
LIGHT = "#EEF3F8"
GREEN = "#4C956C"

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_box(ax, xy, width, height, title, detail, *, fc=LIGHT, ec=BLUE):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.25,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height * 0.68, title, ha="center", va="center", weight="bold", color=NAVY, fontsize=9.0)
    ax.text(x + width / 2, y + height * 0.30, detail, ha="center", va="center", fontsize=7.0, color="#263238")
    return box


def arrow(ax, start, end, *, color=GRAY, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.15,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(ax, (0.02, 0.58), 0.15, 0.25, "Causal market\nstate", "return lags, |r|, r^2,\nHAR-RV features", fc="#F7F9FC")
    add_box(ax, (0.22, 0.68), 0.16, 0.19, "GARCH prior", "known 20/21-day\nvolatility structure", fc="#FFF7E6", ec=GOLD)
    add_box(ax, (0.22, 0.31), 0.16, 0.25, "Fixed QRC\nreservoir", "IQM grid / Rydberg array /\nsigned-Ising sampler", fc="#EAF4FB", ec=BLUE)
    add_box(ax, (0.43, 0.31), 0.16, 0.25, "Shot\nobservables", "<Z_i>, <Z_i Z_j>\n(no quantum training)", fc="#E8F5F2", ec=TEAL)
    add_box(ax, (0.64, 0.31), 0.15, 0.25, "Transfer layer", "QREM + label-free\naffine alignment", fc="#F2EEF8", ec="#7E57C2")
    add_box(ax, (0.84, 0.31), 0.14, 0.25, "Ridge residual\nhead", "validation-locked alpha\nand correction strength", fc="#FCEFEF", ec=RED)
    add_box(ax, (0.67, 0.70), 0.25, 0.18, "Final volatility forecast", r"$\log \widehat{RV}=\log RV_{GARCH}+c f_{QRC}(x)$", fc="#ECF6EF", ec=GREEN)

    arrow(ax, (0.17, 0.705), (0.22, 0.775))
    arrow(ax, (0.17, 0.66), (0.22, 0.45))
    arrow(ax, (0.38, 0.435), (0.43, 0.435))
    arrow(ax, (0.59, 0.435), (0.64, 0.435))
    arrow(ax, (0.79, 0.435), (0.84, 0.435))
    arrow(ax, (0.91, 0.56), (0.84, 0.70), color=RED, rad=-0.08)
    arrow(ax, (0.38, 0.775), (0.67, 0.79), color=GOLD)

    ax.text(0.50, 0.08, "Train only the classical head; keep the physical reservoir fixed and evaluate chronologically.",
            ha="center", va="center", fontsize=9.2, color=NAVY, weight="bold")
    save(fig, "phase3_hybrid_architecture.png")


def simulator_evidence_figure() -> None:
    full = pd.read_csv(ROOT / "results" / "phase2_final_summary.csv")
    hybrid = pd.read_csv(ROOT / "results" / "phase2_garch_hybrid.csv")
    common = pd.read_csv(ROOT / "results" / "hybrid_showcase_common_400_120.csv")

    rows = [
        ("Hybrid QRC\n(simulated TFIM)", float(hybrid.loc[hybrid["config"] == "B-qrc-ensemble", "RMSE"].iloc[0]), TEAL),
        ("GARCH(1,1)", float(full.loc[full["Model"] == "GARCH(1,1)", "RMSE"].iloc[0]), GOLD),
        ("Persistence", float(full.loc[full["Model"] == "Persistence", "RMSE"].iloc[0]), GRAY),
        ("ESN-200", float(full.loc[full["Model"] == "ESN (200 nodes)", "RMSE"].iloc[0]), BLUE),
        ("LSTM", float(full.loc[full["Model"] == "LSTM (1L, 32h)", "RMSE"].iloc[0]), RED),
    ]
    rows.sort(key=lambda x: x[1])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), gridspec_kw={"width_ratios": [1.05, 1.0]})
    ax = axes[0]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [r[2] for r in rows]
    y = np.arange(len(rows))
    ax.barh(y, vals, color=colors, height=0.62)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (lower is better)")
    ax.set_title("A. Full 746-day benchmark")
    ax.set_xlim(0.0074, 0.0112)
    ax.grid(axis="x", alpha=0.22)
    for yi, v in zip(y, vals):
        ax.text(v + 0.00005, yi, f"{v:.5f}", va="center", fontsize=8.5)

    ax = axes[1]
    nisq = common[common["Model"].isin([
        "Pasqal analog QRC hybrid", "Gate QRC hybrid", "QuEra analog QRC hybrid", "Signed-Ising hybrid"
    ])].copy()
    short = {"Pasqal analog QRC hybrid": "Pasqal\nanalog", "Gate QRC hybrid": "Gate\nQRC",
             "QuEra analog QRC hybrid": "QuEra\nanalog", "Signed-Ising hybrid": "Signed\nIsing"}
    nisq["short"] = nisq["Model"].map(short)
    ax.bar(nisq["short"], nisq["Gain_vs_RidgeResidual_%"], color=[TEAL, BLUE, "#5B8FF9", GRAY], width=0.65)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_ylabel("RMSE improvement over linear residual (%)")
    ax.set_title("B. Nonlinear value on common 120-day window")
    ax.set_ylim(0, 2.75)
    ax.grid(axis="y", alpha=0.22)
    for i, v in enumerate(nisq["Gain_vs_RidgeResidual_%"]):
        ax.text(i, v + 0.08, f"{v:.2f}%", ha="center", va="bottom", fontsize=8.8)

    fig.suptitle("Simulator evidence: advantage candidate and hardware-ready ablations", fontsize=12.5, weight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    save(fig, "phase3_simulator_evidence.png")


def hardware_evidence_figure() -> None:
    iqm9 = pd.read_csv(ROOT / "results" / "iqm_emerald_q9_grid_rz_500_summary.csv")
    iqm12 = pd.read_csv(ROOT / "results" / "iqm_emerald_q12_grid_rz_500_summary.csv")
    aquila = pd.read_csv(ROOT / "results" / "quera_aquila_summary_qbraid_aquila_hw_native_neg_23x50.csv")
    amplify = pd.read_csv(ROOT / "results" / "neutral_atom_garch_hybrid_amplify_ae_iqm_locked_400_120.csv")
    toshiba = pd.read_csv(ROOT / "results" / "neutral_atom_garch_hybrid_toshiba_garch_400_120.csv")

    iqm9_best = iqm9[iqm9["Model"] == "hardware_qrem_affine"].iloc[0]
    iqm12_best = iqm12[iqm12["Model"] == "hardware_affine"].iloc[0]
    aq = aquila[aquila["Model"].str.contains("hardware")].iloc[0]
    amp = amplify[amplify["Model"].str.contains("amplify")].iloc[0]
    tos = toshiba[toshiba["Model"].str.contains("toshiba")].iloc[0]

    entries = [
        ("IQM 9q", -float(iqm9_best["Gain_vs_GARCH_%"]), BLUE),
        ("Amplify AE", 100 * (float(amp["RMSE"]) / float(amplify.iloc[0]["RMSE"]) - 1), GRAY),
        ("Aquila 5a", 100 * (float(aq["RMSE"]) / float(aquila[aquila["Model"].str.contains("GARCH")].iloc[0]["RMSE"]) - 1), TEAL),
        ("IQM 12q", -float(iqm12_best["Gain_vs_GARCH_%"]), "#7E57C2"),
        ("Toshiba", 100 * (float(tos["RMSE"]) / float(toshiba.iloc[0]["RMSE"]) - 1), RED),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.35), gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax = axes[0]
    names = [e[0] for e in entries]
    gaps = [e[1] for e in entries]
    colors = [e[2] for e in entries]
    y = np.arange(len(entries))
    ax.barh(y, gaps, color=colors, height=0.62)
    ax.axvline(0, color="#222222", linewidth=0.9)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE gap vs matched GARCH (%)")
    ax.set_title("A. Executed hardware/cloud hybrids")
    ax.grid(axis="x", alpha=0.22)
    ax.set_xlim(-0.16, 0.50)
    for yi, v in zip(y, gaps):
        ax.text(v + 0.012, yi, f"{v:+.3f}%", va="center", ha="left", fontsize=8.6)
    ax.text(-0.145, 4.62, "better", color=GREEN, fontsize=8.5)
    ax.text(0.43, 4.62, "worse", color=RED, fontsize=8.5, ha="right")

    # ---- Panel B: clean 9q -> 12q scaling trade-off schematic ----
    # A twin-axis dual-bar chart made the two metrics ambiguous; a two-column
    # comparison with explicit deltas reads far more clearly at report scale.
    c9 = float(iqm9_best["feature_corr_vs_exact"])
    c12 = float(iqm12_best["feature_corr_vs_exact"])
    g9 = float(iqm9_best["Gain_vs_GARCH_%"])   # +ve = better than GARCH
    g12 = float(iqm12_best["Gain_vs_GARCH_%"])

    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("B. 9q-to-12q scaling trade-off")

    w = 0.195  # box width; arrow endpoints assume half-width w/2

    def value_cell(cx, cy, big, small, *, fc, ec, big_color=NAVY,
                   small_color=None, big_size=12.0):
        h = 0.165
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.006,rounding_size=0.035",
            linewidth=1.3, edgecolor=ec, facecolor=fc))
        ax.text(cx, cy + (0.028 if small else 0.0), big, ha="center", va="center",
                weight="bold", fontsize=big_size, color=big_color)
        if small:
            ax.text(cx, cy - 0.048, small, ha="center", va="center",
                    fontsize=8.0, color=small_color or big_color)

    x9, x12, xlab = 0.58, 0.89, 0.40

    # column headers
    ax.text(x9, 0.915, "9 qubits", ha="center", va="center", weight="bold",
            fontsize=10.5, color=NAVY)
    ax.text(x12, 0.915, "12 qubits", ha="center", va="center", weight="bold",
            fontsize=10.5, color=NAVY)

    # row 1 -- hardware-to-exact feature correlation (blue, improves)
    y1 = 0.63
    ax.text(xlab, y1, "Hardware-to-exact\nfeature correlation", ha="right",
            va="center", fontsize=8.8, color="#263238")
    value_cell(x9, y1, f"{c9:.3f}", None, fc="#EAF4FB", ec=BLUE)
    value_cell(x12, y1, f"{c12:.3f}", None, fc="#EAF4FB", ec=BLUE)
    arrow(ax, (x9 + w / 2 + 0.005, y1), (x12 - w / 2 - 0.005, y1), color=BLUE)
    ax.text((x9 + x12) / 2, y1 + 0.135, f"{c12 - c9:+.3f}", ha="center",
            va="center", fontsize=9.4, weight="bold", color=BLUE)

    # row 2 -- RMSE gap vs GARCH (green better -> red worse)
    y2 = 0.34
    ax.text(xlab, y2, "RMSE gap\nvs GARCH", ha="right", va="center",
            fontsize=8.8, color="#263238")
    value_cell(x9, y2, f"{-g9:+.3f}%", "better", fc="#ECF6EF", ec=GREEN,
               big_color=GREEN, small_color=GREEN, big_size=10.5)
    value_cell(x12, y2, f"{-g12:+.3f}%", "worse", fc="#FCEFEF", ec=RED,
               big_color=RED, small_color=RED, big_size=10.5)
    arrow(ax, (x9 + w / 2 + 0.005, y2), (x12 - w / 2 - 0.005, y2), color=RED)
    ax.text((x9 + x12) / 2, y2 + 0.145, f"{(-g12) - (-g9):.3f} pp worse",
            ha="center", va="center", fontsize=8.6, weight="bold", color=RED)

    # takeaway callout
    ax.add_patch(FancyBboxPatch(
        (0.06, 0.03), 0.88, 0.135,
        boxstyle="round,pad=0.006,rounding_size=0.03",
        linewidth=1.3, edgecolor=GOLD, facecolor="#FFF7E6"))
    ax.text(0.5, 0.097, "Higher feature fidelity did not improve the forecast",
            ha="center", va="center", fontsize=9.0, weight="bold", color="#7A5A00")

    fig.suptitle("Real execution: near-GARCH performance and a topology-aware scaling result", fontsize=12.5, weight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    save(fig, "phase3_hardware_evidence.png")


def main() -> None:
    architecture_figure()
    simulator_evidence_figure()
    hardware_evidence_figure()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
