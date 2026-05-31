"""
Composite results figure for the Phase 2 paper.

Three panels in one PNG so it fits the 3-page submission limit:
  (a) Leaderboard with multi-seed error bar
  (b) Regime-conditional RMSE (calm / normal / turbulent)
  (c) GARCH-hybrid result (the headline)

Loads precomputed numbers from results/ so it's fast and reproducible.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    # ---- Load data ----
    final_df = pd.read_csv(os.path.join(RESULTS_DIR, "phase2_final_summary.csv"))
    regime_df = pd.read_csv(os.path.join(RESULTS_DIR, "phase2_final_regime.csv"))
    hybrid_df = pd.read_csv(os.path.join(RESULTS_DIR, "phase2_garch_hybrid.csv"))
    ann_ens = pd.read_csv(os.path.join(RESULTS_DIR, "phase2_final_ann_ensemble.csv"))

    # ---- Build leaderboard (one row per unique model, ANN as mean +/- std) ----
    base = final_df[~final_df["Model"].isin([
        "ANN 10q a2a t=2.0 a=1.0",
        "ANN 10q a2a (5-seed mean)",
    ])][["Model", "RMSE", "R2"]].copy()
    ann_row = pd.DataFrame([{
        "Model": "QRC Ising 10q (5-seed)",
        "RMSE": float(ann_ens.iloc[0]["RMSE_mean"]),
        "R2":   float(ann_ens.iloc[0]["R2_mean"]),
    }])
    ann_std = float(ann_ens.iloc[0]["RMSE_std"])
    leaderboard = pd.concat([base, ann_row], ignore_index=True).sort_values("RMSE").reset_index(drop=True)
    # Rename for clean display
    leaderboard["Model"] = leaderboard["Model"].replace({
        "QRC v3 gate (5q L2 rand)": "QRC gate-based (5q)",
        "ESN (200 nodes)": "ESN (200 nodes)",
        "LSTM (1L, 32h)": "LSTM (1 layer, 32h)",
        "GARCH(1,1)": "GARCH(1,1)",
        "Persistence": "Persistence",
        "AR(5)": "AR(5)",
        "Ridge": "Ridge",
    })

    # ---- Regime ----
    regime_pivot = regime_df.pivot(index="Model", columns="Regime", values="RMSE")[
        ["calm", "normal", "turbulent"]
    ]
    regime_pivot.index = regime_pivot.index.map({
        "Persistence": "Persistence",
        "AR(5)": "AR(5)",
        "GARCH(1,1)": "GARCH",
        "ESN (200 nodes)": "ESN",
        "QRC v3 gate (5q L2 rand)": "QRC gate-based",
        "ANN 10q a2a (5-seed mean)": "QRC Ising (5-seed)",
    })

    # ---- Hybrid (keep only the ensemble/standalone rows) ----
    hybrid_keep = hybrid_df[hybrid_df["config"].isin([
        "garch_only", "A-ridge", "A-qrc-ensemble", "B-ridge", "B-qrc-ensemble",
    ])].copy()
    hybrid_keep["Model"] = hybrid_keep["Model"].replace({
        "QRC + GARCH-feature (3-seed ensemble)": "QRC + GARCH feature\n(Option A, 3-seed)",
        "QRC on GARCH-residual (3-seed ensemble)": "QRC on GARCH-residual\n(Option B, 3-seed)",
        "Ridge + GARCH-feature (Persistence resid.)": "Ridge + GARCH feature\n(no QRC)",
        "Ridge on GARCH-residual target": "Ridge on GARCH-residual\n(no QRC)",
        "GARCH(1,1) standalone": "GARCH(1,1) standalone",
    })
    hybrid_keep = hybrid_keep.sort_values("RMSE").reset_index(drop=True)

    # ============================================================
    #                       Figure
    # ============================================================
    fig = plt.figure(figsize=(12, 11))
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55, height_ratios=[1.0, 1.0, 1.0])

    # ---- (a) Leaderboard ----
    ax_a = fig.add_subplot(gs[0])
    def _color_a(name):
        if "QRC" in name:    return "#7c3aed"
        return "#d6604d"
    bars = ax_a.barh(leaderboard["Model"], leaderboard["RMSE"],
                      color=[_color_a(m) for m in leaderboard["Model"]],
                      edgecolor="white", linewidth=0.5)
    ax_a.bar_label(bars, fmt="%.5f", padding=3, fontsize=8)
    # Error bar on the QRC Ising row
    qrc_row_idx = leaderboard.index[leaderboard["Model"] == "QRC Ising 10q (5-seed)"][0]
    ax_a.errorbar(leaderboard.loc[qrc_row_idx, "RMSE"], qrc_row_idx,
                   xerr=ann_std, fmt="o", color="black", capsize=4, markersize=4)
    ax_a.set_xlabel("RMSE (annualised volatility)")
    ax_a.set_title("(a) Phase 2 leaderboard — bare QRC vs all required baselines (sorted by RMSE)",
                    fontsize=10, loc="left")
    ax_a.grid(True, axis="x", alpha=0.25)
    ax_a.set_xlim(0, leaderboard["RMSE"].max() * 1.18)

    # ---- (b) Regime-conditional RMSE ----
    ax_b = fig.add_subplot(gs[1])
    x = np.arange(len(regime_pivot.index))
    width = 0.27
    colors_regime = {"calm": "#5dc863", "normal": "#21918c", "turbulent": "#440154"}
    for i, regime in enumerate(["calm", "normal", "turbulent"]):
        ax_b.bar(x + (i - 1) * width, regime_pivot[regime], width,
                  label=regime, color=colors_regime[regime])
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(regime_pivot.index, rotation=15, ha="right", fontsize=9)
    ax_b.set_ylabel("RMSE")
    ax_b.legend(title="VIX regime", loc="upper left", fontsize=9)
    ax_b.set_title("(b) Regime-conditional RMSE (VIX terciles) — GARCH dominates turbulent, "
                    "QRC most consistent",
                    fontsize=10, loc="left")
    ax_b.grid(True, axis="y", alpha=0.25)

    # ---- (c) GARCH-hybrid ----
    ax_c = fig.add_subplot(gs[2])
    def _color_c(name):
        if "QRC" in name: return "#7c3aed"
        if "Ridge" in name: return "#3a7d3a"
        return "#d6604d"
    bars = ax_c.barh(hybrid_keep["Model"], hybrid_keep["RMSE"],
                      color=[_color_c(m) for m in hybrid_keep["Model"]],
                      edgecolor="white", linewidth=0.5)
    ax_c.bar_label(bars, fmt="%.5f", padding=3, fontsize=8)
    ax_c.set_xlabel("RMSE (annualised volatility)")
    ax_c.set_title("(c) QRC + GARCH hybrid — both QRC variants beat GARCH; Ridge alone does not",
                    fontsize=10, loc="left")
    ax_c.grid(True, axis="x", alpha=0.25)
    ax_c.set_xlim(0.0075, hybrid_keep["RMSE"].max() * 1.04)

    # Highlight the headline: best row (QRC on GARCH-residual)
    best_idx = hybrid_keep.index[hybrid_keep["Model"].str.startswith("QRC on GARCH-residual")][0]
    ax_c.add_patch(plt.Rectangle(
        (ax_c.get_xlim()[0], best_idx - 0.4),
        ax_c.get_xlim()[1] - ax_c.get_xlim()[0], 0.8,
        fill=False, edgecolor="#7c3aed", linewidth=2, linestyle="--",
    ))

    fig.suptitle(
        "Phase 2 results — SPY realized volatility (2010–2024 daily, n=746 test days)",
        fontsize=12, weight="bold", y=0.995,
    )

    out = os.path.join(FIG_DIR, "phase2_composite.png")
    out_doc = os.path.join(FIG_DIR, "phase2_composite_doc.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out_doc, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"Saved: {out_doc}  (600 DPI publication copy)")


if __name__ == "__main__":
    main()
