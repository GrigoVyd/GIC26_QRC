"""
Document Figure Generator -- GIC 2026 Phase 2

Produces 8 publication-ready PNGs in results/doc/:

  circuit_ising.png          Fig 1 -- Ising ZZ circuit (5q, 1 layer for clarity)
  circuit_heisenberg.png     Fig 2 -- Heisenberg circuit (5q, 1 layer)
  circuit_iqp.png            Fig 3 -- IQP encoding circuit (5q, 1 layer)
  pipeline_diagram.png       Fig 4 -- data preprocessing pipeline
  strategy_comparison.png    Fig 5 -- grouped bar: strategy x task x primary metric
  scaling_lines.png          Fig 6 -- NMSE vs n_qubits per strategy (financial task)
  noise_robustness.png       Fig 7 -- NMSE vs depolarizing p for all 3 strategies
  financial_predictions.png  Fig 8 -- true vol vs QRC / ESN / AR (document quality)

All figures: 12pt font, 200 dpi, high-contrast palette.

NOTE: Figures 5 and 6 require results/doc/strategy_comparison.csv
      (produced by strategy_tournament.py).
      Figure 7 runs a quick noise sweep internally.
      Figure 8 runs a quick financial evaluation internally.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DOC_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "doc")
os.makedirs(DOC_DIR, exist_ok=True)

# Shared style
RC = {
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "figure.dpi": 200,
}
PALETTE = {
    "Ising ZZ":           "#2166AC",
    "Heisenberg XX+YY+ZZ":"#D6604D",
    "IQP Encoding":       "#4DAC26",
    "classical":          "#888888",
    "true":               "#000000",
    "ESN-100":            "#762A83",
    "AR(5)":              "#E08214",
    "Ridge":              "#AAAAAA",
    "Persistence":        "#CCCCCC",
}


def _save(fig: plt.Figure, name: str) -> str:
    path = os.path.join(DOC_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")
    return path


# ---------------------------------------------------------------------------
# Figures 1-3: Circuit diagrams
# ---------------------------------------------------------------------------

def fig_circuits() -> None:
    """Draw one-layer 5-qubit circuit for each strategy."""
    from strategies import IsingStrategy, HeisenbergStrategy, IQPStrategy

    x_demo = np.array([0.3, -0.5, 0.7, -0.2, 0.1])

    configs = [
        (IsingStrategy,      "circuit_ising.png",       "Fig 1 -- Strategy A: Ising ZZ reservoir"),
        (HeisenbergStrategy, "circuit_heisenberg.png",  "Fig 2 -- Strategy B: Heisenberg XX+YY+ZZ"),
        (IQPStrategy,        "circuit_iqp.png",          "Fig 3 -- Strategy C: IQP Encoding"),
    ]

    with plt.rc_context(RC):
        for Cls, fname, title in configs:
            s = Cls(n_qubits=5, n_layers=1, seed=42)
            qc = s.build_circuit(x_demo)
            fig = qc.draw(output="mpl", style="clifford", fold=40)
            fig.suptitle(title, fontsize=12, y=1.01)
            _save(fig, fname)


# ---------------------------------------------------------------------------
# Figure 4: Data pipeline diagram
# ---------------------------------------------------------------------------

def fig_pipeline() -> None:
    """Matplotlib boxes-and-arrows diagram of the QRC data pipeline."""
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(13, 3.2))
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 3.2)
        ax.axis("off")

        boxes = [
            (0.3,  "Raw prices\n(SPY daily)", "#AEC6E8"),
            (2.2,  "Log returns\nr(t)=Deltalog P", "#AEC6E8"),
            (4.1,  "Realized vol\n21-day rolling sigma", "#AEC6E8"),
            (6.0,  "Delay embed\n[r(t-k)...r(t)]", "#B8E6B8"),
            (7.9,  "Angle encode\nRx(pi·xi)", "#FFD8A8"),
            (9.8,  "QRC reservoir\nIsing / Heis / IQP", "#FFD8A8"),
            (11.7, "Ridge readout\nsigma(t+1)", "#E8C6E8"),
        ]

        bw, bh = 1.7, 1.1
        by = 1.05
        for bx, label, color in boxes:
            rect = mpatches.FancyBboxPatch(
                (bx, by), bw, bh,
                boxstyle="round,pad=0.08",
                facecolor=color, edgecolor="#444444", linewidth=1.2,
            )
            ax.add_patch(rect)
            ax.text(bx + bw / 2, by + bh / 2, label,
                    ha="center", va="center", fontsize=9.5, linespacing=1.4)

        # Arrows
        for i in range(len(boxes) - 1):
            x0 = boxes[i][0] + bw
            x1 = boxes[i + 1][0]
            ax.annotate(
                "", xy=(x1, by + bh / 2), xytext=(x0, by + bh / 2),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.4),
            )

        # Section labels
        for label, x in [("Pre-processing", 3.5), ("Embedding", 6.95),
                          ("Quantum", 9.85), ("Readout", 11.7)]:
            ax.text(x + bw / 2, by + bh + 0.25, label,
                    ha="center", va="bottom", fontsize=9, color="#555555",
                    style="italic")

        ax.set_title("QRC Pipeline -- Financial Volatility Track", fontsize=12, pad=8)
        _save(fig, "pipeline_diagram.png")


# ---------------------------------------------------------------------------
# Figure 5: Strategy x task comparison bar chart
# ---------------------------------------------------------------------------

def fig_strategy_comparison(df: pd.DataFrame) -> None:
    """Grouped bar chart: strategy x task x primary metric."""
    tasks = ["NARMA-5", "Financial", "MNIST-01"]
    strategies = ["Ising ZZ", "Heisenberg XX+YY+ZZ", "IQP Encoding"]

    # For each strategy, use the best n_qubits result per task
    pivot = {}
    for strat in strategies:
        pivot[strat] = []
        for task in tasks:
            sub = df[(df["Strategy"] == strat) & (df["Task"] == task)]
            if sub.empty:
                pivot[strat].append(np.nan)
                continue
            m_name = sub["metric_name"].iloc[0]
            if m_name == "NMSE":
                pivot[strat].append(sub["primary_metric"].min())
            else:
                pivot[strat].append(sub["primary_metric"].max())

    x = np.arange(len(tasks))
    width = 0.22
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, strat in enumerate(strategies):
            vals = pivot[strat]
            bars = ax.bar(x + (i - 1) * width, vals, width,
                          label=strat, color=PALETTE.get(strat, "#999"),
                          edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if v is not None and not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.003,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

        ax.set_xticks(x)
        ax.set_xticklabels(["NARMA-5\n(NMSE (down))", "Financial\n(NMSE (down))", "MNIST-01\n(Accuracy (up))"])
        ax.set_ylabel("Metric (lower NMSE / higher accuracy = better)")
        ax.set_title("Strategy Comparison -- Best n_qubits per Task")
        ax.legend(loc="upper right")
        ax.set_ylim(bottom=0)
        _save(fig, "strategy_comparison.png")


# ---------------------------------------------------------------------------
# Figure 6: Scaling lines -- NMSE vs n_qubits (financial task)
# ---------------------------------------------------------------------------

def fig_scaling_lines(df: pd.DataFrame) -> None:
    """Line plot: NMSE vs n_qubits for each strategy on financial task."""
    fin = df[(df["Task"] == "Financial") & (df["n_qubits"] != "-")].copy()
    fin["n_qubits"] = fin["n_qubits"].astype(int)

    strategies = ["Ising ZZ", "Heisenberg XX+YY+ZZ", "IQP Encoding"]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for strat in strategies:
            sub = fin[fin["Strategy"] == strat].sort_values("n_qubits")
            if sub.empty:
                continue
            ax.plot(sub["n_qubits"], sub["NMSE"], "o-",
                    label=strat, color=PALETTE.get(strat, "#999"),
                    lw=2, markersize=7)

        # Classical ESN reference line
        esn_rows = df[(df["Task"] == "Financial") & (df["Strategy"] == "ESN-100")]
        if not esn_rows.empty:
            esn_nmse = esn_rows["NMSE"].iloc[0]
            ax.axhline(esn_nmse, ls="--", color=PALETTE["ESN-100"],
                       lw=1.5, label="ESN-100 (classical)")

        ax.set_xlabel("Number of qubits")
        ax.set_ylabel("NMSE (lower is better)")
        ax.set_title("QRC Scaling -- Financial Volatility")
        ax.set_xticks([5, 10])
        ax.legend()
        _save(fig, "scaling_lines.png")


# ---------------------------------------------------------------------------
# Figure 7: Noise robustness
# ---------------------------------------------------------------------------

def fig_noise_robustness() -> None:
    """NMSE vs depolarizing p for all 3 strategies (quick internal run)."""
    from strategies import IsingStrategy, HeisenbergStrategy, IQPStrategy
    from src.qrc.noise import depolarizing_noise
    from src.data.loaders import load_narma
    from sklearn.linear_model import Ridge

    X_tr, X_te, y_tr, y_te = load_narma(order=5, n_samples=300, seed=42)
    p_vals = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]

    def _eval(Cls, p):
        nm = depolarizing_noise(p_single=p, p_two=p * 8) if p > 0 else None
        s = Cls(n_qubits=5, n_layers=2, seed=42, noise_model=nm, n_shots=1024)
        sc = StandardScaler()
        R_tr = sc.fit_transform(s.transform(X_tr[:, :5]))
        R_te = sc.transform(s.transform(X_te[:, :5]))
        r = Ridge(alpha=0.1)
        r.fit(R_tr, y_tr)
        y_hat = r.predict(R_te)
        return float(np.mean((y_te - y_hat) ** 2) / (np.var(y_te) + 1e-12))

    from sklearn.preprocessing import StandardScaler

    configs = [
        (IsingStrategy,      "Ising ZZ"),
        (HeisenbergStrategy, "Heisenberg XX+YY+ZZ"),
        (IQPStrategy,        "IQP Encoding"),
    ]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for Cls, label in configs:
            nmses = []
            for p in p_vals:
                print(f"    noise p={p:.3f}  {label} ...", end=" ", flush=True)
                nmse = _eval(Cls, p)
                nmses.append(nmse)
                print(f"NMSE={nmse:.5f}")
            ax.plot(p_vals, nmses, "o-", label=label,
                    color=PALETTE.get(label, "#999"), lw=2, markersize=7)

        ax.set_xlabel("Single-qubit depolarizing probability")
        ax.set_ylabel("NMSE (lower is better)")
        ax.set_title("Noise Robustness -- NARMA-5, 5 qubits")
        ax.legend()
        _save(fig, "noise_robustness.png")


# ---------------------------------------------------------------------------
# Figure 8: Financial predictions (document quality)
# ---------------------------------------------------------------------------

def fig_financial_predictions() -> None:
    """True vol vs QRC (best strategy, 10q) vs ESN vs AR on test set."""
    from strategies import IsingStrategy
    from src.data.loaders import load_financial_data
    from src.baselines.classical import ARBaseline, EchoStateNetwork
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    X_tr, X_te, y_tr, y_te = load_financial_data(delay=5)

    def _qrc_pred(Cls, n_q=10):
        s = Cls(n_qubits=n_q, n_layers=2, seed=42)
        sc = StandardScaler()
        R_tr = sc.fit_transform(s.transform(X_tr[:, :n_q]))
        R_te = sc.transform(s.transform(X_te[:, :n_q]))
        sc2 = StandardScaler()
        R_tr = sc2.fit_transform(R_tr)
        R_te = sc2.transform(R_te)
        r = Ridge(alpha=0.1)
        r.fit(R_tr, y_tr)
        return r.predict(R_te)

    preds = {
        "QRC Ising 10q": _qrc_pred(IsingStrategy, 10),
        "ESN-100":        EchoStateNetwork(n_reservoir=100, seed=42).fit(X_tr, y_tr).predict(X_te),
        "AR(5)":          ARBaseline(order=5).fit(X_tr, y_tr).predict(X_te),
    }

    n_show = min(250, len(y_te))
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(y_te[:n_show], label="True vol.", lw=1.8,
                color=PALETTE["true"], zorder=5)
        ls_cycle = ["-", "--", ":"]
        for (name, yp), ls in zip(preds.items(), ls_cycle):
            c = PALETTE.get(name.split()[0], PALETTE.get(name, "#999"))
            ax.plot(yp[:n_show], label=name, lw=1.4,
                    color=c, alpha=0.85, ls=ls)
        ax.set_xlabel("Trading days (test period)")
        ax.set_ylabel("Realized volatility (annualised)")
        ax.set_title("Financial Volatility Forecast -- QRC vs. Classical Baselines")
        ax.legend(loc="upper right")
        _save(fig, "financial_predictions_doc.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(skip_noise: bool = False) -> None:
    print("=" * 65)
    print("  GENERATING DOCUMENT FIGURES")
    print("=" * 65)

    print("\n[1/8] Circuit diagrams ...")
    fig_circuits()

    print("\n[2/8] Pipeline diagram ...")
    fig_pipeline()

    csv_path = os.path.join(DOC_DIR, "strategy_comparison.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        print("\n[3/8] Strategy comparison bar chart ...")
        fig_strategy_comparison(df)

        print("\n[4/8] Scaling lines (financial) ...")
        fig_scaling_lines(df)
    else:
        warnings.warn(
            "strategy_comparison.csv not found -- skipping Figs 5 & 6. "
            "Run strategy_tournament.py first."
        )

    if not skip_noise:
        print("\n[5/8] Noise robustness (quick sweep) ...")
        fig_noise_robustness()
    else:
        print("\n[5/8] Noise robustness -- SKIPPED (--skip-noise)")

    print("\n[6/8] Financial predictions (doc quality) ...")
    fig_financial_predictions()

    print(f"\n  All figures -> {DOC_DIR}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--skip-noise", action="store_true")
    args = p.parse_args()
    run(skip_noise=args.skip_noise)
