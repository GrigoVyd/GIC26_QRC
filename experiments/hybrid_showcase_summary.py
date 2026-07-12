"""Build the final, apples-to-apples hybrid-QRC evidence tables and figure."""

from __future__ import annotations

import os
import pandas as pd

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS, name))


def _row(df: pd.DataFrame, text: str) -> pd.Series:
    hit = df[df["Model"].str.contains(text, regex=False)]
    if hit.empty:
        raise ValueError(f"No model containing {text!r}")
    return hit.iloc[0]


def main() -> None:
    gate = _read("neutral_atom_garch_hybrid_gate_garch_400_120.csv")
    pasqal = _read("neutral_atom_garch_hybrid_pasqal_garch_400_120.csv")
    quera = _read("neutral_atom_garch_hybrid_quera_garch_400_120.csv")
    ising = _read("neutral_atom_garch_hybrid_ising_sa_garch_400_120.csv")

    garch = _row(gate, "GARCH(1,1)")
    ridge = _row(gate, "Ridge on GARCH-residual")
    specs = [
        ("GARCH(1,1)", garch, "classical baseline", "CPU"),
        ("Ridge residual ablation", ridge, "linear residual model", "CPU"),
        ("Pasqal analog QRC hybrid", _row(pasqal, "Pasqal Fresnel"),
         "global Rydberg analog", "Fresnel-ready"),
        ("Gate QRC hybrid", _row(gate, "Gate QRC"),
         "5q L2 sparse gate reservoir", "IQM Garnet-ready"),
        ("QuEra analog QRC hybrid", _row(quera, "QuEra Aquila"),
         "local-detuning Rydberg analog", "Aquila-ready"),
        ("Signed-Ising hybrid", _row(ising, "Ising machine"),
         "classical signed-J sampling", "Amplify-ready"),
    ]
    rows = []
    for label, r, physics, deployment in specs:
        rmse = float(r["RMSE"])
        rows.append({
            "Model": label,
            "Physics": physics,
            "Deployment": deployment,
            "Window": "400 train / 120 test",
            "RMSE": rmse,
            "MAE": float(r["MAE"]),
            "R2": float(r["R2"]),
            "QLIKE": float(r["QLIKE"]),
            "Gap_vs_GARCH_%": 100.0 * (rmse / float(garch["RMSE"]) - 1.0),
            "Gain_vs_RidgeResidual_%": 100.0 * (float(ridge["RMSE"]) - rmse) / float(ridge["RMSE"]),
            "data_source": r.get("data_source", "yfinance:SPY:2010-01-01:2024-12-31"),
        })
    common = pd.DataFrame(rows).sort_values("RMSE")
    common_path = os.path.join(RESULTS, "hybrid_showcase_common_400_120.csv")
    common.to_csv(common_path, index=False)

    # Evidence with different windows is intentionally separate, never ranked as
    # if it were the same test set.
    headline = _read("phase2_garch_hybrid.csv")
    amp = _read("neutral_atom_garch_hybrid_amplify.csv")
    evidence = pd.DataFrame([
        {
            "System": "Transverse-field annealer QRC (3-seed)",
            "Window": "full 746 test",
            "Status": "simulation advantage candidate",
            "RMSE": float(_row(headline, "QRC on GARCH-residual (3-seed ensemble)")["RMSE"]),
            "Claim": "beats full-window GARCH; requires annealing dynamics",
        },
        {
            "System": "Amplify AE signed-Ising hybrid",
            "Window": "50 train / 10 test",
            "Status": "executed cloud GPU smoke test",
            "RMSE": float(_row(amp, "Ising machine signed-J")["RMSE"]),
            "Claim": "execution proof; classical Ising ablation, not quantum advantage",
        },
    ])
    evidence_path = os.path.join(RESULTS, "hybrid_showcase_other_windows.csv")
    evidence.to_csv(evidence_path, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot = common.sort_values("RMSE", ascending=False).reset_index(drop=True)
    colors = {
        "CPU": "#777777", "Fresnel-ready": "#2166ac",
        "IQM Garnet-ready": "#7c3aed", "Aquila-ready": "#00a087",
        "Amplify-ready": "#d95f02",
    }
    fig, ax = plt.subplots(figsize=(11, 5.5))
    y = range(len(plot))
    ax.hlines(y, float(garch["RMSE"]), plot["RMSE"], color="#bbbbbb", lw=2)
    ax.scatter(plot["RMSE"], y, s=95,
               color=[colors[d] for d in plot["Deployment"]], zorder=3)
    for yi, value in zip(y, plot["RMSE"]):
        ax.annotate(f"{value:.5f}", (value, yi), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(list(y), plot["Model"])
    ax.axvline(float(garch["RMSE"]), color="black", ls="--", lw=1,
               label=f"GARCH = {float(garch['RMSE']):.5f}")
    ax.set_xlabel("RMSE (annualised realized volatility)")
    ax.set_title("GARCH-residual hybrid QRC — common SPY window (400 train / 120 test)")
    pad = 0.00004
    ax.set_xlim(plot["RMSE"].min() - pad, plot["RMSE"].max() + pad)
    ax.grid(True, axis="x", alpha=0.25); ax.legend(fontsize=8)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS, "hybrid_showcase_common_400_120.png")
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    fig.savefig(fig_path.replace(".png", "_doc.png"), dpi=1200, bbox_inches="tight")
    plt.close(fig)

    print(common[["Model", "RMSE", "Gap_vs_GARCH_%", "Gain_vs_RidgeResidual_%"]]
          .to_string(index=False))
    print(f"\nSaved -> {os.path.relpath(common_path)}")
    print(f"Saved -> {os.path.relpath(evidence_path)}")
    print(f"Saved -> {os.path.relpath(fig_path)}")


if __name__ == "__main__":
    main()
