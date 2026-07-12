"""
Phase 3 — GARCH-hybrid on neutral-atom hardware (QuEra Aquila / Pasqal Fresnel).

Reproduces the Phase 2 headline (docs/phase2_submission/garch_hybrid_explained.md)
on the real neutral-atom platforms: the reservoir predicts only the GARCH
RESIDUAL (y = log RV - log RV_garch), so "predict 0" == GARCH exactly and any
non-zero output is the reservoir's contribution. GARCH proxy is also a readout
feature. The Ridge-on-residual row is the ablation: it should NOT beat GARCH, so
any win is isolable to the reservoir's nonlinear feature expansion.

Same recipe as phase2_garch_hybrid.py, but with the local AHS / Pulser reservoirs
(and ready for the real QuEra hardware path in quera_aquila_qrc.py).

    python experiments/neutral_atom_garch_hybrid.py --atoms 5 --max-train 600 --max-test 250
    python experiments/neutral_atom_garch_hybrid.py --atoms 5 --reservoir pasqal --recurrent

Scaling note: more atoms => more features AND more encoded inputs, but the local
emulators are CPU-bound (QuEra AHS ~1s/program at 5 atoms, ~15s at 8). Larger atom
counts want a GPU emulator (CUDA-Q) or the native QuEra hardware (256 atoms, no
classical sim cost). See docs/quera_aquila_setup.md.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from src.qrc.ising_reservoir import IsingReservoir, _credential
from src.data.loaders import load_financial_data_v2, invert_target
from src.baselines.classical import RidgeBaseline, regression_metrics, print_metrics
from src.baselines.garch import GARCHBaseline

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
warnings.filterwarnings("ignore")

_ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]


def _cv_alpha(R_tr, y_tr, alphas=_ALPHAS, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(R_tr) // 3)))
    best_alpha, best_mse = alphas[0], np.inf
    for alpha in alphas:
        mses = []
        for tr_idx, val_idx in tscv.split(R_tr):
            r = Ridge(alpha=alpha)
            r.fit(R_tr[tr_idx], y_tr[tr_idx])
            mses.append(np.mean((y_tr[val_idx] - r.predict(R_tr[val_idx])) ** 2))
        m = float(np.mean(mses))
        if m < best_mse:
            best_mse, best_alpha = m, alpha
    return best_alpha


def fit_readout(R_tr, X_tr_full, y_tr, alpha_override=None):
    R_aug = np.hstack([R_tr, X_tr_full])
    sc = StandardScaler()
    R_s = sc.fit_transform(R_aug)
    alpha = float(alpha_override) if alpha_override is not None else _cv_alpha(R_s, y_tr)
    ridge = Ridge(alpha=alpha)
    ridge.fit(R_s, y_tr)
    return ridge, sc, alpha


def apply_readout(ridge, sc, R_te, X_te_full):
    return ridge.predict(sc.transform(np.hstack([R_te, X_te_full])))


def _make_reservoir(kind: str, n: int, seed: int, args):
    if kind == "quera":
        from src.qrc.quera_reservoir import QueraReservoir
        return QueraReservoir(n_atoms=n, geometry="random2d", seed=seed)
    if kind == "pasqal":  # real Fresnel: global-only detuning encoding
        from src.qrc.pasqal_reservoir import PasqalReservoir
        return PasqalReservoir(n_atoms=n, geometry="random2d", seed=seed, encoding="global")
    if kind == "pasqal_local":  # MockDevice + DMM: per-site local detuning (QuEra-like)
        from src.qrc.pasqal_reservoir import PasqalReservoir
        return PasqalReservoir(n_atoms=n, geometry="random2d", seed=seed, encoding="local")
    if kind == "ising_sa":  # signed-coupling Ising machine (D-Wave/Amplify substrate), SA backend
        # tuned (ising_tune.py): input_scale=1.0, beta=1.0 beats GARCH on the window
        return IsingReservoir(
            n_spins=n,
            connectivity=args.ising_connectivity,
            density=args.ising_density,
            input_scale=args.input_scale,
            J_scale=args.j_scale,
            beta=args.beta,
            seed=seed,
        )
    if kind == "gate":
        from src.qrc.reservoir import QuantumReservoir
        return QuantumReservoir(
            n_qubits=n,
            n_layers=args.gate_layers,
            connectivity=args.gate_connectivity,
            seed=seed,
        )
    raise ValueError(kind)


def _reservoir_features(res, X_in_all, shots, n_jobs, recurrent, ising_backend="sa",
                        timeout_ms=1000):
    if isinstance(res, IsingReservoir):     # signed-coupling: SA / Toshiba / Amplify / D-Wave
        kw = {"timeout_ms": timeout_ms} if ising_backend != "sa" else {}
        return res.transform(
            X_in_all,
            backend=ising_backend,
            num_reads=shots,
            verbose=ising_backend != "sa",
            **kw,
        )
    # Gate QRC uses an exact statevector here. Shot/device effects are evaluated
    # separately by qbraid_hardware_qrc.py on the identical circuit family.
    if res.__class__.__name__ == "QuantumReservoir":
        if recurrent and hasattr(res, "transform_sequential"):
            return res.transform_sequential(X_in_all)
        return res.transform(X_in_all)
    if recurrent and hasattr(res, "transform_sequential"):
        # sequential: state carries across samples (no parallelism)
        return res.transform_sequential(X_in_all, shots=shots)
    return res.transform(X_in_all, device="local", shots=shots, n_jobs=n_jobs)


def run(args) -> None:
    print("=" * 78)
    print("  GIC 2026 PHASE 3 -- GARCH-hybrid on neutral-atom reservoirs")
    print("  Reservoir predicts the GARCH residual (predict 0 == GARCH)")
    print("=" * 78)

    # GARCH-residual config: proxy as feature AND as the residual baseline.
    d = load_financial_data_v2(
        delay=5, log_target=True, include_har=True, include_log_har=True,
        include_garch_proxy=True, garch_residual_target=True,
    )
    paid_cloud = (args.reservoir == "ising_sa"
                  and args.ising_backend not in ("sa", "exact")
                  and not args.dry_run_cloud)
    if paid_cloud and not str(d.get("data_source", "")).startswith("yfinance:"):
        raise RuntimeError(
            "Refusing cloud solves because real SPY data was unavailable; "
            f"loader returned {d.get('data_source', 'unknown')!r}."
        )
    X_tr_full, X_te_full = d["X_train"], d["X_test"]
    y_tr = d["y_train"]
    n = args.atoms
    shots = None if args.noiseless else args.shots   # None => exact/infinite-shot (Pasqal only)

    if args.max_train:
        X_tr_full = X_tr_full[-args.max_train:]; y_tr = y_tr[-args.max_train:]
    n_te = min(args.max_test, len(X_te_full)) if args.max_test else len(X_te_full)
    sl = slice(len(X_te_full) - n_te, len(X_te_full))
    X_te_full = X_te_full[sl]
    y_te_raw = d["y_test_raw"][sl]
    log_gp_te = d["log_garch_proxy_test"][sl]
    to_vol = lambda y: invert_target(y, "residual_log_garch", log_garch_proxy=log_gp_te)
    k_cal = args.hardware_calibration_rows
    if not (0 <= k_cal < n_te):
        raise ValueError("--hardware-calibration-rows must be in [0, max-test-1]")
    eval_sl = slice(k_cal, None)

    print(f"\nWindow: {len(X_tr_full)} train / {n_te} test  |  atoms={n}  "
          f"shots={'noiseless' if shots is None else shots}  recurrent={args.recurrent}\n")
    print(f"Data source: {d.get('data_source', 'unknown')}\n")

    if args.dry_run_cloud:
        if args.reservoir != "ising_sa" or args.ising_backend in ("sa", "exact"):
            raise ValueError("--dry-run-cloud requires --reservoir ising_sa and a cloud backend")
        probe = _make_reservoir("ising_sa", n, args.seed, args)
        h, J = probe._ising(np.asarray(X_tr_full[0, :n], dtype=float))
        cloud_solves = n_te if args.cloud_test_only else len(X_tr_full) + n_te
        token_name = ("AMPLIFY_AE_TOKEN" if args.ising_backend == "amplify"
                      else f"{args.ising_backend.upper()}_TOKEN")
        token_ready = bool(_credential(token_name) or _credential("AMPLIFY_TOKEN"))
        print("[CLOUD DRY RUN] No solves submitted")
        print(f"  backend={args.ising_backend} spins={n} fields={len(h)} couplers={len(J)}")
        print(f"  planned_cloud_solves={cloud_solves} cloud_test_only={args.cloud_test_only}")
        token_hint = token_name if token_name == "AMPLIFY_TOKEN" else f"{token_name} or AMPLIFY_TOKEN"
        print(f"  credential_present={token_ready} ({token_hint})")
        return

    rows = []
    preds = {}

    def record(name, y_pred, extra=None):
        m = regression_metrics(y_te_raw[eval_sl], np.asarray(y_pred)[eval_sl])
        print_metrics(f"{name:<38}", m)
        rows.append({"Model": name, "data_source": d.get("data_source", "unknown"),
                     "n_train": len(X_tr_full), "n_test": n_te,
                     **m, **(extra or {})})
        preds[name] = y_pred

    # ---- Reference: GARCH(1,1) standalone (the strong baseline to beat) ----
    g = GARCHBaseline(vol_window=d["vol_window"])
    g.set_returns(d["log_returns_full"], n_train=d["test_first_ret_idx"])
    g.fit(d["X_train"], d["y_train"])
    record("GARCH(1,1) standalone", g.predict(d["X_test"])[sl])

    # ---- Ablation: Ridge on the GARCH residual (should NOT beat GARCH) ----
    r = RidgeBaseline(alpha=1.0); r.fit(d["X_train"], d["y_train"])
    record("Ridge on GARCH-residual (ablation)",
           invert_target(r.predict(X_te_full), "residual_log_garch", log_garch_proxy=log_gp_te))

    # ---- Neutral-atom reservoirs on the GARCH residual ----
    sc_in = StandardScaler().fit(X_tr_full[:, :n])
    n_tr = len(X_tr_full)
    X_in_all = np.vstack([sc_in.transform(X_tr_full[:, :n]),
                          sc_in.transform(X_te_full[:, :n])])

    if args.reservoir == "both":
        kinds = ["quera", "pasqal"]
    elif args.reservoir == "all":
        kinds = ["quera", "pasqal", "gate", "ising_sa"]
    else:
        kinds = [args.reservoir]
    if shots is None and "quera" in kinds:
        print("  [note] --noiseless is Pasqal-only (QuEra AHS sim is shot-based); skipping QuEra.")
        kinds = [k for k in kinds if k != "quera"]
    labels = {"quera": "QuEra Aquila", "pasqal": "Pasqal Fresnel (global)",
              "pasqal_local": "Per-site local-detuning (MockDevice)",
              "ising_sa": f"Ising machine signed-J ({args.ising_backend})",
              "gate": f"Gate QRC {n}q L{args.gate_layers} ({args.gate_connectivity})"}
    for kind in kinds:
        base = labels[kind] + (" +recurrent" if args.recurrent else "") + (" [noiseless]" if shots is None else "")
        seed_preds = []
        for s_off in range(args.n_seeds):
            seed_i = args.seed + s_off
            res = _make_reservoir(kind, n, seed_i, args)
            t0 = time.time()
            feature_extra = {}
            if (kind == "ising_sa" and args.cloud_test_only
                    and args.ising_backend not in ("sa", "exact")):
                # Economical hybrid: train on the local signed-Ising sampler and
                # spend cloud GPU solves only on the held-out test window.
                train_proxy = args.cloud_train_proxy
                R_tr = res.transform(X_in_all[:n_tr], backend=train_proxy, num_reads=shots)
                R_te_ref = res.transform(X_in_all[n_tr:], backend=train_proxy, num_reads=shots)
                R_te = _reservoir_features(
                    res, X_in_all[n_tr:], shots, args.n_jobs, args.recurrent,
                    ising_backend=args.ising_backend, timeout_ms=args.timeout_ms,
                )
                feature_extra = {
                    "feature_corr_vs_train_proxy": float(np.corrcoef(R_te.ravel(), R_te_ref.ravel())[0, 1]),
                    "n_cloud_solves": n_te,
                    "train_feature_source": f"local_{train_proxy}",
                }
                if res.last_sampling_stats:
                    returned = np.array([x["returned_states"] for x in res.last_sampling_stats])
                    unique = np.array([x["unique_states"] for x in res.last_sampling_stats])
                    feature_extra.update({
                        "mean_returned_states": float(returned.mean()),
                        "min_returned_states": float(returned.min()),
                        "mean_unique_states": float(unique.mean()),
                        "min_unique_states": float(unique.min()),
                    })
            else:
                R_all = _reservoir_features(
                    res, X_in_all, shots, args.n_jobs, args.recurrent,
                    ising_backend=args.ising_backend, timeout_ms=args.timeout_ms,
                )
                R_tr, R_te = R_all[:n_tr], R_all[n_tr:]
            ridge, sc_r, alpha = fit_readout(
                R_tr, X_tr_full, y_tr, alpha_override=args.readout_alpha
            )
            pred_residual = apply_readout(ridge, sc_r, R_te, X_te_full)
            correction_strength = args.correction_strength
            if k_cal:
                grid = np.linspace(args.strength_min, args.strength_max, args.strength_steps)
                scores = [np.sqrt(np.mean(
                    (to_vol(s * pred_residual)[:k_cal] - y_te_raw[:k_cal]) ** 2
                )) for s in grid]
                correction_strength = float(grid[int(np.argmin(scores))])
            pred = to_vol(correction_strength * pred_residual)
            feature_extra.update({
                "correction_strength": correction_strength,
                "hardware_calibration_rows": k_cal,
                "evaluation_rows": n_te - k_cal,
            })
            seed_preds.append(pred)
            if args.save_features:
                tag = args.result_tag or kind
                feat_path = os.path.join(
                    RESULTS_DIR, f"hybrid_features_{tag}_{kind}_seed{seed_i}.npz"
                )
                np.savez_compressed(
                    feat_path, R_train=R_tr, R_test=R_te,
                    X_train=X_tr_full, X_test=X_te_full,
                    y_train=y_tr, y_test_raw=y_te_raw,
                    data_source=d.get("data_source", "unknown"),
                    reservoir=kind, backend=args.ising_backend,
                )
                print(f"    features -> {os.path.relpath(feat_path)}")
            if args.n_seeds > 1:
                print(f"    seed {seed_i}: {time.time()-t0:.1f}s")
        if args.n_seeds > 1:
            record(f"{base} on GARCH-residual ({args.n_seeds}-seed ens)",
                   np.mean(np.stack(seed_preds), axis=0),
                   extra={"ising_backend": args.ising_backend} if kind == "ising_sa" else None)
        else:
            extra = {"alpha": alpha, **feature_extra}
            if kind == "ising_sa":
                extra["ising_backend"] = args.ising_backend
                extra["input_scale"] = args.input_scale
                extra["beta"] = args.beta
                extra["J_scale"] = args.j_scale
                extra["connectivity"] = args.ising_connectivity
            record(f"{base} on GARCH-residual", seed_preds[0], extra=extra)

    # ---- Save + leaderboard ----
    df = pd.DataFrame(rows)
    if args.result_tag:
        suffix = f"_{args.result_tag}"
    else:
        suffix = f"_{args.ising_backend}" if args.reservoir == "ising_sa" else ""
    out_csv = os.path.join(RESULTS_DIR, f"neutral_atom_garch_hybrid{suffix}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n  Saved -> {os.path.relpath(out_csv)}")
    garch_rmse = df[df["Model"] == "GARCH(1,1) standalone"]["RMSE"].iloc[0]
    print(f"\n  Leaderboard by RMSE (GARCH standalone = {garch_rmse:.5f}):")
    for _, r in df.sort_values("RMSE").iterrows():
        flag = "  <- beats GARCH" if r["RMSE"] < garch_rmse and "GARCH(1,1)" not in r["Model"] else ""
        print(f"    {r['Model']:<40} RMSE={r['RMSE']:.5f}  R2={r['R2']:.4f}  QLIKE={r['QLIKE']:.5f}{flag}")

    _plot(df, garch_rmse, suffix=suffix)
    print("\n--- Done ---")


def _plot(df, garch_rmse, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = df.sort_values("RMSE")
    def _c(m):
        if "QuEra" in m: return "#7c3aed"
        if "Pasqal" in m: return "#2166ac"
        if "Ridge" in m: return "#3a7d3a"
        return "#d6604d"
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(order["Model"], order["RMSE"], color=[_c(m) for m in order["Model"]])
    ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=8)
    ax.axvline(garch_rmse, color="black", ls="--", lw=1, label=f"GARCH = {garch_rmse:.5f}")
    ax.set_xlabel("RMSE (annualised vol)"); ax.invert_yaxis()
    ax.set_title("Phase 3 -- GARCH-hybrid on neutral-atom reservoirs (predict GARCH residual)")
    ax.legend(fontsize=8); ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"neutral_atom_garch_hybrid{suffix}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.replace(".png", "_doc.png"), dpi=1200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {os.path.relpath(out)}  (+ _doc.png @ 1200 DPI)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reservoir", default="both",
                   choices=["quera", "pasqal", "pasqal_local", "ising_sa", "gate", "both", "all"])
    p.add_argument("--atoms", type=int, default=5)
    p.add_argument("--shots", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train", type=int, default=600)
    p.add_argument("--max-test", type=int, default=250)
    p.add_argument("--n-jobs", type=int, default=4, dest="n_jobs")
    p.add_argument("--n-seeds", type=int, default=1, dest="n_seeds",
                   help="ensemble size (Phase 2 headline used 3)")
    p.add_argument("--ising-backend", default="sa", dest="ising_backend",
                   choices=[
                       "sa", "exact", "dwave", "amplify", "toshiba", "fujitsu",
                       "hitachi", "nec", "dwave_amplify",
                   ],
                   help="sampler for --reservoir ising_sa (cloud backends need a token)")
    p.add_argument("--ising-connectivity", default="all-to-all",
                   choices=["all-to-all", "random", "linear"],
                   help="coupling graph for --reservoir ising_sa")
    p.add_argument("--ising-density", type=float, default=0.5,
                   help="edge fraction when --ising-connectivity=random")
    p.add_argument("--input-scale", type=float, default=1.0,
                   help="input local-field scale for --reservoir ising_sa")
    p.add_argument("--j-scale", type=float, default=1.0,
                   help="signed coupling scale for --reservoir ising_sa")
    p.add_argument("--beta", type=float, default=1.0,
                   help="inverse-temperature / anneal beta for --reservoir ising_sa")
    p.add_argument("--gate-layers", type=int, default=2,
                   help="layers for --reservoir gate")
    p.add_argument("--gate-connectivity", default="random",
                   choices=["linear", "random", "all-to-all"],
                   help="coupling graph for --reservoir gate")
    p.add_argument("--timeout-ms", type=int, default=1000,
                   help="cloud backend time limit per Ising solve")
    p.add_argument("--cloud-test-only", action="store_true",
                   help="for cloud Ising backends, train on local SA features and submit only test rows")
    p.add_argument("--cloud-train-proxy", choices=["sa", "ground"], default="sa",
                   help="local training feature source in --cloud-test-only mode")
    p.add_argument("--readout-alpha", type=float, default=None,
                   help="lock Ridge alpha instead of selecting it by training CV")
    p.add_argument("--save-features", action="store_true",
                   help="save reusable train/test reservoir matrices as compressed NPZ")
    p.add_argument("--dry-run-cloud", action="store_true",
                   help="validate a cloud-Ising footprint and credentials without submitting solves")
    p.add_argument("--correction-strength", type=float, default=1.0,
                   help="initial multiplier on the predicted GARCH residual")
    p.add_argument("--hardware-calibration-rows", type=int, default=0,
                   help="first test rows used only to calibrate correction amplitude")
    p.add_argument("--strength-min", type=float, default=-0.25)
    p.add_argument("--strength-max", type=float, default=1.25)
    p.add_argument("--strength-steps", type=int, default=301)
    p.add_argument("--result-tag", default="",
                   help="optional suffix for result CSV/PNG names")
    p.add_argument("--recurrent", action="store_true", help="use sequential memory feedback")
    p.add_argument("--noiseless", action="store_true",
                   help="exact/infinite-shot features (Pasqal only) -- Phase 2 methodology")
    run(p.parse_args())
