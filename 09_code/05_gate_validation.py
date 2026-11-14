"""
Part B, step 2: ML leakage of the gates, validated against SPICE.

For every gate case and corner in data/gates_spice.csv:
  1. Take the node voltages SPICE recorded (output and internal nodes).
  2. Give every transistor its terminal voltages and evaluate it with the
     transistor model of its state (trained_models/selection.json).
  3. Build the gate leakage, I(VDD) + I(inputs at VDD), from those currents.
     The ON transistors' channel currents come from Kirchhoff's law (see gates.py).
  4. Compare with the SPICE leakage.

Usage:
    python3 05_gate_validation.py            # needs the full 18 x 200 run of 04_simulate_gates.py
    python3 05_gate_validation.py --quick    # on data/quick/, results in results/gates_quick/

Outputs in results/gates/:
    gate_results.md / .csv   18-row table: SPICE median, ML median, bias, median, P95 and mean error (deliverable 6)
    gate_parity.png          log-log ML vs SPICE for all 3,600 points (deliverable 7)
    gate_predictions.csv     every case and corner: SPICE, ML, errors, node voltages
"""
import argparse
import json
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import gates
import ml_model as mm

COLORS = {"INV": "#2a6fdb", "NAND2": "#e08214", "NOR2": "#1b9e77", "AND2": "#8c6bb1", "OR2": "#d62728"}


def load_models(quick):
    sel_path = config.MODEL_DIR / "selection.json"
    if quick and not sel_path.exists() and (config.MODEL_DIR / "quick" / "selection.json").exists():
        sel_path = config.MODEL_DIR / "quick" / "selection.json"
    if not sel_path.exists():
        sys.exit("trained_models/selection.json not found - run 03_train_models.py first")
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    if sel.get("_info", {}).get("spec_fingerprint") != config.spec_fingerprint():
        sys.exit("the trained models were made from datasets with different settings - re-run 03_train_models.py")
    missing = [c for c in config.CASES if c not in sel]
    if missing:
        sys.exit(f"no trained model for: {', '.join(missing)} - run 03_train_models.py for all four cases")
    models = {c: joblib.load(sel_path.parent / sel[c]["file"]) for c in config.CASES}
    for c, m in models.items():
        print(f"  {c:9s} {m.summary()}")
    return models


def load_spice(quick):
    data_dir = config.QUICK_DIR if quick else config.DATA_DIR
    path = data_dir / "gates_spice.csv"
    if not path.exists():
        sys.exit(f"{path.relative_to(config.ROOT)} not found - run 04_simulate_gates.py{' --quick' if quick else ''} first")
    meta = config.check_meta(data_dir / "gates_spice.meta.json", "data/gates_spice.csv")
    spice = pd.read_csv(path, dtype={"inputs": str})
    want = len(gates.GATE_CASES) * gates.N_CORNERS
    if not quick and (meta.get("quick") or len(spice) != want):
        sys.exit(f"data/gates_spice.csv has {len(spice)} rows, expected {want} (18 cases x {gates.N_CORNERS} corners). "
                 f"Run python3 04_simulate_gates.py (without --quick) first.")
    return spice


def feature_range_check(spice):
    """How far the features of the gate transistors fall outside the range of the training data,
    in standard deviations of that feature. The models are only trustworthy inside that range."""
    ranges = {}
    for case in config.CASES:
        df = pd.read_csv(config.dataset_path(case))
        split = config.RESULTS_DIR / "ml" / "splits" / f"{case}_split.npz"
        idx = np.load(split)["train"] if split.exists() else np.arange(len(df))
        F = mm.make_features(df.iloc[idx], config.CASES[case]["dev"])
        sd = F.std()
        ranges[case] = (F.min(), F.max(), sd.where(sd > 1e-12))      # constant features are skipped
    worst = {}
    for (gate, pattern), d in spice.groupby(["gate", "inputs"], sort=False):
        for dv, case, X in gates.device_inputs(gate, pattern, d):
            lo, hi, sd = ranges[case]
            F = mm.make_features(X, dv["dev"])
            z = (np.maximum(lo - F, F - hi) / sd).max()
            for f, v in z.items():
                if np.isfinite(v) and v > worst.get(f, (-np.inf, ""))[0]:
                    worst[f] = (float(v), f"{gate} {pattern} {dv['name']} ({case.replace('_', ' ')})")
    W = pd.DataFrame([{"feature": f, "std outside the training range": v, "where": w}
                      for f, (v, w) in worst.items()]).sort_values("std outside the training range", ascending=False)
    return W


def md_table(df):
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


# The report figure is drawn at the size it is printed (5.2 in wide), so its text prints at
# 8.5-10 pt without scaling.
PRINT_RC = {"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False}


def draw_gate_parity(P, path):
    """ML vs SPICE gate leakage for every gate case and corner, coloured by gate."""
    a = np.abs(P.i_leak_ml - P.i_leak_spice) / P.i_leak_spice * 100
    with plt.rc_context(PRINT_RC):
        fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
        lo = min(P.i_leak_spice.min(), P.i_leak_ml.min()) * 0.7
        hi = max(P.i_leak_spice.max(), P.i_leak_ml.max()) * 1.4
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.9)
        ax.plot([lo, hi], [lo * 1.1, hi * 1.1], color="grey", lw=0.8, ls=":", label="±10 %")
        ax.plot([lo, hi], [lo / 1.1, hi / 1.1], color="grey", lw=0.8, ls=":")
        for gname, d in P.groupby("gate", sort=False):
            ax.scatter(d.i_leak_spice, d.i_leak_ml, s=6, alpha=0.55, color=COLORS[gname], linewidths=0, label=gname)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("gate leakage, SPICE (A)"); ax.set_ylabel("gate leakage, ML (A)")
        ax.set_title(f"{config.TECH}: 18 gate cases × {P.corner.nunique()} corners\n"
                     f"median error {np.median(a):.2f} %, P95 {np.percentile(a, 95):.1f} %, mean {np.mean(a):.2f} %")
        ax.grid(True, which="major", alpha=0.25)
        ax.legend(markerscale=2.2, loc="upper left", frameon=False)
        fig.savefig(path, dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    spice = load_spice(args.quick)
    print("Transistor models:")
    models = load_models(args.quick)
    out = config.RESULTS_DIR / ("gates_quick" if args.quick else "gates")
    out.mkdir(parents=True, exist_ok=True)

    preds, rows = [], []
    for gate, pattern in gates.GATE_CASES:
        d = spice[(spice.gate == gate) & (spice.inputs.str.zfill(len(pattern)) == pattern)]
        if d.empty:
            sys.exit(f"no SPICE rows for {gate} {pattern}")
        ml = gates.ml_leakage(gate, pattern, d, models)
        sp = d["i_leak_spice"].to_numpy(float)
        rel = (ml["i_leak_ml"].to_numpy() - sp) / sp * 100
        rel_direct = (ml["i_leak_ml_direct"].to_numpy() - sp) / sp * 100
        preds.append(pd.DataFrame({
            "gate": gate, "inputs": pattern, "corner": d["corner"].to_numpy(), "vdd": d["vdd"].to_numpy(),
            "temp": d["temp"].to_numpy(), "i_leak_spice": sp, "i_leak_ml": ml["i_leak_ml"].to_numpy(),
            "i_leak_ml_direct": ml["i_leak_ml_direct"].to_numpy(), "err_%": rel, "err_direct_%": rel_direct,
            "kcl_residual": ml["kcl_residual"].to_numpy(),
            **{c: d[c].to_numpy() for c in gates.NODE_COLUMNS}, "states": ml["states"].to_numpy()}))
        a = np.abs(rel)
        rows.append({"Gate": gate, "Inputs (A B)": pattern, "Corners": len(d),
                     "SPICE median (A)": f"{np.median(sp):.3e}",
                     "ML median (A)": f"{np.median(ml['i_leak_ml']):.3e}",
                     "Bias (%)": f"{np.mean(rel):+.2f}",
                     "Median error (%)": f"{np.median(a):.2f}",
                     "P95 error (%)": f"{np.percentile(a, 95):.2f}",
                     "Mean error (%)": f"{np.mean(a):.2f}",
                     "Max error (%)": f"{np.max(a):.1f}",
                     "Median error, direct (%)": f"{np.median(np.abs(rel_direct)):.2f}"})
    P = pd.concat(preds, ignore_index=True)
    P.to_csv(out / "gate_predictions.csv", index=False, float_format="%.8e")
    table = pd.DataFrame(rows)
    table.to_csv(out / "gate_results.csv", index=False)

    a, adir = np.abs(P["err_%"].to_numpy()), np.abs(P["err_direct_%"].to_numpy())
    order = table.assign(m=table["Median error (%)"].astype(float)).sort_values("m")
    fmt = lambda d: ", ".join(f"{r['Gate']} {r['Inputs (A B)']} ({r['m']:.2f} %)" for _, r in d.iterrows())
    best, worst = fmt(order.head(3)), fmt(order.tail(3).iloc[::-1])
    text = [f"# Gate-level validation ({config.TECH})", "",
            "Leakage = current drawn from VDD + from inputs held at VDD (= static power / VDD).",
            "ML = transistor models evaluated at the SPICE node voltages; ON-transistor channel currents "
            "from Kirchhoff's law. 'Direct' = ON-transistor id/is taken straight from the models instead.",
            "Bias = mean signed error (ML - SPICE) / SPICE. Errors are |ML - SPICE| / SPICE over the corners.",
            "", md_table(table),
            f"All {len(P)} points: median error {np.median(a):.2f} %, P95 {np.percentile(a, 95):.2f} %, "
            f"mean {np.mean(a):.2f} % (mean accuracy {100 - np.mean(a):.2f} %), bias {np.mean(P['err_%']):+.2f} %. "
            f"Direct: median {np.median(adir):.2f} %, P95 {np.percentile(adir, 95):.2f} %.", "",
            f"Most accurate cases (median error): {best}.",
            f"Least accurate cases: {worst}.", "",
            f"Unbalanced node current (at internal nodes that touch only OFF transistors, Kirchhoff has nothing "
            f"to adjust, so the OFF-transistor models' error remains): largest {P['kcl_residual'].max() * 100:.1f} % "
            f"of the gate leakage, median {P['kcl_residual'].median() * 100:.2f} %.", ""]
    W = feature_range_check(spice)
    W.to_csv(out / "feature_range_check.csv", index=False, float_format="%.3f")
    top = W.iloc[0]
    text += ["Feature range check: the gate transistors' model inputs compared with the training data. "
             f"Largest excursion outside the training range: {top['std outside the training range']:.2f} standard "
             f"deviations ({top['feature']}, {top['where']}). "
             + ("All inputs are inside or at the edge of the training range."
                if top["std outside the training range"] < 1 else
                "WARNING: some gate transistors are outside the training range, so the models extrapolate there."),
             ""]
    (out / "gate_results.md").write_text("\n".join(text), encoding="utf-8")
    print("\n" + "\n".join(text))

    draw_gate_parity(P, out / "gate_parity.png")
    print(f"Wrote {out.relative_to(config.ROOT)}/ (table, parity plot, {len(P)} predictions)")
    if not args.quick:
        print("Next: python3 06_simulate_cla.py")


if __name__ == "__main__":
    main()
