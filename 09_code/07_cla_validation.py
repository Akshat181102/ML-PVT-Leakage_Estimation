"""
Part C, step 2: ML leakage of the 3-bit CLA, validated against SPICE.

For every input vector, a logic simulation gives the input pattern of each of
the 34 gates. The circuit's ML leakage at a corner is the sum of the gate-level
ML estimates (Part B, results/gates/gate_predictions.csv) for those gate types
and patterns at that same corner. This assumes every gate input sits exactly at
0 or VDD and ignores how gates load each other.

To separate the two error sources, a second estimate is built the same way from
the SPICE leakage of the stand-alone gates (data/gates_spice.csv):
  * SPICE gate-sum vs circuit SPICE  -> error of the gate-composition assumption alone
  * ML vs circuit SPICE              -> total error (composition + ML)

Usage:
    python3 07_cla_validation.py            # needs the full runs of 04-06
    python3 07_cla_validation.py --quick    # on the quick runs, results in results/cla_quick/

Outputs in results/cla/:
    cla_results.md              summary metrics (deliverable 8) and leakage share per gate type
    cla_parity.png              log-log ML vs SPICE (and SPICE gate-sum vs SPICE) for every vector and corner
    cla_predictions.csv         every vector and corner
    cla_per_vector.csv          metrics per input vector
"""
import argparse
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np
import pandas as pd

import cla
import config
import gates


def _tick(v, _):
    """Label only 1, 2 and 5 x 10^n so log axes spanning ~1 decade stay readable."""
    if v <= 0:
        return ""
    e = int(np.floor(np.log10(v) + 1e-9))
    m = round(v / 10 ** e, 6)
    return f"{m:g}e{e}" if m in (1, 2, 5) else ""


# The report figure is drawn at the size it is printed (6.7 in wide), so its text prints at
# 8.5-10 pt without scaling.
PRINT_RC = {"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "figure.titlesize": 10.5,
            "axes.spines.top": False, "axes.spines.right": False}


def draw_cla_parity(P, path):
    """CLA leakage: ML estimate (left) and SPICE gate-sum (right) against full-circuit SPICE."""
    with plt.rc_context(PRINT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.5), constrained_layout=True)
        for ax, col, title in ((axes[0], "i_leak_ml", "ML (sum of gate ML estimates)"),
                               (axes[1], "i_leak_spice_gate_sum", "SPICE gate-sum (composition only)")):
            x, y = P.i_leak_spice.to_numpy(), P[col].to_numpy()
            lo, hi = min(x.min(), y.min()) * 0.8, max(x.max(), y.max()) * 1.25
            ax.plot([lo, hi], [lo, hi], color="black", lw=0.9)
            ax.plot([lo, hi], [lo * 1.1, hi * 1.1], color="grey", lw=0.8, ls=":", label="±10 %")
            ax.plot([lo, hi], [lo / 1.1, hi / 1.1], color="grey", lw=0.8, ls=":")
            sc = ax.scatter(x, y, s=2.5, c=P.temp, cmap="coolwarm", alpha=0.55, linewidths=0)
            rel = np.abs(y - x) / x * 100
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
                axis.set_minor_locator(LogLocator(base=10, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0)))
                axis.set_major_formatter(FuncFormatter(_tick))
                axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
            ax.set_xlabel("CLA leakage, full-circuit SPICE (A)"); ax.set_ylabel(f"{title.split(' (')[0]} (A)")
            ax.set_title(f"{title}\nmedian error {np.median(rel):.2f} %, P95 {np.percentile(rel, 95):.2f} %")
            ax.grid(True, which="major", alpha=0.25)
            ax.legend(loc="upper left", frameon=False)
        fig.colorbar(sc, ax=axes, label="temperature (°C)", shrink=0.85, pad=0.02)
        fig.suptitle(f"{config.TECH} 3-bit CLA: {P.vector.nunique()} vectors × {P.corner.nunique()} corners")
        fig.savefig(path, dpi=200)
        plt.close(fig)


def need(path, hint):
    if not path.exists():
        sys.exit(f"{path.relative_to(config.ROOT)} not found - {hint}")
    return pd.read_csv(path, dtype={"inputs": str})


def metrics(est, ref):
    rel = (est - ref) / ref * 100
    a = np.abs(rel)
    return {"SPICE median (A)": f"{np.median(ref):.3e}", "Estimate median (A)": f"{np.median(est):.3e}",
            "Bias (%)": f"{np.mean(rel):+.2f}", "Median error (%)": f"{np.median(a):.2f}",
            "P95 error (%)": f"{np.percentile(a, 95):.2f}", "Mean error (%)": f"{np.mean(a):.2f}",
            "Max error (%)": f"{np.max(a):.2f}"}


def md_table(df):
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    q = " --quick" if args.quick else ""
    data_dir = config.QUICK_DIR if args.quick else config.DATA_DIR
    OUT = config.RESULTS_DIR / ("cla_quick" if args.quick else "cla")
    circ = need(data_dir / "cla_spice.csv", f"run 06_simulate_cla.py{q} first")
    meta = config.check_meta(data_dir / "cla_spice.meta.json", "the CLA SPICE results")
    gml = need(config.RESULTS_DIR / ("gates_quick" if args.quick else "gates") / "gate_predictions.csv",
               f"run 05_gate_validation.py{q} first")
    gsp = need(data_dir / "gates_spice.csv", f"run 04_simulate_gates.py{q} first")
    want = len(cla.vectors()) * gates.N_CORNERS
    if not args.quick and (meta.get("quick") or len(circ) != want or gml.corner.nunique() != gates.N_CORNERS):
        sys.exit(f"incomplete full-run data (CLA rows {len(circ)} of {want}, gate corners {gml.corner.nunique()} "
                 f"of {gates.N_CORNERS}). Run 04, 05 and 06 without --quick first.")
    OUT.mkdir(parents=True, exist_ok=True)

    def table(df, col):
        t = df.pivot_table(index="corner", columns=["gate", "inputs"], values=col)
        return t

    ml_t = table(gml, "i_leak_ml")
    sp_t = table(gsp, "i_leak_spice")
    vecs = cla.vectors()

    parts = []
    share = Counter()
    for k, d in circ.groupby("vector"):
        counts = Counter((g, p) for _, g, p in cla.gate_patterns(vecs[k]))
        corners = d["corner"].to_numpy()
        ml = np.zeros(len(d))
        sp = np.zeros(len(d))
        for (g, p), n in counts.items():
            if (g, p) not in ml_t.columns:
                sys.exit(f"no gate-level result for {g} {p}")
            gm = ml_t[(g, p)].reindex(corners).to_numpy()
            ml += n * gm
            sp += n * sp_t[(g, p)].reindex(corners).to_numpy()
            share[g] += float(np.sum(n * gm))
        parts.append(pd.DataFrame({"vector": k, "corner": corners, "vdd": d["vdd"].to_numpy(),
                                   "temp": d["temp"].to_numpy(), "i_leak_spice": d["i_leak_spice"].to_numpy(),
                                   "i_leak_ml": ml, "i_leak_spice_gate_sum": sp}))
    P = pd.concat(parts, ignore_index=True)
    if P[["i_leak_ml", "i_leak_spice_gate_sum"]].isna().any().any():
        sys.exit("missing gate-level results for some corners - were 04/05 run with all 200 corners?")
    P["err_ml_%"] = (P.i_leak_ml - P.i_leak_spice) / P.i_leak_spice * 100
    P["err_gate_sum_%"] = (P.i_leak_spice_gate_sum - P.i_leak_spice) / P.i_leak_spice * 100
    P.to_csv(OUT / "cla_predictions.csv", index=False, float_format="%.8e")

    per_vec = P.groupby("vector").agg(
        spice_median=("i_leak_spice", "median"), ml_median=("i_leak_ml", "median"),
        ml_bias_pct=("err_ml_%", "mean"), ml_median_err_pct=("err_ml_%", lambda e: np.median(np.abs(e))),
        gate_sum_bias_pct=("err_gate_sum_%", "mean")).reset_index()
    bits = pd.DataFrame([{**v, "vector": k} for k, v in enumerate(vecs)])
    per_vec = bits.merge(per_vec, on="vector")
    per_vec.to_csv(OUT / "cla_per_vector.csv", index=False, float_format="%.6g")

    summary = pd.DataFrame([
        {"Estimate": "ML (sum of gate-level ML estimates)", **metrics(P.i_leak_ml.to_numpy(), P.i_leak_spice.to_numpy())},
        {"Estimate": "SPICE gate-sum (stand-alone gate SPICE)",
         **metrics(P.i_leak_spice_gate_sum.to_numpy(), P.i_leak_spice.to_numpy())},
    ])
    ga = np.abs(gml["err_%"].to_numpy())
    gate_line = (f"For comparison, gate level ({len(gml)} points): median error {np.median(ga):.2f} %, "
                 f"P95 {np.percentile(ga, 95):.2f} %, mean {np.mean(ga):.2f} %, bias {gml['err_%'].mean():+.2f} %.")
    cla_a = np.abs(P["err_ml_%"].to_numpy())
    acc_line = (f"CLA mean accuracy (100 % - mean error): {100 - np.mean(cla_a):.2f} %; "
                f"{np.mean(cla_a <= 5) * 100:.0f} % of points within 5 % of SPICE, {np.mean(cla_a <= 10) * 100:.0f} % within 10 %.")
    tot = sum(share.values())
    share_df = pd.DataFrame([{"Gate type": g, "Instances": sum(1 for _, gg, _, _ in cla.CIRCUIT if gg == g),
                              "Share of CLA leakage (ML)": f"{100 * v / tot:.1f} %"} for g, v in share.most_common()])
    worst = per_vec.reindex(per_vec.ml_bias_pct.abs().sort_values(ascending=False).index).head(5)
    text = [f"# 3-bit CLA validation ({config.TECH})", "",
            f"{len(cla.CIRCUIT)} gates / {cla.n_transistors()} transistors; {P.vector.nunique()} input vectors x "
            f"{P.corner.nunique()} corners = {len(P)} points.",
            "Leakage = I(VDD) + I(inputs at VDD). Errors relative to the full-circuit SPICE leakage.", "",
            md_table(summary),
            "The SPICE gate-sum row isolates the error of composing the circuit from stand-alone gates "
            "(ideal 0/VDD inputs, no loading). The ML row adds the gate-level ML error on top.", "",
            acc_line, "", gate_line, "",
            "## Leakage share per gate type (ML, all vectors and corners)", "", md_table(share_df),
            "## Vectors with the largest ML bias", "",
            md_table(worst[["vector", *cla.INPUTS, "ml_bias_pct", "gate_sum_bias_pct"]].round(2)), ""]
    (OUT / "cla_results.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))

    draw_cla_parity(P, OUT / "cla_parity.png")
    print(f"Wrote {OUT.relative_to(config.ROOT)}/ ({len(P)} points)")


if __name__ == "__main__":
    main()
