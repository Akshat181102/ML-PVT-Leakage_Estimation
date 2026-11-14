"""
Part A data check (Project doc, Section 6, deliverable 3).

1. Kirchhoff check per case: id + ig + is + ib should be ~0 (PASS if the largest
   residual is below 0.1 % of the largest terminal current of that sample).
2. The three trend plots of deliverable 3:
     - OFF current vs temperature   (NMOS OFF, PMOS OFF)
     - OFF current vs vth0          (NMOS OFF, PMOS OFF)
     - gate current vs toxe         (NMOS ON, PMOS ON, NMOS OFF, PMOS OFF)
   Each panel shows
     * every dataset sample (faint dots) and its binned median (dashed line),
       with currents divided by W so width does not spread the cloud;
       for the OFF plots only samples with |Vds| >= 0.5 VDD are used, so that
       sub-threshold conduction dominates over gate tunnelling;
     * a controlled SPICE sweep (solid line) where only the plotted parameter
       changes and everything else is nominal (27 C, VDD = 1 V, W = 2L).
       The gate-current sweeps use the bias the transistor has inside a gate:
       ON with drain and source at the body potential (full VDD across the oxide),
       OFF with full VDD between drain and source (gate-drain edge tunnelling).
   The Spearman rank correlation of the dataset cloud is a numeric trend check.

Usage:
    python3 02_data_check.py                  # all four cases (ngspice needed for the sweeps)
    python3 02_data_check.py --no-sweep       # dataset-only plots

Outputs in results/data_check/:
    kirchhoff.csv, kirchhoff.md, off_current_vs_temp.png, off_current_vs_vth0.png,
    gate_current_vs_toxe.png, summary.md
"""
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import fet
import spice_utils as su

OUT = config.RESULTS_DIR / "data_check"
COLORS = {"nmos": "#2a6fdb", "pmos": "#d9480f"}


def load(case):
    path = config.dataset_path(case)
    if not path.exists():
        sys.exit(f"missing {path.relative_to(config.ROOT)} - run 01_generate_transistor_data.py first")
    config.check_meta(config.meta_path(case), f"the {case} dataset")
    return pd.read_csv(path)


def md_table(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.tolist()) + " |")
    return "\n".join(lines) + "\n"


def kirchhoff(data):
    rows = []
    for case, df in data.items():
        cur = df[["id", "ig", "is", "ib"]]
        s = cur.sum(axis=1).abs()
        rel = s / cur.abs().max(axis=1)
        rows.append({
            "case": case, "rows": len(df),
            "median abs(sum) (A)": s.median(), "max abs(sum) (A)": s.max(),
            "median relative": rel.median(), "max relative": rel.max(),
            "median abs(id) (A)": df["id"].abs().median(),
            "median abs(ig) (A)": df["ig"].abs().median(),
            "median abs(ib) (A)": df["ib"].abs().median(),
            "nan rows": int(cur.isna().any(axis=1).sum()),
            "result": "PASS" if rel.max() < 1e-3 and not cur.isna().any().any() else "FAIL",
        })
    return pd.DataFrame(rows)


def binned_median(x, y, bins=15):
    edges = np.linspace(x.min(), x.max(), bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
    centers, med = [], []
    for b in range(bins):
        sel = idx == b
        if sel.sum() >= 5:
            centers.append(x[sel].mean())
            med.append(np.median(y[sel]))
    return np.array(centers), np.array(med)


def spice_sweep(case, param, values, cards, nominal, bias, ycol):
    ys = []
    for v in values:
        row = fet.nominal_row(case, nominal, bias=bias, **{param: v})
        ys.append(abs(fet.simulate(case, row, cards)[ycol]) / (row["w"] * 1e6))
    return np.array(ys)


# The report figures are drawn at the size they are printed (6.7 in wide), so their text
# prints at 8.5-10 pt without scaling.
PRINT_RC = {"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "figure.titlesize": 10.5,
            "axes.spines.top": False, "axes.spines.right": False}


def trend_plot(data, cases, xcol, ycol, xlabel, ylabel, fname, title, sweep_ctx,
               xscale=1.0, xabs=False, vds_filter=True, bias="channel"):
    """One row of panels (`cases` = list of cases) or a grid (`cases` = list of rows).
    `bias` is the SPICE-sweep bias of every panel, or a {case: bias} dict."""
    if not cases:
        return []
    rows = [list(r) for r in cases] if isinstance(cases[0], (list, tuple)) else [list(cases)]
    rows = [[c for c in r if c in data] for r in rows]   # plot whichever cases are available
    rows = [r for r in rows if r]
    if not rows:
        return []
    nrow, ncol = len(rows), max(len(r) for r in rows)
    lines = []
    with plt.rc_context(PRINT_RC):
        fig, grid = plt.subplots(nrow, ncol, figsize=(6.7, 3.0 if nrow == 1 else 2.6 * nrow + 0.5),
                                 constrained_layout=True, squeeze=False)
        panels = []
        for i, r in enumerate(rows):
            for j in range(ncol):
                if j < len(r):
                    panels.append((grid[i][j], r[j], i == nrow - 1))
                else:
                    grid[i][j].set_visible(False)
        for ax, case, bottom in panels:
            dev = config.CASES[case]["dev"]
            case_bias = bias[case] if isinstance(bias, dict) else bias
            df = data[case]
            if vds_filter:
                df = df[(df["vd"] - df["vs"]).abs() >= 0.5 * df["vdd"]]
            x = df[xcol].to_numpy() * xscale
            x = np.abs(x) if xabs else x
            y = df[ycol].abs().to_numpy() / (df["w"].to_numpy() * 1e6)   # A per um of width
            keep = y > 0
            x, y = x[keep], y[keep]
            rho = pd.Series(x).corr(pd.Series(np.log10(y)), method="spearman")
            ax.scatter(x, y, s=2.5, alpha=0.22, color=COLORS[dev], linewidths=0, label="dataset samples")
            cx, my = binned_median(x, y)
            ax.plot(cx, my, color="black", lw=1.4, ls="--", label="dataset binned median")
            sweep_txt = ""
            if sweep_ctx is not None:
                cards, nominal = sweep_ctx
                if xcol == "temp":
                    sv = np.linspace(*config.TEMP_RANGE, 12)
                else:
                    sv = nominal[dev][xcol] * np.linspace(1 - config.PROC_REL, 1 + config.PROC_REL, 9)
                sy = spice_sweep(case, xcol, sv, cards, nominal, case_bias, ycol)
                sx = np.abs(sv * xscale) if xabs else sv * xscale
                ax.plot(sx, sy, color=COLORS[dev], lw=2, marker="o", ms=3, label="SPICE sweep, rest nominal")
                direction = "+" if sy[np.argmax(sx)] > sy[np.argmin(sx)] else "-"
                sweep_txt = f"{direction} (x{max(sy) / min(sy):.3g} over range)"
            ax.set_yscale("log")
            if bottom:
                ax.set_xlabel(xlabel)
            ax.set_title(f"{case.replace('_', ' ').upper()} (n = {len(x):,}, Spearman ρ = {rho:+.2f})", fontsize=8.5)
            ax.grid(True, which="major", alpha=0.25)
            lines.append(f"| {title} | {case} | {len(x)} | {rho:+.3f} | {sweep_txt or 'not run'} |")
        for i in range(nrow):
            grid[i][0].set_ylabel(ylabel)
        handles, labels = grid[0][0].get_legend_handles_labels()
        leg = fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False,
                         markerscale=2.5, handlelength=1.6)
        for h in getattr(leg, "legend_handles", getattr(leg, "legendHandles", [])):
            h.set_alpha(0.9)                   # the legend's copy only; the plotted dots stay faint
        fig.suptitle(f"{config.TECH}: {title}")
        fig.savefig(OUT / fname, dpi=200)
        plt.close(fig)
    return lines


def current_ranges(data):
    """How many decades each terminal current spans and how often it is negative.
    Useful for choosing the ML target transform (e.g. signed log)."""
    rows = []
    for case, df in data.items():
        r = {"case": case}
        for t in ("id", "ig", "is", "ib"):
            a = df[t].abs()
            a = a[a > 0]
            r[f"{t} decades"] = f"{np.log10(a.max() / a.min()):.1f}"
            r[f"{t} % negative"] = f"{100 * (df[t] < 0).mean():.1f}"
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sweep", action="store_true", help="skip the controlled SPICE sweeps")
    ap.add_argument("--cases", nargs="+", choices=list(config.CASES), default=list(config.CASES))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cases = args.cases
    data = {case: load(case) for case in cases}
    print(f"Checking: {', '.join(cases)}\n")

    k = kirchhoff(data)
    k.to_csv(OUT / "kirchhoff.csv", index=False)
    k_md = k.copy()
    for c in k.columns:
        if c not in ("case", "rows", "nan rows", "result"):
            k_md[c] = k_md[c].map("{:.3e}".format)
    (OUT / "kirchhoff.md").write_text(md_table(k_md), encoding="utf-8")
    print("Kirchhoff check (id + ig + is + ib):")
    print(k_md.to_string(index=False))

    sweep_ctx = None
    if not args.no_sweep:
        if su.ngspice_version() is None:
            print("ngspice not found - plotting without the SPICE sweep lines")
        else:
            cards = su.read_model_cards()
            sweep_ctx = (cards, su.nominal_process(cards))

    trend = ["| plot | case | samples | Spearman rho (dataset) | SPICE sweep direction |",
             "|---|---|---|---|---|"]
    trend += trend_plot(data, ["nmos_off", "pmos_off"], "temp", "id", "Temperature (°C)",
                        "|Id| / W (A/µm)", "off_current_vs_temp.png",
                        "OFF current vs temperature", sweep_ctx)
    trend += trend_plot(data, ["nmos_off", "pmos_off"], "vth0", "id", "|vth0| (V)",
                        "|Id| / W (A/µm)", "off_current_vs_vth0.png",
                        "OFF current vs vth0", sweep_ctx, xabs=True)
    trend += trend_plot(data, [["nmos_on", "pmos_on"], ["nmos_off", "pmos_off"]], "toxe", "ig",
                        "toxe (nm)", "|Ig| / W (A/µm)", "gate_current_vs_toxe.png",
                        "Gate current vs toxe", sweep_ctx, xscale=1e9, vds_filter=False,
                        bias={"nmos_on": "gate", "pmos_on": "gate",
                              "nmos_off": "channel", "pmos_off": "channel"})

    ranges = md_table(current_ranges(data))
    print("\nCurrent ranges (decades spanned, % of samples negative):")
    print(current_ranges(data).to_string(index=False))

    summary = ["# Data check", "", f"Model file: {config.TECH}, VDD_nom = {config.VDD_NOM} V", "",
               f"Cases checked: {', '.join(cases)}", "",
               "## Kirchhoff check", "", (OUT / "kirchhoff.md").read_text(encoding="utf-8"), "",
               "## Trend checks (expected direction: temperature +, vth0 -, toxe -)", "",
               *trend, "",
               f"Junction areas on every transistor: AS = AD = W x Ldiff, PS = PD = 2 (W + Ldiff), "
               f"Ldiff = {config.DIFFUSION_LENGTH * 1e9:.0f} nm.", "",
               "## Current ranges (sign convention: positive = into the terminal)", "",
               ranges, ""]
    (OUT / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("\nTrend checks (expected direction: temperature +, vth0 -, toxe -):")
    print("\n".join(trend))
    print(f"\nKirchhoff: {', '.join(f'{r.case} {r.result}' for r in k.itertuples())}")
    print(f"Wrote results to {OUT.relative_to(config.ROOT)}/")


if __name__ == "__main__":
    main()
