"""
Step 5: error analysis - where the largest errors occur and why.

Uses only the outputs of steps 1-4 (no new SPICE runs):
  A. Transistor models (test set): error vs how close (Vd, Vs) is to a corner of
     the sampled square (both at a supply rail, where every gate transistor sits),
     vs temperature, and for currents that are tiny compared with the device's
     largest current (close to a sign change).
  B. Gates: error and bias vs the corner conditions (temperature, VDD, NMOS and
     PMOS vth0), and the bias of every gate case per temperature range.
  C. CLA: bias vs temperature, and how much of the error varies between corners
     versus between input vectors.
  D. Model-to-model variance: the final model of every case is an ensemble of
     three fits that differ only in their random initial weights. The gate and CLA
     errors are recomputed with each fit alone and with the ensemble, which shows
     how much of the error comes from the particular fit and how much averaging
     removes. (If the final models are single fits, two refits with other seeds
     are made instead, ~30 min.)

Usage:
    python3 08_error_analysis.py              # ~1 min
    python3 08_error_analysis.py --no-seeds   # parts A-C only

Outputs in results/error_analysis/:
    error_analysis.md                 every table, ready for the report
    transistor_error_vs_corner.png    part A
    gate_error_vs_conditions.png      part B
    gate_bias_vs_temperature.png      part B
    cla_error_vs_temperature.png      part C
    seed_variance.png                 part D
    *.csv                             the numbers behind each table
"""
import argparse
import json
import sys
import time
from collections import Counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cla
import config
import gates
import ml_model as mm

OUT = config.RESULTS_DIR / "error_analysis"
SEED_OFFSETS = (1, 2)
T_BINS = [-40, 0, 40, 80, 125]
CASE_COLORS = {"nmos_on": "#2a6fdb", "nmos_off": "#7fa7e8", "pmos_on": "#d9480f", "pmos_off": "#f0a07a"}


def label(case):
    return case.replace("_", " ").upper()


def md_table(df, floatfmt="{:.2f}"):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                       for v in r.tolist()) + " |")
    return "\n".join(lines) + "\n"


def need(path, hint):
    if not path.exists():
        sys.exit(f"{path.relative_to(config.ROOT)} not found - {hint}")
    return path


def stats(e):
    a = np.abs(np.asarray(e, float))
    return {"points": len(a), "median error (%)": np.median(a), "P95 error (%)": np.percentile(a, 95),
            "mean error (%)": np.mean(a)}


def temp_bin(t):
    return pd.cut(t, T_BINS, include_lowest=True,
                  labels=[f"{a:.0f} to {b:.0f} °C" for a, b in zip(T_BINS[:-1], T_BINS[1:])])


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_models():
    sel_path = need(config.MODEL_DIR / "selection.json", "run 03_train_models.py first")
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    if sel.get("_info", {}).get("spec_fingerprint") != config.spec_fingerprint():
        sys.exit("the trained models were made with different settings - re-run 03_train_models.py")
    return {c: joblib.load(config.MODEL_DIR / sel[c]["file"]) for c in config.CASES}


def load_inputs():
    config.check_meta(config.DATA_DIR / "gates_spice.meta.json", "data/gates_spice.csv")
    config.check_meta(config.DATA_DIR / "cla_spice.meta.json", "data/cla_spice.csv")
    gp = pd.read_csv(need(config.RESULTS_DIR / "gates" / "gate_predictions.csv", "run 05_gate_validation.py first"),
                     dtype={"inputs": str})
    corners = pd.read_csv(need(config.DATA_DIR / "gate_corners.csv", "run 04_simulate_gates.py first"))
    gsp = pd.read_csv(config.DATA_DIR / "gates_spice.csv", dtype={"inputs": str})
    cp = pd.read_csv(need(config.RESULTS_DIR / "cla" / "cla_predictions.csv", "run 07_cla_validation.py first"))
    csp = pd.read_csv(config.DATA_DIR / "cla_spice.csv")
    if gp.corner.nunique() != gates.N_CORNERS or len(csp) != len(cla.vectors()) * gates.N_CORNERS:
        sys.exit("the gate or CLA results are not from the full 200-corner runs - run 04-07 without --quick first")
    return gp, corners, gsp, cp, csp


# ----------------------------------------------------------------------------
# A. Transistor models
# ----------------------------------------------------------------------------
def part_a(text):
    rows, trows, nrows, curves = [], [], [], {}
    edges = [0, 0.05, 0.10, 0.20, 0.50]
    names = ["< 5 %", "5-10 %", "10-20 %", "> 20 %"]
    for case in config.CASES:
        tp = pd.read_csv(need(config.RESULTS_DIR / "ml" / f"test_predictions_{case}.csv",
                              "run 03_train_models.py first"))
        true = tp[mm.TERMINALS].to_numpy(float)
        pred = tp[[f"{t}_ml" for t in mm.TERMINALS]].to_numpy(float)
        err = mm.errors(pred, true)                                   # (n, 4)
        vdd = tp.vdd.to_numpy()
        dv = np.minimum(tp.vd, vdd - tp.vd) / vdd
        ds = np.minimum(tp.vs, vdd - tp.vs) / vdd
        cdist = np.maximum(dv, ds).to_numpy()                         # distance to the nearest (Vd, Vs) corner
        idx = np.clip(np.digitize(cdist, edges) - 1, 0, len(names) - 1)
        r = {"case": label(case)}
        for k, nm in enumerate(names):
            e = err[idx == k].ravel()
            r[f"{nm}: samples"] = int((idx == k).sum())
            r[f"{nm}: median / P95 (%)"] = f"{np.median(e):.2f} / {np.percentile(e, 95):.1f}" if len(e) else "-"
        rows.append(r)
        curves[case] = (edges, [np.median(err[idx == k]) if (idx == k).any() else np.nan for k in range(len(names))])
        # temperature
        tb = temp_bin(tp.temp)
        r = {"case": label(case)}
        for b in tb.cat.categories:
            e = err[(tb == b).to_numpy()].ravel()
            r[str(b)] = f"{np.median(e):.2f} / {np.percentile(e, 95):.1f}"
        trows.append(r)
        # tiny currents (close to a sign change): |I| < 1 % of the device's largest current
        ratio = np.abs(true) / np.abs(true).max(axis=1, keepdims=True)
        small = ratio < 1e-2
        big_err = err > 10
        nrows.append({"case": label(case), "currents with error > 10 %": int(big_err.sum()),
                      "of which |I| < 1 % of the device's largest current": int((big_err & small).sum()),
                      "P95 error, |I| >= 1 % of largest (%)": np.percentile(err[~small], 95),
                      "P95 error, |I| < 1 % of largest (%)": np.percentile(err[small], 95) if small.any() else np.nan})
    A = pd.DataFrame(rows)
    T = pd.DataFrame(trows)
    N = pd.DataFrame(nrows)
    A.to_csv(OUT / "transistor_error_vs_corner.csv", index=False)
    text += ["## A. Transistor models (test set, chosen models, all four terminals)", "",
             "### Error vs distance of (Vd, Vs) from the nearest corner of the sampled square", "",
             "Distance = max(distance of Vd, distance of Vs) to the nearer supply rail, as a fraction of VDD. "
             "Most gate transistors sit at distance ~0 (see part B).", "", md_table(A),
             "### Error vs temperature (median / P95, %)", "", md_table(T),
             "### Errors on currents that are tiny compared with the device's largest current", "",
             md_table(N, "{:.1f}")]

    plot_corner_distance(names, {label(c): (CASE_COLORS[c], med) for c, (_, med) in curves.items()},
                         OUT / "transistor_error_vs_corner.png")


# All report figures from this script are drawn at the size they are printed (6.0-6.7 in
# wide), so their text prints at 9-10 pt without scaling.
PRINT_FONT = 9.5


def plot_corner_distance(names, curves, path):
    """Median test error of each transistor model vs the distance of (Vd, Vs) from the nearest
    corner of the sampled square. curves = {case label: (colour, medians per bin)}."""
    fig, ax = plt.subplots(figsize=(6.0, 3.4), constrained_layout=True)
    x = np.arange(len(names))
    for name, (color, med) in curves.items():
        ax.plot(x, med, marker="o", ms=6, lw=2, color=color, label=name)
    ax.set_xticks(x, names, fontsize=PRINT_FONT)
    ax.tick_params(axis="y", labelsize=PRINT_FONT)
    ax.set_xlabel("distance of (Vd, Vs) from the nearest corner (fraction of VDD)", fontsize=PRINT_FONT)
    ax.set_ylabel("median error, test set (%)", fontsize=PRINT_FONT)
    ax.set_title("Transistor models are least accurate near the (Vd, Vs) corners", fontsize=11)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=PRINT_FONT, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def gate_transistor_corner_distance(gp):
    """Distance of every gate transistor's (Vd, Vs) from the nearest corner (median over the corners),
    from the SPICE node voltages. Returns a DataFrame, one row per transistor of every gate case."""
    rows = []
    for (g, p), d in gp.groupby(["gate", "inputs"], sort=False):
        V = gates._node_voltages(g, p, d)
        vdd = V["vdd"]
        for dv in gates.GATES[g]["devices"]:
            vd, vs = V[dv["d"]], V[dv["s"]]
            dd = np.minimum(np.abs(vd), np.abs(vdd - vd)) / vdd
            ds = np.minimum(np.abs(vs), np.abs(vdd - vs)) / vdd
            rows.append({"case": f"{g} {p}", "transistor": dv["name"], "drain": dv["d"], "source": dv["s"],
                         "distance (% of VDD)": float(np.median(np.maximum(dd, ds))) * 100})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# B. Gates
# ----------------------------------------------------------------------------
def part_b(text, gp, corners):
    nom = {dev: {p: corners[f"{dev[0]}_{p}"].mean() for p in config.PROC_PARAMS} for dev in ("nmos", "pmos")}
    G = gp.merge(corners[["corner", "n_vth0", "p_vth0"]], on="corner")
    G["n_vth0_pct"] = (G.n_vth0 / nom["nmos"]["vth0"] - 1) * 100
    G["p_vth0_pct"] = (G.p_vth0 / nom["pmos"]["vth0"] - 1) * 100       # |vth0| change (PMOS vth0 < 0)
    groups = [("Temperature", temp_bin(G.temp)),
              ("VDD", pd.cut(G.vdd, [0.9, 0.9667, 1.0333, 1.1], include_lowest=True,
                             labels=["0.90-0.97 V", "0.97-1.03 V", "1.03-1.10 V"])),
              ("NMOS vth0", pd.cut(G.n_vth0_pct, [-10, -3.33, 3.33, 10], include_lowest=True,
                                   labels=["-10 to -3 %", "-3 to +3 %", "+3 to +10 %"])),
              ("PMOS |vth0|", pd.cut(G.p_vth0_pct, [-10, -3.33, 3.33, 10], include_lowest=True,
                                     labels=["-10 to -3 %", "-3 to +3 %", "+3 to +10 %"]))]
    rows = []
    for name, b in groups:
        for cat in b.cat.categories:
            e = G["err_%"][b == cat]
            rows.append({"condition": name, "range": str(cat), "points": len(e),
                         "median error (%)": np.median(np.abs(e)), "P95 error (%)": np.percentile(np.abs(e), 95),
                         "bias (%)": np.mean(e)})
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "gate_error_vs_conditions.csv", index=False)
    tb = temp_bin(G.temp)
    H = G.assign(tb=tb).pivot_table(index=["gate", "inputs"], columns="tb", values="err_%", aggfunc="mean",
                                    observed=True, sort=False)
    H = H[[c for c in tb.cat.categories if c in H.columns]]
    H.to_csv(OUT / "gate_bias_vs_temperature.csv")
    Hm = H.reset_index()
    Hm.insert(0, "case", Hm.gate + " " + Hm.inputs)
    Hm = Hm.drop(columns=["gate", "inputs"])
    Hm.columns = [str(c) for c in Hm.columns]
    D = gate_transistor_corner_distance(gp)
    D.to_csv(OUT / "gate_transistor_corner_distance.csv", index=False)
    far = D[D["distance (% of VDD)"] >= 1].sort_values("distance (% of VDD)", ascending=False)
    text += ["## B. Gates (3,600 points)", "",
             f"Where the gate transistors sit: {(D['distance (% of VDD)'] < 1).mean() * 100:.0f} % of them have "
             f"(Vd, Vs) within 1 % of VDD of a corner of the sampled square (median over the corners). The others "
             f"touch an internal node of a stack:", "", md_table(far.head(12)),
             "### Error and bias vs corner conditions", "", md_table(B),
             "### Bias (mean signed error, %) of each gate case per temperature range", "", md_table(Hm, "{:+.1f}")]

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), constrained_layout=True, sharey=True)
    for ax, (name, _) in zip(axes, groups):
        d = B[B.condition == name]
        x = np.arange(len(d))
        ax.bar(x - 0.2, d["median error (%)"], 0.4, color="#2a6fdb", label="median error")
        ax.bar(x + 0.2, d["bias (%)"], 0.4, color="#d9480f", label="bias")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x, d["range"], fontsize=8)
        ax.set_title(name, fontsize=10); ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("%"); axes[0].legend(fontsize=8)
    fig.suptitle("Gate leakage: ML error vs corner conditions (all 18 cases)", fontsize=11)
    fig.savefig(OUT / "gate_error_vs_conditions.png", dpi=150)
    plt.close(fig)

    plot_bias_heatmap(H, OUT / "gate_bias_vs_temperature.png")


def plot_bias_heatmap(H, path):
    """Bias of each gate case per temperature range, drawn at print size (16 cm wide) so the
    cell values stay readable in the report: 9 pt values, white text on the dark cells, and
    a rule between gate types."""
    vals = H.to_numpy(float)
    lim = np.ceil(np.nanmax(np.abs(vals)))
    fig, ax = plt.subplots(figsize=(6.3, 6.9), constrained_layout=True)
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(H.shape[1]), [str(c) for c in H.columns], fontsize=9.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(H.shape[0]), [f"{g}  {p}" for g, p in H.index], fontsize=9.5)
    ax.tick_params(length=0)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=9.5,
                    color="white" if abs(v) > 0.55 * lim else "#1a1a1a",
                    fontweight="bold" if abs(v) > 0.55 * lim else "normal")
    gates = [g for g, _ in H.index]
    for i in range(1, len(gates)):
        if gates[i] != gates[i - 1]:
            ax.axhline(i - 0.5, color="white", lw=2.5)
    for j in range(1, vals.shape[1]):
        ax.axvline(j - 0.5, color="white", lw=1)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("bias, (ML − SPICE) / SPICE (%)", fontsize=9.5)
    cb.ax.tick_params(labelsize=9)
    ax.set_title("Gate leakage bias per case and corner temperature", fontsize=11, pad=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# C. CLA
# ----------------------------------------------------------------------------
def part_c(text, cp):
    tb = temp_bin(cp.temp)
    rows = []
    for cat in tb.cat.categories:
        e = cp["err_ml_%"][tb == cat]
        rows.append({"temperature": str(cat), "points": len(e), "bias (%)": e.mean(),
                     "median error (%)": np.median(np.abs(e)), "P95 error (%)": np.percentile(np.abs(e), 95)})
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "cla_error_vs_temperature.csv", index=False)
    per_corner = cp.groupby("corner")["err_ml_%"].mean()
    within = cp.groupby("corner")["err_ml_%"].std().mean()
    per_vec = cp.groupby("vector")["err_ml_%"].mean()
    text += ["## C. 3-bit CLA (25,600 points)", "", "### Bias and error vs temperature", "", md_table(C),
             f"Spread of the error: the mean error of a corner varies by {per_corner.std():.2f} % (standard deviation) "
             f"across the 200 corners, but the error varies by only {within:.2f} % across the 128 vectors within a "
             f"corner, and the mean error of a vector varies by {per_vec.std():.2f} % across vectors. The CLA error is "
             f"set by the PVT corner, not by the input vector.", ""]
    corner_t = cp.groupby("corner")["temp"].first()
    plot_cla_corner_error(corner_t, per_corner, C, OUT / "cla_error_vs_temperature.png")


def plot_cla_corner_error(corner_t, per_corner, C, path):
    """Mean signed CLA error of each corner vs its temperature, with the mean of each
    temperature band (all vectors) drawn as a segment."""
    fig, ax = plt.subplots(figsize=(6.0, 3.4), constrained_layout=True)
    ax.scatter(corner_t, per_corner, s=16, color="#2a6fdb", alpha=0.8, label="one corner (mean over 128 vectors)")
    ax.axhline(0, color="black", lw=0.8)
    edges = [-40, 0, 40, 80, 125]
    for k, b in enumerate(C["bias (%)"]):
        ax.plot(edges[k:k + 2], [b, b], color="#d9480f", lw=2.5, label="mean of the band" if k == 0 else None)
    ax.set_xlabel("corner temperature (°C)", fontsize=PRINT_FONT)
    ax.set_ylabel("mean CLA error (%)", fontsize=PRINT_FONT)
    ax.tick_params(labelsize=PRINT_FONT)
    ax.set_title("CLA: ML error of each corner vs its temperature", fontsize=11)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=PRINT_FONT, frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# D. Model-to-model variance
# ----------------------------------------------------------------------------
class Ensemble:
    """Average of several fits in log space; signs from the first member (all share the sign models)."""

    def __init__(self, members):
        self.members = members

    def predict(self, df):
        L = np.mean([m.predict_log10(df) for m in self.members], axis=0)
        return pd.DataFrame(self.members[0].predict_signs(df) * 10.0 ** L, columns=mm.TERMINALS, index=df.index)


def gate_errors(models, gsp):
    out = []
    for gate, pattern in gates.GATE_CASES:
        d = gsp[(gsp.gate == gate) & (gsp.inputs.str.zfill(len(pattern)) == pattern)]
        ml = gates.ml_leakage(gate, pattern, d, models)["i_leak_ml"].to_numpy()
        out.append(pd.DataFrame({"gate": gate, "inputs": pattern, "corner": d.corner.to_numpy(),
                                 "i_leak_ml": ml, "err_%": (ml / d.i_leak_spice.to_numpy() - 1) * 100}))
    return pd.concat(out, ignore_index=True)


def cla_errors(gerr, csp):
    tab = gerr.pivot_table(index="corner", columns=["gate", "inputs"], values="i_leak_ml")
    vecs = cla.vectors()
    errs = []
    for k, d in csp.groupby("vector"):
        counts = Counter((g, p) for _, g, p in cla.gate_patterns(vecs[k]))
        est = sum(n * tab[(g, p)].reindex(d.corner).to_numpy() for (g, p), n in counts.items())
        errs.append((est / d.i_leak_spice.to_numpy() - 1) * 100)
    return np.concatenate(errs)


def part_d(text, models, gsp, csp):
    test_rows = []
    if all(hasattr(m, "members") for m in models.values()):
        n = min(len(m.members) for m in models.values())
        first = next(iter(config.CASES))
        sets = {f"single fit, seed {models[first].members[k].seed}": {c: models[c].members[k] for c in config.CASES}
                for k in range(n)}
        sets["ensemble of the fits (final model)"] = models
        for case in config.CASES:
            df = pd.read_csv(config.dataset_path(case))
            te = df.iloc[np.load(config.RESULTS_DIR / "ml" / "splits" / f"{case}_split.npz")["test"]]
            for name, ms in sets.items():
                a = mm.evaluate(ms[case], te)[0].set_index("terminal").loc["all"]
                test_rows.append({"case": label(case), "models": name, "test median error (%)": a["median_err_%"],
                                  "test P95 error (%)": a["p95_err_%"], "test mean error (%)": a["mean_err_%"]})
    else:
        sets = {"chosen fit (seed 0)": models}
        seed_models = {c: [models[c]] for c in config.CASES}
        for off in SEED_OFFSETS:
            sm = {}
            for case in config.CASES:
                base = models[case]
                df = pd.read_csv(config.dataset_path(case))
                split = np.load(config.RESULTS_DIR / "ml" / "splits" / f"{case}_split.npz")
                t0 = time.time()
                m = mm.TransistorModel(case, base.model_type, base.params, seed=config.SEED + off).fit(
                    df.iloc[split["train"]], base.sign_models)
                (config.MODEL_DIR / "seed_check").mkdir(parents=True, exist_ok=True)
                joblib.dump(m, config.MODEL_DIR / "seed_check" / f"{case}_seed{off}.joblib")
                a = mm.evaluate(m, df.iloc[split["test"]])[0].set_index("terminal").loc["all"]
                test_rows.append({"case": label(case), "models": f"seed {off}",
                                  "test median error (%)": a["median_err_%"], "test P95 error (%)": a["p95_err_%"],
                                  "test mean error (%)": a["mean_err_%"]})
                print(f"  refit {label(case):9s} seed {off}: test median {a['median_err_%']:.3f} %  "
                      f"({time.time() - t0:.0f} s)", flush=True)
                sm[case] = m
                seed_models[case].append(m)
            sets[f"seed {off}"] = sm
        sets["ensemble of 3 (average)"] = {c: Ensemble(seed_models[c]) for c in config.CASES}

    summary, per_case = [], {}
    for name, ms in sets.items():
        ge = gate_errors(ms, gsp)
        ce = cla_errors(ge, csp)
        per_case[name] = ge.groupby(["gate", "inputs"], sort=False)["err_%"].mean()
        gs, cs = stats(ge["err_%"]), stats(ce)
        summary.append({"models": name, "gate median (%)": gs["median error (%)"], "gate P95 (%)": gs["P95 error (%)"],
                        "gate mean (%)": gs["mean error (%)"], "gate bias (%)": float(np.mean(ge["err_%"])),
                        "CLA median (%)": cs["median error (%)"], "CLA P95 (%)": cs["P95 error (%)"],
                        "CLA mean (%)": cs["mean error (%)"], "CLA bias (%)": float(np.mean(ce))})
    S = pd.DataFrame(summary)
    V = pd.DataFrame(per_case)
    singles = [c for c in V.columns if not c.startswith("ensemble")]
    V["spread between single fits (max - min)"] = V[singles].max(axis=1) - V[singles].min(axis=1)
    V = V.reset_index()
    V.insert(0, "case", V.gate + " " + V.inputs)
    V = V.drop(columns=["gate", "inputs"])
    S.to_csv(OUT / "seed_variance_summary.csv", index=False)
    V.to_csv(OUT / "seed_variance_per_case.csv", index=False)
    text += ["## D. Model-to-model variance (same data and settings, different random initial weights)", "",
             "### Transistor-level test error", "", md_table(pd.DataFrame(test_rows)),
             "### Gate and CLA error with each fit alone and with their average", "", md_table(S),
             "### Gate bias (mean signed error, %) per case", "", md_table(V, "{:+.2f}")]

    plot_seed_variance(V, OUT / "seed_variance.png")


def plot_seed_variance(V, path):
    """Gate bias of each gate case with each single fit and with their average (final model)."""
    fig, ax = plt.subplots(figsize=(6.7, 3.7), constrained_layout=True)
    names = [c for c in V.columns if c not in ("case", "spread between single fits (max - min)")]
    w = 0.84 / len(names)
    x = np.arange(len(V))
    colors = ["#6f9ce6", "#9dbdf0", "#c9dbf7"]
    for k, nm in enumerate(names):
        final = nm.startswith("ensemble")
        ax.bar(x + (k - (len(names) - 1) / 2) * w, V[nm], w,
               label="average of the three (final model)" if final else nm.replace("single fit, ", ""),
               color="#d9480f" if final else colors[k % 3], edgecolor="none")
    ax.axhline(0, color="black", lw=0.8)
    gates_ = [c.split()[0] for c in V.case]
    for i in range(1, len(gates_)):
        if gates_[i] != gates_[i - 1]:
            ax.axvline(i - 0.5, color="#999999", lw=0.8, ls=":")
    ax.set_xticks(x, V.case, rotation=55, ha="right", rotation_mode="anchor", fontsize=8.5)
    ax.set_xlim(-0.6, len(V) - 0.4)
    ax.tick_params(axis="y", labelsize=PRINT_FONT)
    ax.set_ylabel("gate bias (%)", fontsize=PRINT_FONT)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8.5, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.2, columnspacing=1.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Gate bias with each single fit of the transistor models and with their average",
                 fontsize=11, pad=22)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-seeds", action="store_true", help="skip part D (the refits)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gp, corners, gsp, cp, csp = load_inputs()
    t0 = time.time()
    text = [f"# Error analysis ({config.TECH})", "",
            "Errors are |ML - SPICE| / SPICE (transistor currents: / max(|SPICE|, 1 fA)). "
            "Bias = mean signed error.", ""]
    part_a(text)
    part_b(text, gp, corners)
    part_c(text, cp)
    if not args.no_seeds:
        print("Part D: model-to-model variance ...", flush=True)
        part_d(text, load_models(), gsp, csp)
    (OUT / "error_analysis.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    print(f"Wrote {OUT.relative_to(config.ROOT)}/ in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
