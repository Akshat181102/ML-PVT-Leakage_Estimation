"""
Part A, step 2: train and compare the four model types for every case.

For each case:
  1. Split the 10,000 samples 70 / 15 / 15 (train / validation / test) with a
     fixed seed. The split is saved, so every model sees exactly the same rows.
  2. Fit the sign classifiers on the training rows (shared by all model types).
  3. For each model type, fit every hyperparameter setting in its grid on the
     training rows and keep the one with the lowest validation mean absolute
     error of log10|I| (a robust form of the mean relative error).
  4. Compare the tuned models on the VALIDATION set: R2, median and
     95th-percentile error (deliverable 4).
  5. Choose the model type for the case = the lowest validation mean absolute
     error of log10|I| (the same score as the tuning; it covers the whole error
     distribution, not just the median).
  6. Final model: if the chosen type depends on its random initial weights (the
     MLP), fit it twice more with other seeds and average the three (an ensemble);
     otherwise the tuned model itself. Score the final model ONCE on the test set
     (deliverable 5). The test set plays no part in any choice.
  7. Learning curve: refit the chosen model type (single fits) on 500 ... 7,000
     training rows to show how much data the accuracy needs.

Errors are |I_ML - I_SPICE| / max(|I_SPICE|, 1 fA); see ml_model.py for why.

Usage:
    python3 03_train_models.py                    # full run (~3 h)
    python3 03_train_models.py --quick            # one setting per model type, outputs in *_quick/
    python3 03_train_models.py --no-learning-curve
    python3 03_train_models.py --reselect         # no training: re-score the saved models, choose again,
                                                  # rewrite the tables/plots (learning curve only where the
                                                  # choice changed)

Outputs:
    trained_models/<case>/<model_type>.joblib       every tuned model (single fit)
    trained_models/<case>/mlp_ensemble.joblib       the final model of the case (deliverable 5)
    trained_models/selection.json                   the final model file per case, with versions
    results/ml/model_comparison.md                  the tables for the report (deliverables 4 and 5)
    results/ml/model_comparison_detail.csv          every metric, per terminal, validation + test
    results/ml/tuning_log.csv                       every hyperparameter setting tried
    results/ml/model_comparison.png                 validation median / P95 error per model and case
    results/ml/parity_<case>.png                    ML vs SPICE, final model, test set (deliverable 5)
    results/ml/test_predictions_<case>.csv          inputs, SPICE and ML currents of the test rows
    results/ml/learning_curve.csv / .png            error vs number of training samples
    results/ml/splits/<case>_split.npz              train / validation / test row indices
"""
import argparse
import json
import platform
import sys
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import sklearn

import config
import ml_model as mm

SPLIT = (0.70, 0.15, 0.15)
LC_SIZES = [500, 1000, 2000, 4000]          # plus the full 7,000 (the chosen model itself)
COLORS = {"poly_ridge": "#8c6bb1", "hist_gb": "#e08214", "mlp": "#2a6fdb", "kernel_ridge": "#1b9e77"}
CASE_COLORS = {"nmos_on": "#2a6fdb", "nmos_off": "#7fa7e8", "pmos_on": "#d9480f", "pmos_off": "#f0a07a"}


def label(case):
    return case.replace("_", " ").upper()


def make_split(n):
    rng = np.random.default_rng(config.SPLIT_SEED)
    idx = rng.permutation(n)
    n_tr, n_va = int(round(SPLIT[0] * n)), int(round(SPLIT[1] * n))
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def load(case):
    path = config.dataset_path(case)
    if not path.exists():
        sys.exit(f"{path.relative_to(config.ROOT)} not found - run 01_generate_transistor_data.py first")
    config.check_meta(config.meta_path(case), f"the {case} dataset")
    return pd.read_csv(path)


# ----------------------------------------------------------------------------
# One case
# ----------------------------------------------------------------------------
def train_case(case, model_types, quick, out, model_dir):
    df = load(case)
    tr, va, te = make_split(len(df))
    (out / "splits").mkdir(parents=True, exist_ok=True)
    np.savez(out / "splits" / f"{case}_split.npz", train=tr, validation=va, test=te)
    d_tr, d_va, d_te = df.iloc[tr], df.iloc[va], df.iloc[te]
    dev = config.CASES[case]["dev"]
    print(f"\n=== {label(case)}  train {len(d_tr)}  validation {len(d_va)}  test {len(d_te)}", flush=True)

    t0 = time.time()
    signs = mm.fit_sign_models(d_tr, dev)
    sign_txt = ", ".join(f"{t}: {'classifier' if isinstance(s, dict) else ('always +' if s > 0 else 'always -')}"
                         for t, s in signs.items())
    print(f"  signs ({time.time() - t0:.1f} s): {sign_txt}", flush=True)

    tuning, fitted = [], {}
    (model_dir / case).mkdir(parents=True, exist_ok=True)
    for mt in model_types:
        spec = mm.MODEL_TYPES[mt]
        best = None
        for params in ([spec["quick"]] if quick else spec["grid"]):
            t1 = time.time()
            m = mm.TransistorModel(case, mt, params).fit(d_tr, signs)
            fit_s = time.time() - t1
            sc = mm.validation_score(m, d_va)
            tuning.append({"case": case, "model": mt, "setting": mm.describe(mt, params),
                           "fit_seconds": round(fit_s, 1), **{f"val_{k}": v for k, v in sc.items()}})
            print(f"  {spec['label']:28s} {mm.describe(mt, params):42s} val MAE {sc['mae_decades']:.4f} dec, "
                  f"median {sc['median_err_%']:.2f} %, P95 {sc['p95_err_%']:.1f} %  ({fit_s:5.1f} s)", flush=True)
            if best is None or sc["mae_decades"] < best[0]:
                best = (sc["mae_decades"], m, fit_s)
        _, m, fit_s = best
        m.fit_seconds = fit_s
        joblib.dump(m, model_dir / case / f"{mt}.joblib")
        fitted[mt] = m
    detail, summary, chosen = score_and_choose(case, fitted, d_va, d_te)
    detail += build_final(case, chosen, fitted, d_tr, d_va, d_te, out, model_dir, reuse=False)
    return tuning, detail, summary, chosen, (d_tr, d_va, signs)


def score_and_choose(case, fitted, d_va, d_te):
    """Validation and test metrics of every tuned model; choose the model type of the case
    (lowest validation MAE of log10|I|)."""
    detail, summary = [], []
    for mt, m in fitted.items():
        sc = mm.validation_score(m, d_va)
        res = {}
        for split_name, d in (("validation", d_va), ("test", d_te)):
            metrics, _ = mm.evaluate(m, d)
            for _, r in metrics.iterrows():
                detail.append({"case": case, "model": mt, "split": split_name, **r.to_dict(),
                               "val_mae_decades": sc["mae_decades"], "setting": mm.describe(mt, m.params),
                               "fit_seconds": round(getattr(m, "fit_seconds", float("nan")), 1)})
            res[split_name] = metrics[metrics.terminal == "all"].iloc[0].to_dict()
        res["validation"]["mae_decades"] = sc["mae_decades"]
        summary.append({"case": case, "model": mt, "val": res["validation"], "test": res["test"],
                        "setting": mm.describe(mt, m.params)})
        v = res["validation"]
        print(f"  -> {mm.MODEL_TYPES[mt]['label']}: validation MAE {v['mae_decades']:.4f} dec, "
              f"median {v['median_err_%']:.3f} %, P95 {v['p95_err_%']:.2f} %, R2 {v['r2_log']:.5f}", flush=True)

    chosen = min(summary, key=lambda s: s["val"]["mae_decades"])
    chosen["params_obj"] = fitted[chosen["model"]].params
    print(f"  CHOSEN TYPE: {fitted[chosen['model']].summary()}", flush=True)
    return detail, summary, chosen


def build_final(case, chosen, fitted, d_tr, d_va, d_te, out, model_dir, reuse):
    """The final model of the case: an ensemble of ENSEMBLE_SIZE fits (the tuned one plus fits with
    other seeds) if the chosen type depends on its seed, else the tuned model. Scores it on the
    validation and test sets, saves it, its test predictions and its parity plot."""
    mt = chosen["model"]
    base = fitted[mt]
    final, path = base, model_dir / case / f"{mt}.joblib"
    ens_rows = []
    if mm.MODEL_TYPES[mt].get("seeded"):
        path = model_dir / case / f"{mt}_ensemble.joblib"
        final = None
        if reuse and path.exists():
            e = joblib.load(path)
            if e.model_type == mt and e.params == base.params and len(e.members) == mm.ENSEMBLE_SIZE:
                final = e
                print(f"  final model: saved ensemble of {len(e.members)} reused", flush=True)
        if final is None:
            members = [base]
            for k in range(1, mm.ENSEMBLE_SIZE):
                t1 = time.time()
                m = mm.TransistorModel(case, mt, base.params, seed=config.SEED + k).fit(d_tr, base.sign_models)
                m.fit_seconds = time.time() - t1
                members.append(m)
                print(f"  ensemble member {k + 1}/{mm.ENSEMBLE_SIZE} (seed {config.SEED + k}) fitted in "
                      f"{m.fit_seconds:.0f} s", flush=True)
            final = mm.EnsembleModel(members)
            joblib.dump(final, path)
        for m in final.members + [final]:
            name = "ensemble (average)" if m is final else f"seed {m.seed}"
            row = {"case": case, "fit": name}
            for split_name, d in (("validation", d_va), ("test", d_te)):
                a = mm.evaluate(m, d)[0].set_index("terminal").loc["all"]
                row.update({f"{split_name} median": a["median_err_%"], f"{split_name} P95": a["p95_err_%"],
                            f"{split_name} mean": a["mean_err_%"]})
            ens_rows.append(row)

    detail = []
    res = {}
    for split_name, d in (("validation", d_va), ("test", d_te)):
        metrics, _ = mm.evaluate(final, d)
        for _, r in metrics.iterrows():
            detail.append({"case": case, "model": "final", "split": split_name, **r.to_dict(),
                           "setting": final.summary()})
        res[split_name] = metrics[metrics.terminal == "all"].iloc[0].to_dict()
    chosen.update({"final_summary": final.summary(), "final_file": str(path.relative_to(model_dir)).replace("\\", "/"),
                   "final_val": res["validation"], "final_test": res["test"], "ens_rows": ens_rows,
                   "ensemble": len(getattr(final, "members", [final]))})
    t = res["test"]
    print(f"  FINAL: {final.summary()}\n  test: median {t['median_err_%']:.3f} %, P95 {t['p95_err_%']:.2f} %, "
          f"mean {t['mean_err_%']:.2f} % (mean accuracy {t['mean_acc_%']:.2f} %)", flush=True)
    _, pred = mm.evaluate(final, d_te)
    tp = d_te.copy()
    for c in mm.TERMINALS:
        tp[f"{c}_ml"] = pred[c]
    tp.insert(0, "row", d_te.index)
    tp.to_csv(out / f"test_predictions_{case}.csv", index=False, float_format="%.10e")
    parity_plot(case, final, d_te, out)
    return detail


def reselect_case(case, model_types, out, model_dir):
    """--reselect: load the saved tuned models, score them and choose again (no training)."""
    df = load(case)
    tr, va, te = make_split(len(df))
    saved = out / "splits" / f"{case}_split.npz"
    if saved.exists() and not np.array_equal(np.load(saved)["test"], te):
        sys.exit(f"{saved.relative_to(config.ROOT)} does not match the split - run the full training instead")
    print(f"\n=== {label(case)} (re-scoring the saved models)", flush=True)
    fitted = {}
    for mt in model_types:
        path = model_dir / case / f"{mt}.joblib"
        if not path.exists():
            sys.exit(f"{path.relative_to(config.ROOT)} not found - run the full training first")
        fitted[mt] = joblib.load(path)
    detail, summary, chosen = score_and_choose(case, fitted, df.iloc[va], df.iloc[te])
    detail += build_final(case, chosen, fitted, df.iloc[tr], df.iloc[va], df.iloc[te], out, model_dir, reuse=True)
    return detail, summary, chosen, (df.iloc[tr], df.iloc[va], None)


# ----------------------------------------------------------------------------
# Learning curve
# ----------------------------------------------------------------------------
def learning_curve(case, chosen, data, sizes):
    d_tr, d_va, _ = data
    rows = []
    params = chosen["params_obj"]
    for n in sizes:
        if n >= len(d_tr):
            continue
        sub = d_tr.iloc[:n]                  # the split order is already random
        t1 = time.time()
        signs = mm.fit_sign_models(sub, config.CASES[case]["dev"])
        m = mm.TransistorModel(case, chosen["model"], params).fit(sub, signs)
        sc = mm.validation_score(m, d_va)
        rows.append({"case": case, "model": chosen["model"], "n_train": n, **sc})
        print(f"  learning curve {label(case):9s} n={n:5d}: validation median {sc['median_err_%']:.3f} %, "
              f"P95 {sc['p95_err_%']:.2f} %, mean {sc['mean_err_%']:.2f} %  ({time.time() - t1:.0f} s)", flush=True)
    v = chosen["val"]
    rows.append({"case": case, "model": chosen["model"], "n_train": len(d_tr),
                 "mae_decades": float("nan"), "rmse_decades": v["rmse_decades"],
                 "median_err_%": v["median_err_%"], "p95_err_%": v["p95_err_%"], "mean_err_%": v["mean_err_%"]})
    return rows


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
# Report figures are drawn at the size they are printed (6.3-6.7 in wide), so their text
# prints at 8.5-10 pt without scaling.
PRINT_RC = {"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "figure.titlesize": 10.5,
            "axes.spines.top": False, "axes.spines.right": False}


def parity_plot(case, model, d_te, out):
    _, pred = mm.evaluate(model, d_te)
    note = (f"average of {len(model.members)} {mm.MODEL_TYPES[model.model_type]['label'].split()[0]} fits"
            if hasattr(model, "members") else model.summary())
    draw_parity(f"{config.TECH} {label(case)}: final model ({note}), test set, {len(d_te):,} samples",
                d_te[mm.TERMINALS], pred[mm.TERMINALS], out / f"parity_{case}.png")


def draw_parity(title, true, pred, path):
    """Predicted vs SPICE |I| for the four terminal currents of one case (2 x 2 panels)."""
    err = mm.errors(pred.to_numpy(float), true.to_numpy(float))
    with plt.rc_context(PRINT_RC):
        fig, axes = plt.subplots(2, 2, figsize=(6.7, 4.4), constrained_layout=True)
        for j, (ax, t) in enumerate(zip(axes.ravel(), mm.TERMINALS)):
            y, p = true[t].to_numpy(float), pred[t].to_numpy(float)
            resolved = np.abs(y) >= mm.I_RES
            wrong = (np.sign(y) != np.sign(p)) & resolved
            a, b = np.maximum(np.abs(y), 1e-18), np.maximum(np.abs(p), 1e-18)
            lo, hi = min(a.min(), b.min()) * 0.5, max(a.max(), b.max()) * 2
            if lo < mm.I_RES:
                ax.axvspan(lo, mm.I_RES, color="grey", alpha=0.13, lw=0, label="SPICE < 1 fA")
            ax.plot([lo, hi], [lo, hi], color="black", lw=0.9)
            ax.plot([lo, hi], [lo * 1.1, hi * 1.1], color="grey", lw=0.8, ls=":", label="±10 %")
            ax.plot([lo, hi], [lo / 1.1, hi / 1.1], color="grey", lw=0.8, ls=":")
            ax.scatter(a[~wrong], b[~wrong], s=3, alpha=0.45, color="#2a6fdb", linewidths=0, label="sign correct")
            if wrong.any():
                ax.scatter(a[wrong], b[wrong], s=14, color="#d62728", marker="x", lw=1,
                           label=f"sign wrong ({wrong.sum()})")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_title(f"{t.capitalize()}: median {np.median(err[:, j]):.2f} %, "
                         f"P95 {np.percentile(err[:, j], 95):.1f} %, mean {np.mean(err[:, j]):.2f} %", fontsize=8.3)
            ax.grid(True, which="major", alpha=0.25)
            ax.legend(loc="upper left", fontsize=7.5, handlelength=1.2, borderaxespad=0.3, framealpha=0.85)
        fig.supxlabel("|I|, SPICE (A)", fontsize=9)
        fig.supylabel("|I|, ML (A)", fontsize=9)
        fig.suptitle(title, fontsize=10)
        fig.savefig(path, dpi=200)
        plt.close(fig)


def _pct_axis(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if f"{v:.0e}"[0] in "25" else ""))
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.grid(True, axis="y", which="both", alpha=0.25)


def comparison_plot(all_summary, cases, model_types, out):
    vals = {(c, mt): next(s["val"] for s in all_summary if s["case"] == c and s["model"] == mt)
            for c in cases for mt in model_types}
    draw_comparison(vals, cases, model_types, out / "model_comparison.png")


def draw_comparison(vals, cases, model_types, path):
    """Validation median and P95 error of each model type in each case.
    vals[(case, model_type)] = {"median_err_%": ..., "p95_err_%": ...}"""
    with plt.rc_context(PRINT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.9), constrained_layout=True)
        w = 0.8 / len(model_types)
        x = np.arange(len(cases))
        for ax, key, title in ((axes[0], "median_err_%", "Median error (validation)"),
                               (axes[1], "p95_err_%", "95th-percentile error (validation)")):
            for k, mt in enumerate(model_types):
                ax.bar(x + (k - (len(model_types) - 1) / 2) * w, [vals[(c, mt)][key] for c in cases], w,
                       color=COLORS[mt], label=mm.MODEL_TYPES[mt]["label"])
            ax.set_xticks(x, [label(c) for c in cases])
            ax.set_ylabel("error (%)"); ax.set_title(title)
            _pct_axis(ax)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=len(model_types), frameon=False,
                   handlelength=1.2, columnspacing=1.2)
        fig.suptitle(f"{config.TECH}: model comparison on the validation set (dashed line = 1 %)")
        fig.savefig(path, dpi=200)
        plt.close(fig)


def learning_curve_plot(lc, out):
    draw_learning_curve(lc, out / "learning_curve.png")


def draw_learning_curve(lc, path):
    with plt.rc_context(PRINT_RC):
        fig, axes = plt.subplots(1, 3, figsize=(6.7, 2.7), constrained_layout=True)
        sizes = sorted(lc.n_train.unique())
        for ax, key, title in zip(axes, ("median_err_%", "p95_err_%", "mean_err_%"),
                                  ("Median error", "95th-percentile error", "Mean error")):
            for case, d in lc.groupby("case", sort=False):
                ax.plot(d.n_train, d[key], marker="o", ms=3.5, lw=1.6, color=CASE_COLORS[case], label=label(case))
            ax.set_xscale("log")
            ax.set_xticks(sizes, [f"{n / 1000:g}k" for n in sizes])
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
            ax.set_xlabel("training samples"); ax.set_title(title)
            _pct_axis(ax)
        axes[0].set_ylabel("validation error (%)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, handlelength=1.5)
        fig.suptitle(f"{config.TECH}: learning curve of the chosen model type, single fits (dashed line = 1 %)")
        fig.savefig(path, dpi=200)
        plt.close(fig)


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def write_tables(all_summary, chosen, detail, lc, cases, model_types, args, out):
    head = "| Model | " + " | ".join(label(c) for c in cases) + " |"
    lines = [f"# Transistor models ({config.TECH})", "",
             "Errors are |I_ML - I_SPICE| / max(|I_SPICE|, 1 fA), pooled over the four terminal currents "
             "(id, ig, is, ib). R2 is on log10 max(|I|, 1 fA).", "",
             "## Model comparison, validation set (deliverable 4)", "",
             "Each cell: R2 / median error / 95th-percentile error. Bold = chosen: the lowest validation mean absolute "
             "error of log10|I| (table below), the same score used to tune each model.", "",
             head, "|---|" + "---|" * len(cases)]
    for mt in model_types:
        cells = []
        for c in cases:
            s = next(s for s in all_summary if s["case"] == c and s["model"] == mt)["val"]
            cell = f"{s['r2_log']:.4f} / {s['median_err_%']:.2f} % / {s['p95_err_%']:.1f} %"
            cells.append(f"**{cell}**" if chosen[c]["model"] == mt else cell)
        lines.append(f"| {mm.MODEL_TYPES[mt]['label']} | " + " | ".join(cells) + " |")

    lines += ["", "Selection score, validation mean absolute error of log10|I| (decades; 0.0043 decades ~ 1 %):", "",
              head, "|---|" + "---|" * len(cases)]
    for mt in model_types:
        cells = []
        for c in cases:
            v = next(s for s in all_summary if s["case"] == c and s["model"] == mt)["val"]["mae_decades"]
            cells.append(f"**{v:.4f}**" if chosen[c]["model"] == mt else f"{v:.4f}")
        lines.append(f"| {mm.MODEL_TYPES[mt]['label']} | " + " | ".join(cells) + " |")

    lines += ["", "## Final models, test set (deliverable 5)", "",
              "| Case | Final model | R2 | Median error | P95 error | Mean error | Mean accuracy | Sign accuracy |",
              "|---|---|---|---|---|---|---|---|"]
    for c in cases:
        s, t = chosen[c], chosen[c]["final_test"]
        lines.append(f"| {label(c)} | {s['final_summary']} | {t['r2_log']:.4f} | "
                     f"{t['median_err_%']:.2f} % | {t['p95_err_%']:.2f} % | {t['mean_err_%']:.2f} % | "
                     f"{t['mean_acc_%']:.2f} % | {t['sign_acc_%']:.2f} % |")

    ens = [r for c in cases for r in chosen[c]["ens_rows"]]
    if ens:
        lines += ["", "## Ensemble members and their average", "",
                  "Same setting and training data, different random initial weights. "
                  "Each cell: median / P95 / mean error.", "",
                  "| Case | Fit | Validation | Test |", "|---|---|---|---|"]
        for r in ens:
            lines.append(f"| {label(r['case'])} | {r['fit']} | "
                         f"{r['validation median']:.2f} / {r['validation P95']:.2f} / {r['validation mean']:.2f} % | "
                         f"{r['test median']:.2f} / {r['test P95']:.2f} / {r['test mean']:.2f} % |")

    lines += ["", "## Final models per terminal, test set", "",
              "| Case | Terminal | Median error | P95 error | Mean error | Sign accuracy | SPICE values below 1 fA |",
              "|---|---|---|---|---|---|---|"]
    D = pd.DataFrame(detail)
    for c in cases:
        d = D[(D.case == c) & (D.model == "final") & (D.split == "test") & (D.terminal != "all")]
        for _, r in d.iterrows():
            lines.append(f"| {label(c)} | {r['terminal']} | {r['median_err_%']:.2f} % | {r['p95_err_%']:.2f} % | "
                         f"{r['mean_err_%']:.2f} % | {r['sign_acc_%']:.2f} % | {r['below_1fA_%']:.1f} % |")

    if lc is not None and len(lc):
        sizes = sorted(lc.n_train.unique())
        lines += ["", "## Learning curve, validation set (chosen model type, single fits)", "",
                  "Each cell: median / P95 / mean error.", "",
                  "| Case | " + " | ".join(f"{n:,} samples" for n in sizes) + " |", "|---|" + "---|" * len(sizes)]
        for c in cases:
            d = lc[lc.case == c].set_index("n_train")
            cells = [f"{d.loc[n, 'median_err_%']:.2f} / {d.loc[n, 'p95_err_%']:.1f} / {d.loc[n, 'mean_err_%']:.2f} %"
                     if n in d.index else "-" for n in sizes]
            lines.append(f"| {label(c)} | " + " | ".join(cells) + " |")

    lines += ["", "## Tuned settings", "", "| Case | Model | Setting |", "|---|---|---|"]
    for s in all_summary:
        lines.append(f"| {label(s['case'])} | {mm.MODEL_TYPES[s['model']]['label']} | {s['setting']} |")
    lines += ["", f"Run options: quick={args.quick}, reselect={args.reselect}; scikit-learn {sklearn.__version__}, "
                  f"Python {platform.python_version()}; settings fingerprint {config.spec_fingerprint()}.", ""]
    (out / "model_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", choices=list(config.CASES), default=list(config.CASES))
    ap.add_argument("--models", nargs="+", choices=list(mm.MODEL_TYPES), default=list(mm.MODEL_TYPES))
    ap.add_argument("--quick", action="store_true", help="one setting per model type; outputs go to *_quick/")
    ap.add_argument("--no-learning-curve", action="store_true")
    ap.add_argument("--reselect", action="store_true",
                    help="no training: re-score the saved models, choose again and rewrite the outputs")
    args = ap.parse_args()

    out = config.RESULTS_DIR / ("ml_quick" if args.quick else "ml")
    model_dir = config.MODEL_DIR / "quick" if args.quick else config.MODEL_DIR
    out.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"scikit-learn {sklearn.__version__} | cases: {', '.join(args.cases)} | models: {', '.join(args.models)}"
          f"{' | QUICK (results in ' + str(out.relative_to(config.ROOT)) + ')' if args.quick else ''}"
          f"{' | RESELECT (no training)' if args.reselect else ''}", flush=True)

    t0 = time.time()
    all_tuning, all_detail, all_summary, chosen, data = [], [], [], {}, {}
    for case in args.cases:
        if args.reselect:
            detail, summary, best, d = reselect_case(case, args.models, out, model_dir)
        else:
            tuning, detail, summary, best, d = train_case(case, args.models, args.quick, out, model_dir)
            all_tuning += tuning
        all_detail += detail; all_summary += summary
        chosen[case], data[case] = best, d

    lc = None
    lc_path = out / "learning_curve.csv"
    if not args.no_learning_curve:
        sizes = [500, 1000] if args.quick else LC_SIZES
        old_lc = pd.read_csv(lc_path) if (args.reselect and lc_path.exists()) else None
        rows = []
        print("\n=== Learning curve (chosen model type, refit on fewer training samples) ===", flush=True)
        for case in args.cases:
            prev = old_lc[old_lc.case == case] if old_lc is not None else None
            if prev is not None and len(prev) and (prev.model == chosen[case]["model"]).all():
                print(f"  learning curve {label(case):9s}: unchanged model type, kept", flush=True)
                rows += prev.to_dict("records")
            else:
                rows += learning_curve(case, chosen[case], data[case], sizes)
        lc = pd.DataFrame(rows)
        lc.to_csv(lc_path, index=False, float_format="%.6g")
        learning_curve_plot(lc, out)
    elif lc_path.exists():
        lc = pd.read_csv(lc_path)

    if all_tuning:
        pd.DataFrame(all_tuning).to_csv(out / "tuning_log.csv", index=False, float_format="%.6g")
    pd.DataFrame(all_detail).to_csv(out / "model_comparison_detail.csv", index=False, float_format="%.6g")
    comparison_plot(all_summary, args.cases, args.models, out)
    table = write_tables(all_summary, chosen, all_detail, lc, args.cases, args.models, args, out)

    sel_path = model_dir / "selection.json"
    selection = json.loads(sel_path.read_text(encoding="utf-8")) if sel_path.exists() else {}
    for c, s in chosen.items():
        t = s["final_test"]
        selection[c] = {"model": s["model"], "file": s["final_file"], "setting": s["setting"],
                        "ensemble_size": s["ensemble"], "final_model": s["final_summary"],
                        "test_median_err_%": round(t["median_err_%"], 4), "test_p95_err_%": round(t["p95_err_%"], 3),
                        "test_mean_err_%": round(t["mean_err_%"], 3)}
    selection["_info"] = {"spec_fingerprint": config.spec_fingerprint(), "scikit-learn": sklearn.__version__,
                          "python": platform.python_version(), "error_scale_A": mm.I_RES,
                          "target_floor_A": mm.LOG_FLOOR,
                          "note": "load the .joblib files with the same scikit-learn version"}
    sel_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")

    print("\n" + table)
    print(f"Total time {time.time() - t0:.0f} s. Results in {out.relative_to(config.ROOT)}/, "
          f"models in {model_dir.relative_to(config.ROOT)}/")


if __name__ == "__main__":
    main()
