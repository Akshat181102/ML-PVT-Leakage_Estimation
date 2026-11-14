"""
Part A, step 1: generate the single-transistor datasets.

For each case (NMOS ON, NMOS OFF, PMOS ON, PMOS OFF) draw 10,000 random samples of
VDD, Vd, Vs, W, temperature and the five process parameters, simulate the single
FET in ngspice (DC operating point, one voltage source per terminal, junction
areas AS = AD = W * Ldiff) and store the inputs with the four terminal currents.

A short self-test runs first and the generation stops if it fails.

Usage:
    python3 01_generate_transistor_data.py                  # self-test + all 4 cases, 10,000 each
    python3 01_generate_transistor_data.py --selftest       # self-test only (~20 simulations)
    python3 01_generate_transistor_data.py --cases nmos_off # one case

Outputs:
    data/32nm_MGK_<case>.csv                 columns as in config.CSV_COLUMNS (deliverable 1)
    data/32nm_MGK_<case>.meta.json           settings fingerprint, versions, run time
    netlists/transistor/<case>_example.cir   the netlist of sample 0 of each case (deliverable 2)
    results/data_check/selftest.txt          the self-test output
"""
import argparse
import contextlib
import datetime
import io
import sys
import time

import numpy as np
import pandas as pd

import config
import fet
import sampling
import spice_utils as su


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------
def sample_inputs(case, n, nominal):
    """Draw n samples for one case. VDD is drawn first because Vd and Vs depend on it."""
    spec = config.CASES[case]
    rng = np.random.default_rng(config.CASE_SEEDS[case])
    vdd = sampling.draw_vdd(rng, n)
    vd = rng.uniform(0.0, vdd)
    vs = rng.uniform(0.0, vdd)
    w = sampling.draw_width(rng, n)
    temp = sampling.draw_temp(rng, n)
    proc = sampling.draw_process(rng, nominal[spec["dev"]], n)
    vg = vdd.copy() if spec["vg"] == "vdd" else np.zeros(n)
    vb = vdd.copy() if spec["vb"] == "vdd" else np.zeros(n)
    df = pd.DataFrame({"vdd": vdd, "temp": temp, "w": w, "vd": vd, "vs": vs,
                       "vg": vg, "vb": vb, **proc})
    return df[[c for c in config.CSV_COLUMNS if c in df.columns]]


# ----------------------------------------------------------------------------
# One full case
# ----------------------------------------------------------------------------
def run_case(case, n, workers, cards, nominal, ngspice_ver):
    df = sample_inputs(case, n, nominal)
    rows = df.to_dict("records")

    config.NETLIST_DIR.joinpath("transistor").mkdir(parents=True, exist_ok=True)
    example = config.NETLIST_DIR / "transistor" / f"{case}_example.cir"
    example.write_text(fet.fet_netlist(case, rows[0], cards, "(sample 0)"), encoding="utf-8")

    print(f"\n[{case}] simulating {n} samples with {workers} parallel ngspice processes", flush=True)
    t0 = time.time()
    results, errors = su.run_parallel(fet.simulate, [(case, r, cards) for r in rows], workers)
    if errors:
        print(f"  WARNING: {len(errors)} sample(s) failed and are left out. First error:\n"
              f"{next(iter(errors.values()))}")
        if len(errors) == n:
            sys.exit("Every simulation failed - fix the error above before continuing.")

    ok = [i for i in range(n) if results[i] is not None]
    cur = pd.DataFrame([results[i] for i in ok], index=ok)
    out = pd.concat([df.loc[ok], cur], axis=1)[config.CSV_COLUMNS]
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = config.dataset_path(case)
    out.to_csv(path, index=False, float_format="%.12e")
    seconds = time.time() - t0
    config.write_meta(config.meta_path(case), case=case, rows=len(out), failed=len(errors),
                      ngspice=ngspice_ver, numpy=np.__version__, pandas=pd.__version__,
                      diffusion_length_m=config.DIFFUSION_LENGTH, seed=config.CASE_SEEDS[case],
                      generated_at=datetime.datetime.now().isoformat(timespec="seconds"),
                      generation_seconds=round(seconds, 1))
    print(f"  wrote {path.relative_to(config.ROOT)}  ({len(out)} rows, {seconds:.0f} s)", flush=True)
    return out


# ----------------------------------------------------------------------------
# Self-test: a handful of runs that prove the pipeline and the parameter edits work
# ----------------------------------------------------------------------------
def selftest(cards, nominal):
    print(f"ngspice: {su.ngspice_version()}")
    sim = lambda case, **kw: fet.simulate(case, fet.nominal_row(case, nominal, **kw), cards)

    print("\nNominal corner (VDD=1 V, 27 C, W=2L, |Vds|=VDD):")
    print(f"  {'case':9s} {'id':>12s} {'ig':>12s} {'is':>12s} {'ib':>12s} {'sum/max':>10s}")
    base = {}
    for case in config.CASES:
        c = sim(case)
        base[case] = c
        mx = max(abs(v) for v in c.values())
        print(f"  {case:9s} " + " ".join(f"{c[k]:12.4e}" for k in ("id", "ig", "is", "ib"))
              + f" {abs(sum(c.values())) / mx:10.2e}")

    # Junction leakage: OFF transistor with half of VDD on the drain, so GIDL (which needs
    # V_dg > egidl = 0.8 V) is off and |Ib| is the reverse-biased drain junction.
    print(f"\nJunction leakage, |Ib| of an OFF transistor with |Vdb| = VDD/2 "
          f"(AS = AD = W*Ldiff, Ldiff = {config.DIFFUSION_LENGTH * 1e9:.0f} nm):")
    print(f"  {'case':9s} {'T (C)':>6s} {'|ib| (A)':>12s}")
    junc = {}
    for case in ("nmos_off", "pmos_off"):
        half = ({"vd": 0.5 * config.VDD_NOM} if case == "nmos_off"
                else {"vd": 0.5 * config.VDD_NOM, "vs": config.VDD_NOM})
        for t in (-40.0, 27.0, 125.0):
            junc[(case, t)] = abs(sim(case, temp=t, **half)["ib"])
            print(f"  {case:9s} {t:6.0f} {junc[(case, t)]:12.4e}")

    vn, vp = nominal["nmos"], nominal["pmos"]
    checks = []

    def check(name, ok, detail):
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print("\nChecks:")
    kir = max(abs(sum(c.values())) / max(abs(v) for v in c.values()) for c in base.values())
    check("Kirchhoff (id+ig+is+ib = 0)", kir < 1e-3, f"worst relative residual {kir:.1e}")
    check("sign convention (NMOS ON id > 0 for Vd > Vs)", base["nmos_on"]["id"] > 0,
          f"id = {base['nmos_on']['id']:.3e} A")
    for dev in ("nmos", "pmos"):
        ratio = abs(base[f"{dev}_on"]["id"]) / abs(base[f"{dev}_off"]["id"])
        check(f"{dev.upper()} ON/OFF current ratio > 100", ratio > 100, f"{ratio:.3g}")

    lo, hi = sim("nmos_off", vth0=vn["vth0"] * 0.9)["id"], sim("nmos_off", vth0=vn["vth0"] * 1.1)["id"]
    check("vth0 edit applied (NMOS OFF: lower vth0 -> more leakage)", abs(lo) > abs(hi),
          f"|id| {abs(lo):.3e} (0.9x) vs {abs(hi):.3e} (1.1x)")
    lo, hi = sim("pmos_off", vth0=vp["vth0"] * 0.9)["id"], sim("pmos_off", vth0=vp["vth0"] * 1.1)["id"]
    check("vth0 edit applied (PMOS OFF: smaller |vth0| -> more leakage)", abs(lo) > abs(hi),
          f"|id| {abs(lo):.3e} (0.9x) vs {abs(hi):.3e} (1.1x)")
    cold, hot = sim("nmos_off", temp=-40.0)["id"], sim("nmos_off", temp=125.0)["id"]
    check("temperature applied (NMOS OFF: 125 C leaks more than -40 C)", abs(hot) > abs(cold),
          f"|id| {abs(hot):.3e} vs {abs(cold):.3e}")
    thin = sim("nmos_on", bias="gate", toxe=vn["toxe"] * 0.9)["ig"]
    thick = sim("nmos_on", bias="gate", toxe=vn["toxe"] * 1.1)["ig"]
    check("toxe edit applied (NMOS ON: thinner oxide -> more gate current)",
          abs(thin) > abs(thick), f"|ig| {abs(thin):.3e} vs {abs(thick):.3e}")
    for case in ("nmos_off", "pmos_off"):
        check(f"junction leakage rises with temperature ({case.upper()})",
              junc[(case, 125.0)] > junc[(case, 27.0)],
              f"{junc[(case, 125.0)]:.3e} A at 125 C vs {junc[(case, 27.0)]:.3e} A at 27 C")

    print("\nSelf-test " + ("PASSED." if all(checks)
                          else "FAILED - send the output above before generating data."))
    return all(checks)


def run_selftest(cards, nominal):
    """Run the self-test, show it and save it to results/data_check/selftest.txt."""
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)
            return len(s)

        def flush(self):
            sys.__stdout__.flush()

    with contextlib.redirect_stdout(Tee()):
        ok = selftest(cards, nominal)
    out = config.RESULTS_DIR / "data_check"
    out.mkdir(parents=True, exist_ok=True)
    (out / "selftest.txt").write_text(buf.getvalue(), encoding="utf-8")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", choices=list(config.CASES), default=list(config.CASES))
    ap.add_argument("--n", type=int, default=config.N_SAMPLES)
    ap.add_argument("--workers", type=int, default=config.WORKERS)
    ap.add_argument("--selftest", action="store_true", help="run only the self-test")
    args = ap.parse_args()

    ver = su.ngspice_version()
    if ver is None:
        sys.exit(f"ngspice not found ('{config.NGSPICE}'). Install it (Ubuntu: sudo apt install ngspice) "
                 f"or set NGSPICE=/path/to/ngspice.")

    cards = su.read_model_cards()
    nominal = su.nominal_process(cards)
    print(f"Model file: {config.MODEL_FILE.name}   VDD_nom = {config.VDD_NOM} V   L = {config.L * 1e9:.0f} nm   "
          f"Ldiff = {config.DIFFUSION_LENGTH * 1e9:.0f} nm   settings fingerprint {config.spec_fingerprint()}")
    for dev, p in nominal.items():
        print(f"  nominal {dev}: " + ", ".join(f"{k}={v:.6g}" for k, v in p.items()))

    print("\n=== Self-test ===")
    ok = run_selftest(cards, nominal)
    if args.selftest:
        sys.exit(0 if ok else 1)
    if not ok:
        sys.exit("Stopped: the self-test failed (see above). No data was generated.")
    if args.n != config.N_SAMPLES:
        print(f"\nNOTE: --n {args.n} differs from the project's {config.N_SAMPLES} samples per case.")

    print("\n=== Generating the datasets ===")
    t0 = time.time()
    for case in args.cases:
        run_case(case, args.n, args.workers, cards, nominal, ver)
    print(f"\nDone in {time.time() - t0:.0f} s. Next: python3 02_data_check.py")


if __name__ == "__main__":
    main()
