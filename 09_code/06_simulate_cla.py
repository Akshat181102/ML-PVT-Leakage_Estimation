"""
Part C, step 1: SPICE simulation of the 3-bit CLA at the 200 shared corners.

Every input vector (128, with carry-in) is simulated at the same 200 corners
used for the gates (data/gate_corners.csv, written by 04_simulate_gates.py).
Recorded per vector and corner: the four outputs, the current delivered by VDD
and by the inputs held at VDD, and the leakage
    I_leak = I(VDD) + I(inputs at VDD)   (= static power / VDD, as for the gates).

Usage:
    python3 06_simulate_cla.py            # 128 x 200 = 25,600 simulations
    python3 06_simulate_cla.py --quick    # 128 x 5, written to data/quick/ (never mixed with the full run)

Outputs:
    data/cla_spice.csv (+ .meta.json)          one row per vector and corner
    netlists/cla/cla_vector<k>_example.cir     netlist of one vector at corner 0 (deliverable 2)
"""
import argparse
import datetime
import sys
import time

import pandas as pd

import cla
import config
import gates
import spice_utils as su


QUICK_CORNERS = 5


def simulate(k, vec, corner, cards):
    vecs = su.run_netlist(cla.cla_netlist(vec, corner, cards))
    return cla.spice_result(vecs, vec)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help=f"only {QUICK_CORNERS} corners, into data/quick/")
    ap.add_argument("--workers", type=int, default=config.WORKERS)
    args = ap.parse_args()

    ver = su.ngspice_version()
    if ver is None:
        sys.exit(f"ngspice not found ('{config.NGSPICE}')")
    cla.check_logic()
    cards = su.read_model_cards()
    data_dir = config.QUICK_DIR if args.quick else config.DATA_DIR
    cpath = data_dir / "gate_corners.csv"
    if not cpath.exists():
        sys.exit(f"{cpath.relative_to(config.ROOT)} not found - run 04_simulate_gates.py{' --quick' if args.quick else ''} "
                 f"first (the CLA uses the same corners as the gates)")
    config.check_meta(data_dir / "gates_spice.meta.json", str(cpath.relative_to(config.ROOT)))
    corners = pd.read_csv(cpath)
    if args.quick:
        corners = corners.head(QUICK_CORNERS)
    elif len(corners) != gates.N_CORNERS:
        sys.exit(f"data/gate_corners.csv has {len(corners)} corners, expected {gates.N_CORNERS} - "
                 f"run python3 04_simulate_gates.py (without --quick) first")
    vecs = cla.vectors()
    print(f"ngspice: {ver}\n3-bit CLA: {len(cla.CIRCUIT)} gates, {cla.n_transistors()} transistors; "
          f"{len(vecs)} vectors x {len(corners)} corners, {args.workers} parallel ngspice processes"
          f"{'  [QUICK -> data/quick/]' if args.quick else ''}")

    if not args.quick:
        ex_dir = config.NETLIST_DIR / "cla"
        ex_dir.mkdir(parents=True, exist_ok=True)
        k_ex = 0b1011101                                   # a = 5, b = 3, c0 = 1  ->  s = 1, c3 = 1
        (ex_dir / f"cla_vector{k_ex}_example.cir").write_text(
            cla.cla_netlist(vecs[k_ex], corners.iloc[0], cards, f"(vector {k_ex}, corner 0)"), encoding="utf-8")

    jobs = [(k, vecs[k], c, cards) for k in range(len(vecs)) for _, c in corners.iterrows()]
    t0 = time.time()
    res, errors = su.run_parallel(simulate, jobs, args.workers)
    if errors:
        print(f"WARNING: {len(errors)} simulation(s) failed. First error:\n{next(iter(errors.values()))}")
        if len(errors) == len(jobs):
            sys.exit("every simulation failed")

    rows = []
    for i, (k, v, c, _) in enumerate(jobs):
        if res[i] is not None:
            rows.append({"vector": k, **v, "corner": int(c["corner"]), "vdd": c["vdd"], "temp": c["temp"], **res[i]})
    df = pd.DataFrame(rows)
    df.to_csv(data_dir / "cla_spice.csv", index=False, float_format="%.12e")
    seconds = time.time() - t0
    config.write_meta(data_dir / "cla_spice.meta.json", corners=len(corners), vectors=len(vecs), rows=len(df),
                      failed=len(errors), quick=args.quick, ngspice=ver,
                      generated_at=datetime.datetime.now().isoformat(timespec="seconds"),
                      generation_seconds=round(seconds, 1))

    # logic check against the exhaustive truth table
    wrong = 0
    for k, d in df.groupby("vector"):
        want = cla.evaluate(vecs[k])
        for o in cla.OUTPUTS:
            wrong += int(((d[f"v_{o}"] - want[o] * d["vdd"]).abs() > 0.1 * d["vdd"]).sum())
    print(f"\nLogic check: {'all outputs correct' if wrong == 0 else f'{wrong} wrong output values'}")
    print(f"Leakage check: {'all positive' if (df.i_leak_spice > 0).all() else 'NON-POSITIVE values found'}")
    print(f"CLA leakage over all vectors and corners: median {df.i_leak_spice.median():.3e} A, "
          f"range {df.i_leak_spice.min():.3e} - {df.i_leak_spice.max():.3e} A")
    print(f"Wrote {(data_dir / 'cla_spice.csv').relative_to(config.ROOT)} ({len(df)} rows) in {seconds:.0f} s")
    if not args.quick:
        print("Next: python3 07_cla_validation.py")


if __name__ == "__main__":
    main()
