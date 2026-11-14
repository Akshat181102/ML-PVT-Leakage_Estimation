"""
Part B, step 1: SPICE simulation of the gates at 200 shared random corners.

18 gate / input cases (INV 0,1; NAND2, NOR2, AND2, OR2 with A B = 00, 01, 10, 11),
each simulated at the same 200 corners. A corner is one random draw of VDD, T,
the NMOS process parameters and the PMOS process parameters. All NMOS in a gate
share the NMOS draw; all PMOS share the PMOS draw. AND2 and OR2 are simulated
as complete gates (NAND2/NOR2 followed by an inverter in one netlist).

Recorded per case and corner: the output and internal node voltages, the
current delivered by VDD and by each input source, and the gate leakage
    I_leak = I(VDD) + I(inputs held at VDD)   (= static power / VDD).
These SPICE results are only used to validate the ML estimate, never to train it.

Usage:
    python3 04_simulate_gates.py            # 18 x 200 = 3,600 simulations
    python3 04_simulate_gates.py --quick    # 18 x 10, written to data/quick/ (never mixed with the full run)

Outputs:
    data/gate_corners.csv                      the 200 corners (also used by the test circuit)
    data/gates_spice.csv (+ .meta.json)        one row per case and corner
    netlists/gates/<GATE>_<AB>_example.cir     corner-0 netlist of every case (deliverable 2)
"""
import argparse
import datetime
import sys
import time

import pandas as pd

import config
import gates
import spice_utils as su


def simulate(gate, pattern, corner, cards):
    vec = su.run_netlist(gates.gate_netlist(gate, pattern, corner, cards))
    return gates.spice_result(vec, gate, pattern)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help=f"only {gates.QUICK_CORNERS} corners, into data/quick/")
    ap.add_argument("--workers", type=int, default=config.WORKERS)
    args = ap.parse_args()

    ver = su.ngspice_version()
    if ver is None:
        sys.exit(f"ngspice not found ('{config.NGSPICE}')")
    data_dir = config.QUICK_DIR if args.quick else config.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    cards = su.read_model_cards()
    corners = gates.draw_corners(su.nominal_process(cards))
    if args.quick:
        corners = corners.head(gates.QUICK_CORNERS)
    corners.to_csv(data_dir / "gate_corners.csv", index=False, float_format="%.12e")
    print(f"ngspice: {ver}\n{len(gates.GATE_CASES)} cases x {len(corners)} corners, "
          f"{args.workers} parallel ngspice processes{'  [QUICK -> data/quick/]' if args.quick else ''}")
    print("Sizes (multiples of Lmin): " + ", ".join(f"{g} N{s['n']}/P{s['p']}" for g, s in gates.SIZES.items())
          + "; AND2/OR2 = NAND2/NOR2 + INV")

    if not args.quick:
        ex_dir = config.NETLIST_DIR / "gates"
        ex_dir.mkdir(parents=True, exist_ok=True)
        c0 = corners.iloc[0]
        for gate, pattern in gates.GATE_CASES:
            (ex_dir / f"{gate}_{pattern}_example.cir").write_text(
                gates.gate_netlist(gate, pattern, c0, cards, "(corner 0)"), encoding="utf-8")

    jobs = [(g, p, c, cards) for g, p in gates.GATE_CASES for _, c in corners.iterrows()]
    t0 = time.time()
    rows, errors = su.run_parallel(simulate, jobs, args.workers)
    if errors:
        print(f"WARNING: {len(errors)} simulation(s) failed. First error:\n{next(iter(errors.values()))}")
        if len(errors) == len(jobs):
            sys.exit("every simulation failed")

    out = [{"gate": g, "inputs": p, **c.to_dict(), **rows[i]}
           for i, (g, p, c, _) in enumerate(jobs) if rows[i] is not None]
    df = pd.DataFrame(out)
    df["corner"] = df["corner"].astype(int)
    df.to_csv(data_dir / "gates_spice.csv", index=False, float_format="%.12e")
    seconds = time.time() - t0
    config.write_meta(data_dir / "gates_spice.meta.json", corners=len(corners), rows=len(df), failed=len(errors),
                      quick=args.quick, ngspice=ver, generated_at=datetime.datetime.now().isoformat(timespec="seconds"),
                      generation_seconds=round(seconds, 1))

    # sanity checks: output at the expected logic level, positive leakage
    bad = []
    for (g, p), d in df.groupby(["gate", "inputs"], sort=False):
        want = gates.expected_output(g, p) * d["vdd"]
        if ((d["v_out"] - want).abs() > 0.05 * d["vdd"]).any():
            bad.append(f"{g} {p}")
    print("\nLogic check: " + ("every output is at the expected level" if not bad
                               else f"UNEXPECTED output level in {', '.join(bad)}"))
    neg = (df["i_leak_spice"] <= 0).sum()
    print(f"Leakage check: {'all positive' if neg == 0 else f'{neg} non-positive values'}")
    print("\nMedian SPICE leakage per case (A):")
    med = df.groupby(["gate", "inputs"], sort=False)["i_leak_spice"].median()
    for (g, p), v in med.items():
        print(f"  {g:6s} {p:3s} {v:.3e}")
    print(f"\nWrote {(data_dir / 'gates_spice.csv').relative_to(config.ROOT)} ({len(df)} rows) in {seconds:.0f} s")
    if not args.quick:
        print("Next: python3 05_gate_validation.py")


if __name__ == "__main__":
    main()
