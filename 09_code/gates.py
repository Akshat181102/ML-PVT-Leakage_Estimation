"""
Part B building blocks: gate topologies and sizes, the 200 shared random corners,
SPICE netlists, and the ML (transistor-model) leakage estimate.

Gate leakage (agreed definition)
    I_leak = current drawn from VDD + current drawn from every input held at VDD
           = static power / VDD
  Inputs at 0 V deliver no power, so they are not counted. Current that flows
  into a low input came from VDD and is already in the VDD current.

ML estimate (Kirchhoff-based)
  Every transistor is evaluated with the model of its state (NMOS ON/OFF,
  PMOS ON/OFF), using the terminal voltages SPICE recorded for that corner.
  - OFF transistors: all four terminal currents come from their model.
  - ON transistors: ig and ib come from their model. Their channel current
    (id, is) is NOT taken from the model. An ON transistor sits at Vd ~ Vs,
    where the model is unreliable, and its channel current is simply whatever
    the rest of the circuit pushes through it. It is solved from Kirchhoff's
    current law at the gate's internal and output nodes.
  The same terminal currents are then summed at VDD and at the high inputs.
  The "direct" estimate, which takes the ON transistors' id/is straight from
  their model, is also reported for comparison.
"""
import numpy as np
import pandas as pd

import config
import sampling
import spice_utils as su

N_CORNERS = 200
QUICK_CORNERS = 10

# Transistor widths in multiples of Lmin (all inside the 1-8 x range of the training data)
SIZES = {"INV": {"n": 2, "p": 4}, "NAND2": {"n": 4, "p": 4}, "NOR2": {"n": 2, "p": 8}}


def _fet(name, dev, d, g, s, b, mult):
    return {"name": name, "dev": dev, "d": d, "g": g, "s": s, "b": b, "w": mult * config.L, "mult": mult}


def _inv(inp, out, k):
    sz = SIZES["INV"]
    return [_fet(f"MP{k}", "pmos", out, inp, "vdd", "vdd", sz["p"]),
            _fet(f"MN{k}", "nmos", out, inp, "0", "0", sz["n"])]


def _nand2(out):
    """Input A drives the top NMOS (next to the output), B the bottom one; x is between them."""
    sz = SIZES["NAND2"]
    return [_fet("MP1", "pmos", out, "a", "vdd", "vdd", sz["p"]),
            _fet("MP2", "pmos", out, "b", "vdd", "vdd", sz["p"]),
            _fet("MN1", "nmos", out, "a", "x", "0", sz["n"]),
            _fet("MN2", "nmos", "x", "b", "0", "0", sz["n"])]


def _nor2(out):
    """Input A drives the top PMOS (next to VDD), B the bottom one; y is between them."""
    sz = SIZES["NOR2"]
    return [_fet("MP1", "pmos", "y", "a", "vdd", "vdd", sz["p"]),
            _fet("MP2", "pmos", out, "b", "y", "vdd", sz["p"]),
            _fet("MN1", "nmos", out, "a", "0", "0", sz["n"]),
            _fet("MN2", "nmos", out, "b", "0", "0", sz["n"])]


GATES = {
    "INV":   {"inputs": ["a"], "devices": _inv("a", "out", 1), "internal": [], "logic": lambda a: 1 - a},
    "NAND2": {"inputs": ["a", "b"], "devices": _nand2("out"), "internal": ["x"],
              "logic": lambda a, b: 1 - (a & b)},
    "NOR2":  {"inputs": ["a", "b"], "devices": _nor2("out"), "internal": ["y"],
              "logic": lambda a, b: 1 - (a | b)},
    "AND2":  {"inputs": ["a", "b"], "devices": _nand2("n1") + _inv("n1", "out", 3), "internal": ["x", "n1"],
              "logic": lambda a, b: a & b},
    "OR2":   {"inputs": ["a", "b"], "devices": _nor2("n1") + _inv("n1", "out", 3), "internal": ["y", "n1"],
              "logic": lambda a, b: a | b},
}
GATE_CASES = [(g, p) for g in GATES for p in (["0", "1"] if len(GATES[g]["inputs"]) == 1
                                               else ["00", "01", "10", "11"])]   # 18 cases; pattern = A B
NODE_COLUMNS = ["v_out", "v_x", "v_y", "v_n1"]


def case_label(gate, pattern):
    return f"{gate} {pattern}"


def expected_output(gate, pattern):
    return GATES[gate]["logic"](*[int(c) for c in pattern])


# ----------------------------------------------------------------------------
# Corners
# ----------------------------------------------------------------------------
def draw_corners(nominal, n=N_CORNERS):
    """One random draw of VDD, T, NMOS process and PMOS process per corner (Project doc, Part B)."""
    rng = np.random.default_rng(config.CORNER_SEED)
    vdd = sampling.draw_vdd(rng, n)
    temp = sampling.draw_temp(rng, n)
    pn = sampling.draw_process(rng, nominal["nmos"], n)
    pp = sampling.draw_process(rng, nominal["pmos"], n)
    df = pd.DataFrame({"corner": np.arange(n), "vdd": vdd, "temp": temp,
                       **{f"n_{k}": v for k, v in pn.items()}, **{f"p_{k}": v for k, v in pp.items()}})
    return df


# ----------------------------------------------------------------------------
# SPICE
# ----------------------------------------------------------------------------
def gate_netlist(gate, pattern, corner, cards, title_extra=""):
    g = GATES[gate]
    vdd = float(corner["vdd"])
    vin = {inp: (vdd if bit == "1" else 0.0) for inp, bit in zip(g["inputs"], pattern)}
    ncard = su.set_params(cards["nmos"], {p: corner[f"n_{p}"] for p in config.PROC_PARAMS})
    pcard = su.set_params(cards["pmos"], {p: corner[f"p_{p}"] for p in config.PROC_PARAMS})
    sizes = ", ".join(f"{d['name']} W={d['mult']}L" for d in g["devices"])
    lines = [
        f"* {config.TECH} {gate} inputs {pattern} (A B) {title_extra}".rstrip(),
        f"* corner {int(corner['corner'])}: VDD={vdd:.6g} V  T={corner['temp']:.6g} C",
        f"* sizes: {sizes} (L={config.L * 1e9:.0f} nm); junctions AS=AD=W*Ldiff, PS=PD=2(W+Ldiff), "
        f"Ldiff={config.DIFFUSION_LENGTH * 1e9:.0f} nm",
        config.SPICE_OPTIONS,
        f".options temp={corner['temp']:.10g}",
        "",
        f"VDD vdd 0 DC {vdd:.10g}",
        *[f"VIN{inp.upper()} {inp} 0 DC {v:.10g}" for inp, v in vin.items()],
        *[su.mos_instance(d["name"], d["d"], d["g"], d["s"], d["b"], d["dev"], d["w"]) for d in g["devices"]],
        "",
        "* model cards from " + config.MODEL_FILE.name + ", process parameters of this corner",
        ncard.rstrip(),
        pcard.rstrip(),
        su.control_block(),
        ".end",
        "",
    ]
    return "\n".join(lines)


def node_voltage(vectors, node):
    for key in (f"v({node})", node):
        if key in vectors:
            return vectors[key]
    raise KeyError(f"no voltage for node '{node}' in {sorted(vectors)}")


def spice_result(vectors, gate, pattern):
    """Node voltages and leakage from one gate simulation."""
    g = GATES[gate]
    res = {c: np.nan for c in NODE_COLUMNS}
    for node in ["out"] + g["internal"]:
        res[f"v_{node}"] = node_voltage(vectors, node)
    # ngspice branch current flows into the source's + node, so "delivered" = -branch
    res["i_vdd"] = -su.branch_current(vectors, "VDD")
    total = res["i_vdd"]
    for inp, bit in zip(["a", "b"], pattern if len(pattern) == 2 else pattern + "-"):
        if inp in g["inputs"]:
            res[f"i_in_{inp}"] = -su.branch_current(vectors, f"VIN{inp.upper()}")
            if bit == "1":
                total += res[f"i_in_{inp}"]
        else:
            res[f"i_in_{inp}"] = np.nan
    res["i_leak_spice"] = total
    return res


# ----------------------------------------------------------------------------
# ML estimate
# ----------------------------------------------------------------------------
def _node_voltages(gate, pattern, rows):
    g = GATES[gate]
    vdd = rows["vdd"].to_numpy(float)
    V = {"vdd": vdd, "0": np.zeros(len(rows))}
    for inp, bit in zip(g["inputs"], pattern):
        V[inp] = vdd.copy() if bit == "1" else np.zeros(len(rows))
    for node in ["out"] + g["internal"]:
        V[node] = rows[f"v_{node}"].to_numpy(float)
    return V


def device_inputs(gate, pattern, rows):
    """For every transistor of one gate case: (device, case such as 'nmos_on', model inputs).
    The model inputs are the transistor's terminal voltages (SPICE node voltages), its
    width and the corner's temperature, VDD and process parameters."""
    g = GATES[gate]
    n = len(rows)
    V = _node_voltages(gate, pattern, rows)
    out = []
    for dv in g["devices"]:
        dev = dv["dev"]
        vg = V[dv["g"]]
        on = vg > 0.5 * V["vdd"] if dev == "nmos" else vg < 0.5 * V["vdd"]
        if not (on.all() or (~on).all()):
            raise ValueError(f"{gate} {pattern}: {dv['name']} changes state between corners")
        pre = "n_" if dev == "nmos" else "p_"
        X = pd.DataFrame({"vdd": V["vdd"], "temp": rows["temp"].to_numpy(float),
                          "w": np.full(n, dv["w"]), "vd": V[dv["d"]], "vs": V[dv["s"]],
                          "vg": V[dv["g"]], "vb": V[dv["b"]],
                          **{p: rows[pre + p].to_numpy(float) for p in config.PROC_PARAMS}},
                         index=rows.index)
        out.append((dv, f"{dev}_{'on' if on[0] else 'off'}", X))
    return out


def ml_leakage(gate, pattern, rows, models):
    """ML leakage for every corner of one gate case.

    rows   : DataFrame with vdd, temp, n_*/p_* process columns and the SPICE node voltages
    models : {"nmos_on": TransistorModel, ...}
    Returns a DataFrame with i_leak_ml (Kirchhoff-based), i_leak_ml_direct,
    kcl_residual (largest node-current mismatch / i_leak_ml) and the device states.
    """
    g = GATES[gate]
    n = len(rows)
    V = _node_voltages(gate, pattern, rows)
    cur, state = {}, {}
    for dv, case, X in device_inputs(gate, pattern, rows):
        state[dv["name"]] = case.endswith("_on")
        pred = models[case].predict(X)
        cur[dv["name"]] = {t: pred[c].to_numpy(float) for t, c in
                           (("d", "id"), ("g", "ig"), ("s", "is"), ("b", "ib"))}

    def leakage(currents):
        tot = np.zeros(n)
        for dv in g["devices"]:
            for t in ("d", "g", "s", "b"):
                if dv[t] == "vdd":
                    tot += currents[dv["name"]][t]
        for inp, bit in zip(g["inputs"], pattern):
            if bit == "1":
                for dv in g["devices"]:
                    if dv["g"] == inp:
                        tot += currents[dv["name"]]["g"]
        return tot

    direct = leakage(cur)

    # Kirchhoff: unknown channel current c_k of each ON transistor,
    #   id = c_k - h_k,  is = -c_k - h_k,  h_k = (ig + ib) / 2   (so id + ig + is + ib = 0)
    on_devs = [dv["name"] for dv in g["devices"] if state[dv["name"]]]
    col = {name: j for j, name in enumerate(on_devs)}
    nodes = ["out"] + g["internal"]
    A = np.zeros((len(nodes), len(on_devs)))
    rhs = np.zeros((len(nodes), n))
    for i, node in enumerate(nodes):
        for dv in g["devices"]:
            name = dv["name"]
            for t in ("d", "g", "s", "b"):
                if dv[t] != node:
                    continue
                if state[name] and t in ("d", "s"):
                    h = 0.5 * (cur[name]["g"] + cur[name]["b"])
                    A[i, col[name]] += 1.0 if t == "d" else -1.0
                    rhs[i] += h                      # known part (-h) moved to the right-hand side
                else:
                    rhs[i] -= cur[name][t]
    C = np.linalg.pinv(A) @ rhs if on_devs else np.zeros((0, n))
    resid = np.abs(A @ C - rhs).max(axis=0) if len(nodes) else np.zeros(n)
    kcl = {k: dict(v) for k, v in cur.items()}
    for name in on_devs:
        h = 0.5 * (cur[name]["g"] + cur[name]["b"])
        kcl[name]["d"] = C[col[name]] - h
        kcl[name]["s"] = -C[col[name]] - h
    i_kcl = leakage(kcl)

    out = pd.DataFrame({"i_leak_ml": i_kcl, "i_leak_ml_direct": direct,
                        "kcl_residual": resid / np.abs(i_kcl)}, index=rows.index)
    out["states"] = " ".join(f"{k}:{'on' if v else 'off'}" for k, v in state.items())
    return out
