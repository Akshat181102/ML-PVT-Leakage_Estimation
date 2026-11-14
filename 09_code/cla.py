"""
Part C: the 3-bit carry-lookahead adder (test circuit), built only from the
five characterised gates (INV, NAND2, NOR2, AND2, OR2).

Inputs  a2 a1 a0, b2 b1 b0, carry-in c0        -> 2^7 = 128 input vectors
Outputs s2 s1 s0 (sum), c3 (carry-out)

Structure (34 gates, 180 transistors):
  per bit i:  n_i = NAND2(a_i, b_i)
              g_i = INV(n_i)                    generate   G_i = a_i b_i
              o_i = OR2(a_i, b_i)
              p_i = AND2(o_i, n_i)              propagate  P_i = a_i XOR b_i
  carries (lookahead, every carry computed from G, P and c0 directly, sharing terms):
              t0 = P0 c0
              c1 = G0 + t0
              c2 = (G1 + P1 G0) + P1 t0
              c3 = ((G2 + P2 G1) + P2 P1 G0) + P2 P1 t0
  sums:       s_i = P_i XOR c_i, built like P_i: AND2(OR2(P_i, c_i), NAND2(P_i, c_i))

XOR is not in the characterised gate set, so it is built as
(x + y) AND NOT(x y) from OR2, NAND2 and AND2 (3 gates).
"""
import itertools

import config
import gates
import spice_utils as su

INPUTS = ["a0", "a1", "a2", "b0", "b1", "b2", "c0"]
OUTPUTS = ["s0", "s1", "s2", "c3"]

# (instance, gate type, input nets in pin order (A, B), output net); topologically ordered
CIRCUIT = []


def _add(inst, gtype, ins, out):
    CIRCUIT.append((inst, gtype, ins, out))


for _i in range(3):
    _a, _b = f"a{_i}", f"b{_i}"
    _add(f"N{_i}", "NAND2", [_a, _b], f"n{_i}")
    _add(f"G{_i}", "INV", [f"n{_i}"], f"g{_i}")
    _add(f"O{_i}", "OR2", [_a, _b], f"o{_i}")
    _add(f"P{_i}", "AND2", [f"o{_i}", f"n{_i}"], f"p{_i}")
_add("T0", "AND2", ["p0", "c0"], "t0")      # P0 c0
_add("K1", "OR2", ["g0", "t0"], "c1")       # c1 = G0 + P0 c0
_add("U1", "AND2", ["p1", "g0"], "u1")      # P1 G0
_add("U2", "AND2", ["p1", "t0"], "u2")      # P1 P0 c0
_add("V1", "OR2", ["g1", "u1"], "v1")
_add("K2", "OR2", ["v1", "u2"], "c2")       # c2 = G1 + P1 G0 + P1 P0 c0
_add("W1", "AND2", ["p2", "g1"], "w1")      # P2 G1
_add("W2", "AND2", ["p2", "p1"], "w2")      # P2 P1
_add("W3", "AND2", ["w2", "g0"], "w3")      # P2 P1 G0
_add("W4", "AND2", ["p2", "u2"], "w4")      # P2 P1 P0 c0
_add("X1", "OR2", ["g2", "w1"], "x1")
_add("X2", "OR2", ["x1", "w3"], "x2")
_add("K3", "OR2", ["x2", "w4"], "c3")       # carry-out
for _i, _c in enumerate(["c0", "c1", "c2"]):
    _add(f"SN{_i}", "NAND2", [f"p{_i}", _c], f"sn{_i}")
    _add(f"SO{_i}", "OR2", [f"p{_i}", _c], f"so{_i}")
    _add(f"S{_i}", "AND2", [f"so{_i}", f"sn{_i}"], f"s{_i}")


def vectors():
    """All 128 input vectors as dicts; vector index k has bit j = INPUTS[j]."""
    out = []
    for k in range(2 ** len(INPUTS)):
        out.append({name: (k >> j) & 1 for j, name in enumerate(INPUTS)})
    return out


def evaluate(vec):
    """Logic value of every net for one input vector."""
    v = dict(vec)
    for _, g, ins, out in CIRCUIT:
        v[out] = gates.GATES[g]["logic"](*[v[n] for n in ins])
    return v


def gate_patterns(vec):
    """[(instance, gate type, input pattern 'AB')] for one input vector."""
    v = evaluate(vec)
    return [(inst, g, "".join(str(v[n]) for n in ins)) for inst, g, ins, _ in CIRCUIT]


def check_logic():
    """Exhaustive check: s2 s1 s0 + 8 c3 == a + b + c0 for every vector."""
    for vec in vectors():
        v = evaluate(vec)
        a = vec["a0"] + 2 * vec["a1"] + 4 * vec["a2"]
        b = vec["b0"] + 2 * vec["b1"] + 4 * vec["b2"]
        got = v["s0"] + 2 * v["s1"] + 4 * v["s2"] + 8 * v["c3"]
        if got != a + b + vec["c0"]:
            raise AssertionError(f"CLA logic error for {vec}: got {got}, want {a + b + vec['c0']}")
    return True


def n_transistors():
    return sum(len(gates.GATES[g]["devices"]) for _, g, _, _ in CIRCUIT)


def cla_netlist(vec, corner, cards, title_extra=""):
    vdd = float(corner["vdd"])
    ncard = su.set_params(cards["nmos"], {p: corner[f"n_{p}"] for p in config.PROC_PARAMS})
    pcard = su.set_params(cards["pmos"], {p: corner[f"p_{p}"] for p in config.PROC_PARAMS})
    bits = " ".join(f"{k}={vec[k]}" for k in INPUTS)
    lines = [f"* {config.TECH} 3-bit CLA {title_extra}".rstrip(),
             f"* inputs {bits}",
             f"* corner {int(corner['corner'])}: VDD={vdd:.6g} V  T={corner['temp']:.6g} C;"
             f" {len(CIRCUIT)} gates, {n_transistors()} transistors",
             f"* junctions on every transistor: AS=AD=W*Ldiff, PS=PD=2(W+Ldiff), "
             f"Ldiff={config.DIFFUSION_LENGTH * 1e9:.0f} nm",
             config.SPICE_OPTIONS,
             f".options temp={corner['temp']:.10g}",
             "",
             f"VDD vdd 0 DC {vdd:.10g}"]
    lines += [f"VIN{k.upper()} {k} 0 DC {vdd if vec[k] else 0.0:.10g}" for k in INPUTS]
    for inst, g, ins, out in CIRCUIT:
        pins = {"vdd": "vdd", "0": "0", "out": out, "a": ins[0]}
        if len(ins) > 1:
            pins["b"] = ins[1]
        for node in gates.GATES[g]["internal"]:
            pins[node] = f"{inst.lower()}_{node}"
        lines.append(f"* {inst}: {g}({', '.join(ins)}) -> {out}")
        for d in gates.GATES[g]["devices"]:
            lines.append(su.mos_instance(f"M{inst}_{d['name']}", pins[d["d"]], pins[d["g"]], pins[d["s"]],
                                         pins[d["b"]], d["dev"], d["w"]))
    lines += ["", "* model cards from " + config.MODEL_FILE.name + ", process parameters of this corner",
              ncard.rstrip(), pcard.rstrip(), su.control_block(), ".end", ""]
    return "\n".join(lines)


def spice_result(vectors_, vec):
    """Outputs and leakage = I(VDD) + I(inputs held at VDD)."""
    res = {f"v_{o}": gates.node_voltage(vectors_, o) for o in OUTPUTS}
    res["i_vdd"] = -su.branch_current(vectors_, "VDD")
    res["i_inputs_high"] = sum(-su.branch_current(vectors_, f"VIN{k.upper()}") for k in INPUTS if vec[k])
    res["i_leak_spice"] = res["i_vdd"] + res["i_inputs_high"]
    return res
