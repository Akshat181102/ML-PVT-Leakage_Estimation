"""
Single-FET netlist and simulation (used by Part A data generation and the data check).
"""
import config
import spice_utils as su

TERMINAL_SOURCES = {"id": "VD", "ig": "VG", "is": "VS", "ib": "VB"}


def fet_netlist(case, row, cards, title_extra=""):
    """Netlist for one sample. `row` needs vdd, temp, w, vd, vs, vg, vb and the
    five process parameters for this case's device type."""
    dev = config.CASES[case]["dev"]
    card = su.set_params(cards[dev], {p: row[p] for p in config.PROC_PARAMS})
    proc_txt = ", ".join(f"{p}={row[p]:.6g}" for p in config.PROC_PARAMS)
    return "\n".join([
        f"* {config.TECH} single-FET characterisation: {case.upper()} {title_extra}".rstrip(),
        f"* VDD={row['vdd']:.6g} V  T={row['temp']:.6g} C  W={row['w'] * 1e9:.4g} nm  "
        f"L={config.L * 1e9:.4g} nm",
        f"* junctions: AS = AD = W*Ldiff, PS = PD = 2(W+Ldiff), Ldiff = {config.DIFFUSION_LENGTH * 1e9:.4g} nm",
        f"* {dev} process: {proc_txt}",
        config.SPICE_OPTIONS,
        f".options temp={row['temp']:.10g}",
        "",
        "* every terminal has its own voltage source; currents are read from these",
        f"VD nd 0 DC {row['vd']:.10g}",
        f"VG ng 0 DC {row['vg']:.10g}",
        f"VS ns 0 DC {row['vs']:.10g}",
        f"VB nb 0 DC {row['vb']:.10g}",
        su.mos_instance("M1", "nd", "ng", "ns", "nb", dev, row["w"]),
        "",
        f"* {dev} model card from {config.MODEL_FILE.name}, process parameters replaced",
        card.rstrip(),
        su.control_block(),
        ".end",
        "",
    ])


def simulate(case, row, cards):
    """Return {'id','ig','is','ib'}: currents INTO the device terminals (A).
    ngspice's branch current flows into the source's + node, i.e. OUT of the
    device terminal, so each one is negated."""
    vec = su.run_netlist(fet_netlist(case, row, cards))
    return {k: -su.branch_current(vec, src) for k, src in TERMINAL_SOURCES.items()}


def nominal_row(case, nominal, vdd=config.VDD_NOM, bias="channel", **overrides):
    """A nominal-corner sample (27 C, W = 2L, nominal process) for quick checks.
    bias='channel': full VDD across the channel (sub-threshold / ON current).
    bias='gate'   : drain and source at the same potential as the body, so the
                    full VDD sits across the oxide (gate tunnelling), as in the
                    slides' single-MOS setup."""
    spec = config.CASES[case]
    dev = spec["dev"]
    row = {"vdd": vdd, "temp": 27.0, "w": 2 * config.L, **nominal[dev]}
    row["vg"] = vdd if spec["vg"] == "vdd" else 0.0
    row["vb"] = vdd if spec["vb"] == "vdd" else 0.0
    if bias == "channel":
        row.update(vd=vdd, vs=0.0) if dev == "nmos" else row.update(vd=0.0, vs=vdd)
    else:
        row.update(vd=row["vb"], vs=row["vb"])
    row.update(overrides)
    return row
