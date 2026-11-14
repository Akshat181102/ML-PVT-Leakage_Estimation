"""
Project configuration (single run of the whole project).

DVD course project: ML-driven PVT-variation-aware leakage estimation.
Assignment: model file 32nm_MGK, nominal VDD = 1.0 V, test circuit = 3-bit CLA.

Every script imports its settings from here, so changing the assignment means
changing this file only.
"""
import hashlib
import json
import os
import platform
import shutil
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "models" / "32nm_MGK.pm"
DATA_DIR = ROOT / "data"
QUICK_DIR = DATA_DIR / "quick"          # quick-check runs (--quick) never touch the full data
NETLIST_DIR = ROOT / "netlists"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "trained_models"


def _find_ngspice():
    """NGSPICE environment variable if set, else ngspice on the PATH, else the
    usual install locations (Windows: the console build ngspice_con.exe)."""
    if os.environ.get("NGSPICE"):
        return os.environ["NGSPICE"]
    names = ["ngspice_con", "ngspice"] if os.name == "nt" else ["ngspice"]
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    for p in [r"C:\Spice64\bin\ngspice_con.exe", r"C:\Program Files\Spice64\bin\ngspice_con.exe",
              "/opt/homebrew/bin/ngspice", "/usr/local/bin/ngspice", "/usr/bin/ngspice"]:
        if Path(p).exists():
            return p
    return "ngspice"


# ngspice executable. Override with:  export NGSPICE=/path/to/ngspice
NGSPICE = _find_ngspice()

# Parallel ngspice processes (default: all CPU cores). Each process runs single-threaded
# (spice_utils.SPICE_THREADS), so this is one ngspice per core.
WORKERS = int(os.environ.get("WORKERS", os.cpu_count() or 2))

# ----------------------------------------------------------------------------
# Assignment
# ----------------------------------------------------------------------------
TECH = "32nm_MGK"
VDD_NOM = 1.0            # V
L = 32e-9                # channel length = feature size of the node (m)
W_MIN = 1 * L            # widths: 1 x Lmin ... 8 x Lmin
W_MAX = 8 * L
TEST_CIRCUIT = "3-bit CLA"

# ----------------------------------------------------------------------------
# Sampling ranges (Project doc, Section 4)
# ----------------------------------------------------------------------------
VDD_REL = 0.10                     # VDD uniform in VDD_NOM * (1 +/- 10 %)
TEMP_RANGE = (-40.0, 125.0)        # degC, uniform
PROC_PARAMS = ["vth0", "toxe", "ndep", "u0", "xj"]
PROC_REL = 0.10                    # each process parameter uniform in nominal * (1 +/- 10 %)

N_SAMPLES = 10_000                 # per case

# Fixed seeds so every result is reproducible
SEED = 2026                        # ML models
CASE_SEEDS = {                     # transistor datasets
    "nmos_on": SEED + 1,
    "nmos_off": SEED + 2,
    "pmos_on": SEED + 3,
    "pmos_off": SEED + 4,
}
SPLIT_SEED = SEED + 100            # train / validation / test split
CORNER_SEED = SEED + 200           # the 200 gate / test-circuit corners

# Gate and body voltages are fixed by the case (Project doc, Section 4).
# "vdd" means "this sample's VDD".
CASES = {
    "nmos_on":  {"dev": "nmos", "vg": "vdd", "vb": "0"},
    "nmos_off": {"dev": "nmos", "vg": "0",   "vb": "0"},
    "pmos_on":  {"dev": "pmos", "vg": "0",   "vb": "vdd"},
    "pmos_off": {"dev": "pmos", "vg": "vdd", "vb": "vdd"},
}

# ----------------------------------------------------------------------------
# SPICE settings
# ----------------------------------------------------------------------------
# Required in every netlist (Project doc, Section 4, "Netlist requirements")
SPICE_OPTIONS = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"

# Source/drain junction geometry on every transistor (transistor, gate and test-circuit
# netlists alike):  AS = AD = W * Ldiff,  PS = PD = 2 * (W + Ldiff).
# Ldiff = 2 x Lmin, the same ratio as the TA's example netlist (.PARAM Ldiff=44n for 22 nm).
DIFFUSION_LENGTH = 2 * L

SPICE_TIMEOUT_S = 120              # per ngspice run

# ----------------------------------------------------------------------------
# Dataset format (Project doc, Section 6, deliverable 1)
# ----------------------------------------------------------------------------
# Units: vdd, vd, vs, vg, vb in V; temp in degC; w in m; vth0 in V; toxe in m;
# ndep in cm^-3 (BSIM4 convention); u0 in m^2/Vs; xj in m; currents in A.
# Current sign convention: positive = current flowing INTO the device terminal,
# so id + ig + is + ib = 0 (Kirchhoff).
CSV_COLUMNS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb",
               "vth0", "toxe", "ndep", "u0", "xj",
               "id", "ig", "is", "ib"]


def dataset_path(case: str) -> Path:
    return DATA_DIR / f"{TECH}_{case}.csv"


def meta_path(case: str) -> Path:
    return DATA_DIR / f"{TECH}_{case}.meta.json"


# ----------------------------------------------------------------------------
# Run fingerprint: a short code that changes if anything that affects the SPICE
# data changes (ranges, seeds, options, junction geometry, model file). Every
# data file records it, and later steps refuse data made with other settings.
# ----------------------------------------------------------------------------
def spec_dict():
    model_text = MODEL_FILE.read_text(encoding="utf-8").replace("\r\n", "\n")
    return {
        "tech": TECH, "vdd_nom": VDD_NOM, "L": L, "w_min": W_MIN, "w_max": W_MAX,
        "vdd_rel": VDD_REL, "temp_range": list(TEMP_RANGE),
        "proc_params": PROC_PARAMS, "proc_rel": PROC_REL,
        "n_samples": N_SAMPLES, "case_seeds": CASE_SEEDS, "corner_seed": CORNER_SEED,
        "cases": CASES, "spice_options": SPICE_OPTIONS, "diffusion_length": DIFFUSION_LENGTH,
        "csv_columns": CSV_COLUMNS,
        "model_file_sha256": hashlib.sha256(model_text.encode()).hexdigest(),
    }


def spec_fingerprint():
    blob = json.dumps(spec_dict(), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def write_meta(path, **extra):
    """Write a small JSON next to a data file: fingerprint, versions, anything extra."""
    meta = {"spec_fingerprint": spec_fingerprint(), "python": platform.python_version(),
            "platform": platform.platform(), **extra}
    Path(path).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def check_meta(path, what):
    """Stop with a clear message if a data file was made with different settings."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path.relative_to(ROOT)} not found - {what} was not generated by this version "
                         f"of the scripts. Re-run the step that makes it.")
    got = json.loads(path.read_text(encoding="utf-8")).get("spec_fingerprint")
    if got != spec_fingerprint():
        raise SystemExit(f"{what} was generated with different settings (fingerprint {got}, "
                         f"current {spec_fingerprint()}). Re-run the step that makes it.")
    return json.loads(path.read_text(encoding="utf-8"))
