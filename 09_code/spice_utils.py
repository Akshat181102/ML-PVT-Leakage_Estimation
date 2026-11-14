"""
Shared SPICE helpers: model-card editing, netlist building, running ngspice and
reading its ASCII raw output. Used by the transistor, gate and test-circuit scripts.
"""
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import config

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


# ----------------------------------------------------------------------------
# Model cards
# ----------------------------------------------------------------------------
def read_model_cards(path=config.MODEL_FILE):
    """Split a PTM .pm file into {'nmos': card_text, 'pmos': card_text}."""
    text = Path(path).read_text(encoding="utf-8")
    cards, current, buf = {}, None, []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(".model"):
            if current is not None:
                cards[current] = "\n".join(buf).rstrip() + "\n"
            name = stripped.split()[1].lower()
            current, buf = name, [line]
        elif current is not None:
            buf.append(line)
    if current is not None:
        cards[current] = "\n".join(buf).rstrip() + "\n"
    missing = {"nmos", "pmos"} - set(cards)
    if missing:
        raise ValueError(f"{path}: model card(s) not found: {missing}")
    return cards


def _param_regex(name):
    # Exact parameter name: not preceded by a word character (so 'lvth0' or
    # 'ku0' never match) and followed by '=' (so 'xjbvs' never matches 'xj').
    return re.compile(rf"(?<![\w])({re.escape(name)})(\s*=\s*)({_NUM})", re.IGNORECASE)


def get_param(card, name):
    matches = _param_regex(name).findall(card)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one '{name}' in model card, found {len(matches)}")
    return float(matches[0][2])


def set_params(card, values):
    """Return a copy of the card with the given {param: value} replaced."""
    for name, value in values.items():
        rx = _param_regex(name)
        new_card, n = rx.subn(lambda m: f"{m.group(1)}{m.group(2)}{value:.10g}", card)
        if n != 1:
            raise ValueError(f"expected exactly one '{name}' in model card, replaced {n}")
        card = new_card
    return card


def nominal_process(cards=None):
    """{'nmos': {vth0: .., toxe: .., ...}, 'pmos': {...}} read from the model file."""
    cards = cards or read_model_cards()
    return {dev: {p: get_param(cards[dev], p) for p in config.PROC_PARAMS}
            for dev in ("nmos", "pmos")}


# ----------------------------------------------------------------------------
# Netlist pieces
# ----------------------------------------------------------------------------
def mos_instance(name, d, g, s, b, dev, w, l=config.L):
    """One MOSFET instance line with its junction geometry:
    AS = AD = W * Ldiff, PS = PD = 2 * (W + Ldiff), Ldiff = config.DIFFUSION_LENGTH."""
    ld = config.DIFFUSION_LENGTH
    return (f"{name} {d} {g} {s} {b} {dev} W={w:.10g} L={l:.10g}"
            f" AS={w * ld:.10g} AD={w * ld:.10g} PS={2 * (w + ld):.10g} PD={2 * (w + ld):.10g}")


def control_block(raw_name="out.raw"):
    """Run one DC operating point and write every node voltage / branch current
    to an ASCII raw file (full double precision)."""
    return ("\n.control\n"
            "set filetype=ascii\n"
            "op\n"
            f"write {raw_name}\n"
            "quit\n"
            ".endc\n")


# ----------------------------------------------------------------------------
# Running ngspice
# ----------------------------------------------------------------------------
class SpiceError(RuntimeError):
    pass


def ngspice_version():
    try:
        out = subprocess.run([config.NGSPICE, "-v"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace",
                             timeout=30, stdin=subprocess.DEVNULL)
        text = (out.stdout + out.stderr).strip()
        lines = [l.strip(" *") for l in text.splitlines() if "ngspice" in l.lower()]
        with_digit = [l for l in lines if any(ch.isdigit() for ch in l)]
        return (with_digit or lines or ["unknown version"])[0]
    except FileNotFoundError:
        return None


# Threads per ngspice process. The scripts already run one ngspice per CPU core in
# parallel, so each process should use a single thread: an OpenMP build of ngspice
# would otherwise start several threads per process and oversubscribe the CPU.
# Override with  export SPICE_THREADS=4  (or "default" to leave ngspice's own setting).
SPICE_THREADS = os.environ.get("SPICE_THREADS", "1")


def run_parallel(fn, jobs, workers=None, label="", n_progress=20):
    """Run fn(*job) for every job, `workers` at a time (each call starts its own ngspice).

    Returns (results, errors): results[i] is fn's return value (None if it failed),
    errors = {i: exception}. Failed jobs are retried once. Ctrl+C drops the queued
    simulations and exits within a few seconds (the running ngspice processes get the
    same Ctrl+C and stop at once)."""
    workers = workers or config.WORKERS
    n = len(jobs)
    results, errors = [None] * n, {}
    t0 = time.time()
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futs = {pool.submit(fn, *job): i for i, job in enumerate(jobs)}
        for done, fut in enumerate(as_completed(futs), 1):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:          # keep going, retry once at the end
                errors[i] = exc
            if done % max(1, n // n_progress) == 0 or done == n:
                el = time.time() - t0
                print(f"  {label}{done:6d}/{n}  elapsed {el:6.0f} s  ETA {el / done * (n - done):6.0f} s  "
                      f"failures {len(errors)}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted: cancelled the queued simulations; exiting.", flush=True)
        pool.shutdown(wait=False, cancel_futures=True)
        raise SystemExit(130)
    pool.shutdown(wait=True)
    for i in list(errors):
        try:
            results[i] = fn(*jobs[i])
            del errors[i]
        except Exception as exc:
            errors[i] = exc
    return results, errors


def run_netlist(netlist_text, raw_name="out.raw", keep_dir=None, threads=None):
    """Run a netlist in batch mode in a private temp dir; return {vector_name: value}."""
    threads = SPICE_THREADS if threads is None else str(threads)
    tmp = Path(tempfile.mkdtemp(prefix="dvd_spice_"))
    try:
        cir = tmp / "netlist.cir"
        cir.write_text(netlist_text, encoding="utf-8")
        env = dict(os.environ)
        if threads != "default":
            # ngspice reads .spiceinit from its working directory at start-up
            (tmp / ".spiceinit").write_text(f"set num_threads={threads}\n", encoding="utf-8")
            env["OMP_NUM_THREADS"] = threads
        proc = subprocess.run([config.NGSPICE, "-b", cir.name], cwd=tmp, env=env,
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=config.SPICE_TIMEOUT_S, stdin=subprocess.DEVNULL)
        raw = tmp / raw_name
        if not raw.exists():
            log = (proc.stdout + "\n" + proc.stderr).strip()
            raise SpiceError(f"ngspice produced no output (exit {proc.returncode}).\n"
                             f"--- last lines of the ngspice log ---\n"
                             + "\n".join(log.splitlines()[-25:]))
        return parse_ascii_raw(raw)
    finally:
        if keep_dir is not None:
            shutil.copytree(tmp, keep_dir, dirs_exist_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)


def parse_ascii_raw(path):
    """Read the first point of an ngspice ASCII raw file -> {name: value} (names lower-case)."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    names, i, n_vars = [], 0, None
    while i < len(lines):
        line = lines[i].strip()
        low = line.lower()
        if low.startswith("no. variables:"):
            n_vars = int(line.split(":", 1)[1])
        elif low.startswith("variables:"):
            i += 1
            while i < len(lines) and not lines[i].strip().lower().startswith("values:"):
                parts = lines[i].split()
                if len(parts) >= 2:
                    names.append(parts[1].lower())
                i += 1
            break
        i += 1
    if n_vars is not None and len(names) != n_vars:
        raise SpiceError(f"{path}: expected {n_vars} variables, read {len(names)}")
    tokens = []
    for line in lines[i + 1:]:
        if line.strip().lower().startswith(("title:", "plotname:")):
            break  # a second plot follows; we only need the first
        tokens.extend(line.replace(",", " ").split())
    values = [float(t) for t in tokens[1:1 + len(names)]]  # tokens[0] is the point index
    if len(values) != len(names):
        raise SpiceError(f"{path}: expected {len(names)} values, read {len(values)}")
    return dict(zip(names, values))


def branch_current(vectors, source):
    """Current through voltage source `source` (SPICE convention: flowing from its
    + node through the source to its - node)."""
    s = source.lower()
    for key in (f"{s}#branch", f"i({s})"):
        if key in vectors:
            return vectors[key]
    raise KeyError(f"no branch current for source '{source}' in {sorted(vectors)}")
