# ML-Driven PVT-Variation-Aware Leakage Estimation (32nm_MGK, 1.0 V, 3-bit CLA)

Digital VLSI Design course project. Every script reads its settings from `config.py`, and every result is regenerated from the model file in `models/` by running the numbered scripts in order. All random draws use fixed seeds.

## Folder

```
09_code/
├── README.md  requirements.txt
├── models/32nm_MGK.pm          PTM 32 nm metal-gate / high-k model file (the only input)
├── config.py                   assignment, ranges, seeds, SPICE options, junction geometry
├── sampling.py  spice_utils.py  fet.py           Part A building blocks
├── ml_model.py                                   features, model types, metrics
├── gates.py  cla.py                              Part B / C building blocks
├── 01_ … 08_ *.py              the steps, run in order (see below; deliverable 9)
├── data/                       SPICE datasets (deliverable 1) and gate / CLA simulation results
├── netlists/                   example netlists (deliverable 2)
├── trained_models/             <case>/mlp_ensemble.joblib = the four final models (deliverable 5);
│                               selection.json names them
└── results/                    data check, model comparison, gate, CLA and error-analysis tables and plots
                                (deliverables 3, 4, 6, 7, 8)
```

`data/`, `netlists/`, `trained_models/` and `results/` are created when the scripts run. The submitted copies of these outputs are in the numbered folders next to `09_code/`, and the report is in `10_report/`.

## Steps

Run each command on its own and wait for it to finish before starting the next.

| Step | Command | What it does | Time (12-thread laptop) |
|---|---|---|---|
| 1 | `python3 01_generate_transistor_data.py` | Self-test, then 4 × 10,000 single-transistor simulations | ~1 min |
| 1 | `python3 02_data_check.py` | Kirchhoff check and trend plots (deliverable 3) | ~1 min |
| 2 | `python3 03_train_models.py` | 4 model types × 4 cases, selection, final model (ensemble of 3 MLP fits), test scores, learning curve (`--reselect` re-scores the models a full run saved, without training) | ~3 h |
| 3 | `python3 04_simulate_gates.py` | 18 gate cases × 200 corners in SPICE | ~1 min |
| 3 | `python3 05_gate_validation.py` | ML gate leakage vs SPICE (deliverables 6, 7) | < 1 min |
| 4 | `python3 06_simulate_cla.py` | 128 vectors × 200 corners of the full CLA in SPICE | ~7 min |
| 4 | `python3 07_cla_validation.py` | ML CLA leakage vs SPICE (deliverable 8) | < 1 min |
| 5 | `python3 08_error_analysis.py` | where the errors occur and why | ~1 min |

**Quick checks.** The simulation scripts accept `--quick` for a short trial. Quick runs write to `data/quick/` and `results/*_quick/`, so they can never be mixed up with the full results. The validation scripts refuse to run on incomplete data.

**Stopping a run.** Ctrl+C stops a simulation run within a second.

**Settings fingerprint.** Every data file records a fingerprint of the settings that produced it (ranges, seeds, SPICE options, junction geometry, model file). Later steps stop if the data was made with different settings, so stale data cannot slip in.

## Conventions

- Terminal currents are positive **into** the device terminal, so Id + Ig + Is + Ib = 0.
- Every netlist has `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9`; temperature is set with `.options temp`; the analysis is a DC operating point.
- Every transistor has junction areas: AS = AD = W × Ldiff, PS = PD = 2(W + Ldiff), Ldiff = 2 Lmin = 64 nm.
- Leakage of a gate or circuit = current drawn from VDD + current drawn from the inputs held at VDD (= static power / VDD).

## Requirements and operating systems

You need ngspice (tested with ngspice-45.2) and Python 3 with the packages in `requirements.txt`. The scripts are the same on Linux, macOS and Windows; only the setup and the command name differ. They have been run on Ubuntu, and nothing in them is specific to one system.

| | Linux (Ubuntu) | macOS | Windows |
|---|---|---|---|
| Install ngspice | `sudo apt install ngspice` | `brew install ngspice` | download it from ngspice.sourceforge.io and add the folder with `ngspice_con.exe` to PATH |
| Install the Python packages | `pip install -r requirements.txt` | same | same |
| Run a script | `python3 01_generate_transistor_data.py` | same | `python 01_generate_transistor_data.py` (or `py`) |
| If ngspice is not found | `export NGSPICE=/path/to/ngspice` | same | PowerShell: `$env:NGSPICE="C:\path\to\ngspice_con.exe"` |

If your system Python refuses `pip install`, create a virtual environment first (`python3 -m venv .venv`, then activate it). On Windows the scripts look for `ngspice_con.exe`, the console version of ngspice.

The fixed seeds give the same samples and corners on every system. A different ngspice version, or retraining on another system (whose floating-point libraries differ slightly), can change the last digits of the results. The saved models give the same predictions on any system with scikit-learn 1.7.2.
