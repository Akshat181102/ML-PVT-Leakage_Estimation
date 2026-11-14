# ML-Driven PVT Variation Aware Leakage Estimation

Machine-learning models that estimate the leakage current of CMOS transistors, logic gates and a 3-bit carry-lookahead adder under process, voltage and temperature (PVT) variation, trained only on single-transistor SPICE simulations. Course project for Digital VLSI Design at IIIT Hyderabad (course instructor: Dr. Zia Abbas).

**Assignment:** PTM 32 nm metal-gate/high-k model file (`32nm_MGK`), nominal VDD = 1.0 V, test circuit a 3-bit carry-lookahead adder (CLA). Simulator: ngspice-45.2.

## How it works

1. **Transistor data.** Simulate NMOS and PMOS transistors, ON and OFF, 10,000 times each in SPICE, with random VDD, temperature, width, drain and source voltages and five process parameters.
2. **Transistor models.** Train and compare four model types per case; the final model of each case averages three multi-layer perceptron (MLP) fits.
3. **Gates.** Estimate the leakage of INV, NAND2, NOR2, AND2 and OR2 from the transistor models, for every input combination at 200 random PVT corners.
4. **Test circuit.** Estimate the adder's leakage as the sum of its 34 gates, for all 128 input vectors at the same 200 corners.

Each level is checked against its own SPICE simulation; no gate or circuit SPICE result is used in training.

## Results

| Level | Median error | 95th-percentile error | Mean accuracy |
|---|---|---|---|
| Transistor models (1,500 test samples per case) | 0.37–0.49 % | 2.1–4.5 % | 97.4–99.3 % |
| Gates (18 gate/input cases × 200 corners) | 3.39 % | 12.8 % | 95.4 % |
| 3-bit CLA (128 vectors × 200 corners) | 2.32 % | 8.0 % | 97.1 % |

Method, figures and error analysis: [the report](10_report/ML_PVT_Leakage_Estimation_Report.pdf).

## Deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Datasets | [`01_datasets/`](01_datasets): `32nm_MGK_<case>.csv`, 10,000 rows each |
| 2 | Example netlists | [`02_example_netlists/`](02_example_netlists): `transistor/` (4), `gates/` (18), `cla/` (1) |
| 3 | Data check | [`03_data_check/`](03_data_check): `kirchhoff.md` (and `.csv`), the three trend plots, `summary.md` |
| 4 | Model comparison table | [`04_05_models/model_comparison.md`](04_05_models/model_comparison.md) (validation set), `model_comparison.png` |
| 5 | Chosen models | test-set metrics in [`04_05_models/model_comparison.md`](04_05_models/model_comparison.md) ("Final models, test set"); `parity_<case>.png`; `saved_models/<case>/mlp_ensemble.joblib` |
| 6 | Gate results table | [`06_07_gate_results/gate_results.md`](06_07_gate_results/gate_results.md) (and `.csv`); every case and corner in `gate_predictions.csv` |
| 7 | Gate parity plot | [`06_07_gate_results/gate_parity.png`](06_07_gate_results/gate_parity.png) |
| 8 | Test-circuit results | [`08_test_circuit_results/`](08_test_circuit_results): `cla_results.md`, `cla_parity.png`; every vector and corner in `cla_predictions.csv` |
| 9 | Code | [`09_code/`](09_code) |
| 10 | Report | [`10_report/ML_PVT_Leakage_Estimation_Report.pdf`](10_report/ML_PVT_Leakage_Estimation_Report.pdf) |

## Reproducing the results

You need ngspice (tested with 45.2) and Python 3 with the packages in [`09_code/requirements.txt`](09_code/requirements.txt); the saved models load with scikit-learn 1.7.2. From `09_code/`, run the eight scripts in order, each after the previous one finishes:

```
python3 01_generate_transistor_data.py
python3 02_data_check.py
python3 03_train_models.py
python3 04_simulate_gates.py
python3 05_gate_validation.py
python3 06_simulate_cla.py
python3 07_cla_validation.py
python3 08_error_analysis.py
```

Every random choice uses a fixed seed, so a rerun on the same setup reproduces every number. The outputs are written inside `09_code/` (`data/`, `netlists/`, `trained_models/`, `results/`). See [`09_code/README.md`](09_code/README.md) for details.

**Windows, macOS or Linux.** The scripts are the same on all three. On Windows, type `python` instead of `python3` and install ngspice from [ngspice.sourceforge.io](https://ngspice.sourceforge.io), with the folder containing `ngspice_con.exe` on PATH; on macOS, `brew install ngspice`; on Ubuntu, `sudo apt install ngspice`. [`09_code/README.md`](09_code/README.md) has the full setup for each system. Results reproduce exactly on the same setup; another ngspice version or a retrain on another system can change the last digits.

## Team

Simon Shibu, Ramit Nag, Akshat Agrawal, Kanuthala Harshith.
