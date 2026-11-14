# Gate-level validation (32nm_MGK)

Leakage = current drawn from VDD + from inputs held at VDD (= static power / VDD).
ML = transistor models evaluated at the SPICE node voltages; ON-transistor channel currents from Kirchhoff's law. 'Direct' = ON-transistor id/is taken straight from the models instead.
Bias = mean signed error (ML - SPICE) / SPICE. Errors are |ML - SPICE| / SPICE over the corners.

| Gate | Inputs (A B) | Corners | SPICE median (A) | ML median (A) | Bias (%) | Median error (%) | P95 error (%) | Mean error (%) | Max error (%) | Median error, direct (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| INV | 0 | 200 | 1.244e-07 | 1.354e-07 | +5.58 | 5.66 | 14.19 | 6.09 | 18.2 | 64.79 |
| INV | 1 | 200 | 1.293e-07 | 1.190e-07 | -7.61 | 7.33 | 16.03 | 7.80 | 25.7 | 7.33 |
| NAND2 | 00 | 200 | 1.752e-07 | 1.697e-07 | -0.16 | 2.55 | 8.52 | 3.32 | 14.6 | 69.28 |
| NAND2 | 01 | 200 | 2.739e-07 | 2.741e-07 | -0.34 | 3.99 | 12.61 | 4.86 | 20.7 | 40.40 |
| NAND2 | 10 | 200 | 1.370e-07 | 1.395e-07 | -0.04 | 1.56 | 5.70 | 2.17 | 10.0 | 63.47 |
| NAND2 | 11 | 200 | 3.552e-07 | 3.215e-07 | -7.70 | 7.90 | 15.26 | 7.88 | 21.5 | 7.90 |
| NOR2 | 00 | 200 | 4.299e-07 | 4.414e-07 | +2.99 | 3.63 | 9.44 | 3.98 | 15.4 | 47.08 |
| NOR2 | 01 | 200 | 3.790e-07 | 3.704e-07 | -2.68 | 2.77 | 6.55 | 3.20 | 17.6 | 47.31 |
| NOR2 | 10 | 200 | 1.622e-07 | 1.491e-07 | -7.34 | 6.86 | 14.64 | 7.38 | 24.4 | 6.86 |
| NOR2 | 11 | 200 | 1.261e-07 | 1.152e-07 | -9.72 | 8.94 | 21.15 | 9.73 | 26.4 | 8.94 |
| AND2 | 00 | 200 | 3.140e-07 | 3.073e-07 | -2.75 | 2.71 | 8.07 | 3.36 | 14.1 | 50.13 |
| AND2 | 01 | 200 | 4.034e-07 | 3.953e-07 | -2.41 | 3.06 | 13.24 | 4.18 | 21.9 | 25.96 |
| AND2 | 10 | 200 | 2.689e-07 | 2.590e-07 | -3.37 | 3.05 | 9.83 | 3.66 | 18.6 | 34.92 |
| AND2 | 11 | 200 | 4.802e-07 | 4.734e-07 | -3.67 | 3.28 | 9.81 | 4.03 | 18.1 | 21.10 |
| OR2 | 00 | 200 | 5.613e-07 | 5.731e-07 | +0.47 | 1.93 | 5.70 | 2.39 | 12.8 | 32.31 |
| OR2 | 01 | 200 | 5.093e-07 | 5.136e-07 | -0.54 | 1.58 | 5.03 | 2.01 | 12.5 | 51.91 |
| OR2 | 10 | 200 | 2.904e-07 | 2.869e-07 | -1.39 | 1.72 | 6.51 | 2.40 | 15.9 | 30.15 |
| OR2 | 11 | 200 | 2.604e-07 | 2.512e-07 | -2.34 | 3.27 | 11.16 | 4.04 | 21.6 | 35.63 |

All 3600 points: median error 3.39 %, P95 12.81 %, mean 4.58 % (mean accuracy 95.42 %), bias -2.39 %. Direct: median 32.40 %, P95 97.71 %.

Most accurate cases (median error): NAND2 10 (1.56 %), OR2 01 (1.58 %), OR2 10 (1.72 %).
Least accurate cases: NOR2 11 (8.94 %), NAND2 11 (7.90 %), INV 1 (7.33 %).

Unbalanced node current (at internal nodes that touch only OFF transistors, Kirchhoff has nothing to adjust, so the OFF-transistor models' error remains): largest 9.1 % of the gate leakage, median 0.00 %.

Feature range check: the gate transistors' model inputs compared with the training data. Largest excursion outside the training range: 0.26 standard deviations (vgd_eff, NAND2 00 MP1 (pmos on)). All inputs are inside or at the edge of the training range.
