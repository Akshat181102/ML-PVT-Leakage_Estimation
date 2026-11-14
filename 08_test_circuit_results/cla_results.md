# 3-bit CLA validation (32nm_MGK)

34 gates / 180 transistors; 128 input vectors x 200 corners = 25600 points.
Leakage = I(VDD) + I(inputs at VDD). Errors relative to the full-circuit SPICE leakage.

| Estimate | SPICE median (A) | Estimate median (A) | Bias (%) | Median error (%) | P95 error (%) | Mean error (%) | Max error (%) |
|---|---|---|---|---|---|---|---|
| ML (sum of gate-level ML estimates) | 1.199e-05 | 1.171e-05 | -2.22 | 2.32 | 7.99 | 2.94 | 18.67 |
| SPICE gate-sum (stand-alone gate SPICE) | 1.199e-05 | 1.197e-05 | -0.20 | 0.21 | 0.81 | 0.29 | 1.69 |

The SPICE gate-sum row isolates the error of composing the circuit from stand-alone gates (ideal 0/VDD inputs, no loading). The ML row adds the gate-level ML error on top.

CLA mean accuracy (100 % - mean error): 97.06 %; 84 % of points within 5 % of SPICE, 98 % within 10 %.

For comparison, gate level (3600 points): median error 3.39 %, P95 12.81 %, mean 4.58 %, bias -2.39 %.

## Leakage share per gate type (ML, all vectors and corners)

| Gate type | Instances | Share of CLA leakage (ML) |
|---|---|---|
| OR2 | 12 | 44.8 % |
| AND2 | 13 | 40.2 % |
| NAND2 | 6 | 11.9 % |
| INV | 3 | 3.2 % |

## Vectors with the largest ML bias

| vector | a0 | a1 | a2 | b0 | b1 | b2 | c0 | ml_bias_pct | gate_sum_bias_pct |
|---|---|---|---|---|---|---|---|---|---|
| 87 | 1 | 1 | 1 | 0 | 1 | 0 | 1 | -3.23 | -0.34 |
| 79 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | -3.2 | -0.37 |
| 71 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | -3.18 | -0.38 |
| 115 | 1 | 1 | 0 | 0 | 1 | 1 | 1 | -3.14 | -0.33 |
| 94 | 0 | 1 | 1 | 1 | 1 | 0 | 1 | -3.14 | -0.33 |

