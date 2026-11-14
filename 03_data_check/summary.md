# Data check

Model file: 32nm_MGK, VDD_nom = 1.0 V

Cases checked: nmos_on, nmos_off, pmos_on, pmos_off

## Kirchhoff check

| case | rows | median abs(sum) (A) | max abs(sum) (A) | median relative | max relative | median abs(id) (A) | median abs(ig) (A) | median abs(ib) (A) | nan rows | result |
|---|---|---|---|---|---|---|---|---|---|---|
| nmos_on | 10000 | 8.162e-17 | 1.540e-15 | 1.174e-12 | 3.067e-05 | 7.405e-05 | 1.158e-08 | 2.640e-11 | 0 | PASS |
| nmos_off | 10000 | 5.432e-22 | 8.267e-20 | 8.947e-14 | 8.554e-13 | 3.147e-09 | 6.844e-09 | 1.990e-17 | 0 | PASS |
| pmos_on | 10000 | 9.309e-18 | 1.237e-16 | 3.301e-13 | 3.045e-07 | 2.940e-05 | 1.304e-08 | 1.014e-10 | 0 | PASS |
| pmos_off | 10000 | 3.631e-16 | 8.013e-15 | 5.222e-08 | 5.819e-05 | 2.669e-09 | 4.930e-09 | 2.776e-17 | 0 | PASS |


## Trend checks (expected direction: temperature +, vth0 -, toxe -)

| plot | case | samples | Spearman rho (dataset) | SPICE sweep direction |
|---|---|---|---|---|
| OFF current vs temperature | nmos_off | 2475 | +0.274 | + (x14.3 over range) |
| OFF current vs temperature | pmos_off | 2563 | +0.260 | + (x9.84 over range) |
| OFF current vs vth0 | nmos_off | 2475 | -0.128 | - (x3.84 over range) |
| OFF current vs vth0 | pmos_off | 2563 | -0.081 | - (x3.07 over range) |
| Gate current vs toxe | nmos_on | 10000 | -0.445 | - (x13.2 over range) |
| Gate current vs toxe | pmos_on | 10000 | -0.481 | - (x12.3 over range) |
| Gate current vs toxe | nmos_off | 10000 | -0.566 | - (x12 over range) |
| Gate current vs toxe | pmos_off | 10000 | -0.603 | - (x12.1 over range) |

Junction areas on every transistor: AS = AD = W x Ldiff, PS = PD = 2 (W + Ldiff), Ldiff = 64 nm.

## Current ranges (sign convention: positive = into the terminal)

| case | id decades | id % negative | ig decades | ig % negative | is decades | is % negative | ib decades | ib % negative |
|---|---|---|---|---|---|---|---|---|
| nmos_on | 9.7 | 50.5 | 5.9 | 0.0 | 8.4 | 49.5 | 8.7 | 100.0 |
| nmos_off | 6.5 | 11.8 | 5.3 | 100.0 | 5.9 | 12.5 | 12.7 | 70.4 |
| pmos_on | 8.2 | 50.7 | 5.6 | 100.0 | 7.8 | 49.2 | 8.9 | 0.0 |
| pmos_off | 6.2 | 86.7 | 4.8 | 0.0 | 6.2 | 86.4 | 8.4 | 12.7 |

