# Transistor models (32nm_MGK)

Errors are |I_ML - I_SPICE| / max(|I_SPICE|, 1 fA), pooled over the four terminal currents (id, ig, is, ib). R2 is on log10 max(|I|, 1 fA).

## Model comparison, validation set (deliverable 4)

Each cell: R2 / median error / 95th-percentile error. Bold = chosen: the lowest validation mean absolute error of log10|I| (table below), the same score used to tune each model.

| Model | NMOS ON | NMOS OFF | PMOS ON | PMOS OFF |
|---|---|---|---|---|
| Polynomial ridge | 0.9975 / 0.75 % / 6.3 % | 0.9660 / 4.05 % / 59.8 % | 0.9945 / 0.45 % / 4.6 % | 0.9506 / 3.91 % / 61.3 % |
| Histogram gradient boosting | 0.9981 / 4.85 % / 23.9 % | 0.9893 / 4.03 % / 28.3 % | 0.9977 / 4.49 % / 23.0 % | 0.9828 / 3.92 % / 28.6 % |
| MLP (neural network) | **0.9998 / 0.70 % / 3.6 %** | **0.9986 / 0.74 % / 5.9 %** | **0.9999 / 0.51 % / 3.1 %** | **0.9983 / 0.70 % / 6.5 %** |
| Kernel ridge (RBF) | 0.9988 / 0.86 % / 7.6 % | 0.9696 / 3.92 % / 54.3 % | 0.9987 / 0.58 % / 5.3 % | 0.9589 / 3.75 % / 57.6 % |

Selection score, validation mean absolute error of log10|I| (decades; 0.0043 decades ~ 1 %):

| Model | NMOS ON | NMOS OFF | PMOS ON | PMOS OFF |
|---|---|---|---|---|
| Polynomial ridge | 0.0100 | 0.0621 | 0.0090 | 0.0662 |
| Histogram gradient boosting | 0.0332 | 0.0346 | 0.0314 | 0.0365 |
| MLP (neural network) | **0.0053** | **0.0067** | **0.0042** | **0.0073** |
| Kernel ridge (RBF) | 0.0102 | 0.0586 | 0.0078 | 0.0626 |

## Final models, test set (deliverable 5)

| Case | Final model | R2 | Median error | P95 error | Mean error | Mean accuracy | Sign accuracy |
|---|---|---|---|---|---|---|---|
| NMOS ON | MLP (neural network), ensemble of 3 (64x64x64 tanh, L-BFGS, alpha 1e-06) | 0.9997 | 0.49 % | 2.81 % | 1.33 % | 98.67 % | 99.92 % |
| NMOS OFF | MLP (neural network), ensemble of 3 (64x64x64 tanh, L-BFGS, alpha 1e-05) | 0.9980 | 0.49 % | 4.16 % | 2.55 % | 97.45 % | 99.39 % |
| PMOS ON | MLP (neural network), ensemble of 3 (64x64x64 tanh, L-BFGS, alpha 1e-06) | 0.9999 | 0.37 % | 2.07 % | 0.73 % | 99.27 % | 99.97 % |
| PMOS OFF | MLP (neural network), ensemble of 3 (64x64x64 tanh, L-BFGS, alpha 1e-05) | 0.9988 | 0.46 % | 4.51 % | 2.07 % | 97.93 % | 99.72 % |

## Ensemble members and their average

Same setting and training data, different random initial weights. Each cell: median / P95 / mean error.

| Case | Fit | Validation | Test |
|---|---|---|---|
| NMOS ON | seed 2026 | 0.70 / 3.65 / 1.33 % | 0.71 / 3.70 / 1.63 % |
| NMOS ON | seed 2027 | 0.59 / 3.05 / 1.17 % | 0.58 / 3.15 / 1.42 % |
| NMOS ON | seed 2028 | 0.71 / 3.78 / 1.45 % | 0.69 / 4.02 / 1.74 % |
| NMOS ON | ensemble (average) | 0.48 / 2.66 / 1.03 % | 0.49 / 2.81 / 1.33 % |
| NMOS OFF | seed 2026 | 0.74 / 5.94 / 2.16 % | 0.76 / 5.92 / 3.11 % |
| NMOS OFF | seed 2027 | 0.72 / 5.61 / 8.59 % | 0.70 / 5.43 / 2.86 % |
| NMOS OFF | seed 2028 | 0.75 / 5.74 / 2.91 % | 0.72 / 5.93 / 3.20 % |
| NMOS OFF | ensemble (average) | 0.47 / 4.40 / 2.18 % | 0.49 / 4.16 / 2.55 % |
| PMOS ON | seed 2026 | 0.51 / 3.14 / 0.98 % | 0.49 / 2.61 / 0.94 % |
| PMOS ON | seed 2027 | 0.56 / 3.19 / 1.03 % | 0.55 / 2.82 / 1.00 % |
| PMOS ON | seed 2028 | 0.58 / 2.89 / 1.01 % | 0.56 / 2.82 / 1.00 % |
| PMOS ON | ensemble (average) | 0.37 / 2.41 / 0.75 % | 0.37 / 2.07 / 0.73 % |
| PMOS OFF | seed 2026 | 0.70 / 6.55 / 2.54 % | 0.68 / 6.27 / 2.51 % |
| PMOS OFF | seed 2027 | 0.72 / 7.19 / 2.60 % | 0.71 / 6.39 / 2.60 % |
| PMOS OFF | seed 2028 | 0.73 / 7.21 / 2.74 % | 0.72 / 6.72 / 2.74 % |
| PMOS OFF | ensemble (average) | 0.47 / 4.73 / 2.05 % | 0.46 / 4.51 / 2.07 % |

## Final models per terminal, test set

| Case | Terminal | Median error | P95 error | Mean error | Sign accuracy | SPICE values below 1 fA |
|---|---|---|---|---|---|---|
| NMOS ON | id | 0.43 % | 2.01 % | 1.63 % | 99.87 % | 0.0 % |
| NMOS ON | ig | 0.48 % | 3.92 % | 1.03 % | 100.00 % | 0.0 % |
| NMOS ON | is | 0.43 % | 1.92 % | 1.60 % | 99.80 % | 0.0 % |
| NMOS ON | ib | 0.65 % | 3.33 % | 1.07 % | 100.00 % | 0.1 % |
| NMOS OFF | id | 0.40 % | 3.08 % | 4.54 % | 99.00 % | 0.0 % |
| NMOS OFF | ig | 0.29 % | 1.30 % | 0.43 % | 100.00 % | 0.0 % |
| NMOS OFF | is | 0.41 % | 4.12 % | 2.97 % | 99.00 % | 0.0 % |
| NMOS OFF | ib | 1.05 % | 8.56 % | 2.27 % | 100.00 % | 73.5 % |
| PMOS ON | id | 0.34 % | 1.65 % | 0.74 % | 99.93 % | 0.0 % |
| PMOS ON | ig | 0.33 % | 1.98 % | 0.61 % | 100.00 % | 0.0 % |
| PMOS ON | is | 0.33 % | 1.52 % | 0.70 % | 99.93 % | 0.0 % |
| PMOS ON | ib | 0.52 % | 2.81 % | 0.88 % | 100.00 % | 0.0 % |
| PMOS OFF | id | 0.42 % | 2.65 % | 1.31 % | 99.80 % | 0.0 % |
| PMOS OFF | ig | 0.28 % | 1.29 % | 0.44 % | 100.00 % | 0.0 % |
| PMOS OFF | is | 0.41 % | 2.54 % | 3.68 % | 99.27 % | 0.0 % |
| PMOS OFF | ib | 1.19 % | 10.96 % | 2.84 % | 100.00 % | 71.5 % |

## Learning curve, validation set (chosen model type, single fits)

Each cell: median / P95 / mean error.

| Case | 500 samples | 1,000 samples | 2,000 samples | 4,000 samples | 7,000 samples |
|---|---|---|---|---|---|
| NMOS ON | 3.24 / 19.2 / 23.84 % | 1.31 / 8.8 / 4.06 % | 0.82 / 5.4 / 2.03 % | 0.73 / 4.3 / 1.70 % | 0.70 / 3.6 / 1.33 % |
| NMOS OFF | 8.15 / 93.9 / 59.32 % | 2.68 / 53.1 / 22.01 % | 1.03 / 15.2 / 5.70 % | 0.86 / 7.8 / 3.04 % | 0.74 / 5.9 / 2.16 % |
| PMOS ON | 2.53 / 19.1 / 6.47 % | 1.14 / 8.4 / 2.93 % | 0.75 / 5.0 / 1.64 % | 0.67 / 4.3 / 1.45 % | 0.51 / 3.1 / 0.98 % |
| PMOS OFF | 8.14 / 95.2 / 44.38 % | 2.99 / 57.0 / 20.14 % | 1.34 / 26.8 / 7.76 % | 0.93 / 11.1 / 3.81 % | 0.70 / 6.5 / 2.54 % |

## Tuned settings

| Case | Model | Setting |
|---|---|---|
| NMOS ON | Polynomial ridge | degree 3, alpha 1e-06 |
| NMOS ON | Histogram gradient boosting | learning rate 0.05, 15 leaves, early stopping |
| NMOS ON | MLP (neural network) | 64x64x64 tanh, L-BFGS, alpha 1e-06 |
| NMOS ON | Kernel ridge (RBF) | RBF gamma 0.01, alpha 0.0001 |
| NMOS OFF | Polynomial ridge | degree 3, alpha 0.01 |
| NMOS OFF | Histogram gradient boosting | learning rate 0.05, 15 leaves, early stopping |
| NMOS OFF | MLP (neural network) | 64x64x64 tanh, L-BFGS, alpha 1e-05 |
| NMOS OFF | Kernel ridge (RBF) | RBF gamma 0.03, alpha 0.01 |
| PMOS ON | Polynomial ridge | degree 3, alpha 1e-06 |
| PMOS ON | Histogram gradient boosting | learning rate 0.05, 15 leaves, early stopping |
| PMOS ON | MLP (neural network) | 64x64x64 tanh, L-BFGS, alpha 1e-06 |
| PMOS ON | Kernel ridge (RBF) | RBF gamma 0.01, alpha 0.0001 |
| PMOS OFF | Polynomial ridge | degree 3, alpha 0.01 |
| PMOS OFF | Histogram gradient boosting | learning rate 0.05, 63 leaves, early stopping |
| PMOS OFF | MLP (neural network) | 64x64x64 tanh, L-BFGS, alpha 1e-05 |
| PMOS OFF | Kernel ridge (RBF) | RBF gamma 0.03, alpha 0.01 |

Run options: quick=False, reselect=False; scikit-learn 1.7.2, Python 3.14.4; settings fingerprint 43eb2541a1e3.
