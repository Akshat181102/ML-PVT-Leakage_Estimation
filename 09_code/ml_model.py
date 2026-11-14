"""
Transistor surrogate models (Part A step 2). Also used by the gate-level and
test-circuit scripts, which load the saved models.

For one case (e.g. NMOS ON), a TransistorModel predicts the four terminal
currents id, ig, is, ib (A, positive = into the terminal) from the same inputs
as the CSV: vdd, temp, w, vd, vs, vg, vb, vth0, toxe, ndep, u0, xj.

How a current is modelled
  * Magnitude: one regressor predicts log10|I| of all four terminals. Currents
    span 5-10 decades, so the log is what can be fitted to ~1 %.
  * Sign: if a terminal (almost) never changes sign in the training data, the
    sign is a constant. Otherwise a classifier predicts it. The same sign
    classifier is used for all four model types, so the comparison is purely
    about the magnitude regressor.

Inputs: the raw columns plus physics-based features (make_features), all
standardised. Targets: log10|I| standardised per terminal.

Currents below SPICE's resolution
  With junction areas, the OFF-state body current is often far below 1 fA
  (median ~2e-17 A). There it is set by simulator artefacts, not device physics:
  the required gmin = 1e-18 S across each junction, and for PMOS a floating-point
  step of 2^-58 A (~3.5e-18 A, with 5 % of samples exactly 0). So
    * targets are log10(max(|I|, LOG_FLOOR)), LOG_FLOOR = 1e-17 A, and
    * every error is |I_ml - I_spice| / max(|I_spice|, I_RES), I_RES = 1 fA:
      plain relative error above 1 fA, an absolute error on a 1 fA scale below it
      (the same reltol/abstol idea SPICE itself uses). Only the OFF-state Ib is
      affected (72 % of its samples are below 1 fA); every other current is above it.
    * sign accuracy and the sign classifier use only currents above 1 fA.

The four model types (MODEL_TYPES): polynomial ridge, histogram gradient
boosting, multi-layer perceptron and kernel ridge with an RBF kernel. Each has
a small hyperparameter grid that 03_train_models.py searches on the
validation split.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import config
import spice_utils as su

TERMINALS = ["id", "ig", "is", "ib"]
INPUTS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj"]
K_OVER_Q = 8.617333262e-5      # V/K
N_SLOPE = 1.4                  # nominal sub-threshold slope factor used in the EKV-style features
TINY = 1e-30                   # numerical floor inside the features
I_RES = 1e-15                  # A: error scale below which SPICE does not resolve a current (see above)
LOG_FLOOR = 1e-17              # A: training targets are log10(max(|I|, LOG_FLOOR))


def _gidl_constants():
    """egidl, bgidl of each device from the model card (not varied by the sampling)."""
    cards = su.read_model_cards()
    return {dev: (su.get_param(cards[dev], "egidl"), su.get_param(cards[dev], "bgidl")) for dev in ("nmos", "pmos")}


GIDL = _gidl_constants()


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
def make_features(df, dev):
    """Raw inputs plus physics-based combinations.

    Every feature is bounded and finite, including at Vd = Vs exactly, which the
    gate-level inputs hit. "Effective source/drain" is the lower/higher of
    (Vd, Vs) for NMOS (the reverse for PMOS), and the effective voltages are
    signed so that "more ON" is positive for both device types.
    """
    p = 1.0 if dev == "nmos" else -1.0
    vd, vs, vg, vb = (df[c].to_numpy(float) for c in ("vd", "vs", "vg", "vb"))
    T = df["temp"].to_numpy(float) + 273.15
    vt = K_OVER_Q * T
    v_src = np.minimum(vd, vs) if dev == "nmos" else np.maximum(vd, vs)
    v_drn = np.maximum(vd, vs) if dev == "nmos" else np.minimum(vd, vs)
    vds_eff = np.abs(vd - vs)
    vgs_eff = p * (vg - v_src)
    vgd_eff = p * (vg - v_drn)
    vth = np.abs(df["vth0"].to_numpy(float))
    tox_nm = df["toxe"].to_numpy(float) * 1e9

    # EKV-style channel current: F(x) = ln^2(1 + e^(x/2)) covers sub-threshold
    # (exponential) and strong inversion (square law) with one smooth function.
    F = lambda x: np.logaddexp(0.0, x / 2.0) ** 2
    f_fwd = F((vgs_eff - vth) / (N_SLOPE * vt))
    f_rev = F((vgd_eff - vth) / (N_SLOPE * vt))
    i_ch = np.maximum(f_fwd - f_rev, 1e-2 * f_fwd)        # floor keeps Vd = Vs finite

    # GIDL / GISL (BSIM4): ~ x * exp(-3 * toxe * bgidl / x), x = V_dg - egidl > 0 (drain side;
    # source side with V_sg). Log form, clipped so it stays finite when GIDL is off.
    egidl, bgidl = GIDL[dev]
    tox_m = df["toxe"].to_numpy(float)

    def log_gidl(v_xg):
        x = np.maximum(v_xg - egidl, 1e-2)
        return np.maximum(np.log(x) - 3.0 * tox_m * bgidl / x, -60.0)

    X = pd.DataFrame({
        # raw inputs, in convenient units
        "vdd": df["vdd"].to_numpy(float),
        "inv_T": 1000.0 / T,
        "log_w": np.log(df["w"].to_numpy(float) / config.L),
        "vth0": vth,
        "toxe_nm": tox_nm,
        "ndep_1e18": df["ndep"].to_numpy(float) * 1e-18,
        "u0_1e3": df["u0"].to_numpy(float) * 1e3,
        "xj_nm": df["xj"].to_numpy(float) * 1e9,
        # terminal voltages relative to each other
        "vds": p * (vd - vs),
        "vds_eff": vds_eff,
        "vgs_eff": vgs_eff,
        "vgd_eff": vgd_eff,
        "vsb_eff": p * (v_src - vb),
        "vdb_eff": p * (v_drn - vb),
        "vgb": p * (vg - vb),
        # sub-threshold physics
        "overdrive_vt": (vgs_eff - vth) / vt,
        "log_1mexp": np.log(np.maximum(-np.expm1(-vds_eff / vt), 1e-2)),
        "log_ekv_fwd": np.log(f_fwd + TINY),
        "log_ekv_rev": np.log(f_rev + TINY),
        "log_ekv_ch": np.log(i_ch + TINY),
        # gate tunnelling: oxide voltages and fields
        "log_vox_gs": np.log(np.abs(vg - vs) + 1e-3),
        "log_vox_gd": np.log(np.abs(vg - vd) + 1e-3),
        "fox_gs": np.abs(vg - vs) / tox_nm,
        "fox_gd": np.abs(vg - vd) / tox_nm,
        # band-to-band (GIDL / GISL) leakage
        "log_gidl_d": log_gidl(p * (vd - vg)),
        "log_gidl_s": log_gidl(p * (vs - vg)),
    }, index=df.index)
    return X


# ----------------------------------------------------------------------------
# The four model types and their hyperparameter grids
# ----------------------------------------------------------------------------
MODEL_TYPES = {
    "poly_ridge": {
        "label": "Polynomial ridge",
        "grid": [{"degree": d, "alpha": a} for d in (2, 3) for a in (1e-6, 1e-4, 1e-2)],
        "quick": {"degree": 3, "alpha": 1e-4},
    },
    "hist_gb": {
        "label": "Histogram gradient boosting",
        "grid": [{"learning_rate": lr, "max_leaf_nodes": n} for lr in (0.05, 0.1) for n in (15, 31, 63)],
        "quick": {"learning_rate": 0.1, "max_leaf_nodes": 31},
    },
    "mlp": {
        "label": "MLP (neural network)",
        "grid": [{"hidden": (64, 64, 64), "alpha": 1e-5},
                 {"hidden": (64, 64, 64), "alpha": 1e-6},
                 {"hidden": (96, 96), "alpha": 1e-6}],
        "quick": {"hidden": (64, 64, 64), "alpha": 1e-5},
        "seeded": True,          # the fit depends on the random initial weights -> final model is an ensemble
    },
    "kernel_ridge": {
        "label": "Kernel ridge (RBF)",
        "grid": [{"gamma": g, "alpha": a} for g in (0.003, 0.01, 0.03) for a in (1e-6, 1e-4, 1e-2)],
        "quick": {"gamma": 0.01, "alpha": 1e-5},
    },
}


def build_regressor(model_type, params, seed=config.SEED):
    """Unfitted multi-output regressor (standardised X -> standardised log10|I| of 4 terminals)."""
    if model_type == "poly_ridge":
        return make_pipeline(PolynomialFeatures(degree=params["degree"], include_bias=False),
                             StandardScaler(), Ridge(alpha=params["alpha"]))
    if model_type == "hist_gb":
        return MultiOutputRegressor(HistGradientBoostingRegressor(
            learning_rate=params["learning_rate"], max_leaf_nodes=params["max_leaf_nodes"],
            max_iter=3000, min_samples_leaf=10, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=50, random_state=seed))
    if model_type == "mlp":
        return MLPRegressor(hidden_layer_sizes=tuple(params["hidden"]), activation="tanh",
                            solver="lbfgs", alpha=params["alpha"], max_iter=6000,
                            max_fun=60000, tol=1e-10, random_state=seed)
    if model_type == "kernel_ridge":
        return KernelRidge(kernel="rbf", gamma=params["gamma"], alpha=params["alpha"])
    raise ValueError(f"unknown model type {model_type}")


def describe(model_type, params):
    if model_type == "poly_ridge":
        return f"degree {params['degree']}, alpha {params['alpha']:g}"
    if model_type == "hist_gb":
        return f"learning rate {params['learning_rate']:g}, {params['max_leaf_nodes']} leaves, early stopping"
    if model_type == "mlp":
        layers = "x".join(str(h) for h in params["hidden"])
        return f"{layers} tanh, L-BFGS, alpha {params['alpha']:g}"
    if model_type == "kernel_ridge":
        return f"RBF gamma {params['gamma']:g}, alpha {params['alpha']:g}"
    return str(params)


# ----------------------------------------------------------------------------
# Sign models (shared by all model types of a case)
# ----------------------------------------------------------------------------
def fit_sign_models(df, dev, seed=config.SEED, min_class=20):
    """terminal -> +1.0 / -1.0 if the sign is (almost) constant, else a fitted classifier.
    Only currents above I_RES count (below it the sign is simulator noise). A terminal
    whose minority sign has fewer than `min_class` such samples is treated as constant."""
    F = make_features(df, dev)
    cols = [c for c in F.columns if F[c].std() > 0]
    out = {}
    for t in TERMINALS:
        i = df[t].to_numpy(float)
        keep = np.abs(i) >= I_RES
        s = np.sign(i[keep])
        n_pos, n_neg = int((s > 0).sum()), int((s < 0).sum())
        if min(n_pos, n_neg) < min_class:
            out[t] = 1.0 if n_pos >= n_neg else -1.0
        else:
            clf = HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.1, max_leaf_nodes=31,
                                                 early_stopping=True, n_iter_no_change=50,
                                                 random_state=seed)
            clf.fit(F[cols].to_numpy()[keep], (s > 0).astype(int))
            out[t] = {"columns": cols, "clf": clf}
    return out


# ----------------------------------------------------------------------------
# The model object that is trained, evaluated and saved
# ----------------------------------------------------------------------------
class TransistorModel:
    """Predicts id, ig, is, ib (A, positive into the terminal) for one case."""

    def __init__(self, case, model_type, params, seed=config.SEED):
        self.case = case
        self.dev = config.CASES[case]["dev"]
        self.model_type = model_type
        self.params = dict(params)
        self.seed = seed

    # -- training ------------------------------------------------------------
    def fit(self, df, sign_models):
        F = make_features(df, self.dev)
        # drop features that are constant in this case (e.g. vgb is fixed by the bias)
        self.feature_columns = [c for c in F.columns if F[c].std() > 1e-12]
        X = F[self.feature_columns].to_numpy()
        Y = np.log10(np.maximum(np.abs(df[TERMINALS].to_numpy(float)), LOG_FLOOR))
        self.x_scaler = StandardScaler().fit(X)
        self.y_scaler = StandardScaler().fit(Y)
        self.regressor = build_regressor(self.model_type, self.params, self.seed)
        self.regressor.fit(self.x_scaler.transform(X), self.y_scaler.transform(Y))
        self.sign_models = sign_models
        self.n_train = len(df)
        return self

    # -- prediction ----------------------------------------------------------
    def predict_log10(self, df):
        """log10|I| for the four terminals, shape (n, 4) in TERMINALS order."""
        X = make_features(df, self.dev)[self.feature_columns].to_numpy()
        Z = self.regressor.predict(self.x_scaler.transform(X))
        return self.y_scaler.inverse_transform(np.asarray(Z).reshape(len(df), -1))

    def predict_signs(self, df):
        F = None
        out = np.empty((len(df), len(TERMINALS)))
        for j, t in enumerate(TERMINALS):
            s = self.sign_models[t]
            if isinstance(s, float):
                out[:, j] = s
            else:
                if F is None:
                    F = make_features(df, self.dev)
                out[:, j] = np.where(s["clf"].predict(F[s["columns"]].to_numpy()) == 1, 1.0, -1.0)
        return out

    def predict(self, df):
        """DataFrame with columns id, ig, is, ib (A)."""
        I = self.predict_signs(df) * 10.0 ** self.predict_log10(df)
        return pd.DataFrame(I, columns=TERMINALS, index=df.index)

    def summary(self):
        return f"{MODEL_TYPES[self.model_type]['label']} ({describe(self.model_type, self.params)})"


ENSEMBLE_SIZE = 3


class EnsembleModel:
    """The final model of a case when its model type depends on a random seed (the MLP):
    ENSEMBLE_SIZE fits with the same setting and different random initial weights, averaged
    in log space. Inside the training range the fits agree closely; at the edges of the range
    (where gate transistors sit) each fit errs differently, and the average cancels much of it.
    All members share the sign classifiers."""

    def __init__(self, members):
        self.members = list(members)
        m0 = self.members[0]
        self.case, self.dev = m0.case, m0.dev
        self.model_type, self.params = m0.model_type, m0.params
        self.sign_models = m0.sign_models
        self.n_train = m0.n_train
        self.seeds = [m.seed for m in self.members]

    def predict_log10(self, df):
        return np.mean([m.predict_log10(df) for m in self.members], axis=0)

    def predict_signs(self, df):
        return self.members[0].predict_signs(df)

    def predict(self, df):
        I = self.predict_signs(df) * 10.0 ** self.predict_log10(df)
        return pd.DataFrame(I, columns=TERMINALS, index=df.index)

    def summary(self):
        return (f"{MODEL_TYPES[self.model_type]['label']}, ensemble of {len(self.members)} "
                f"({describe(self.model_type, self.params)})")


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def r2_score(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def errors(pred, true):
    """Error in % (see the module docstring): plain relative error above I_RES,
    absolute error on an I_RES scale below it."""
    return np.abs(pred - true) / np.maximum(np.abs(true), I_RES) * 100


def log_res(x):
    """log10 of a current, with everything below I_RES treated as I_RES."""
    return np.log10(np.maximum(np.abs(x), I_RES))


def evaluate(model, df):
    """Metrics per terminal and pooled over all four ('all').

    err %        = |I_ml - I_spice| / max(|I_spice|, 1 fA) * 100   (the headline error)
    mean_acc %   = 100 - mean err %      (the brief's "mean accuracy")
    R2           = on log10 max(|I|, 1 fA), the quantity the regressor fits
    sign_acc %   = share of currents above 1 fA whose sign is right
    plain err %  = |I_ml - I_spice| / |I_spice|, no 1 fA scale (for reference)
    dev-rel err  = |I_ml - I_spice| / max(|id|,|ig|,|is|,|ib|) of that sample: the error
                   compared with the device's largest current.
    """
    true = df[TERMINALS].to_numpy(float)
    pred = model.predict(df)[TERMINALS].to_numpy(float)
    err = errors(pred, true)
    plain = np.abs(pred - true) / np.maximum(np.abs(true), TINY) * 100
    dev_rel = np.abs(pred - true) / np.abs(true).max(axis=1, keepdims=True) * 100
    lt, lp = log_res(true), log_res(pred)
    resolved = np.abs(true) >= I_RES
    sign_ok = (np.sign(pred) == np.sign(true)) | ~resolved
    rows = []
    for j, t in enumerate(TERMINALS):
        rows.append(_metric_row(t, err[:, j], plain[:, j], dev_rel[:, j], r2_score(lt[:, j], lp[:, j]),
                                sign_ok[resolved[:, j], j].mean() * 100 if resolved[:, j].any() else float("nan"),
                                np.sqrt(np.mean((lp[:, j] - lt[:, j]) ** 2)), (~resolved[:, j]).mean() * 100))
    rows.append(_metric_row("all", err.ravel(), plain.ravel(), dev_rel.ravel(),
                            float(np.mean([r["r2_log"] for r in rows])),
                            sign_ok[resolved].mean() * 100, float(np.sqrt(np.mean((lp - lt) ** 2))),
                            (~resolved).mean() * 100))
    return pd.DataFrame(rows), pd.DataFrame(pred, columns=TERMINALS, index=df.index)


def _metric_row(t, err, plain, dev_rel, r2, sign_acc, rmse_dec, below_res):
    return {"terminal": t, "r2_log": r2, "median_err_%": float(np.median(err)),
            "p95_err_%": float(np.percentile(err, 95)), "mean_err_%": float(np.mean(err)),
            "mean_acc_%": 100.0 - float(np.mean(err)), "max_err_%": float(np.max(err)),
            "median_err_plain_%": float(np.median(plain)), "mean_err_plain_%": float(np.mean(plain)),
            "median_devrel_%": float(np.median(dev_rel)), "p95_devrel_%": float(np.percentile(dev_rel, 95)),
            "sign_acc_%": sign_acc, "rmse_decades": rmse_dec, "below_1fA_%": below_res}


def validation_score(model, df):
    """Scores used for hyperparameter selection, on the validation rows.

    mae_decades  : mean |log10 I_ml - log10 I_spice| over all terminals and samples, both
                   floored at 1 fA - the selection score. It is a robust form of the mean
                   relative error (0.0043 decades ~ 1 %), and unlike the RMSE it is not
                   dominated by the few samples whose current is close to zero.
    rmse_decades, median_err_%, p95_err_%, mean_err_% : recorded for reference.
    """
    true = df[TERMINALS].to_numpy(float)
    pred = model.predict(df)[TERMINALS].to_numpy(float)
    d = log_res(pred) - log_res(true)
    err = errors(pred, true).ravel()
    return {"mae_decades": float(np.mean(np.abs(d))), "rmse_decades": float(np.sqrt(np.mean(d ** 2))),
            "median_err_%": float(np.median(err)), "p95_err_%": float(np.percentile(err, 95)),
            "mean_err_%": float(np.mean(err))}
