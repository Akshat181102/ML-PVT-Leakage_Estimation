"""
Random draws of supply, temperature and process parameters (Project doc, Section 4).
Shared by the transistor datasets (Part A) and the gate / test-circuit corners (Parts B, C).
"""
import numpy as np

import config


def draw_vdd(rng, n):
    return rng.uniform(config.VDD_NOM * (1 - config.VDD_REL),
                       config.VDD_NOM * (1 + config.VDD_REL), n)


def draw_temp(rng, n):
    return rng.uniform(*config.TEMP_RANGE, n)


def draw_width(rng, n):
    return rng.uniform(config.W_MIN, config.W_MAX, n)


def draw_process(rng, nominal_dev, n):
    """nominal_dev = {'vth0': .., 'toxe': .., ...} for ONE device type.
    Each parameter is uniform in nominal * (1 +/- PROC_REL), drawn independently.
    (A negative PMOS vth0 keeps its sign; its magnitude varies by +/- 10 %.)"""
    return {p: nominal_dev[p] * rng.uniform(1 - config.PROC_REL, 1 + config.PROC_REL, n)
            for p in config.PROC_PARAMS}
