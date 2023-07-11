import numpy as np


def log_multinomial(fmu, trace, Tmu, Tem, cutoff=1e-20):
    mu = np.maximum(fmu * Tmu + (1 - fmu) * Tem, cutoff)  # log > 0
    lnL = -np.sum(trace * np.log(mu))
    # dln/dfmu
    dmu = Tmu - Tem
    jac = -np.sum(trace * dmu / mu)
    return lnL, jac
