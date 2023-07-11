#!/usr/bin/env python
"""
make lookup table of the integral of the energy spectrum of muons from
MPD framework to get signal per particle in VEM.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from time_templates import data_path, package_path
from numba import njit


@njit
def I_x(x, x0, i=0, gamma=2.6, kappa=0.8):
    return x ** (2 - gamma + i) * (1 - x0 / x) ** kappa * np.exp(-x)


@njit
def total_integral_below_Evem(
    x, l, r, Evem=0.45, gamma=2.6, kappa=0.8, Q=0.17, pa=0.0002, m=0.105
):
    x0 = pa * r / Q
    pal = pa * l
    sina = r / l
    I1 = Q ** 2 / Evem ** 2 * sina ** -2 * I_x(x, x0, i=2, gamma=gamma, kappa=kappa)
    I2 = (
        -2
        * pal
        * Q
        / Evem ** 2
        * sina ** -1
        * I_x(x, x0, i=1, gamma=gamma, kappa=kappa)
    )
    I3 = pal ** 2 / Evem ** 2 * I_x(x, x0, i=0, gamma=gamma, kappa=kappa)
    return I1 + I2 + I3


def eval_integral_below_Evem(
    l, r, Evem=0.45, gamma=2.6, kappa=0.8, Q=0.17, pa=0.0002, m=0.105
):
    def I_below_Evem(x):
        return total_integral_below_Evem(x, l, r, Evem, gamma, kappa, Q, pa, m)

    sina = r / l
    a = m / Q * sina + pa * r / Q
    b = Evem / Q * sina + pa * r / Q
    return quad(I_below_Evem, a, b)


def eval_integral_above_Evem(
    l, r, Evem=0.45, gamma=2.6, kappa=0.8, Q=0.17, pa=0.0002, m=0.105
):
    b = Evem / Q * r / l + pa * r / Q

    def I_above_Evem(x):
        return I_x(x, pa * r / Q, gamma=gamma, kappa=kappa)

    return quad(I_above_Evem, b, np.inf)


def eval_integral_E(
    l, r, Evem=0.45, gamma=2.6, kappa=0.8, Q=0.17, pa=0.0002, m=0.105, TL=1
):
    out = (
        eval_integral_below_Evem(l, r, Evem, gamma, kappa, Q, pa, m)[0]
        + TL * eval_integral_above_Evem(l, r, Evem, gamma, kappa, Q, pa, m)[0]
    )
    return out


nr, nl = 100, 100

Evems = np.linspace(0.1, 1, 10)  # np.array([0.2, 0.3, 0.4, 0.5, 0.6])
lgrs = np.linspace(2, 4, nr)
lgls = np.linspace(0, 5.5, nl)
nEvems = len(Evems)

for gamma in [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1]:
    print(f"gamma = {gamma}")
    out = np.zeros((nl, nr, nEvems))
    for i, l in enumerate(lgls):
        for j, r in enumerate(lgrs):
            for k, Evem in enumerate(Evems):
                out[i, j, k] = eval_integral_E(10 ** l, 10 ** r, Evem=Evem)

    np.savez(
        package_path
        + f"/data/muon_E_integrals/I_integral_vs_l_r_Evem_gamma_{gamma}.npz",
        lgl=lgls,
        lgr=lgrs,
        Evem=Evems,
        array=out,
    )
