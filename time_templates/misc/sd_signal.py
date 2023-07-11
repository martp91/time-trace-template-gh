import numpy as np


def poisson_factor(theta):
    "so that var(S) = p^2 * S"
    return 0.34 + 0.46 / np.cos(theta)  # A. Aab et al 2020 JINST 15 P10021


def wcd_total_signal_variance(S, theta):
    return poisson_factor(theta) ** 2 * S


# var_S = f**2 * S


def wcd_total_signal_uncertainty(S, theta):
    return np.sqrt(wcd_total_signal_variance(S, theta))


def wcd_neff_particles(S, theta):
    return S / poisson_factor(theta) ** 2


# n = S / (var_S/S) = S**2/var_S
