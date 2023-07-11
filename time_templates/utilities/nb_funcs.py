import numpy as np
from numba import njit

# Could be compiled, but not sure how to do types correctly


@njit(fastmath=True, cache=True, nogil=True, error_model="numpy")
def interp(x, xp, yp):
    return np.interp(x, xp, yp)


@njit(fastmath=True, cache=True, nogil=True, error_model="numpy")
def normalize(y, dx=1):
    ysum = np.sum(y) * dx
    if ysum > 0:
        return y / ysum
    return y


@njit(fastmath=True, cache=True, nogil=True, error_model="numpy")
def make_cdf(pdf, dt):
    return np.cumsum(pdf) * dt
