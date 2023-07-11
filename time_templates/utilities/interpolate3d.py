import numpy as np
from numba import njit, prange


class interp3d:
    """
    NOTE: the first axis can be vectorized (not any other)
    Same as regular grid interpolator from scipy but ~50 times faster
    """

    def __init__(self, x, y, z, array, vectorized=True, fill_value=0):
        self.x = x
        self.y = y
        self.z = z
        self.array = array
        self.xmin = x.min()
        self.xmax = x.max()
        self.ymin = y.min()
        self.ymax = y.max()
        self.zmin = z.min()
        self.zmax = z.max()
        self.nx = len(x)
        self.ny = len(y)
        self.nz = len(z)
        self.fill_value = fill_value
        self.vectorized = vectorized

    def __call__(self, x, y, z):
        if self.vectorized:
            return do_interp_vx(
                x,
                y,
                z,
                self.x,
                self.y,
                self.z,
                self.array,
                self.nx,
                self.ny,
                self.nz,
                self.xmin,
                self.ymin,
                self.zmin,
                self.xmax,
                self.ymax,
                self.zmax,
                self.fill_value,
            )
        return do_interp(
            x,
            y,
            z,
            self.x,
            self.y,
            self.z,
            self.array,
            self.nx,
            self.ny,
            self.nz,
            self.xmin,
            self.ymin,
            self.zmin,
            self.xmax,
            self.ymax,
            self.zmax,
            self.fill_value,
        )


@njit(fastmath=True)
def get_ix(x, xarr):
    i = np.searchsorted(xarr, x)
    if i == 0:
        return 0
    return i - 1


@njit(fastmath=True)
def check_bounds(x, xmin, xmax):
    if x < xmin or x > xmax:
        return False
    return True


@njit(fastmath=True)
def do_interp(
    x,
    y,
    z,
    xarr,
    yarr,
    zarr,
    array,
    nx,
    ny,
    nz,
    xmin,
    ymin,
    zmin,
    xmax,
    ymax,
    zmax,
    fill_value,
):
    if not check_bounds(x, xmin, xmax):
        return fill_value
    if not check_bounds(y, ymin, ymax):
        return fill_value
    if not check_bounds(z, zmin, zmax):
        return fill_value
    i = get_ix(x, xarr)
    j = get_ix(y, yarr)
    k = get_ix(z, zarr)

    if i >= nx:
        i -= 1
    x0, x1 = xarr[i], xarr[i + 1]
    #    if np.isnan(x0) or np.isnan(x1):
    #        return np.nan

    if j >= ny - 1:
        j -= 1
    y0, y1 = yarr[j], yarr[j + 1]
    #    if np.isnan(y0) or np.isnan(y1):
    #        return np.nan

    if k >= nz - 1:
        k -= 1
    z0, z1 = zarr[k], zarr[k + 1]
    #    if np.isnan(z0) or np.isnan(z1):
    #        return np.nan

    xd = (x - x0) / (x1 - x0)
    yd = (y - y0) / (y1 - y0)
    zd = (z - z0) / (z1 - z0)

    c000 = array[i, j, k]
    c001 = array[i, j, k + 1]
    c010 = array[i, j + 1, k]
    c011 = array[i, j + 1, k + 1]
    c100 = array[i + 1, j, k]
    c101 = array[i + 1, j, k + 1]
    c010 = array[i, j + 1, k]
    c110 = array[i + 1, j + 1, k]
    c111 = array[i + 1, j + 1, k + 1]

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    c = c0 * (1 - zd) + c1 * zd

    return c


# This is about 10x faster than regulargridinterpolator from scipy at about 500 entries
@njit(fastmath=True)
def do_interp_vx(
    xv,
    y,
    z,
    xarr,
    yarr,
    zarr,
    array,
    nx,
    ny,
    nz,
    xmin,
    ymin,
    zmin,
    xmax,
    ymax,
    zmax,
    fill_value,
):
    """ Only vectorized in x direction"""
    size = len(xv)
    out = np.zeros(size)
    for i in range(size):
        out[i] = do_interp(
            xv[i],
            y,
            z,
            xarr,
            yarr,
            zarr,
            array,
            nx,
            ny,
            nz,
            xmin,
            ymin,
            zmin,
            xmax,
            ymax,
            zmax,
            fill_value,
        )

    return out
