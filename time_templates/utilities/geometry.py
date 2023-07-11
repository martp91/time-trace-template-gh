import numpy as np
from time_templates.utilities.constants import *
from numba import njit


# assume axis is normalized


@njit
def calc_shower_plane_position(station, core, axis):
    # CHECKED, but slow
    x = station - core
    ez = -axis
    ey = np.cross(ez, [0, 0, 1.0])
    ey /= np.linalg.norm(ey)
    ex = np.cross(ez, ey)
    return np.array([np.dot(x, ex), np.dot(x, ey), np.dot(x, ez)])


def calc_r_psi_z_SP(station, core, axis):
    r = calc_shower_plane_position(station, core, axis)  # vector
    return np.sqrt(r[0] ** 2 + r[1] ** 2), np.arctan2(r[1], r[0]), r[2]


def calc_shower_plane_angle(station, core, axis):
    r = calc_shower_plane_position(station, core, axis)
    return np.arctan2(r[1], r[0])


def calc_shower_plane_height(station, core, axis):
    "Delta also works vectorized"
    x = station - core
    return np.dot(x, axis)


def calc_shower_plane_front_time(station, core, axis):
    return calc_shower_plane_height(station, core, axis) / C0


def calc_shower_axis_distance(station, core, axis):
    "r also works vectorized"
    # CHECKED
    # always within 1% at least of adst output
    x = station - core
    return np.sqrt(np.sum(x ** 2, axis=-1) - np.dot(x, axis) ** 2)


def ang2vec(phi, zenith):
    """ Get 3-vector from spherical angles.
    Args:
        phi (array): azimuth (pi, -pi), 0 points in x-direction, pi/2 in y-direction
        zenith (array): zenith (0, pi), 0 points in z-direction
    Returns:
        array of 3-vectors
    """
    x = np.sin(zenith) * np.cos(phi)
    y = np.sin(zenith) * np.sin(phi)
    z = np.cos(zenith)
    return np.array([x, y, z]).T


def vec2ang(v):
    """ Get spherical angles phi and zenith from 3-vector
    Args:
        array of 3-vectors
    Returns:
        phi, zenith
        phi (array): azimuth (pi, -pi), 0 points in x-direction, pi/2 in y-direction
        zenith (array): zenith (0, pi), 0 points in z-direction
    """
    x, y, z = v.T
    phi = np.arctan2(y, x)
    zenith = np.pi / 2 - np.arctan2(z, (x ** 2 + y ** 2) ** 0.5)
    return phi, zenith
