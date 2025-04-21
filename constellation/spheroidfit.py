"""Fit the oblate spheroid equation to satellite orbit positions.

The script reads TLE data for the Starlink constellation, computes the (x, y, z)
coordinates of satellite positions, and fits the oblate spheroid equation to
estimate the semi-major and semi-minor axes of the orbit.

Functions:
- oblate_spheroid_equation: Computes the oblate spheroid equation value for
  given 3D coordinates.

Usage:
Run this script to compute and print the fitted parameters `a` and `c`.
"""

import logging

import astropy.coordinates
import astropy.time
import astropy.units
import library
import numpy as np
from scipy.optimize import curve_fit


def oblate_spheroid_equation(  # noqa: D417
    coords: [float, float, float],
    a: float,
    c: float) -> float:
    """Compute the oblate spheroid equation for a given set of 3D coordinates.

    Parameters
    ----------
    coords (tuple[float, float, float]): A tuple containing the
        (x, y, z) coordinates in 3D space. These values are expected to represent
        the position of satellite relative to the center of the earth in meters.
    a (float): The semi-major axis of the oblate spheroid orbit in the x-y plane.
    c (float): The semi-minor axis of the oblate spheroid orbit along the z-axis.


    Returns
    -------
    float: The computed value from the oblate spheroid equation.

    """
    x, y, z = coords
    return ((x**2 + y**2) / a**2) + (z**2 / c**2)-1


starlinks = library.ConstellationFromTLE("starlink_tle.txt")
tonight = astropy.time.Time("2023-05-10T03:00:00")
x, y, z = starlinks.get_teme_position(tonight)
ar_nanx = np.where(np.isfinite(x))
x = x[ar_nanx]
y = y[ar_nanx]
z = z[ar_nanx]


initial_guess = [600000, 600000]
bounds = ([0.0, 0.0], [np.inf, np.inf])

params, covariance = curve_fit(oblate_spheroid_equation, (x, y, z) ,[0]*len(x),
 bounds=bounds, p0=initial_guess)

a_fit, c_fit = params

logging.basicConfig(level=logging.INFO)
logging.info("Fitted Parameters")
logging.info("a: %s", a_fit)
logging.info("c: %s", c_fit)
