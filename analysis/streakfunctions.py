"""A set of helper functions to aid in calculating streaks and plotting."""

import math

import astropy
import astropy.coordinates
import astropy.time
import lumos.conversions
import lumos.functions
import numpy as np


def unit_to_altaz(
    x: float,
    y: float,
    z: float) -> tuple[float, float]:
    """Convert unit vector to altaz coordinate.

    Parameters
    ----------
    - x (float): x-coordinate of the unit vector. Can be a numpy array.
    - y (float): y-coordinate of the unit vector. Can be a numpy array.
    - z (float): z-coordinate of the unit vector. Can be a numpy array.

    Returns
    -------
    Returns the altitude and azimuth coordinates after the conversion.

    """
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    phi = np.arccos(z)
    theta = np.arctan2(y, x)

    azimuth = np.rad2deg(theta)
    azimuth[azimuth < 0] += 360
    altitude = 90 - np.rad2deg(phi)

    return altitude, azimuth


def image_to_altaz(
    x: float,
    y: float,
    z: float,
    tel_alt: float,
    tel_az: float) -> tuple[float, float]:
    """Convert unit vector to altaz coordinate.

    Parameters
    ----------
    - x (float): x-coordinate of the satellite in the image plane. Can be a numpy array.
    - y (float): y-coordinate of the satellite in the image plane. Can be a numpy array.
    - z (float): z-coordinate of the satellite in the image plane. Can be a numpy array.
    - tel_alt (float): Telescope altitude pointing direction.
    - tel_az (float): Telescope azimuth pointing direction.

    Returns
    -------
    Returns the satellite altitude and azimuth coordinates after the conversion.

    """
    x, y, z = lumos.functions.Ry(np.pi / 2 - np.deg2rad(tel_alt), x, y, z)
    sat_x, sat_y, sat_z = lumos.functions.Rz(np.deg2rad(tel_az), x, y, z)

    sat_alt, sat_az = unit_to_altaz(sat_x, sat_y, sat_z)

    return sat_alt, sat_az


def first_n_digits(
    num: float,
    n: int):
    """Get the first n digits of a number.

    Parameters
    ----------
    - num (float): Number that needs to be contracted
    - n (float): Number of digits needed

    Returns
    -------
    The first n digits of number.

    """
    return num // 10 ** (int(math.log10(num)) - n + 1)


def closest(
    xs: np.ndarray,
    ys: np.ndarray,
    tel_alt: np.ndarray,
    tel_az: np.ndarray) -> np.ndarray:
    """Determine if a satellite's path along the image plane intersects the telescope's focal plane.

    This function checks whether the satellite comes within the radius of the focal
    plane by calculating the closest approach to the center of the telescope.

    Parameters
    ----------
    - xs (np.ndarray): Array of x-coordinates of the satellite's path on the image plane
        (floats). The function altaz_to_image converts the satellite's altaz coordinates
        to x-y coordinates.
    - ys (np.ndarray): Array of y-coordinates of the satellite's path on the image plane
        (floats). The function altaz_to_image converts the satellite's altaz coordinates
        to x-y coordinates.
    - tel_alt (np.ndarray): Array of telescope altitude angles in degrees (floats).
    - tel_az (np.ndarray): Array of telescope azimuth angles in degrees (floats).

    Returns
    -------
    - np.ndarray: returns the satellite position and time when it is the closest to the
    center of the telescope's focal plane. The numpy array's columns are:
        1. x-position of the satellite when its the closest to the center of the
            telscope's focal plane.
        2. y-position of the satellite when its the closest to the center of the
            telscope's focal plane.
        3. time of the satellite when its the closest to the center of the
            telscope's focal plane.
        4. Telescope altitude
        5. Telescope azimuth

    """
    tel_alt = np.array(tel_alt)
    tel_az = np.array(tel_az)
    closest_x = np.zeros(3)
    closest_y = np.zeros(3)
    closest_t = np.zeros(3)
    for i in range(len(closest_x)):
        a_x = xs[i]
        a_y = ys[i]
        b_x = xs[i + 1]
        b_y = ys[i + 1]

        t = (a_x**2 - a_x * b_x + a_y**2 - a_y * b_y) / (
            (a_x - b_x) ** 2 + (a_y - b_y) ** 2
        )

        if (t < 0) or (t > 1):
            r_a = a_x**2 + a_y**2
            r_b = b_x**2 + b_y**2

            if r_a < r_b:
                closest_x[i] = a_x
                closest_y[i] = a_y
                closest_t[i] = 5 * i
            else:
                closest_x[i] = b_x
                closest_y[i] = b_y
                closest_t[i] = 5 * i + 5
            continue

        closest_x[i] = a_x + (b_x - a_x) * t
        closest_y[i] = a_y + (b_y - a_y) * t
        closest_t[i] = 5 * i + t * 5

    rs = closest_x**2 + closest_y**2
    closest_index = np.argmin(rs)

    return (
        closest_x[closest_index],
        closest_y[closest_index],
        closest_t[closest_index],
        tel_alt[closest_index],
        tel_az[closest_index],
    )


def altaz_to_radec(
    altitude: float,
    azimuth: float,
    time: astropy.time.Time,
    location: astropy.coordinates.EarthLocation,
) -> tuple[float, float]:
    """Convert altitude and azimuth to right ascension and declination.

    Parameters
    ----------
        altitude (float) : Altitude in HCS frame (degrees)
        azimuth (float) : Azimuth in HCS frame (degrees)
        time (astropy.time.Time) : Time of conversion
        location (astropy.coordinates.EarthLocation) : Location of conversion

    Returns
    -------
        right_ascension, declination (float) : RA and DEC (degrees)

    """
    aa = astropy.coordinates.AltAz(
        az=azimuth * astropy.units.degree,
        alt=altitude * astropy.units.degree,
        location=location,
        obstime=time,
    )

    coord = astropy.coordinates.SkyCoord(aa.transform_to(astropy.coordinates.ICRS()))

    return coord.ra.deg, coord.dec.deg


def altaz_to_image(
    sat_alt: float,
    sat_az: float,
    tel_alt: float,
    tel_az: float) -> (float, float, float):
    """Convert satellite altaz coordinates to image plane coordinates as seen by the telescope.

    Parameters
    ----------
        sat_alt (float): Satellite altitude
        sat_az (float): Satellite azimuth
        tel_alt (float): Telescope altitude
        tel_az (float): Telescope azimuth

    Returns
    -------
        Returns the x,y,z coordinates of the satellite in the image plane coordinates.

    """
    sat_x, sat_y, sat_z = lumos.conversions.altaz_to_unit(sat_alt, sat_az)

    rot_x, rot_y, rot_z = lumos.functions.Rz(np.deg2rad(-tel_az), sat_x, sat_y, sat_z)
    return lumos.functions.Ry(np.deg2rad(tel_alt) - np.pi / 2, rot_x, rot_y, rot_z)


def radec_to_altaz(
    ra: float,
    dec: float,
    time: float,
    location: float) -> tuple[float, float]:
    """Convert RA and DEC to altitude and azimuth.

    Parameters
    ----------
        RA (float) : Right ascension (degrees)
        DEC (float) : Declination (degrees)
        time (astropy.time.Time) : Time of conversion
        location (astropy.coordinates.EarthLocation) : Location of conversion

    Returns
    -------
        right_ascension, declination (float) : altitude and azimuth (degrees)

    """
    coord = astropy.coordinates.SkyCoord(ra=ra, dec=dec, unit=astropy.units.degree)
    aa = astropy.coordinates.AltAz(location=location, obstime=time)
    coord_altaz = coord.transform_to(aa)

    return coord_altaz.alt.degree, coord_altaz.az.degree
