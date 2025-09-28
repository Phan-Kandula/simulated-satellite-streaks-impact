import lumos
import lumos.conversions
import lumos.functions
import numpy as np
import astropy.time
import astropy.units
import astropy.coordinates
import lumos.constants


def altaz_to_image(
    sat_alt: float | np.ndarray,
    sat_az: float | np.ndarray,
    tel_alt: float | np.ndarray,
    tel_az: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts satellite altitude and azimuth coordinates into the telescope image frame.

    This function takes the satellite's altitude and azimuth coordinates along with the
    telescope's pointing direction (altitude and azimuth) and computes the satellite's
    position in the telescope's image frame.

    Parameters:
        sat_alt (float or np.ndarray): Altitude(s) of the satellite(s) in degrees.
        sat_az (float or np.ndarray): Azimuth(s) of the satellite(s) in degrees.
        tel_alt (float or np.ndarray): Altitude of the telescope's pointing direction in degrees.
        tel_az (float or np.ndarray): Azimuth of the telescope's pointing direction in degrees.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Cartesian coordinates of the satellite(s)
        in the telescope's image frame:
            - x-coordinate(s)
            - y-coordinate(s)
            - z-coordinate(s)
    """
    sat_x, sat_y, sat_z = lumos.conversions.altaz_to_unit(sat_alt, sat_az)

    rot_x, rot_y, rot_z = lumos.functions.Rz(np.deg2rad(-tel_az), sat_x, sat_y, sat_z)
    return lumos.functions.Ry(np.deg2rad(tel_alt) - np.pi / 2, rot_x, rot_y, rot_z)


def unit_to_altaz(x, y, z):
    """
    Converts vectors to altaz coordinates. It is an inversion of lumos.conversion.altaz_to_unit

    Parameters:
        x (float or np.ndarray): x-coordinate of the vector
        y (float or np.ndarray): y-coordinate of the vector
        z (float or np.ndarray): z-coordinate of the vector

    Returns:
        tuple[np.ndarray, np.ndarray]: Altitude and azimuth conversion
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


def image_to_altaz(x, y, z, tel_alt, tel_az):
    """
    Converts from image coordinates(as viewed on the focal plane) of the satellite to altaz coordinates.

    Parameters:
        x (float or np.ndarray): x-coordinate of the vector
        y (float or np.ndarray): y-coordinate of the vector
        z (float or np.ndarray): z-coordinate of the vector
        tel_alt (float or np.ndarray): The telescope's pointing altitude
        tel_az (float or np.ndarray): The telescope's pointing azimuth

    Returns:
        tuple[np.ndarray, np.ndarray]: Altitude and azimuth of the satellite in the image
    """

    x, y, z = lumos.functions.Ry(np.pi / 2 - np.deg2rad(tel_alt), x, y, z)
    sat_x, sat_y, sat_z = lumos.functions.Rz(np.deg2rad(tel_az), x, y, z)

    sat_alt, sat_az = unit_to_altaz(sat_x, sat_y, sat_z)

    return sat_alt, sat_az


def altaz_to_radec(
    altitude: float,
    azimuth: float,
    time: astropy.time.Time,
    location: astropy.coordinates.EarthLocation,
) -> tuple[float, float]:
    """
    Converts altitude and azimuth to right ascension and declination

    Parameters:
        altitude (float) : Altitude in HCS frame (degrees)
        azimuth (float) : Azimuth in HCS frame (degrees)
        time (astropy.time.Time) : Time of conversion
        location (astropy.coordinates.EarthLocation) : Location of conversion

    Returns:
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


def radec_to_altaz(ra, dec, time, location) -> tuple[float, float]:
    """
    Converts RA and DEC to altitude and azimuth

    Parameters:
        RA (float) : Right ascension (degrees)
        DEC (float) : Declination (degrees)
        time (astropy.time.Time) : Time of conversion
        location (astropy.coordinates.EarthLocation) : Location of conversion

    Returns:
        right_ascension, declination (float) : altitude and azimuth (degrees)
    """
    coord = astropy.coordinates.SkyCoord(ra=ra, dec=dec, unit=astropy.units.degree)
    aa = astropy.coordinates.AltAz(location=location, obstime=time)
    coord_altaz = coord.transform_to(aa)

    return coord_altaz.alt.degree, coord_altaz.az.degree


def to_hcs(time, earth_location, x, y, z):
    teme = astropy.coordinates.TEME(
        x=x * astropy.units.meter,
        y=y * astropy.units.meter,
        z=z * astropy.units.meter,
        representation_type="cartesian",
        obstime=time,
    )

    itrs_geo = teme.transform_to(astropy.coordinates.ITRS(obstime=time))
    topo_itrs_repr = (
        itrs_geo.cartesian.without_differentials()
        - earth_location.get_itrs(time).cartesian
    )
    itrs_topo = astropy.coordinates.ITRS(
        topo_itrs_repr, obstime=time, location=earth_location
    )
    aa = itrs_topo.transform_to(
        astropy.coordinates.AltAz(obstime=time, location=earth_location, pressure=0)
    )

    heights = itrs_geo.earth_location.geodetic.height.value

    return aa.alt.degree, aa.az.degree, heights
