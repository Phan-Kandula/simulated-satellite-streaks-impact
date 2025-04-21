""" Library of Satellite Constellations """

import numpy as np
import sgp4.api
import astropy.time
import astropy.units
import astropy.coordinates
import matplotlib.pyplot as plt
import lumos.constants

EARTH_MASS = 5.97219 * (10 ** 24)
G = 6.6743 * (10 ** -11)
MU_EARTH = 3.986e14  # meters ^ 3 / seconds ^ 2


class Constellation:
    """
    Base class for all constellations
    """

    def get_teme_position(self, time):
        """
        Gets positions of all satellites in TEME coordinate frame

        :param time: Time at which to get positions
        :type time: :class:`astropy.time.Time`
        :return: (x, y, z) position measured in meters
        :rtype: tuple[:class:`np.ndarray]
        """

        jd1 = np.array([time.jd1])
        jd2 = np.array([time.jd2])

        _, teme_pos, _ = self.constellation.sgp4(jd1, jd2)
        teme_pos = teme_pos[:, 0, :]
        x, y, z = teme_pos[:, 0], teme_pos[:, 1], teme_pos[:, 2]

        return x * 1000, y * 1000, z * 1000

    def get_hcs_position(self, time, earth_location):
        """
        Gets position of all satellites in constellation in HCS frame

        :param time: Time at which to get positions
        :type time: :class:`astropy.time.Time`
        :param earth_location: Location at which to get positions in HCS frame
        :type earth_location: :class:`astropy.coordinates.EarthLocation`
        :return: altitudes (degrees), azimuths (degrees), heights (meters)
        :rtype: tuple[:class:`np.ndarray`]
        """

        x, y, z = self.get_teme_position(time)
        teme = astropy.coordinates.TEME(
            x=x * astropy.units.meter,
            y=y * astropy.units.meter,
            z=z * astropy.units.meter,
            representation_type='cartesian',
            obstime=time)

        itrs_geo = teme.transform_to(astropy.coordinates.ITRS(obstime=time))
        topo_itrs_repr = itrs_geo.cartesian.without_differentials() \
            - earth_location.get_itrs(time).cartesian
        itrs_topo = astropy.coordinates.ITRS(
            topo_itrs_repr, obstime=time, location=earth_location)
        aa = itrs_topo.transform_to(
            astropy.coordinates.AltAz(obstime=time,
                                      location=earth_location,
                                      pressure=0)
        )

        heights = itrs_geo.earth_location.geodetic.height.value

        return aa.alt.degree, aa.az.degree, heights

    def plot_teme(self, ax, time):
        """
        Plots satellite constellation in TEME coordinate frame

        :param ax: Axis for plotting onto (must be 3D projection)
        :type ax: :class:`matplotlib.pyplot.Axes`
        :param time: Time at which to plot constellation
        :type time: :class:`astropy.time.Time`
        """

        x, y, z = self.get_teme_position(time)
        x, y, z = x / 1000, y / 1000, z / 1000

        ax.scatter(x, y, z, s=1, alpha=0.5)
        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
        ax.xaxis.set_pane_color((1, 1, 1, 0))
        ax.yaxis.set_pane_color((1, 1, 1, 0))
        ax.zaxis.set_pane_color((1, 1, 1, 0))
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_zlabel("z (km)")

    def plot_hcs(self, ax, time, location):
        """
        Plots satellite constellation in HCS coordinate frame

        :param ax: Axis for plotting onto (must be polar projection)
        :type ax: :class:`matplotlib.pyplot.Axes`
        :param time: Time at which to plot constellation
        :type time: :class:`astropy.time.Time`
        :param location: Location on Earth at which to plot constellation
        :type location: :class:`astropy.coordinates.EarthLocation`
        """

        altitudes, azimuths, _ = self.get_hcs_position(time, location)
        ax.plot(np.deg2rad(azimuths), 90 - altitudes, '.')
        ax.set_rmax(90)
        ax.set_yticklabels([])
        ax.set_theta_zero_location('N')
        ax.set_rticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_xticklabels(['N', 'E', 'S', 'W'])


class ConstellationFromTLE(Constellation):
    """
    This class builds a constellation from a TLE file

    :param tle_file_path: Path to a TLE text file
    :type tle_file_path: str
    """

    def __init__(self, tle_file_path):
        """
        Constructor Method
        """

        with open(tle_file_path, 'r') as file:
            lines = file.read().splitlines()

        TLES = [(l0, l1, l2)
                for l0, l1, l2 in zip(lines[::3], lines[1::3], lines[2::3])]

        satellite_ids = []
        constellation = []

        for tle in TLES:
            satellite_ids.append(tle[0].strip())
            constellation.append(
                sgp4.api.Satrec.twoline2rv(tle[1], tle[2])
            )

        self.constellation = sgp4.api.SatrecArray(constellation)
        self.satellite_ids = tuple(satellite_ids)
        self.constellation_size = len(self.satellite_ids)


class WalkerConstellation(Constellation):
    def __init__(self,
                 height: float,
                 inclination: float,
                 totalSatellites: int,
                 planes: int,
                 phase: int,
                 t0: astropy.time.Time):
        """
        This class builds a Walker constellation

        :param height: Distance from the surface of earth to the satellites (meters)
        :type height: float
        :param inclination: Inclination of the constellation (degrees)
        :param totalSatellites: Total number of satellites in the constellation
        :param planes: Total number of geometry planes
        :param phase: Phase difference between satellites in adjcent planes
        """

        self.radius = lumos.constants.EARTH_RADIUS + height
        self.inclination = inclination
        self.totalSatellites = totalSatellites
        self.planes = planes
        self.planeSatellites = int(totalSatellites / planes)
        self.phase = phase
        # self.meanMotion = np.sqrt(MU_EARTH * 3600 * (self.radius)**(-3))
        self.sat_index, self.plane_index = np.meshgrid(np.linspace(0, self.planeSatellites - 1, self.planeSatellites),
                                                       np.linspace(0, self.planes - 1, self.planes))
        self.t0 = t0
        self.period = 2 * np.pi * (self.radius)**(1.5) / np.sqrt(MU_EARTH)

    def get_teme_position(self, t: int | astropy.time.core.Time):
        """
        Returns the position of the satellites in the walker constellation

        :param time: time at which to get position
        :type time: astropy.time.Time
        :return: (x, y, z) position of all satellites meters
        :rtype: tuple[:class:`np.ndarray]
        """
        if type(t) == astropy.time.core.Time:
            time = t.unix - self.t0
        else:
            time = t - self.t0
        """
        phis = 2 * np.pi * ((time / self.period) + (self.sat_index / self.planeSatellites) + \ 
            ((self.plane_index * self.phase) / (self.totalSatellites)))
        """
        phis = 2 * np.pi * ((time / self.period) + (self.sat_index / self.planeSatellites) +
                            (self.plane_index * self.phase / (self.totalSatellites * self.planes)))
        alpha = 2 * np.pi * (self.plane_index) / self.planes

        x = np.cos(alpha) * np.cos(phis) - np.sin(alpha) * \
            np.cos(np.deg2rad(self.inclination)) * np.sin(phis)
        y = np.sin(alpha) * np.cos(phis) + np.cos(alpha) * \
            np.cos(np.deg2rad(self.inclination)) * np.sin(phis)
        z = np.sin(np.deg2rad(self.inclination)) * np.sin(phis)

        x = x * self.radius
        y = y * self.radius
        z = z * self.radius

        return x.flatten(), y.flatten(), z.flatten()

    def mean_motion(self):
        """
        returns the mean motion of satellite in degrees
        """

#        return self.meanMotion

        return 360/self.period

class FrozenWalkerConstellation(Constellation):
    def __init__(self,
                 height: float,
                 inclination: float,
                 totalSatellites: int,
                 planes: int,
                 phase: int,
                 t0: astropy.time.Time,
                 a: float,
                 c: float):
        """
        This class builds a Walker constellation

        :param height: Distance from the surface of earth to the satellites (meters)
        :type height: float
        :param inclination: Inclination of the constellation (degrees)
        :param totalSatellites: Total number of satellites in the constellation
        :param planes: Total number of geometry planes
        :param phase: Phase difference between satellites in adjcent planes
        """

        self.radius = lumos.constants.EARTH_RADIUS + height
        self.inclination = inclination
        self.totalSatellites = totalSatellites
        self.planes = planes
        self.planeSatellites = int(totalSatellites / planes)
        self.phase = phase
        # self.meanMotion = np.sqrt(MU_EARTH * 3600 * (self.radius)**(-3))
        self.sat_index, self.plane_index = np.meshgrid(np.linspace(0, self.planeSatellites - 1, self.planeSatellites),
                                                       np.linspace(0, self.planes - 1, self.planes))
        self.t0 = t0
        self.period = 2 * np.pi * (self.radius)**(1.5) / np.sqrt(MU_EARTH)
        self.a = a + height
        self.c = c + height

    def get_teme_position(self, t: int | astropy.time.core.Time):
        """
        Returns the position of the satellites in the walker constellation

        :param time: time at which to get position
        :type time: astropy.time.Time
        :return: (x, y, z) position of all satellites meters
        :rtype: tuple[:class:`np.ndarray]
        """
        if type(t) == astropy.time.core.Time:
            time = t.unix - self.t0
        else:
            time = t - self.t0
        """
        phis = 2 * np.pi * ((time / self.period) + (self.sat_index / self.planeSatellites) + \ 
            ((self.plane_index * self.phase) / (self.totalSatellites)))
        """
        phis = 2 * np.pi * ((time / self.period) + (self.sat_index / self.planeSatellites) +
                            (self.plane_index * self.phase / (self.totalSatellites * self.planes)))
        alpha = 2 * np.pi * (self.plane_index) / self.planes

        x = np.cos(alpha) * np.cos(phis) - np.sin(alpha) * \
            np.cos(np.deg2rad(self.inclination)) * np.sin(phis)
        y = np.sin(alpha) * np.cos(phis) + np.cos(alpha) * \
            np.cos(np.deg2rad(self.inclination)) * np.sin(phis)
        z = np.sin(np.deg2rad(self.inclination)) * np.sin(phis)

        x = x * self.a
        y = y * self.a
        z = z * self.c

        return x.flatten(), y.flatten(), z.flatten()

    def mean_motion(self):
        """
        returns the mean motion of satellite in degrees
        """

#        return self.meanMotion

        return 360/self.period


class ConstellationGroup(Constellation):
    def __init__(self, constellations):
        """
        A group of constellations.

        :param constellations: A list of constellations
        :type constellations: python list
        """
        self.constellations = constellations
        self.size = 0
        for i in range(len(self.constellations)):
            self.size += self.constellations[i].totalSatellites
        self.satellite_ids = np.arange(0, self.size, 1)

    def get_teme_position(self, time):
        """Returns the position of the satellites in the constellation in teme coordinates

        :param time: time at which to get position
        :type time: astropy.time.Time
        :return: (x, y, z) position of all satellites meters
        :rtype: tuple[:class:`np.ndarray]
        """
        x = np.zeros(0)
        y = np.zeros(0)
        z = np.zeros(0)
        for constellation in self.constellations:
            teme = constellation.get_teme_position(time)
            # print(teme[0].shape)
            x = np.append(x, teme[0])
            y = np.append(y, teme[1])
            z = np.append(z, teme[2])
        return x, y, z

    def get_hcs_position2(self, time, location):
        """
        Returns the position of the satellites in the constellation in teme coordinates

        :param time: time at which to get position
        :type time: astropy.time.Time
        :param location: Location of the telescope
        :type location: :class:`astropy.coordinates.EarthLocation`
        :return: (x, y, z) position of all satellites meters
        :rtype: tuple[:class:`np.ndarray]
        """
        alt = np.zeros(0)
        az = np.zeros(0)
        height = np.zeros(0)
        for constellation in self.constellations:
            hcs = constellation.get_hcs_position(time, location)
            alt = np.append(alt, hcs[0])
            az = np.append(az, hcs[1])
            height = np.append(height, hcs[2])

        return alt, az, height

    def teme_across_time(self, start, end, images):
        """Returns the position of the satellites in the constellation in
        teme coordinates over a time interval.

        :param start: start time at which to get position
        :type start: astropy.time.Time
        :param end: stop time at which to get position
        :type end: astropy.time.Time
        :param images: Number of time instances within the interval where position is calculated
        :type images: int
        :return: [(x, y, z)] position of all satellites meters. 
            - Each column represents a different time in the interval.
            - Each row is for a different satellite.
        :rtype: tuple[:class:`np.ndarray]
        """
        times = np.linspace(start, end, images)
        x = np.zeros((self.size, len(times)))
        y = np.zeros((self.size, len(times)))
        z = np.zeros((self.size, len(times)))
        for i in range(len(times)):
            teme = self.get_teme_position(times[i])
            x[:, i] = teme[0]
            y[:, i] = teme[1]
            z[:, i] = teme[2]

        return x, y, z

    def hcs_across_time(self, start, end, images, earth_location):
        """Returns the position of the satellites in the constellation in
        hcs coordinates over a time interval.

        :param start: start time at which to get position
        :type start: astropy.time.Time
        :param end: stop time at which to get position
        :type end: astropy.time.Time
        :param images: Number of time instances within the interval where position is calculated
        :type images: int
        :param location: Location of the telescope
        :type location: :class:`astropy.coordinates.EarthLocation`
        :return: [(x, y, z)] position of all satellites meters. 
            - Each column represents a different time in the interval.
            - Each row is for a different satellite.
        :rtype: tuple[:class:`np.ndarray]
        """
        times = np.linspace(start, end, images)
        alts = np.zeros((self.size, len(times)))
        azs = np.zeros((self.size, len(times)))
        heights = np.zeros((self.size, len(times)))
        for i in range(len(times)):
            hcs = self.get_hcs_position(
                time=times[i], earth_location=earth_location)
            alts[:, i] = hcs[0]
            azs[:, i] = hcs[1]
            heights[:, i] = hcs[2]

        return alts, azs, heights

    def mean_motion(self):
        """
        returns the mean motion of satellite in degrees
        """
        mean_motion = np.zeros(0)
        for i, constellation in enumerate(self.constellations):
            mean_motion = np.append(mean_motion, np.full(
                constellation.totalSatellites, constellation.mean_motion()))
        return mean_motion
