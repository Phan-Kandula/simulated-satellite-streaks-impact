"""Creates a database of Starlink satellites that enter the telescope's focal plane.

This script simulates Starlink satellite streaks using the constellation's parameters
to create a database of satellites that traverse the region of the sky that is observed
by the LSST.

Usage:
The main section of this file contains the Starlink walker constellation configuration.
For the purpose of our work, we use the walker constellation described in the Starlink
white paper. If a different satellite constellation is necessary, the walker
constellation config can be changed, or a new custom constellation class can be
implemented in 'library.py'. The script in the 'data' folder must be executed first to
download and format the scheduler database to simulate streaks. The schedule is broken
into monthly chunks to make it more organizable and easier to save data. This script then
iterates through each month, computes satellite streaks,and saves the results in the
`data/results` folder.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import astropy
import astropy.units as u
import functions
import lumos
import numpy as np
import pandas as pd
from library import ConstellationGroup, WalkerConstellation
from tqdm import tqdm

MU_EARTH = 3.986e14  # Gravitational earth constant (meters ^ 3 / seconds ^ 2)


def is_in(  # noqa: D417
    xs: np.ndarray,
    ys: np.ndarray,
    tel_alt: np.ndarray,
    tel_az: np.ndarray) -> bool:
    """Determine if a satellite's path intersects the telescope's field of view.

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
    - bool: True if the satellite intersects the focal plane, otherwise False.

    """
    tel_alt = np.array(tel_alt)
    tel_az = np.array(tel_az)

    # The closest arrays contain the position and time of the satellite when it is the
    # closest to the center of the telescope focal plane frame for each 5 second time
    # interval.
    closest_x = np.zeros(3)
    closest_y = np.zeros(3)
    closest_t = np.zeros(3)
    for i in range(len(closest_x)):
        a_x = xs[i]
        a_y = ys[i]
        b_x = xs[i + 1]
        b_y = ys[i + 1]

        # Calculate the time at which satellite's position in the image frame is closest
        # to the center of the focal plane. This is calculated through finding the
        # extrema of distance from satellite position to center of the telescope's focal
        # plane as a function of time.
        t = (a_x**2 - a_x * b_x + a_y**2 - a_y * b_y) / (
            (a_x - b_x) ** 2 + (a_y - b_y) ** 2
        )

        # Checking if t is within the 5 second time interval chunk, and if it is not
        # check if the edges at the interval are the extrema.
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

        # Add to closest array
        closest_x[i] = a_x + (b_x - a_x) * t
        closest_y[i] = a_y + (b_y - a_y) * t
        closest_t[i] = 5 * i + t * 5

    rs = closest_x**2 + closest_y**2
    closest_index = np.argmin(rs)
    return np.arcsin(rs[closest_index]) < np.deg2rad(1.75)


def show_sky(
    start_time: astropy.time.core.Time,
    tel_ra: float,
    tel_dec: float,
    location: astropy.coordinates.earth.EarthLocation,
    group: ConstellationGroup,
    hour: float,
    night: float,
    tel_index: float,
) -> np.ndarray:
    """Determine which satellites are in the telescope's focal plane.

    This function computes the position of all satellites in the constellation to
    check which satellites would go through the telescope's focal plane given the
    telescope's pointing direction and time.

    The satellite positions are checked at 5-second intervals over the 15-second
    obsrevation window. By accounting for the speed of the satellite, we can calculate
    if the satellite will be in the focal plane of the telescope withing those 5 seconds
    before the next frame where the satellite's position will be calculated.
    The function returns a numpy array table of 4 rows corresponding the each 5 second
    interval within the 15-second observation window with information about satellites
    that enter the focal plane.

    Parameters
    ----------
    - start_time (astropy.time.core.Time): The time at which the observation started.
    - tel_ra (float): Telescope right ascension
    - tel_dec (float): Telescope declination
    - location (astropy.coordinates.earth.EarthLocation): Telescope's location on
        earth coordinates.
    - group (ConstellationGroup): Satellite constellation group object. If its a single
        constellation, a ConstellationGroup object can be made with just one
        constellation.
    - hour (float): The hour at which this observation takes place. This isn't used for
        calculations but rather used for the table returned at the end.
    - night (float): The night at which this observation takes place. This isn't used
        for calculations but rather used for the table returned at the end.
    - tel_index (float): The telescope observation index for logging purposes of the
        table that is returned.

    Returns
    -------
    np.ndarray
        A table containing satellite observation data with the following columns:

        1. Satellite ID
        2. Satellite elevation (degrees)
        3. Satellite azimuth (degrees)
        4. Satellite height (meters)
        5. Time of observation (Modified Julian Date)
        6. Telescope right ascension (degrees)
        7. Telescope declination (degrees)
        8. Telescope altitude (degrees)
        9. Telescope azimuth (degrees)
        10. Boolean indicating if
        11. Hour of observation
        12. Night of observation
        13. Telescope observation index

    """
    images = 4
    stop_time = start_time + 15 * u.second
    times = np.linspace(start_time, stop_time, images)
    alts, azs, heights = group.hcs_across_time(start_time, stop_time, images, location)
    observing_ids = set([])
    ins = np.zeros([len(group.satellite_ids), images])
    dt = 5
    tel_alts = np.zeros(0)
    tel_azs = np.zeros(0)
    # Each loop calculates satellite positions at 5 second intervals if we take 4 images.
    for i in range(images):
        tel_alt, tel_az = functions.radec_to_altaz(tel_ra, tel_dec, times[i], location)
        tel_alts = np.append(tel_alts, tel_alt)
        tel_azs = np.append(tel_azs, tel_az)
        alt, az, height = alts[:, i], azs[:, i], heights[:, i]
        ids = group.satellite_ids[alt > 0]
        mean_motion = group.mean_motion()[ids]
        x, y, z = functions.altaz_to_image(alt[ids], az[ids], tel_alt, tel_az)
        ids_close = ids[2 * np.arccos(z) <= np.deg2rad(3.5) + 1.5 * np.deg2rad(mean_motion) * (lumos.constants.EARTH_RADIUS / (lumos.constants.EARTH_RADIUS + height[ids])) * dt]
        ins[ids_close, i] = 1
        observing_ids.update(ids_close)

    observing_ids = np.array(list(observing_ids))
    out_of_ids = []
    for i in range(len(observing_ids)):
        xs, ys, _ = functions.altaz_to_image(
            alts[observing_ids[i]], azs[observing_ids[i]], tel_alt, tel_az
        )
        if not is_in(xs, ys, tel_alt, tel_az):
            out_of_ids.append(i)
    observing_ids = np.delete(observing_ids, out_of_ids)

    table = np.zeros([len(observing_ids) * images, 13])
    for i in range(len(observing_ids)):
        table[i * images : (i + 1) * images, 0] = observing_ids[i]
        table[i * images : (i + 1) * images, 1] = alts[observing_ids[i]]
        table[i * images : (i + 1) * images, 2] = azs[observing_ids[i]]
        table[i * images : (i + 1) * images, 3] = heights[observing_ids[i]]
        table[i * images : (i + 1) * images, 4] = times.mjd
        table[i * images : (i + 1) * images, 5] = tel_ra
        table[i * images : (i + 1) * images, 6] = tel_dec
        table[i * images : (i + 1) * images, 7] = tel_alts
        table[i * images : (i + 1) * images, 8] = tel_azs
        table[i * images : (i + 1) * images, 9] = ins[observing_ids[i]]
        table[i * images : (i + 1) * images, 10] = hour
        table[i * images : (i + 1) * images, 11] = night
        table[i * images : (i + 1) * images, 12] = tel_index

    return table


def make_streak_file(
    df: pd.DataFrame,
    month: int,
    year: int,
    height: float):
    """Create and saves files containing streaks.

    Parameters
    ----------
    - df (pd.DataFrame): Scheduler database in csv format.
    - month (int) : Current month being worked on
    - year (int) : Current year being worked on
    - height (float): height of the satellites in km

    """
    times = astropy.time.Time(df["observationStartMJD"], format="mjd")

    table = np.empty((0, 13), int)

    print("month: " + str(month))  # noqa: T201

    num_cores = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_cores) as executor:  # noqa: SIM117
        with tqdm(total=len(df)) as progress:
            futures = []

            for i in range(len(df)):
                future = executor.submit(
                    show_sky,
                    times[i],
                    df["fieldRA"][i],
                    df["fieldDec"][i],
                    loc,
                    group,
                    df["hours"][i],
                    df["night"][i],
                    df["observationId"][i],
                )
                futures.append(future)

            for f in as_completed(futures):
                progress.update()
                table = np.append(table, f.result(), axis=0)

    table = pd.DataFrame(
        table,
        columns=[
            "id",
            "alts",
            "azs",
            "heights",
            "times",
            "tel_ra",
            "tel_dec",
            "tel_alt",
            "tel_az",
            "ins",
            "hours",
            "night",
            "tel_index",
        ],
    )

    name = f"../data/results/year_{year}_streaks_{month}_{height}km.csv"
    table.to_csv(name, index=False)


if __name__ == "__main__":

    # Starlink constellation configuration taken from the white paper. Each element
    # represents a singular walker constellation. Together they make up a group of
    # walker constellations.
    alts = [350, 350, 350, 350, 350, 350, 350, 350, 350]
    inc = [53, 46, 38, 96.9, 53, 43, 33, 148, 115.7]
    planes = [48, 48, 48, 30, 28, 28, 28, 12, 18]
    sats = [110, 110, 110, 120, 120, 120, 120, 12, 18]
    walkers = []

    # number of Satellite positions to collect during the interval before confirming if
    # the satellite goes through the telescope's focal plane.
    images = 4

    # Creating the group constellation that is just a set of walker constellations.
    for i in range(len(alts)):
        walkers.append(  # noqa: PERF401
            WalkerConstellation(
                alts[i] * 1000,
                inc[i],
                sats[i] * planes[i],
                planes[i],
                1,
                0  # noqa: COM812
            )  # noqa: COM812
        )
    group = ConstellationGroup(walkers)

    # Location of the LSST.
    loc = astropy.coordinates.EarthLocation(
        lon=-70.7494 * u.deg, lat=-30.2444 * u.deg, height=2650.0 * u.m
    )
    # A random time to start the constellation.
    t0 = astropy.time.Time(60218.031143, format="mjd")

    # uncomment the next two lines for calculating the streaks for all 10 years 
    """
    for year in range(10):
        for month in range(1, 13):
            scheduler = pd.read_csv(f"../data/scheduler_year{year}/scheduler_month\
                _{month}.csv")
            make_streak_file(scheduler, month, year, 550)
    """

    # uncomment the next two lines for calculating the streaks for just the summer

    for year in range(3):
        scheduler = pd.read_csv(f"../data/summer_scheduler_year{year}/summer_schedule.csv")
        make_streak_file(scheduler, "summer", year, 350)

