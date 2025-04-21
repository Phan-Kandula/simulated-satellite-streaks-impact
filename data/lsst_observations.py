"""Formats the LSST scheduler into a structured database for further analysis.

This script converts the scheduler to a csv format and then breaks it down into monthly
chunks. This can help the next script that calculates streak positions by clearing out
the memory after each month. Furthermore, the next script can also use each month as a
check point to save the progress.

Usage:
Install scheduler file from [https://s3df.slac.stanford.edu/data/rubin/sim-data/sims_featureScheduler_runs3.2/baseline/baseline_v3.2_10yrs.db]
and save it to the data folder. Running this script will make a subdirectory for each
year containing scheduler positions for each month. Lines 189 to 192 must be uncommented
for only the summer schedule.
"""
import os
import shutil
import sqlite3

import astropy
import astropy.units as u
import lumos.calculator
import numpy as np
import pandas as pd
import pytz
from tqdm import tqdm


def make_month_csv(
    month_start: list,
    df: pd.DataFrame,
    file_name: str):
    """Split the dataframe into monthly CSV files.

    Parameters
    ----------
    - month_start (list): A list containing the days a new month starts. The list
        doesn't use real days a month starts, but rather splits the year into 12 equal
        parts and finds when a month needs to start for each part.
    - df (pd.DataFrame): Dataframe containing observation data.
    - file_name (str): Prefix for the output CSV file names.

    """
    for i in tqdm(range(1, len(month_start))):
        month = df.loc[df["night"] < month_start[i]]
        month = month.loc[month["night"] >= month_start[i - 1]]
        month.reset_index(drop=True, inplace=True)

        # Location of the LSST.
        loc = astropy.coordinates.EarthLocation(
            lon=-70.7494 * u.deg, lat=-30.2444 * u.deg, height=2650.0 * u.m
        )
        times = astropy.time.Time(month["observationStartMJD"], format="mjd")
        sun_alt, sun_az = lumos.calculator.get_sun_alt_az(times, loc)

        #filter out daytime observations.
        times = np.delete(times, np.where(sun_alt > 0)[0])
        month = month.drop(np.where(sun_alt > 0)[0])
        month.reset_index(drop=True, inplace=True)
        chile_tz = pytz.timezone("Chile/Continental")
        datetimes = times.to_datetime(chile_tz)

        # Vectorize function to return hour of observation
        convertdt = np.vectorize(lambda x: x.hour)
        convertdt_month = np.vectorize(lambda x: x.month)

        month["hours"] = convertdt(datetimes)
        month["month"] = convertdt(datetimes)

        #Save file
        name = file_name + str(i-1) + ".csv"
        month.to_csv(name, index=False)


def create_directory(
    folder_name: str):
    """Create new directories for each month.

    The permission to delete folders is quite powerful and dangerous. It is commented
    out for safety. If you run this script only once, it is nothing to worry about. If
    you run this script multiple times, you will get an error if the folder already
    exists. Either manually delete those folders or uncomment the piece of code that
    deletes it.

    Parameters
    ----------
    - folder_name: name of the folder to create.

    """
    """
    # Check if the directory exists
    if os.path.exists(folder_name):
        # Remove the directory
        shutil.rmtree(folder_name)
        print(f"Directory '{folder_name}' already exists and has been deleted.")
    # Create the new directory
    """
    os.mkdir(folder_name)
    print(f"Directory '{folder_name}' has been created.")


def make_summer_csv(
    df: pd.DataFrame,
    file_name: str):
    """Split the dataframe into just the summer months.

    Parameters
    ----------
    - month_start (list): A list containing the days a new month starts. The list
        doesn't use real days a month starts, but rather splits the year into 12 equal
        parts and finds when a month needs to start for each part.
    - df (pd.DataFrame): Dataframe containing observation data.
    - file_name (str): Prefix for the output CSV file names.

    """
    loc = astropy.coordinates.EarthLocation(
        lon=-70.7494 * u.deg, lat=-30.2444 * u.deg, height=2650.0 * u.m
    )
    times = astropy.time.Time(df["observationStartMJD"], format="mjd")
    sun_alt, sun_az = lumos.calculator.get_sun_alt_az(times, loc)

    times = np.delete(times, np.where(sun_alt > 0)[0])
    df = df.drop(np.where(sun_alt > 0)[0])
    df.reset_index(drop=True, inplace=True)
    chile_tz = pytz.timezone("Chile/Continental")
    datetimes = times.to_datetime(chile_tz)

    # Vectorize function to return hour of observation
    convertdt = np.vectorize(lambda x: x.hour)
    convertdt_month = np.vectorize(lambda x: x.month)


    df["month"] = convertdt_month(datetimes)
    df["hours"] = convertdt(datetimes)
    df = df.loc[(df["month"] == 11) | (df["month"] == 12) | (df["month"] == 1)]

    #Save file
    name = file_name + ".csv"
    df.to_csv(name, index=False)


def create_directory(
    folder_name: str):
    """Create new directories for each month.

    The permission to delete folders is quite powerful and dangerous. It is commented
    out for safety. If you run this script only once, it is nothing to worry about. If
    you run this script multiple times, you will get an error if the folder already
    exists. Either manually delete those folders or uncomment the piece of code that
    deletes it.

    Parameters
    ----------
    - folder_name: name of the folder to create.

    """
    # Check if the directory exists
    if os.path.exists(folder_name):
        # Remove the directory
        shutil.rmtree(folder_name)
        print(f"Directory '{folder_name}' already exists and has been deleted.")
    # Create the new directory

    os.mkdir(folder_name)
    print(f"Directory '{folder_name}' has been created.")

if __name__ == "__main__":
    # connect to database. And read the columns listed
    conn = sqlite3.connect("baseline_v3.2_10yrs.db")

    query = "select observationId,observationStartMJD, night, fieldRA, fieldDec,\
    altitude, azimuth, filter, fiveSigmaDepth from observations"

    obs = pd.read_sql(query, conn)
    conn.close()

    # Process data for each year.
    for year in range(3):
        print(year)
        year_data = obs.loc[
            (obs["night"] >= year * 365) & (obs["night"] <= (year + 1) * 365)
        ]
        year_data = year_data.loc[year_data["filter"] != "u"]
        year_data = year_data.reset_index(drop=True)

        nights = year_data["night"].unique()

        monthdays = [int(i) for i in np.linspace(nights.min(), nights.max() + 1, 13)]

        # Uncomment the next two set of lines for the summer schedule
        """
        create_directory(f"summer_scheduler_year{year}")
        make_summer_csv(year_data, f"summer_scheduler_year{year}/summer_schedule")
        """

        # Comment the next two lines disable formatting the entire 10-year survey
        # database. Useful if you are only interested in the summer and don't want
        # to spend extra time running this script.
        create_directory(f"scheduler_year{year}")
        make_month_csv(monthdays, year_data, f"scheduler_year{year}/scheduler_month_")
