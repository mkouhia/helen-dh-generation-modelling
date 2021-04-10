import logging
from pathlib import Path

import pandas as pd
import requests
from pandas import DataFrame, DatetimeIndex

from dh_modelling.data import intermediate_data_path, raw_data_path

HELEN_DATA_URL = (
    "https://www.helen.fi/globalassets/helen-oy/vastuullisuus/hki_dh_2015_2020_a.csv"
)
HELEN_DATA_FILENAME = "hki_dh_2015_2020_a.csv"
HELEN_INTERMEDIATE_FILENAME = "helen_cleaned.feather"


def get_and_save(
    url: str = HELEN_DATA_URL, raw_file_path: Path = raw_data_path / HELEN_DATA_FILENAME
):
    """
    Get DH data CSV, save to file

    :param url: online location for data
    :param raw_file_path: local file path for raw data
    """
    logging.info(f"Downloading Helen DH data from {HELEN_DATA_URL} to {raw_file_path}")
    r = requests.get(url)
    with raw_file_path.open("wb") as f:
        f.write(r.content)


def load_and_clean(
    raw_file_path: Path = raw_data_path / HELEN_DATA_FILENAME,
) -> DataFrame:
    """
    Load dataset from disk, clean up features

    :param raw_file_path: local file path for raw data
    :return: Pandas dataframe, with index column 'date_time' and feature column 'dh_MWh'
    """
    logging.info("Load Helen raw dataset from CSV, clean up file")
    df = pd.read_csv(
        raw_file_path, sep=";", decimal=",", parse_dates=["date_time"], dayfirst=True
    ).set_index("date_time")
    df.index = df.index.tz_localize(tz="Europe/Helsinki", ambiguous="infer")
    return df


def save_intermediate(
    df: DataFrame,
    intermediate_file_path: Path = intermediate_data_path / HELEN_INTERMEDIATE_FILENAME,
):
    """
    Save intermediate representation of dataframe to disk

    :param df: dataframe to be saved
    :param intermediate_file_path: intermediate file location
    """
    logging.info(f"Save cleaned intermediate Helen dataset to {intermediate_file_path}")
    df.reset_index().to_feather(intermediate_file_path)


def load_intermediate(
    intermediate_file_path: Path = intermediate_data_path / HELEN_INTERMEDIATE_FILENAME,
) -> DataFrame:
    """
    Load pre-saved intermediate file from disk

    :param intermediate_file_path: intermediate file location
    :return: loaded dataframe, with 'date_time' as index
    """
    df: DataFrame = pd.read_feather(intermediate_file_path)
    df.index = DatetimeIndex(df["date_time"]).tz_convert("Europe/Helsinki")
    df = df.drop("date_time", axis=1)
    return df


def get_cleaned_data(
    url: str = HELEN_DATA_URL,
    raw_file_path: Path = raw_data_path / HELEN_DATA_FILENAME,
    intermediate_file_path: Path = intermediate_data_path / HELEN_INTERMEDIATE_FILENAME,
) -> DataFrame:
    """
    Get cleaned district heat generation data

    Try to load intermediate data from disk. If it does not exist, download raw data and clean it.

    :param url: online location for data
    :param raw_file_path: local file path for raw data
    :param intermediate_file_path: intermediate file location
    :return: Cleaned district heat generation dataframe
    """
    logging.info("Verify that Helen raw & intermediate data exists")

    if intermediate_file_path.exists():
        return load_intermediate(intermediate_file_path=intermediate_file_path)

    else:

        if not raw_file_path.exists():
            get_and_save(url=url, raw_file_path=raw_file_path)

        df: DataFrame = load_and_clean(raw_file_path=raw_file_path)
        save_intermediate(df, intermediate_file_path=intermediate_file_path)
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    get_cleaned_data()
