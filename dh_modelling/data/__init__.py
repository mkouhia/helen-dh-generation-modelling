import logging
from pathlib import Path

import pandas as pd
from pandas import DataFrame, DatetimeIndex

raw_data_path = (Path(__file__) / "../../../data/raw").resolve()
intermediate_data_path = (Path(__file__) / "../../../data/intermediate").resolve()
processed_data_path = (Path(__file__) / "../../../data/processed").resolve()


def save_intermediate(
    df: DataFrame, intermediate_file_path: Path, reset_datetime_index: bool = True
):
    """
    Save intermediate representation of dataframe to disk

    :param df: dataframe to be saved
    :param intermediate_file_path: intermediate file location
    :param reset_datetime_index: reset datetime index to normal column, convert timestamp to UTC
    """
    logging.info(f"Save intermediate dataset to {intermediate_file_path}")
    if reset_datetime_index:
        index_name: str = df.index.name
        df = df.reset_index()
        df[index_name] = DatetimeIndex(df[index_name]).tz_convert("UTC")
    df.to_feather(intermediate_file_path)


def load_intermediate(
    intermediate_file_path: Path,
    set_datetime_index: bool = True,
    date_time_column: str = "date_time",
    timezone: str = "Europe/Helsinki",
) -> DataFrame:
    """
    Load pre-saved intermediate file from disk

    :param intermediate_file_path: intermediate file location
    :param set_datetime_index: set DatetimeIndex from column 'date_time_column' with timezone 'timezone'
    :param date_time_column: column, which should be set as index
    :param timezone: timezone, at which date_time_column is converted
    :return: loaded dataframe, with 'date_time' as index
    """
    df: DataFrame = pd.read_feather(intermediate_file_path)
    if set_datetime_index:
        df.index = DatetimeIndex(df[date_time_column]).tz_convert(timezone)
        df = df.drop("date_time", axis=1)
    return df
