import logging
from pathlib import Path

from pandas import DataFrame, DatetimeIndex, read_feather


def save_intermediate(df: DataFrame, path: Path, reset_datetime_index: bool = True):
    """
    Save intermediate representation of dataframe to disk

    :param df: dataframe to be saved
    :param path: file location
    :param reset_datetime_index: reset datetime index to normal column, convert timestamp to UTC
    """
    logging.info(f"Save dataset to {path}")
    if reset_datetime_index:
        index_name: str = df.index.name
        df = df.reset_index()
        df[index_name] = DatetimeIndex(df[index_name]).tz_convert("UTC")
    df.to_feather(path)


def load_intermediate(
    path: Path,
    set_datetime_index: bool = True,
    date_time_column: str = "date_time",
    timezone: str = "Europe/Helsinki",
) -> DataFrame:
    """
    Load dataset from disk

    :param path: file location
    :param set_datetime_index: set DatetimeIndex from column 'date_time_column' with timezone 'timezone'
    :param date_time_column: column, which should be set as index
    :param timezone: timezone, at which date_time_column is converted
    :return: loaded dataframe, with 'date_time_column' as index
    """
    logging.info(f"Load dataset from {path}")
    df: DataFrame = read_feather(path)
    if set_datetime_index:
        df.index = DatetimeIndex(df[date_time_column]).tz_convert(timezone)
        df = df.drop(date_time_column, axis=1)
    return df
