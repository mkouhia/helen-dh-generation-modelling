import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import requests
import yaml
from pandas import DataFrame, DatetimeIndex, concat, merge, read_csv
from pandas._typing import FilePathOrBuffer

HELEN_DATA_URL = (
    "https://www.helen.fi/globalassets/helen-oy/vastuullisuus/hki_dh_2015_2020_a.csv"
)
HELEN_DATA_FILENAME = "hki_dh_2015_2020_a.csv"

HELEN_INTERMEDIATE_FILENAME = "helen_cleaned.feather"
WEATHER_INTERMEDIATE_FILENAME = "weather_{station}.feather"
MASTER_INTERMEDIATE_FILENAME = "master.feather"

raw_data_path = (Path(__file__) / "../../../data/raw").resolve()
intermediate_data_path = (Path(__file__) / "../../../data/intermediate").resolve()
processed_data_path = (Path(__file__) / "../../../data/processed").resolve()


def merge_deep_dict(user, default):
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            user[k] = v if k not in user else merge_deep_dict(user[k], v)
    return user


default_params = {"prepare": {"split": 0.2}}
if Path("params.yaml").exists():
    with open("params.yaml", "r") as fd:
        params = yaml.safe_load(fd)
        params = merge_deep_dict(params, default_params)
else:
    params = default_params


class GenerationData:
    def __init__(self, raw_file_path: Path = raw_data_path / HELEN_DATA_FILENAME):
        self.raw_file_path = raw_file_path

    @classmethod
    def from_url(
        cls,
        url: str = HELEN_DATA_URL,
        raw_file_path: Path = raw_data_path / HELEN_DATA_FILENAME,
    ):
        """
        Get DH data CSV, save to file

        :param url: online location for data
        :param raw_file_path: local file path for raw data
        """
        logging.info(f"Downloading Helen DH data from {url} to {raw_file_path}")
        r = requests.get(url)
        with raw_file_path.open("wb") as f:
            f.write(r.content)
        return cls(raw_file_path)

    def load_and_clean(self) -> DataFrame:
        """
        Load dataset from disk, clean up features

        :param raw_file_path: local file path for raw data
        :return: Pandas dataframe, with index column 'date_time' and feature column 'dh_MWh'
        """
        logging.info("Load Helen raw dataset from CSV, clean up file")
        df = pd.read_csv(
            self.raw_file_path,
            sep=";",
            decimal=",",
            parse_dates=["date_time"],
            dayfirst=True,
        ).set_index("date_time")
        df.index = df.index.tz_localize(tz="Europe/Helsinki", ambiguous="infer")
        return df


fmi_weather_files = {
    "Kaisaniemi": [
        "csv-f2faff3e-672d-41a0-8bf3-278ab00d8796.csv",
        "csv-270db165-c228-466b-a438-69caa8c8f075.csv",
        "csv-c1047dd7-2a2c-4046-bae8-9835353aced4.csv",
        "csv-9c60edf9-247b-4dc5-8b3d-d50137de4bdc.csv",
        "csv-5bba8028-c209-48d8-a958-0764f0e927b1.csv",
        "csv-563b0676-9825-4a09-a9fd-7095ade8dec6.csv",
        "csv-f6c18948-8fac-45ac-ab2a-bb51173800dc.csv",
    ]
}


class FmiData:
    def __init__(self, *raw_file_paths: Path):
        """
        Create data loader, from multiple data files mapping to **same weather station**

        :param raw_file_paths: Location of CSV files, downloaded from FMI data service
        """
        self.raw_file_paths = raw_file_paths

    def load_and_clean(self) -> DataFrame:
        """
        Load data files from disk, clean up features
        :return: Pandas dataframe, with index column 'datetime'
        """
        logging.info("Load and clean up data files")
        frames = [self.read_file(f) for f in self.raw_file_paths]
        df = concat(frames)
        df = df.loc[~df.index.duplicated()]

        df["Ilman lämpötila (degC)"].interpolate(inplace=True)

        return df

    @staticmethod
    def read_file(filename: FilePathOrBuffer) -> DataFrame:
        """
        Read FMI weather data into pandas data frame
        """
        logging.info(f"Read FMI weather data: {filename=}")
        d = read_csv(filename, parse_dates=[["Vuosi", "Kk", "Pv", "Klo"]]).rename(
            columns={"Vuosi_Kk_Pv_Klo": "date_time"}
        )

        assert (d["Aikavyöhyke"] == "UTC").all()

        d = d.drop(["Aikavyöhyke"], axis=1).set_index("date_time")

        d.index = d.index.tz_localize("UTC").tz_convert("Europe/Helsinki")
        return d


def train_test_split_time(
    df: DataFrame, test_size: float = params["prepare"]["split"]
) -> Tuple[DataFrame, DataFrame]:
    all_data = df.sort_index()
    split_idx = int(len(all_data) * (1 - test_size))
    train = df[:split_idx]
    test = df[split_idx:]
    return train, test


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


def read_fmi(
    station_name="Kaisaniemi",
    intermediate_file_path: Path = intermediate_data_path
    / WEATHER_INTERMEDIATE_FILENAME.format(station="Kaisaniemi"),
) -> DataFrame:
    logging.info("Read FMI data")

    if intermediate_file_path.exists():
        logging.info("Intermediate file exists, read from there")
        return load_intermediate(intermediate_file_path=intermediate_file_path)

    logging.info("Process from raw data")
    fmi_data = FmiData(*[(raw_data_path / i) for i in fmi_weather_files[station_name]])
    df: DataFrame = fmi_data.load_and_clean()
    save_intermediate(df, intermediate_file_path=intermediate_file_path)
    return df


def read_helen(
    intermediate_file_path: Path = intermediate_data_path / HELEN_INTERMEDIATE_FILENAME,
) -> DataFrame:
    logging.info("Read Helen data")

    if intermediate_file_path.exists():
        logging.info("Intermediate file exists, read from there")
        return load_intermediate(intermediate_file_path=intermediate_file_path)

    logging.info("Process from raw data")
    generation_data = GenerationData()
    df: DataFrame = generation_data.load_and_clean()
    save_intermediate(df, intermediate_file_path=intermediate_file_path)
    return df


def merge_dataframes(
    df_helen: DataFrame,
    df_fmi: DataFrame,
    save_master_data: bool = True,
    file_path: Path = processed_data_path / MASTER_INTERMEDIATE_FILENAME,
) -> DataFrame:

    df = merge(df_helen, df_fmi, how="left", left_index=True, right_index=True)

    if save_master_data:
        save_intermediate(df, intermediate_file_path=file_path)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df_fmi = read_fmi()[["Ilman lämpötila (degC)"]]
    df_helen = read_helen()
    merge_dataframes(df_helen=df_helen, df_fmi=df_fmi)
