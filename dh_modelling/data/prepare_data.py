import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from pandas import DataFrame, DatetimeIndex, concat, merge, read_csv

PathLike = Union[Path, os.PathLike]

HELEN_DATA_FILENAME = "hki_dh_2015_2020_a.csv"

HELEN_INTERMEDIATE_FILENAME = "helen_cleaned.feather"
WEATHER_INTERMEDIATE_FILENAME = "weather_{station}.feather"
MASTER_INTERMEDIATE_FILENAME = "master.feather"
TEST_FILENAME = "test.feather"

raw_data_path = (Path(__file__) / "../../../data/raw").resolve()
intermediate_data_path = (Path(__file__) / "../../../data/intermediate").resolve()
processed_data_path = (Path(__file__) / "../../../data/processed").resolve()


class GenerationData:
    def __init__(self, raw_file_path):
        self.raw_file_path = raw_file_path

    def load_and_clean(self) -> DataFrame:
        """
        Load dataframe from disk, clean up features

        :return: Pandas dataframe, with index column 'date_time' and feature column 'dh_MWh'
        """
        logging.info("Load Helen raw dataframe from CSV, clean up file")
        df = pd.read_csv(
            self.raw_file_path,
            sep=";",
            decimal=",",
            parse_dates=["date_time"],
            date_parser=lambda x: datetime.strptime(x, "%d.%m.%Y %H:%M"),
        ).set_index("date_time")
        df.index = df.index.tz_localize(tz="Europe/Helsinki", ambiguous="infer")
        return df

    @staticmethod
    def read_helen(
        raw_file_path: Path = raw_data_path / HELEN_DATA_FILENAME,
    ) -> DataFrame:
        logging.info(f"Read Helen generation data, {raw_file_path=}")
        generation_data = GenerationData(raw_file_path=raw_file_path)
        return generation_data.load_and_clean()


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
    def __init__(self, *raw_file_paths: PathLike):
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
    def read_file(filepath_or_buffer: PathLike) -> DataFrame:
        """
        Read FMI weather data into pandas data frame
        """
        logging.info(f"Read FMI weather data: {filepath_or_buffer=}")
        d = read_csv(
            filepath_or_buffer, parse_dates=[["Vuosi", "Kk", "Pv", "Klo"]]
        ).rename(columns={"Vuosi_Kk_Pv_Klo": "date_time"})

        assert (d["Aikavyöhyke"] == "UTC").all()

        d = d.drop(["Aikavyöhyke"], axis=1).set_index("date_time")

        d.index = d.index.tz_localize("UTC").tz_convert("Europe/Helsinki")
        return d

    @staticmethod
    def read_fmi(station_name="Kaisaniemi") -> DataFrame:
        logging.info(f"Read FMI data, {station_name=}")
        fmi_data = FmiData(
            *[(raw_data_path / i) for i in fmi_weather_files[station_name]]
        )
        return fmi_data.load_and_clean()


def save_dataframe(df: DataFrame, file_path: Path, reset_datetime_index: bool = True):
    """
    Save intermediate representation of dataframe to disk

    :param df: dataframe to be saved
    :param file_path: file location
    :param reset_datetime_index: reset datetime index to normal column, convert timestamp to UTC
    """
    logging.info(f"Save intermediate dataset to {file_path}")
    if reset_datetime_index:
        index_name: str = df.index.name
        df = df.reset_index()
        df[index_name] = DatetimeIndex(df[index_name]).tz_convert("UTC")
    df.to_feather(file_path)


def merge_dataframes(df_helen: DataFrame, df_fmi: DataFrame) -> DataFrame:
    logging.info("Left join fmi on helen")
    return merge(df_helen, df_fmi, how="left", left_index=True, right_index=True)


def train_test_split_sorted(
    df: DataFrame, test_size: float = 0.2
) -> Tuple[DataFrame, DataFrame]:
    """
    Split train/test data, by sorting data on index and taking last data points as test

    :param df: input dataframe
    :param test_size: fraction of test size of all points
    :return: train, test
    """
    logging.info(f"Perform train/test split, {test_size=}")
    all_data = df.sort_index()
    split_idx = int(len(all_data) * (1 - test_size))
    train = all_data[:split_idx]
    test = all_data[split_idx:]
    return train, test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Prepare data from raw sources",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split", help="Test data size, fraction of total", type=float, default=0.2
    )
    parser.add_argument(
        "--test-path",
        help="Where to save test dataframe",
        type=lambda p: Path(p).absolute(),
        default=processed_data_path / TEST_FILENAME,
    )

    args = parser.parse_args()

    df_weather = FmiData.read_fmi()
    df_generation = GenerationData.read_helen()

    gen_train, gen_test = train_test_split_sorted(df_generation, test_size=args.split)
    save_dataframe(gen_test, file_path=args.test_path)

    df_master: DataFrame = merge_dataframes(df_helen=gen_train, df_fmi=df_weather)
    save_dataframe(
        df_master, file_path=processed_data_path / MASTER_INTERMEDIATE_FILENAME
    )
