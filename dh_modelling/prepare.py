from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path

import pandas as pd
from pandas import DataFrame, DatetimeIndex, concat, merge, read_csv


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
    def read_helen(raw_file_path: Path) -> DataFrame:
        logging.info(f"Read Helen generation data, {raw_file_path=}")
        generation_data = GenerationData(raw_file_path=raw_file_path)
        return generation_data.load_and_clean()


class FmiData:
    def __init__(self, station_name: str, *raw_file_paths: PathLike):
        """
        Create data loader, from multiple data files mapping to **same weather station**

        :type station_name: name of station, where the data has been collected
        :param raw_file_paths: Location of CSV files, downloaded from FMI data service
        """
        self.station_name = station_name
        self.raw_file_paths = raw_file_paths

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FmiData):
            return NotImplemented
        return (self.station_name == other.station_name) and (
            self.raw_file_paths == other.raw_file_paths
        )

    def __repr__(self) -> str:
        return f"{self.__class__}({self.__dict__!r})"

    def load_and_clean(self) -> DataFrame:
        """
        Load data files from disk, clean up features
        :return: Pandas dataframe, with index column 'datetime'
        """
        logging.info("Load and clean up data files")
        frames = [self._read_file(f) for f in self.raw_file_paths]
        df = concat(frames)
        df = df.loc[~df.index.duplicated()]

        df["Ilman lämpötila (degC)"].interpolate(inplace=True)

        return df.sort_index()

    @staticmethod
    def _read_file(filepath_or_buffer: PathLike) -> DataFrame:
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
    def read_fmi_files(directory: Path, station_name: str) -> FmiData:
        """
        Read batch of FMI weather files and metadata from a directory

        Go through the folder, find metadata files with pattern "csv-meta-(.+)\\.csv".
        Read metadata files, see which IDs are from the desired station.
        For stations that match, find files in the folder with pattern "csv-{id}.csv".
        Read those files into dataframe.

        :param directory: path to directory, where content is searched
        :param station_name: Name of station, for which to assemble the dataframe
        :return: dataframe, assembled from
        """
        logging.info(f"Read FMI data, {station_name=}")
        all_files = [fp for fp in directory.glob("*.csv") if fp.is_file()]
        meta_pattern = re.compile("csv-meta-(.+)\\.csv")

        meta_dict: dict[str, FmiMeta] = dict()
        for f in all_files:
            if (match := meta_pattern.match(f.name)) is not None:
                meta_dict[match.group(1)] = FmiMeta.from_file(f)

        station_files = [
            (directory / f"csv-{id_str}.csv")
            for id_str, meta_ob in meta_dict.items()
            if (meta_ob.station_name == station_name)
            and ((directory / f"csv-{id_str}.csv").is_file())
        ]

        logging.info(
            f"Found {len(station_files)} metadata-csv file pairs for {station_name=}"
        )

        return FmiData(station_name, *station_files)


@dataclass
class FmiMeta:
    station_name: str
    station_code: int
    latitude: float
    longitude: float
    start_time: datetime
    end_time: datetime
    creation_time: datetime

    @classmethod
    def from_file(cls, path: Path) -> FmiMeta:
        content = path.read_text("utf8")
        lines = content.split("\n")
        assert len(lines) == 2
        keys = lines[0].split(",")
        values = lines[1].split(",")
        d = dict(zip(keys, values))

        def utc_string_to_datetime(s: str) -> datetime:
            return datetime.fromisoformat(s.removesuffix("Z")).replace(
                tzinfo=timezone.utc
            )

        return cls(
            station_name=d["Havaintoasema"],
            station_code=int(d["Asemakoodi"]),
            latitude=float(d["Latitudi (desimaaliasteita)"]),
            longitude=float(d["Longitudi (desimaaliasteita)"]),
            start_time=utc_string_to_datetime(d["Alkuhetki"]),
            end_time=utc_string_to_datetime(d["Loppuhetki"]),
            creation_time=utc_string_to_datetime(d["Datan luontihetki"]),
        )


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
) -> tuple[DataFrame, DataFrame]:
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
        "--helen-path",
        help="Where to read raw generation data",
        type=Path,
        default=Path("data/raw/hki_dh_2015_2020_a.csv"),
    )
    parser.add_argument(
        "--fmi-dir",
        help="Directory, where to read raw FMI weather files",
        type=Path,
        default=Path("data/raw/fmi"),
    )
    parser.add_argument(
        "--fmi-station-name",
        help="Station name, which to use in FMI csv lookup in fmi-dir",
        type=str,
        default="Helsinki Kaisaniemi",
    )
    parser.add_argument(
        "--test-path",
        help="Where to save test dataframe",
        type=Path,
        default=Path("data/processed/test.feather"),
    )
    parser.add_argument(
        "--master-path",
        help="Where to save master dataframe",
        type=Path,
        default=Path("data/intermediate/master.feather"),
    )

    args = parser.parse_args()

    fmi_loader: FmiData = FmiData.read_fmi_files(
        directory=args.fmi_dir.absolute(), station_name=args.fmi_station_name
    )
    df_weather = fmi_loader.load_and_clean()
    df_generation = GenerationData.read_helen(raw_file_path=args.helen_path.absolute())

    gen_train, gen_test = train_test_split_sorted(df_generation, test_size=args.split)
    save_dataframe(gen_test, file_path=args.test_path.absolute())

    df_master: DataFrame = merge_dataframes(df_helen=gen_train, df_fmi=df_weather)
    save_dataframe(df_master, file_path=args.master_path.absolute())
