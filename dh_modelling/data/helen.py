import logging
from pathlib import Path

import pandas as pd
import requests
from pandas import DataFrame

from dh_modelling.data import intermediate_data_path, raw_data_path, save_intermediate

HELEN_DATA_URL = (
    "https://www.helen.fi/globalassets/helen-oy/vastuullisuus/hki_dh_2015_2020_a.csv"
)
HELEN_DATA_FILENAME = "hki_dh_2015_2020_a.csv"
HELEN_INTERMEDIATE_FILENAME = "helen_cleaned.feather"


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Read raw Helen data from disk, process to intermediate file")
    generation_data = GenerationData()
    df: DataFrame = generation_data.load_and_clean()
    save_intermediate(
        df, intermediate_file_path=intermediate_data_path / HELEN_INTERMEDIATE_FILENAME
    )
