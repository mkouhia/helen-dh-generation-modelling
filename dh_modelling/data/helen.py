import logging
import requests

from pandas import DataFrame
import pandas as pd

from pathlib import Path

from dh_modelling.data import raw_data_path, intermediate_data_path

HELEN_DATA_URL = 'https://www.helen.fi/globalassets/helen-oy/vastuullisuus/hki_dh_2015_2020_a.csv'
HELEN_DATA_FILENAME = 'hki_dh_2015_2020_a.csv'
HELEN_INTERMEDIATE_FILENAME = 'helen_cleaned.feather'


def get_and_save(url: str = HELEN_DATA_URL, save_file_path: Path = raw_data_path / HELEN_DATA_FILENAME):
    """
    Get DH data CSV, save to file

    :param url: online location for data
    :param save_file_path: file where data is to be saved
    """
    logging.info(f'Downloading Helen DH data from {HELEN_DATA_URL} to {save_file_path}')
    r = requests.get(url)
    with save_file_path.open('wb') as f:
        f.write(r.content)


def load_and_clean(file_path: Path = raw_data_path / HELEN_DATA_FILENAME) -> DataFrame:
    """
    Load dataset from disk, clean up features

    :param file_path: local file location for data
    :return: Pandas dataframe, with index column 'date_time' and feature column 'dh_MWh'
    """
    logging.info("Load Helen raw dataset from CSV, clean up file")
    df = pd.read_csv(file_path, sep=';', decimal=',', parse_dates=['date_time'], dayfirst=True) \
        .set_index('date_time')
    df.index = df.index.tz_localize(tz='Europe/Helsinki', ambiguous='infer')
    return df


def save_intermediate(df: DataFrame, save_file_path: Path = intermediate_data_path / HELEN_INTERMEDIATE_FILENAME):
    logging.info(f'Save cleaned intermediate Helen dataset to {save_file_path}')
    df.reset_index()\
        .to_feather(save_file_path)


def load_intermediate(load_file_path: Path = intermediate_data_path / HELEN_INTERMEDIATE_FILENAME) -> DataFrame:
    return pd.read_feather(load_file_path)\
        .set_index('date_time')


def main():
    logging.basicConfig(level=logging.INFO)

    logging.info("Verify that Helen raw & intermediate data exists")

    if not (raw_data_path / HELEN_DATA_FILENAME).exists():
        get_and_save()

    if not (intermediate_data_path / HELEN_INTERMEDIATE_FILENAME).exists():
        df: DataFrame = load_and_clean()
        save_intermediate(df)


if __name__ == '__main__':
    main()
