import logging
import requests
from pathlib import Path

from dh_modelling.data import raw_data_path

HELEN_DATA_URL = 'https://www.helen.fi/globalassets/helen-oy/vastuullisuus/hki_dh_2015_2020_a.csv'
HELEN_DATA_FILENAME = 'hki_dh_2015_2020_a.csv'


def get_and_save(url: str = HELEN_DATA_URL, save_file_path: Path = raw_data_path / HELEN_DATA_FILENAME):
    """
    Get DH data CSV, save to file

    :param url: online location for data
    :param save_file_path: file where data is to be saved
    """
    file_path = save_file_path
    logging.info(f'Downloading Helen DH data from {HELEN_DATA_URL} to {file_path}')
    r = requests.get(url)
    with file_path.open('wb') as f:
        f.write(r.content)


def main():
    logging.basicConfig(level=logging.INFO)
    get_and_save()


if __name__ == '__main__':
    main()
