import argparse
import logging
from pathlib import Path

from pandas import DataFrame, DatetimeIndex, read_feather


def featurize(df: DataFrame) -> DataFrame:
    """
    Create features from dataframe

    :param df: input dataframe
    :return: modified dataframe
    """
    return df


def load_master(
    path: Path,
    set_datetime_index: bool = True,
    date_time_column: str = "date_time",
    timezone: str = "Europe/Helsinki",
) -> DataFrame:
    """
    Load master file from disk

    :param path: master file location
    :param set_datetime_index: set DatetimeIndex from column 'date_time_column' with timezone 'timezone'
    :param date_time_column: column, which should be set as index
    :param timezone: timezone, at which date_time_column is converted
    :return: loaded dataframe, with 'date_time' as index
    """
    logging.info(f"Load master from {path=}")
    df: DataFrame = read_feather(path)
    if set_datetime_index:
        df.index = DatetimeIndex(df[date_time_column]).tz_convert(timezone)
        df = df.drop("date_time", axis=1)
    return df


def save_train(df: DataFrame, path: Path, reset_datetime_index: bool = True):
    """
    Save train dataframe to disk

    :param df: dataframe to be saved
    :param path: file location
    :param reset_datetime_index: reset datetime index to normal column, convert timestamp to UTC
    """
    logging.info(f"Save train dataset to {path}")
    if reset_datetime_index:
        index_name: str = df.index.name
        df = df.reset_index()
        df[index_name] = DatetimeIndex(df[index_name]).tz_convert("UTC")
    df.to_feather(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Featurize data in master file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--master-path",
        help="Where to read master dataframe",
        type=Path,
        default=Path("data/intermediate/master.feather"),
    )
    parser.add_argument(
        "--train-path",
        help="Where to save train dataframe",
        type=Path,
        default=Path("data/processed/train.feather"),
    )

    args = parser.parse_args()

    df_master: DataFrame = load_master(path=args.master_path)
    df_train: DataFrame = featurize(df_master)
    save_train(df_train, path=args.train_path)
