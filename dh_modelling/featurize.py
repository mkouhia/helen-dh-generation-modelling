import argparse
import logging
from pathlib import Path

from pandas import DataFrame

from .helpers import load_intermediate, save_intermediate


def featurize(df: DataFrame) -> DataFrame:
    """
    Create features from dataframe

    :param df: input dataframe
    :return: modified dataframe
    """
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Featurize data in master file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        help="Where to read master dataframe",
        type=Path,
        default=Path("data/intermediate/master.feather"),
    )
    parser.add_argument(
        "--output",
        help="Where to save train dataframe",
        type=Path,
        default=Path("data/processed/train.feather"),
    )

    args = parser.parse_args()

    df_master: DataFrame = load_intermediate(path=args.input.absolute())
    df_train: DataFrame = featurize(df_master)
    save_intermediate(df_train, path=args.output.absolute())
