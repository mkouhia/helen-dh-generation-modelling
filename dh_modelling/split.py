import argparse
import logging
from pathlib import Path

from pandas import DataFrame

from dh_modelling.helpers import load_intermediate, save_intermediate


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
        description="Train/test split",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test-size", help="Test data size, fraction of total", type=float, default=0.2
    )
    parser.add_argument(
        "--input",
        help="Where to read input data",
        type=Path,
        default=Path("data/intermediate/master.feather"),
    )
    parser.add_argument(
        "--train-output",
        help="Where to save train dataframe",
        type=Path,
        default=Path("data/processed/train.feather"),
    )
    parser.add_argument(
        "--test-output",
        help="Where to save test dataframe",
        type=Path,
        default=Path("data/processed/test.feather"),
    )

    args = parser.parse_args()

    df_all: DataFrame = load_intermediate(args.input.absolute())

    train, test = train_test_split_sorted(df_all, test_size=args.test_size)
    save_intermediate(train, path=args.train_output.absolute())
    save_intermediate(test, path=args.test_output.absolute())
