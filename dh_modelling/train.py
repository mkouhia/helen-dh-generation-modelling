import argparse
import logging
from pathlib import Path

import numpy as np
from pandas import DataFrame

from .helpers import load_intermediate
from .model import Model, save_model


def train(df: DataFrame, model: Model):
    logging.info("Train model")

    X: np.ndarray = df["Ilman lämpötila (degC)"].to_numpy()
    y: np.ndarray = df["dh_MWh"].to_numpy()

    model.fit(X, y)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Train model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-path",
        help="Where to read train dataframe",
        type=Path,
        default=Path("data/processed/train.feather"),
    )
    parser.add_argument(
        "--model-path",
        help="Where to save model",
        type=Path,
        default=Path("models/model.joblib"),
    )

    args = parser.parse_args()

    df_train: DataFrame = load_intermediate(path=args.train_path.absolute())

    model: Model = Model()
    train(df_train, model)

    save_model(model, args.model_path.absolute())
