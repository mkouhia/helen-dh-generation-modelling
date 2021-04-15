import argparse
import json
import logging
from pathlib import Path

import numpy as np
from pandas import DataFrame
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from .helpers import load_intermediate
from .model import Model, load_model


def evaluate(model: Model, df: DataFrame) -> dict[str, float]:
    logging.info("Evaluating model performance")
    actual: np.ndarray = df["dh_MWh"]

    X: np.ndarray = df["Ilman lämpötila (degC)"].to_numpy()

    predictions: np.ndarray = model.predict(X)
    return calculate_metrics(actual, predictions)


def calculate_metrics(actual: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "mean_absolute_error": mean_absolute_error(actual, predictions),
        "mean_absolute_percentage_error": mean_absolute_percentage_error(
            actual, predictions
        ),
        "root_mean_squared_error": mean_squared_error(
            actual, predictions, squared=False
        ),
    }


def save_metrics(metrics: dict, path: Path):
    logging.info(f"Save metrics to {path=}")
    with open(path, "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Evaluate model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        help="Where to load model",
        type=Path,
        default=Path("models/model.joblib"),
    )
    parser.add_argument(
        "--test-path",
        help="Where to load test dataframe",
        type=Path,
        default=Path("data/processed/test.feather"),
    )
    parser.add_argument(
        "--metrics-path",
        help="Where to save test metrics",
        type=Path,
        default=Path("output/score.json"),
    )

    args = parser.parse_args()

    df_test: DataFrame = load_intermediate(path=args.test_path.absolute())

    model = load_model(args.model_path.absolute())
    metrics: dict = evaluate(model, df_test)

    save_metrics(metrics, args.metrics_path.absolute())
