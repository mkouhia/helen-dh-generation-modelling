import argparse
import logging
from pathlib import Path

import xgboost as xgb
from pandas import DataFrame

from .helpers import load_intermediate


def train(
    df: DataFrame,
    objective,
    num_boost_round,
    tree_method,
    max_depth,
    min_child_weight,
    eta,
    subsample,
    colsample_bytree,
) -> xgb.Booster:
    logging.info("Train model")

    X, y = df.drop("dh_MWh", axis=1), df[["dh_MWh"]]

    dtrain: xgb.DMatrix = xgb.DMatrix(data=X, label=y)

    params = {
        "objective": objective,
        "tree_method": tree_method,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "eta": eta,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
    }

    return xgb.train(params, dtrain, num_boost_round=num_boost_round)


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
        default=Path("models/model.bin"),
    )
    parser.add_argument("--objective", default="reg:squarederror", type=str)
    parser.add_argument("--num-boost-round", default=10, type=int)
    parser.add_argument("--tree-method", default="hist", type=str)
    parser.add_argument("--eval-metric", default="rmse", type=str)
    parser.add_argument("--max-depth", default=5, type=int)
    parser.add_argument("--min-child-weight", default=1.0, type=float)
    parser.add_argument("--eta", default=0.3, type=float)
    parser.add_argument("--subsample", default=1.0, type=float)
    parser.add_argument("--colsample-bytree", default=1.0, type=float)

    args = parser.parse_args()

    df_train: DataFrame = load_intermediate(path=args.train_path.absolute())

    model: xgb.Booster = train(
        df_train,
        objective=args.objective,
        num_boost_round=args.num_boost_round,
        tree_method=args.tree_method,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        eta=args.eta,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
    )

    model.save_model(args.model_path.absolute())
