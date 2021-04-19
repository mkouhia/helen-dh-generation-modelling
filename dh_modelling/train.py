import argparse
import logging
from pathlib import Path
from typing import cast

import optuna
import xgboost as xgb
from optuna.trial import FrozenTrial
from pandas import DataFrame

from .helpers import load_intermediate


def train(train_matrix: xgb.DMatrix, params: dict, num_boost_round: int) -> xgb.Booster:
    logging.info("Train model")
    return xgb.train(params, train_matrix, num_boost_round=num_boost_round)


class Objective(object):
    def __init__(
        self,
        d_matrix_train: xgb.DMatrix,
        num_boost_round: int = 100,
        run_on_gpu: bool = True,
        seed: int = 0,
    ) -> None:
        self.d_train = d_matrix_train
        self.num_boost_round = num_boost_round
        self.run_on_gpu = run_on_gpu
        self.seed = seed

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Perform a trial

        :param trial: optuna trial, as fed in by study.optimize
        :return: a numeric value on how the trial went
        """
        eval_metric = "rmse"
        param = {
            "verbosity": 0,
            "objective": "reg:squarederror",
            "eval_metric": eval_metric,
            ## Others
            "max_depth": trial.suggest_int("max_depth", 1, 9),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 5),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            # "eta": trial.suggest_float("eta", 1e-8, 1.0, log=True),
            "eta": trial.suggest_float("eta", 1e-3, 0.5, log=True),
            # "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            # "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            # "gamma": trial.suggest_float("gamma", 1e-8, 10, log=True)
        }

        if self.run_on_gpu:
            param["tree_method"] = "gpu_hist"
            param["sampling_method"] = "gradient_based"
            param["subsample"] = trial.suggest_float("subsample", 0.05, 1.0)
        else:
            param["tree_method"] = "hist"
            param["subsample"] = trial.suggest_float("subsample", 0.2, 1.0)

        pruning_callback = optuna.integration.XGBoostPruningCallback(
            trial, f"test-{eval_metric}"
        )
        history: DataFrame = xgb.cv(
            param,
            self.d_train,
            num_boost_round=self.num_boost_round,
            nfold=5,
            seed=self.seed,
            maximize=False,
            early_stopping_rounds=50,
            callbacks=[pruning_callback],
        )

        best_eval_idx = history[f"test-{eval_metric}-mean"].argmin()
        trial.set_user_attr("num_rounds", best_eval_idx + 1)

        return history[f"test-{eval_metric}-mean"].iloc[best_eval_idx]


def optuna_optimize_hyperparameters(
    func: optuna.study.ObjectiveFuncType, seed: int = 0, n_trials: int = 100
) -> FrozenTrial:
    """
    Perform hyperparameter optimization

    :return: dictionary of best parameters
    """
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="minimize")
    study.optimize(func, n_trials=n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print(study.trials_dataframe())

    return study.best_trial


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
    parser.add_argument(
        "--optimize-rounds",
        help="Number of hyperparameter optimization trials",
        type=int,
        default=100,
    )
    parser.add_argument("--num-boost-round", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--use-gpu", action="store_true")

    args = parser.parse_args()

    df_train: DataFrame = load_intermediate(path=args.train_path.absolute())
    X, y = df_train.drop("dh_MWh", axis=1), df_train[["dh_MWh"]]

    dtrain: xgb.DMatrix = xgb.DMatrix(data=X, label=y)

    obj = Objective(
        dtrain,
        num_boost_round=args.num_boost_round,
        run_on_gpu=args.use_gpu,
        seed=args.seed,
    )
    best_trial: FrozenTrial = optuna_optimize_hyperparameters(
        obj, seed=args.seed, n_trials=args.optimize_rounds
    )
    best_params: dict = best_trial.params
    num_rounds = best_trial.user_attrs.get("num_rounds")
    best_rounds: int = cast(int, num_rounds)
    print(f"Best rounds: {best_rounds}")

    model: xgb.Booster = train(dtrain, best_params, num_boost_round=best_rounds)
    model.save_model(args.model_path.absolute())
