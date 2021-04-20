import numpy as np
import optuna
import xgboost
from pandas import DataFrame

from dh_modelling.train import Objective, optuna_optimize_hyperparameters, train


def test_train(mocker):
    model = xgboost.Booster()
    X = np.array([[-20, 0, 15, 20, 30]])
    y = np.array([2200, 1000, 400, 300, 300])

    dx = xgboost.DMatrix(data=X, label=y)

    mocker.patch("xgboost.train", return_value=model)

    train(
        dx,
        {},
        num_boost_round=10,
    )
    mocker.patch("xgboost.Booster.predict")


def test_obj_call(mocker):
    X = np.array([[-20, 0, 15, 20, 30]])
    y = np.array([2200, 1000, 400, 300, 300])

    dx = xgboost.DMatrix(data=X, label=y)
    obj = Objective(dx, 10, False)

    test_df = DataFrame({"test-rmse-mean": [0.1, 0.7]})

    mocker.patch("xgboost.cv", return_value=test_df)

    study = optuna.create_study()
    trial = study.ask()
    assert obj(trial) == 0.1


def test_obj_call_gpu(mocker):
    X = np.array([[-20, 0, 15, 20, 30]])
    y = np.array([2200, 1000, 400, 300, 300])

    dx = xgboost.DMatrix(data=X, label=y)
    obj = Objective(dx, 10, run_on_gpu=True)

    test_df = DataFrame({"test-rmse-mean": [0.1, 0.7]})

    mocker.patch("xgboost.cv", return_value=test_df)

    study = optuna.create_study()
    trial = study.ask()
    assert obj(trial) == 0.1


def test_optuna_optimize_hyperparameters():
    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        return x ** 2

    best_trial = optuna_optimize_hyperparameters(objective, 20)

    assert -1.0 < best_trial.params["x"] < 1
