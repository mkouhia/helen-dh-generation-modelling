import xgboost
from pandas import DataFrame

from dh_modelling.train import train


def test_train(mocker):

    model = xgboost.Booster()
    df = DataFrame(
        {
            "Ilman lämpötila (degC)": [-20, 0, 15, 20, 30],
            "dh_MWh": [2200, 1000, 400, 300, 300],
        }
    )

    mocker.patch("xgboost.train", return_value=model)

    train(
        df,
        objective="reg:squarederror",
        num_boost_round=10,
        tree_method="hist",
        max_depth=5,
        min_child_weight=1.0,
        eta=0.3,
        subsample=1.0,
        colsample_bytree=1.0,
    )
    mocker.patch("xgboost.Booster.predict")
