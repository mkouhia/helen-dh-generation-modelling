import json

from pandas import DataFrame

from dh_modelling.evaluate import evaluate, save_metrics
from dh_modelling.model import Model


def test_evaluate(mocker):
    model = Model()
    df = DataFrame({"dh_MWh": [11.7, 12.3], "Ilman lämpötila (degC)": [4.3, -5.7]})

    mae = 11.7
    mape = 0.082
    msqe = 10.5

    mocker.patch("dh_modelling.model.Model.predict")
    mocker.patch("dh_modelling.evaluate.mean_absolute_error", return_value=mae)
    mocker.patch(
        "dh_modelling.evaluate.mean_absolute_percentage_error", return_value=mape
    )
    mocker.patch("dh_modelling.evaluate.mean_squared_error", return_value=msqe)

    received = evaluate(model, df)

    expected = {
        "mean_absolute_error": mae,
        "mean_absolute_percentage_error": mape,
        "root_mean_squared_error": msqe,
    }

    assert received == expected


def test_save_metrics(tmp_path):
    test_path = tmp_path / "file"
    metrics = {"key1": 0.1, "key2": 0.2}
    save_metrics(metrics, test_path)

    with open(test_path, "r") as f:
        received = json.load(f)

    assert received == metrics
