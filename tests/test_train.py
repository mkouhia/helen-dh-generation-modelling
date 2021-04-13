import numpy as np
from pandas import DataFrame

from dh_modelling.model import Model
from dh_modelling.train import train


def test_train():

    model = Model()
    df = DataFrame(
        {
            "Ilman lämpötila (degC)": [-20, 0, 15, 20, 30],
            "dh_MWh": [2200, 1000, 400, 300, 300],
        }
    )

    train(df, model)
    assert isinstance(model.params, np.ndarray)
