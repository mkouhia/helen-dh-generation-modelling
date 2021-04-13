import numpy as np

from dh_modelling.model import Model, load_model, save_model


def test_save_load_model(tmp_path):

    test_path = tmp_path / "test"
    original = Model()
    original.params = [1, 2]

    save_model(original, test_path)
    received = load_model(test_path)

    assert received == original


def test_model():

    model = Model()
    X = np.array([-20, 0, 15, 20, 30])
    y = np.array([2200, 1000, 400, 300, 300])

    model.fit(X, y)
    assert isinstance(model.params, np.ndarray)
    assert model.predict(np.array([10, 25])).shape == (2,)
