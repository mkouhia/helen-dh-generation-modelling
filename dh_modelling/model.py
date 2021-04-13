import logging
from pathlib import Path

import numpy as np
from joblib import dump, load
from scipy import optimize


class Model:
    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Model):
            return NotImplemented
        return self.params == o.params

    def fit(self, X: np.ndarray, y: np.ndarray):
        logging.info("Training model...")
        self.params, _ = optimize.curve_fit(self._piecewise_linear, X, y)

    def predict(self, X) -> np.ndarray:
        return self._piecewise_linear(X, *self.params)

    def _piecewise_linear(self, x, y0, k1) -> np.ndarray:
        x0 = 17
        k2 = 0
        return np.piecewise(
            x,
            [x < x0],
            [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0],
        )


def save_model(model: Model, path: Path):
    logging.info(f"Saving model to {path}")
    dump(model, path)


def load_model(path: Path) -> Model:
    logging.info(f"Load model from {path}")
    return load(path)
