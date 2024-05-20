from sklearn.base import RegressorMixin
from sklearn.utils import estimator_html_repr
from src.model import (
    get_apple_model,
    get_alphabet_model,
    get_tesla_model,
    get_microsoft_model
)
import numpy as np


class AutoRegressor(RegressorMixin):
    def __init__(self,
                 stock_name: str) -> None:
        self._window_size = 30

        match stock_name:
            case 'Apple':
                self._model = get_apple_model()
            case 'Alphabet':
                self._model = get_alphabet_model()
            case 'Microsoft':
                self._model = get_microsoft_model()
            case 'Tesla':
                self._model = get_tesla_model()

    @property
    def html_repr(self) -> str:
        return estimator_html_repr(self._model)

    def predict(self,
                last_data: np.ndarray,
                count_predictions: int = 1) -> np.ndarray:
        assert last_data.size == self._window_size

        if count_predictions == 1:
            return self._model.predict(last_data)
        else:
            predictions = np.zeros(count_predictions)
            for i in range(count_predictions):
                logits = self._model.predict(last_data)
                predictions[i] = logits
                last_data = np.array(last_data[0, 1:].tolist()
                                     + logits.tolist()).reshape(1, -1)
            return predictions
