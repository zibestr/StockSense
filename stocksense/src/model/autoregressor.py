from sklearn.base import RegressorMixin
from src.model import get_apple_model
import numpy as np


class AutoRegressor(RegressorMixin):
    def __init__(self,
                 stock_name: str) -> None:
        self._window_size = 30

        match stock_name:
            case 'Apple':
                self._model, self._html = get_apple_model()

    @property
    def html_repr(self) -> str:
        return self._html

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
