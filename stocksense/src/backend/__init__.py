import numpy as np


def moving_average(data: np.ndarray, window_size: int = 3) -> np.ndarray:
    cumsum = np.cumsum(np.insert(data, 0, 0))
    result = data.copy()
    result[window_size - 1:] = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
    return result
