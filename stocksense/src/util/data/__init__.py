from collections.abc import Sequence
from pandas import Series
import numpy as np


class TimeSeriesDataset(Sequence):
    """_summary_
    """
    def __init__(self, data: Series) -> None:
        """_summary_

        Args:
            data (Series): _description_
        """
        # set window size to 30, because minimum period for analysis is 30 days
        self.__window_size = 30
        self.__data = data
        self.min = data.min()
        self.max = data.max()

    def __getitem__(self, ind: int | slice):
        """_summary_

        Args:
            ind (int | slice): _description_

        Returns:
            _type_: _description_
        """
        if isinstance(ind, slice):
            start = 0 if ind.start is None else ind.start
            stop = len(self) if ind.stop is None else ind.stop
            step = 1 if ind.step is None else ind.step
            return (np.array([self.__get_one(ind)[0]
                              for ind in range(start, stop,
                                               step)]),
                    np.array([self.__get_one(ind)[1]
                              for ind in range(start, stop,
                                               step)]))
        return self.__get_one(ind)

    def __get_one(self, ind: int) -> tuple[np.ndarray,
                                           float]:
        """_summary_

        Args:
            ind (int): _description_

        Returns:
            tuple[np.ndarray, float]: _description_
        """
        # return ((self.__data.iloc[ind:ind + self.__window_size].to_numpy()
        #          - self.__mean) / self.__std,
        #         (self.__data.iloc[ind + self.__window_size]
        #          - self.__mean) / self.__std)
        return (self.__data.iloc[ind:ind + self.__window_size].to_numpy(),
                self.__data.iloc[ind + self.__window_size])

    def __len__(self) -> int:
        return len(self.__data) - self.__window_size

    @property
    def X(self) -> np.ndarray:
        return self[:len(self)][0]

    @property
    def y(self) -> np.ndarray:
        return self[:len(self)][1]
