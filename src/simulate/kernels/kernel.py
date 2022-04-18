"""
Kernel
"""

import numpy as np


class Kernel:
    """
    Kernel base class
    """

    def __init__(self, labels=None) -> None:
        if labels is None:
            labels = {"t": 0, "val": 1}
        self.__labels = labels
        self.__val = 0.0

    def simulate(self, step: float, step_size: float) -> np.ndarray:
        """
        simulate one step
        :param step: current step
        :param step_size: step size
        :return: recordings, packed in a numpy ndarray
        """
        self.__val += step_size
        return np.array([step, self.__val])

    def labels(self) -> dict:
        """
        :return: labels of recordings and their indices in the recordings
        """
        return self.__labels
