"""
Kernel
"""

import numpy as np


class Kernel:
    """
    Kernel base class
    """

    __val = 0.0
    __labels = {"t": 0, "val": 1}

    def callback(self, t, t_step: float) -> np.ndarray:
        """
        simulate one step
        :param t: current step
        :param t_step: step size
        :return: recordings, packed in a numpy ndarray
        """
        self.__val += t_step
        return np.array([t, self.__val])

    def labels(self) -> dict:
        """
        :return: labels of recordings and their indices in the recordings
        """
        return self.__labels
