"""
Kernel
"""

import numpy as np


class Kernel:
    """
    Kernel base class
    """

    def __init__(self, integrator: callable, labels: dict = None) -> None:
        """
        :param labels: dictionary with label names \
                and corresponding index in the simulation result
        """
        self._integrator = integrator

        if labels is None:
            labels = {"step": 0, "val": 1}
        self.__labels = labels

    def simulate(self, step: float, step_size: float) -> np.ndarray:
        """
        Override this method
        simulate one step
        :param step_size: step size
        :param step: current step
        :return: recordings, packed in a numpy ndarray
        """
        raise NotImplementedError

    def labels(self) -> dict:
        """
        :return: labels of recordings and their indices in the recordings
        """
        return self.__labels
