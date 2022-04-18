"""
Simulator
"""
from typing import List

import numpy as np

from simulate.kernels import Kernel


class Simulator:
    """
    Simulator
    performs a simulation specified by a kernel
    using continuous simulation
    """

    def __init__(self, kernel: Kernel, step_size: float, max_steps: float):
        """
        :param kernel: simulation kernel
        :param step_size: step size
        :param max_steps: maximum step
        """
        self.__callback = kernel.simulate
        self.__step_size = step_size
        self.__max_steps = max_steps
        self.__progress: List[np.ndarray] = []

    def run(self) -> None:
        """
        perform the continuous simulation
        :return: progress
        """
        for t in np.arange(0, self.__max_steps, self.__step_size):
            # evaluate step
            current_state = self.__callback(t, self.__step_size)
            # record step
            self.__progress.append(current_state)

    def progress(self) -> np.ndarray:
        """
        :return: progress recordings
        """
        return np.array(self.__progress)
