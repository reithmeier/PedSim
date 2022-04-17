"""
Simulator
"""
import numpy as np

from simulate.kernels import Kernel


class Simulator:
    """
    Simulator
    performs a simulation specified by a kernel
    using continuous simulation
    """

    def __init__(self, kernel: Kernel, t_step: float, t_max: float):
        """
        :param kernel: simulation kernel
        :param t_step: step size
        :param t_max: maximum step
        """
        self.__callback = kernel.callback
        self.__step_size = t_step
        self.__max_steps = t_max
        self.__progress = []

    def run(self) -> np.ndarray:
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
