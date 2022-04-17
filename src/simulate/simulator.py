from typing import Callable
import numpy as np
import math
from simulate.kernels import Kernel


class Simulator:
    def __init__(self, kernel: Kernel, t_step: float, t_max: float):
        self.__callback = kernel.callback
        self.__step_size = t_step
        self.__max_steps = t_max
        self.__progress = []
        self.__progress_labels = {"step": 1}.update(kernel.labels)

    def run(self):
        for t in np.arange(0, self.__max_steps, self.__step_size):
            current_state = self.__callback(self.__step_size)

            row = np.append(np.array([t]), current_state.flat)
            self.__progress.append(row)

    def progress(self):
        return np.array(self.__progress)

    def progress_labels(self):
        return self.__progress_labels
