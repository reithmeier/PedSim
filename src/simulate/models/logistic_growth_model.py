"""
LogisticGrowthModel
"""
import math

import numpy as np

from .model import Model


class LogisticGrowthModel(Model):
    """
    LogisticGrowthModel
    simulates a logistic growth model
    """

    def __init__(
        self,
        integrator: callable,
        alpha: float = 1 / 5,
        beta: float = 1 / 5175,
        start_value: float = 1,
    ):
        """
        :param integrator integrator
        :param alpha multiplier for increasing term
        :param beta multiplier for decreasing term
        """
        super().__init__(integrator=integrator, labels={"step": 0, "value": 1})
        self.__value = start_value
        self.__alpha = alpha
        self.__beta = beta

    def simulate(self, step: float, step_size: float) -> np.ndarray:
        """
        simulates a predator prey model
        :param step_size: step size
        :param step: current step
        :return: recordings [step, value]
        """

        self.__value = self._integrator(
            self.__value,
            lambda value: self.__alpha * value - self.__beta * (math.pow(value, 2)),
            step_size,
        )

        return np.array([step, self.__value])
