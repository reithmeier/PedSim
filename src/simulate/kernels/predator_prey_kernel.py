"""
PredatorPreyKernel
"""
import numpy as np

from .kernel import Kernel


class PredatorPreyKernel(Kernel):
    """
    PredatorPreyKernel
    simulates a predator prey model
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.002,
        gamma: float = 0.15,
        delta: float = 0.001,
        start_prey: float = 170,
        start_predators: float = 40,
    ):
        """
        :param alpha: predator prey model alpha parameter
        :param beta: predator prey model beta parameter
        :param gamma: predator prey model gamma parameter
        :param delta: predator prey model delta parameter
        :param start_prey: initial number of prey animals
        :param start_predators: initial number of perdator animals
        """
        super().__init__(labels={"step": 0, "prey": 1, "predator": 2})
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma
        self.__delta = delta
        self.__prey = start_prey  # prey
        self.__predators = start_predators  # predators

    def simulate(self, step, step_size) -> np.ndarray:
        """
        simulates a predator prey model
        :param step: current step
        :param step_size: step size
        :return: recordings [prey, predator]
        """
        diff_prey = (
            self.__alpha * self.__prey - self.__beta * self.__prey * self.__predators
        )
        diff_predators = (
            -self.__gamma * self.__predators
            + self.__delta * self.__prey * self.__predators
        )

        self.__prey += diff_prey * step_size
        self.__predators += diff_predators * step_size

        return np.array([step, self.__prey, self.__predators])
