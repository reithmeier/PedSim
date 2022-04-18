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
        integrator: callable,
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
        :param start_predators: initial number of predator animals
        """
        super().__init__(
            integrator=integrator, labels={"step": 0, "prey": 1, "predator": 2}
        )
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma
        self.__delta = delta
        self.__prey = start_prey
        self.__predators = start_predators

    def simulate(self, step: float, step_size: float) -> np.ndarray:
        """
        simulates a predator prey model
        :param step_size: step size
        :param step: current step
        :return: recordings [prey, predator]
        """
        diff_prey = (
            self.__alpha * self.__prey - self.__beta * self.__prey * self.__predators
        )
        diff_predators = (
            -self.__gamma * self.__predators
            + self.__delta * self.__prey * self.__predators
        )

        self.__prey = self._integrator(self.__prey, diff_prey, step_size)
        self.__predators = self._integrator(self.__predators, diff_predators, step_size)

        return np.array([step, self.__prey, self.__predators])
