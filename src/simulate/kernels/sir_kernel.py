"""
SIRKernel
"""
import numpy as np

from .kernel import Kernel


class SIRKernel(Kernel):
    """
    SIRKernel
    simulates a SIR model
    """

    def __init__(
        self,
        integrator: callable,
        alpha: float = 4.0,
        beta: float = 3.0,
        population: int = 1000,
    ):
        """
        :param alpha: infection rate
        :param beta: recovery rate
        :param population: total population
        """
        super().__init__(
            integrator=integrator,
            labels={"step": 0, "susceptible": 1, "infected": 2, "removed": 3},
        )
        self.__alpha = alpha
        self.__beta = beta
        self.__population = population
        self.__susceptible = self.__population - 1
        self.__infected = 1
        self.__removed = 0

    def simulate(self, step: float, step_size: float) -> np.ndarray:
        """
        simulates a predator prey model
        :param step_size: step size
        :param step: current step
        :return: recordings [prey, predator]
        """
        susceptible_diff = (
            -self.__alpha * self.__susceptible * self.__infected / self.__population
        )
        infected_diff = (
            self.__alpha * self.__susceptible * self.__infected / self.__population
            - self.__beta * self.__infected
        )
        removed_diff = self.__beta * self.__infected

        self.__susceptible = self._integrator(
            self.__susceptible, susceptible_diff, step_size
        )
        self.__infected = self._integrator(self.__infected, infected_diff, step_size)
        self.__removed = self._integrator(self.__removed, removed_diff, step_size)

        return np.array([step, self.__susceptible, self.__infected, self.__removed])
