"""
SIRModel
"""
import numpy as np

from .model import Model


class SIRModel(Model):
    """
    SIRModel
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
        :return: recordings [step, susceptible, infected, removed]
        """

        self.__susceptible = self._integrator(
            self.__susceptible,
            lambda susceptible: -self.__alpha
                                * susceptible
                                * self.__infected
                                / self.__population,
            step_size,
        )
        self.__infected = self._integrator(
            self.__infected,
            lambda infected: self.__alpha
                             * self.__susceptible
                             * infected
                             / self.__population
                             - self.__beta * infected,
            step_size,
        )
        self.__removed = self._integrator(
            self.__removed, lambda removed: self.__beta * self.__infected, step_size
        )

        return np.array([step, self.__susceptible, self.__infected, self.__removed])
