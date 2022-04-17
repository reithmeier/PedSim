import numpy as np

from simulate.kernels import Kernel


class PredatorPreyKernel(Kernel):
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
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma
        self.__delta = delta
        self.__b = start_prey  # prey
        self.__r = start_predators  # predators

    def callback(self, t_step):
        b_ = self.__alpha * self.__b - self.__beta * self.__b * self.__r
        r_ = -self.__gamma * self.__r + self.__delta * self.__b * self.__r

        self.__b += b_ * t_step
        self.__r += r_ * t_step

        return np.array([self.__b, self.__r])

    def labels(self):
        return {"prey": 1, "predator": 2}
