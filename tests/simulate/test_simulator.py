"""
Simulator tests
"""
import math

import numpy as np

from simulate import Simulator
from simulate.kernels import Kernel, PredatorPreyKernel


class SimpleKernel(Kernel):
    """
    Simple Kernel
    """

    def __init__(self):
        super().__init__(labels={"step": 0, "val": 1})
        self.__val = 0

    def simulate(self, step, step_size):
        self.__val += step_size

        return np.array([step, self.__val])


def test_simple_model():
    """
    test simulator with the SimpleKernel
    :return:
    """
    # given
    simple_kernel = SimpleKernel()
    sim = Simulator(simple_kernel, 0.01, 10)

    # when
    sim.run()
    result = sim.progress()

    # then
    for i in np.arange(0, 1000):
        assert math.isclose(i * 0.01, result[i][0])
        assert math.isclose(i * 0.01 + 0.01, result[i][1])


def test_predator_prey_no_throw():
    """
    Test Simulator using PredatorPreyKernel
    :return:
    """
    # given
    predator_prey = PredatorPreyKernel()
    sim = Simulator(predator_prey, 0.01, 100)

    # when
    sim.run()
    result = sim.progress()

    # then
    assert result.shape == (10000, 3)
