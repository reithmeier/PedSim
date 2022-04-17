"""
Simulator tests
"""
import math

from simulate import Simulator

import numpy as np
from simulate.kernels import PredatorPreyKernel


class SimpleKernel:
    def __init__(self):
        self.__val = 0

    def callback(self, t, t_step):
        self.__val += t_step

        return np.array([self.__val])


def test_simple_model():
    # given
    simple_kernel = SimpleKernel()
    sim = Simulator(simple_kernel.callback, 0.01, 10)

    # when
    sim.run()
    result = sim.progress()

    # then
    for i in np.arange(0, 1000):
        assert math.isclose(i * 0.01, result[i][0])
        assert math.isclose(i * 0.01 + 0.01, result[i][1])


def test_predator_prey_no_throw():
    # given
    predator_prey = PredatorPreyKernel()
    sim = Simulator(predator_prey.callback, 0.01, 100)

    # when
    sim.run()
    result = sim.progress()

    # then
    assert result.shape == (10000, 3)
