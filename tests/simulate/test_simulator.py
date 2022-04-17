"""
Simulator tests
"""
import math

import numpy as np

from simulate import Simulator
from simulate.kernels import Kernel, PredatorPreyKernel


class SimpleKernel(Kernel):
    def __init__(self):
        self.__val = 0

    def callback(self, t, t_step):
        self.__val += t_step

        return np.array([t, self.__val])

    def labels(self):
        return {"step": 0, "val": 1}


def test_simple_model():
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
    # given
    predator_prey = PredatorPreyKernel()
    sim = Simulator(predator_prey, 0.01, 100)

    # when
    sim.run()
    result = sim.progress()

    # then
    assert result.shape == (10000, 3)
