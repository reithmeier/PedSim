"""
Simulator tests
"""
import math

import numpy as np

from simulate import Simulator
from simulate.integrators import integration_methods
from simulate.kernels import Kernel, PredatorPreyKernel


def integrate(current: float, delta: float, step_size: float) -> float:
    """sample integration method"""
    return current + delta * step_size


class SampleKernel(Kernel):
    """
    Sample Kernel
    """

    def __init__(self):
        super().__init__(integrator=integrate, labels={"step": 0, "val": 1})
        self.__val = 0

    def simulate(self, step, step_size):
        self.__val += step_size

        return np.array([step, self.__val])


def test_simple_model():
    """
    test simulator with the SimpleKernel
    """
    # given
    simple_kernel = SampleKernel()
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
    """
    # given
    predator_prey = PredatorPreyKernel(integrator=integration_methods.euler)
    sim = Simulator(predator_prey, 0.01, 100)

    # when
    sim.run()
    result = sim.progress()

    # then
    assert result.shape == (10000, 3)
