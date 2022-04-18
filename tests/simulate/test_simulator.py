"""
Simulator tests
"""
import math

import numpy as np

from simulate import Simulator
from simulate.integrators import integration_methods
from simulate.models import Model, PredatorPreyModel


def integrate(current: float, delta: float, step_size: float) -> float:
    """sample integration method"""
    return current + delta * step_size


class SampleModel(Model):
    """
    Sample Model
    """

    def __init__(self):
        super().__init__(integrator=integrate, labels={"step": 0, "val": 1})
        self.__val = 0

    def simulate(self, step, step_size):
        self.__val += step_size

        return np.array([step, self.__val])


def test_simple_model():
    """
    test simulator with the SimpleModel
    """
    # given
    simple_model = SampleModel()
    sim = Simulator(simple_model, 0.01, 10)

    # when
    sim.run()
    result = sim.progress()

    # then
    for i in np.arange(0, 1000):
        assert math.isclose(i * 0.01, result[i][0])
        assert math.isclose(i * 0.01 + 0.01, result[i][1])


def test_predator_prey_no_throw():
    """
    Test Simulator using PredatorPreyModel
    """
    # given
    predator_prey = PredatorPreyModel(integrator=integration_methods.euler)
    sim = Simulator(predator_prey, 0.01, 100)

    # when
    sim.run()
    result = sim.progress()

    # then
    assert result.shape == (10000, 3)
