"""
Kernel Tests
"""
import numpy as np
import pytest

from simulate.integrators import integration_methods
from simulate.kernels import (Kernel, LogisticGrowthKernel, PredatorPreyKernel,
                              SIRKernel)


def integrate(current: float, delta: float, step_size: float) -> float:
    """sample integration method"""
    return current + delta * step_size


def test_kernel_init():
    """are labels used?"""
    # given
    expected = {"a": 1}
    # when
    kernel = Kernel(integrator=integrate, labels={"a": 1})
    result = kernel.labels()
    # then
    assert result == expected


@pytest.mark.parametrize(
    "step, step_size",
    [
        (0.0, 1.0),
        (-1.0, -1.0),
        (12, 12.12),
    ],
)
def test_kernel(step, step_size):
    """Initialize a default Kernel"""
    # given
    kernel = Kernel(integrator=integrate)
    # when
    # then
    with pytest.raises(NotImplementedError):
        kernel.simulate(step, step_size)


def test_predator_prey_kernel_labels():
    """valid inputs"""
    # given
    expected = {"step": 0, "prey": 1, "predator": 2}
    kernel = PredatorPreyKernel(integrator=integration_methods.euler)
    # when
    result = kernel.labels()
    # then
    assert result == expected


@pytest.mark.parametrize(
    "step, step_size, expected",
    [
        [1, 0.1, [1.0, 170.34, 40.08]],
        [0, 0.0, [0.0, 170.0, 40.0]],
    ],
)
def test_predator_prey_kernel_callback(step, step_size, expected):
    """valid inputs"""
    # given
    kernel = PredatorPreyKernel(integration_methods.euler)
    # when
    result = kernel.simulate(step, step_size)
    # then
    assert np.allclose(result, expected)


def test_logistic_growth_kernel_labels():
    """valid inputs"""
    # given
    expected = {"step": 0, "value": 1}
    kernel = LogisticGrowthKernel(integrator=integration_methods.euler)
    # when
    result = kernel.labels()
    # then
    assert result == expected


@pytest.mark.parametrize(
    "step, step_size, expected",
    [
        [1, 1.0, [1.0, 1.19980676]],
        [0, 0.0, [0.0, 1.0]],
    ],
)
def test_logistic_growth_kernel_callback(step, step_size, expected):
    """valid inputs"""
    # given
    kernel = LogisticGrowthKernel(
        alpha=1 / 5, beta=1 / 5175, start_value=1, integrator=integration_methods.euler
    )
    # when
    result = kernel.simulate(step, step_size)
    # then
    assert np.allclose(result, expected)


def test_sir_kernel_labels():
    """valid inputs"""
    # given
    expected = {"step": 0, "susceptible": 1, "infected": 2, "removed": 3}
    kernel = SIRKernel(integrator=integration_methods.euler)
    # when
    result = kernel.labels()
    # then
    assert result == expected


@pytest.mark.parametrize(
    "step, step_size, expected",
    [
        [1, 0.1, [1.0, 9.986004e02, 1.099600e00, 3.000000e-01]],
        [0, 0.0, [0.0, 999.0, 1.0, 0.0]],
    ],
)
def test_sir_kernel_callback(step, step_size, expected):
    """valid inputs"""
    # given
    kernel = SIRKernel(
        alpha=4, beta=3, population=1000, integrator=integration_methods.euler
    )
    # when
    result = kernel.simulate(step, step_size)
    # then
    assert np.allclose(result, expected)
