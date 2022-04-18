"""
Kernel Tests
"""
import numpy as np
import pytest

from simulate.integrators import integration_methods
from simulate.kernels import Kernel, PredatorPreyKernel


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
