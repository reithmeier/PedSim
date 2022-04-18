"""
Kernel Tests
"""
import numpy as np
import pytest

from simulate.kernels import Kernel, PredatorPreyKernel


def test_kernel_init():
    """are labels used?"""
    # given
    expected = {"a": 1}
    # when
    kernel = Kernel({"a": 1})
    result = kernel.labels()
    # then
    assert result == expected


@pytest.mark.parametrize(
    "step, step_size,expected",
    [
        (0.0, 1.0, [0.0, 1.0]),
        (-1.0, -1.0, [-1.0, -1.0]),
        (12, 12.12, [12.0, 12.12]),
    ],
)
def test_kernel(step, step_size, expected):
    """Initialize a default Kernel"""
    # given

    kernel = Kernel()
    # when
    result = kernel.simulate(step, step_size)

    # then
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "step_size",
    [
        "1",
        [],
    ],
)
def test_kernel_throws(step_size):
    """Wrong input types"""
    # given
    kernel = Kernel()
    # when
    # then
    with pytest.raises(TypeError):
        kernel.simulate(0, step_size)


def test_predator_prey_kernel_labels():
    """valid inputs"""
    # given
    expected = {"step": 0, "prey": 1, "predator": 2}
    kernel = PredatorPreyKernel()
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
    kernel = PredatorPreyKernel()
    # when
    result = kernel.simulate(step, step_size)
    # then
    assert np.allclose(result, expected)
