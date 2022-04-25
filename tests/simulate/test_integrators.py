"""
Integration methods tests
"""
import math

import pytest

from simulate import integrators


@pytest.mark.parametrize(
    "current, step_size, expected",
    [
        [1, 0.01, 1.01],
        [0, 0.01, 0.0],
        [1, 0.1, 1.1],
    ],
)
def test_euler(current, step_size, expected):
    """valid inputs"""
    # given
    # when
    result = integrators.euler(
        current=current, diff_func=lambda v: v, step_size=step_size
    )

    # then
    assert math.isclose(result, expected)


@pytest.mark.parametrize(
    "current, step_size, expected",
    [
        [1, 0.01, 1.01005],
        [0, 0.01, 0.0],
        [1, 0.1, 1.105],
    ],
)
def test_heun(current, step_size, expected):
    """valid inputs"""
    # given
    # when
    result = integrators.heun(
        current=current, diff_func=lambda v: v, step_size=step_size
    )

    # then
    assert math.isclose(result, expected)


@pytest.mark.parametrize(
    "current, step_size, expected",
    [
        [1, 0.01, 1.0100501670833333],
        [0, 0.01, 0.0],
        [1, 0.1, 1.1051708333333334],
    ],
)
def test_runge_kutta(current, step_size, expected):
    """valid inputs"""
    # given
    # when
    result = integrators.runge_kutta(
        current=current, diff_func=lambda v: v, step_size=step_size
    )

    # then
    assert math.isclose(result, expected)
