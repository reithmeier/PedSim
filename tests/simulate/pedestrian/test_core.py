import math

import numpy as np
import pytest

from simulate.models.pedestrian.core import Vec2D, length, normalize


@pytest.mark.parametrize(
    "vec, expected",
    [
        [np.array([0.0, 1.0]), 1.0],
        [np.array([1.0, 0.0]), 1.0],
        [np.array([1.0, 1.0]), math.sqrt(2)],
        [np.array([0.0, 0.0]), 0.0],
        [np.array([-1.0, 0.0]), 1.0],
        [np.array([0.0, -1.0]), 1.0],
        [np.array([-1.0, -1.0]), math.sqrt(2)],
    ],
)
def test_length(vec: Vec2D, expected: float):
    """valid inputs"""
    # given
    # when
    result = length(vec)
    # then
    assert math.isclose(result, expected)


@pytest.mark.parametrize(
    "vec, expected",
    [
        [np.array([0.0, 1.0]), np.array([0.0, 1.0])],
        [np.array([1.0, 0.0]), np.array([1.0, 0.0])],
        [np.array([1.0, 1.0]), np.array([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)])],
        [np.array([-1.0, 0.0]), np.array([-1.0, 0.0])],
        [np.array([0.0, -1.0]), np.array([0.0, -1.0])],
        [np.array([-1.0, -1.0]), np.array([-1.0 / math.sqrt(2), -1.0 / math.sqrt(2)])],
    ],
)
def test_normalize(vec: Vec2D, expected: Vec2D):
    """valid inputs"""
    # given
    # when
    result = normalize(vec)
    # then
    assert np.allclose(result, expected)


def test_normalize_throws():
    """invalid inputs"""
    # given
    vec = np.array([0.0, 0.0])
    # when
    # then
    with pytest.raises(ValueError):
        normalize(vec)
