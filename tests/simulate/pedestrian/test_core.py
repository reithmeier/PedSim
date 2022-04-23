"""
Core tests
"""
import math

import numpy as np
import pytest

from simulate.models.pedestrian.core import (Distance, Vec2D,
                                             calc_repelling_force, length,
                                             normalize, random_vector)


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


@pytest.mark.parametrize(
    "position, other_position, comfort_zone, other_comfort_zone, expected",
    [
        [np.array([0.0, 0.0]), np.array([0.5, 0.0]), 0.1, 0.1, np.array([0.0, 0.0])],
        [
            np.array([0.0, 0.0]),
            np.array([100.0, 100.0]),
            1.0,
            1.0,
            np.array([0.0, 0.0]),
        ],
        [np.array([0.0, 0.0]), np.array([0.5, 0.0]), 1.0, 1.0, np.array([0.5, 0.0])],
        [np.array([0.0, 0.0]), np.array([-0.5, 0.0]), 1.0, 1.0, np.array([-0.5, 0.0])],
        [np.array([0.0, 0.0]), np.array([0.0, 0.5]), 1.0, 1.0, np.array([0.0, 0.5])],
        [np.array([0.0, 0.0]), np.array([0.0, -0.5]), 1.0, 1.0, np.array([0.0, -0.5])],
        [np.array([0.0, 0.0]), np.array([0.0, 0.5]), 0.0, 0.0, np.array([0.0, 0.0])],
    ],
)
def test_calc_repelling_force(
    position: Vec2D,
    other_position: Vec2D,
    comfort_zone: Distance,
    other_comfort_zone: Distance,
    expected: Vec2D,
):
    """valid inputs"""
    # given
    # when
    result = calc_repelling_force(
        position=position,
        other_position=other_position,
        comfort_zone=comfort_zone,
        other_comfort_zone=other_comfort_zone,
    )
    # then
    assert np.allclose(result, expected)


def test_calc_repelling_force_gives_random():
    """positions on top of each other"""
    # given
    position = np.array([0.0, 0.0])
    other_position = np.array([0.0, 0.0])
    comfort_zone = 0.1
    other_comfort_zone = 0.1
    # when
    result = calc_repelling_force(
        position=position,
        other_position=other_position,
        comfort_zone=comfort_zone,
        other_comfort_zone=other_comfort_zone,
    )
    # then
    assert math.isclose(length(result), 1.0)


def test_random_vector():
    """random vector must have size 1.0"""
    # given
    # when
    result = random_vector()
    # then
    assert math.isclose(length(result), 1.0)
