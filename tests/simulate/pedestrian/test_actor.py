"""
Actor tests
"""
import math

import numpy as np
import pytest

from simulate.models.pedestrian.actor import Actor


@pytest.mark.parametrize(
    "identifier, position, path, arrival_tolerance, max_speed, comfort_zone",
    [
        [0, np.array([0.0, 0.0]), [], 0.1, 0.3, 0.5],
        [1, np.array([1.0, 0.0]), [], -0.1, 0.3, -0.5],
        [2, np.array([0.0, -1.0]), [], 0.1, -0.3, 0.5],
        [3, np.array([1.0, -1.0]), [], -0.1, 0.3, 0.5],
    ],
)
def test_init(identifier, position, path, arrival_tolerance, max_speed, comfort_zone):
    """valid inputs"""
    # given
    # when
    result = Actor(
        identifier=identifier,
        position=position,
        path=path,
        arrival_tolerance=arrival_tolerance,
        max_speed=max_speed,
        comfort_zone=comfort_zone,
    )
    # then
    assert result.get_id() == identifier
    assert np.allclose(result.position, position)
    assert np.allclose(result.get_path(), [np.array([0.0, 0.0])])
    assert math.isclose(result.get_max_speed(), max_speed)
    assert math.isclose(result.get_comfort_zone(), comfort_zone)
    assert np.allclose(result.get_goal(), np.array([0.0, 0.0]))
    assert (
        result.__str__()
        == f"{identifier} {position} {np.array([0.0, 0.0])} {max_speed}"
    )


@pytest.mark.parametrize(
    "position, arrival_tolerance, path, expected",
    [
        [np.array([0.0, 0.0]), 0.1, [np.array([0.0, 0.0])], True],
        [np.array([0.01, 0.01]), 0.1, [np.array([0.0, 0.0])], True],
        [np.array([0.0, 0.0]), 0.0, [np.array([0.0, 0.0])], False],
        [np.array([1.0, 0.0]), 0.1, [np.array([0.0, 0.0])], False],
        [np.array([0.0, 0.0]), 0.1, [np.array([0.0, 1.0])], False],
        [np.array([0.0, 0.0]), -0.1, [np.array([0.0, 0.0])], False],
        [np.array([1.0, 0.0]), 0.1, [np.array([1.0, 0.0])], True],
        [np.array([1.0, -1.0]), 0.1, [np.array([1.0, -1.0])], True],
    ],
)
def test_has_reached_goal(position, arrival_tolerance, path, expected):
    """valid inputs"""
    # given
    actor = Actor(position=position, arrival_tolerance=arrival_tolerance, path=path)
    # when
    result = actor.has_reached_goal()
    # then
    assert result == expected


@pytest.mark.parametrize(
    "path, expected",
    [
        [[np.array([1.0, 1.0])], True],
        [[np.array([1.0, 1.0]), np.array([2.0, 2.0])], False],
        [[], True],
    ],
)
def test_has_arrived(path, expected):
    """valid inputs"""
    # given
    actor = Actor(path=path)
    actor.update_goal()
    # when
    result = actor.has_arrived()
    # then
    assert result == expected


@pytest.mark.parametrize(
    "path, expected",
    [
        [[np.array([0.0, 0.0])], np.array([0.0, 0.0])],
        [[np.array([0.0, 0.0]), np.array([0.0, 0.0])], np.array([0.0, 0.0])],
        [[np.array([0.0, 0.0]), np.array([1.0, 1.0])], np.array([1.0, 1.0])],
        [
            [np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([2.0, 2.0])],
            np.array([2.0, 2.0]),
        ],
        [[], np.array([0.0, 0.0])],
    ],
)
def test_update_goal(path, expected):
    """valid inputs"""
    # given
    actor = Actor(path=path)
    # when
    actor.update_goal()
    # then
    assert np.allclose(actor.get_goal(), expected)
