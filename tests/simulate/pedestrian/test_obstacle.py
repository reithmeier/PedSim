"""
Obstacle tests
"""
import math

import numpy as np
import pytest

from simulate.models.pedestrian.obstacle import Obstacle


@pytest.mark.parametrize(
    "position, radius, repelling_strength",
    [
        [np.array([0.0, 0.0]), 0.0, 0.0],
        [np.array([-10.0, -10.0]), -10.0, -10.0],
        [np.array([10.0, 10.0]), 10.0, 10.0],
        [np.array([-50.0, 50.0]), -50.0, 50.0],
    ],
)
def test_init(position, radius, repelling_strength):
    """valid inputs"""
    # given
    # when
    result = Obstacle(
        position=position, radius=radius, repelling_strength=repelling_strength
    )
    # then
    assert np.allclose(result.position, position)
    assert math.isclose(result.get_radius(), radius)
    assert math.isclose(result.get_repelling_strength(), repelling_strength)
    assert result.__str__() == f"{position} {radius}"
