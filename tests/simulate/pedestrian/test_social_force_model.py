"""
SocialForceModel tests
"""

import numpy as np

from simulate.models.pedestrian import Actor
from simulate.models.pedestrian.obstacle import Obstacle
from simulate.models.pedestrian.social_force_model import SocialForceModel


def test_init():
    """test empty init"""
    # given
    # when
    model = SocialForceModel(integrator=lambda a: a, actors=[], obstacles=[])
    # then
    assert model.labels()["step"] == 0
    assert model.labels()["actors"] == 1


def test_step_no_movement():
    """test with actor that doesn't move"""
    # given
    actors = [
        Actor(
            identifier=0,
            position=np.zeros(2, dtype=float),
            path=[np.zeros(2, dtype=float)],
        )
    ]
    expected = [
        Actor(
            identifier=0,
            position=np.zeros(2, dtype=float),
            path=[np.zeros(2, dtype=float)],
        )
    ]
    model = SocialForceModel(integrator=lambda a: a, actors=actors, obstacles=[])
    labels = model.labels()

    # when
    result = model.simulate(0, 0.01)

    # then
    i = 0
    for actor in actors:
        assert result[labels["actors"]][i] == actor
        i += 1
    i = 0
    for actor in actors:
        assert np.allclose(expected[i].position, actor.position)
        i += 1


def test_step_movement():
    """test with actors that move"""
    # given
    actors = [
        Actor(
            identifier=0,
            position=np.zeros(2, dtype=float),
            path=[np.array([10.0, 10.0])],
        ),
        Actor(
            identifier=0, position=np.array([0.1, 0.1]), path=[np.array([10.0, 10.0])]
        ),
    ]
    start_positions = [np.zeros(2, dtype=float), np.array([0.1, 0.1])]

    model = SocialForceModel(integrator=lambda a: a, actors=actors, obstacles=[])

    # when
    model.simulate(0, 0.01)

    # then
    i = 0
    for actor in actors:
        assert not np.allclose(start_positions[i], actor.position)
    i += 1


def test_step_movement_on_top_of_each_other():
    """test with actors on top of each other"""
    # given
    actors = [
        Actor(
            identifier=0,
            position=np.zeros(2, dtype=float),
            path=[np.array([10.0, 10.0])],
        ),
        Actor(
            identifier=0,
            position=np.zeros(2, dtype=float),
            path=[np.array([10.0, 10.0])],
        ),
    ]
    start_positions = [np.zeros(2, dtype=float), np.zeros(2, dtype=float)]

    model = SocialForceModel(integrator=lambda a: a, actors=actors, obstacles=[])

    # when
    model.simulate(0, 0.01)

    # then
    i = 0
    for actor in actors:
        assert not np.allclose(start_positions[i], actor.position)
    i += 1


def test_step_with_obstacle_no_movement():
    """test with actor and obstacle, no movement"""
    # given
    actors = [
        Actor(
            identifier=0,
            position=np.zeros(2, dtype=float),
            path=[np.zeros(2, dtype=float)],
        )
    ]
    obstacles = [Obstacle(position=np.array([10.0, 10.0]), radius=1.0)]
    start_positions = [np.zeros(2, dtype=float)]

    model = SocialForceModel(integrator=lambda a: a, actors=actors, obstacles=obstacles)

    # when
    model.simulate(0, 0.01)

    # then
    i = 0
    for actor in actors:
        assert np.allclose(start_positions[i], actor.position)
    i += 1


def test_step_with_obstacle_movement():
    """test with actor and obstacle"""
    # given
    actors = [
        Actor(
            identifier=0,
            position=np.zeros(2, dtype=float),
            path=[np.zeros(2, dtype=float)],
        )
    ]
    obstacles = [Obstacle(position=np.array([0.5, 0.5]), radius=1.0)]
    start_positions = [np.zeros(2, dtype=float)]

    model = SocialForceModel(integrator=lambda a: a, actors=actors, obstacles=obstacles)

    # when
    model.simulate(0, 0.01)

    # then
    i = 0
    for actor in actors:
        assert not np.allclose(start_positions[i], actor.position)
    i += 1


def test_step_with_obstacle_on_top_of_each_other():
    """test with actor and obstacle"""
    # given
    actors = [
        Actor(
            identifier=0,
            position=np.zeros(2, dtype=float),
            path=[np.zeros(2, dtype=float)],
        )
    ]
    obstacles = [Obstacle(position=np.zeros(2, dtype=float), radius=1.0)]
    start_positions = [np.zeros(2, dtype=float)]

    model = SocialForceModel(integrator=lambda a: a, actors=actors, obstacles=obstacles)

    # when
    model.simulate(0, 0.01)

    # then
    i = 0
    for actor in actors:
        assert not np.allclose(start_positions[i], actor.position)
    i += 1
