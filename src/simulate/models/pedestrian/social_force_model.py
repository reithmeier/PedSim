"""
Social Force Model
"""

from typing import List

import numpy as np

from ..model import Model
from .actor import Actor
from .core import Vec2D, length, normalize
from .obstacle import Obstacle


def calc_repelling_force(
    position: Vec2D,
    other_position: Vec2D,
    comfort_zone: float,
    other_comfort_zone: float,
) -> Vec2D:
    """
    calculate the repelling force between 2 actors
    or between an actor and an obstacle
    :param position: position of the first actor
    :param other_position: position of the second actor or the obstacle
    :param comfort_zone: comfort zone of the first actor
    :param other_comfort_zone: comfort zone of the second actor
    :return: repelling force
    """
    min_distance = max(comfort_zone, other_comfort_zone)  # minimum acceptable distance
    repelling_force = other_position - position
    distance = length(repelling_force)
    if distance < min_distance:
        return (
            repelling_force
            / distance  # normalize
            * ((min_distance - repelling_force) / min_distance)
        )
    return np.zeros(2, dtype=float)


class SocialForceModel(Model):
    """
    SocialForceModel
    simulates a social force model
    """

    def __init__(
        self, integrator: callable, actors: List[Actor], obstacles: List[Obstacle]
    ):
        """

        :param integrator:
        :param actors:
        :param obstacles:
        """
        super().__init__(
            integrator=integrator,
            labels={"step": 0, "actors": 1},
        )
        self.__obstacles = obstacles
        self.__actors = actors

    def __move_all(self, step_size: float):
        movements: List[np.ndarray] = []

        for actor in self.__actors:
            # move towards next goal
            if actor.has_reached_goal():
                actor.update_goal()

            # actor walks x [m/s] * step_size per step towards the goal
            goal_attraction_force = actor.get_goal() - actor.position
            # scale goal attraction force
            goal_attraction_force = (
                goal_attraction_force
                / length(goal_attraction_force)
                * actor.get_max_speed()
            )

            # add repelling force towards other actors

            other_actors_repelling_force = np.zeros(2, dtype=float)
            for other_actor in self.__actors:
                if other_actor.get_id() != actor.get_id():
                    other_actors_repelling_force += (
                        calc_repelling_force(
                            actor.position,
                            other_actor.position,
                            actor.get_comfort_zone(),
                            other_actor.get_comfort_zone(),
                        )
                        * actor.get_max_speed()
                    )

            # add repelling force towards obstacles
            obstacle_repelling_force = np.zeros(2, dtype=float)
            for obstacle in self.__obstacles:
                obstacle_repelling_force += (
                    calc_repelling_force(
                        actor.position,
                        obstacle.position,
                        actor.get_comfort_zone(),
                        obstacle.get_radius(),
                    )
                    * obstacle.get_repelling_strength()
                )

            # random force
            random_force = normalize(
                0.5 - np.random.rand(2)
            )  # numbers in interval [-0.5, 0.5]

            # sum up all forces
            movements.append(
                goal_attraction_force
                - other_actors_repelling_force
                - obstacle_repelling_force
                + random_force
            )  # euler

        # update positions
        i = 0
        for actor in self.__actors:
            actor.position = actor.position + movements[i] * step_size
            i += 1

    def simulate(self, step: float, step_size: float) -> np.ndarray:
        """
        simulates a predator prey model
        :param step_size: step size
        :param step: current step
        :return: recordings
        """

        self.__move_all(step_size)
        return np.array([step, self.__actors], dtype=object)
