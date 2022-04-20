"""
Social Force Model
"""
import math
from typing import List

import numpy as np

from ..model import Model
from .actor import Actor
from .core import Position
from .obstacle import Obstacle


def length(a: Position):
    """
    length of a 2 sized vector
    :param a: vector
    :return: length
    """
    return math.sqrt(a[0] * a[0] + a[1] * a[1])


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
        for actor in self.__actors:

            # actor walks x [m/s] * step_size per step towards the goal
            goal_attraction_force = actor.get_goal() - actor.position
            # scale goal attraction force
            goal_attraction_force = (
                goal_attraction_force
                / length(goal_attraction_force)
                * actor.get_max_speed()
            )

            # add repelling force towards other actors
            comfort_zone = 0.5
            other_actors_repelling_force = np.array([0.0, 0.0])
            for other_actor in self.__actors:
                if other_actor.get_id() != actor.get_id():
                    other_position = other_actor.position
                    repelling_force = other_position - actor.position
                    repelling_force_length = length(repelling_force)
                    if repelling_force_length < comfort_zone:
                        repelling_force = (
                            repelling_force
                            / repelling_force_length
                            * ((comfort_zone - repelling_force) / comfort_zone)
                            * actor.get_max_speed()
                        )
                        other_actors_repelling_force += repelling_force

            # add repelling force towards obstacles
            for obstacle in self.__obstacles:
                bbox = obstacle.bbox
                print(bbox)
                # check for collision

            # random force
            random_force = np.random.rand(2)
            random_force = random_force / length(random_force)

            actor.position = (
                actor.position
                + (goal_attraction_force - other_actors_repelling_force + random_force)
                * step_size
            )  # euler

            # move towards next goal
            if actor.has_reached_goal():
                actor.update_goal()

    def simulate(self, step: float, step_size: float) -> np.ndarray:
        """
        simulates a predator prey model
        :param step_size: step size
        :param step: current step
        :return: recordings
        """

        self.__move_all(step_size)
        return np.array([step, self.__actors], dtype=object)
