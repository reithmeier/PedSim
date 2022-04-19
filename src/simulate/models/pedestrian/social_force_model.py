"""
Social Force Model
"""
import math
from typing import List

import numpy as np

from ..model import Model
from .actor import Actor
from .obstacle import Obstacle


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
            goal = actor.get_goal()
            position = actor.position

            # actor walks x [m/s] * step_size per step towards the goal
            movement = np.array([goal[0] - position[0], goal[1] - position[1]])
            movement_length = math.sqrt(
                movement[0] * movement[0] + movement[1] * movement[1]
            )
            movement = movement / movement_length
            movement = movement * actor.get_max_speed()
            # add repelling force towards other actors
            # add repelling force towards obstacles
            for obstacle in self.__obstacles:
                bbox = obstacle.bbox
                print(bbox)
                # check for collision

            next_pos = position + movement * step_size  # euler
            actor.position = next_pos

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
