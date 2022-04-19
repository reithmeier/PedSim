"""
Social Force Model
"""
import math
from typing import List

import numpy as np

from .actor import Actor
from ..model import Model


class SocialForceModel(Model):
    """
    SocialForceModel
    simulates a social force model
    """

    def __init__(
            self,
            integrator: callable,
            actors: List[Actor]
    ):
        """

        :param integrator:
        :param actors:
        """
        super().__init__(
            integrator=integrator,
            labels={"step": 0, "actors": 1},
        )
        self.__actors = actors

    def __move_all(self, step_size: float):
        for actor in self.__actors:
            goal = actor.get_next_goal()
            position = actor.get_position()

            # actor walks x [m/s] * step_size per step towards the goal
            movement = np.array([goal[0] - position[0], goal[1] - position[1]])
            movement_length = math.sqrt(movement[0] * movement[0] + movement[1] * movement[1])
            movement = movement / movement_length
            movement = movement * actor.get_max_speed()
            next_pos = position + movement * step_size  # euler
            actor.set_position(next_pos)

    def simulate(self, step: float, step_size: float) -> np.ndarray:
        """
        simulates a predator prey model
        :param step_size: step size
        :param step: current step
        :return: recordings
        """

        self.__move_all(step_size)
        return np.array([step, self.__actors], dtype=object)
