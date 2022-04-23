"""
Social Force Model
"""

from typing import List

import numpy as np

from ..model import Model
from .actor import Actor
from .core import calc_repelling_force, normalize, random_vector
from .obstacle import Obstacle


class SocialForceModel(Model):
    """
    SocialForceModel
    simulates a social force model
    """

    def __init__(
        self, integrator: callable, actors: List[Actor], obstacles: List[Obstacle]
    ) -> None:
        """
        :param integrator: integrator method
        :param actors: actors
        :param obstacles: obstacles
        """
        super().__init__(
            integrator=integrator,
            labels={"step": 0, "actors": 1},
        )
        self.__obstacles = obstacles
        self.__actors = actors

    def __move_all(self) -> List[np.ndarray]:
        """
        moves all actors
        :return movement of all actors
        """
        movements: List[np.ndarray] = []

        for actor in self.__actors:
            total_force = np.zeros(2, dtype=float)

            if not actor.has_arrived():
                if actor.has_reached_goal():
                    actor.update_goal()

                # move towards next goal
                # actor walks x [m/s] * step_size per step towards the goal
                goal_attraction_force = actor.get_goal() - actor.position
                # add to total_force
                total_force += normalize(goal_attraction_force) * actor.get_max_speed()

                # random force
                # actors move randomly to prevent locks
                total_force += random_vector()
                # numbers in interval [-0.5, 0.5]

            # add repelling force towards other actors
            for other_actor in self.__actors:
                # don't repel self
                if other_actor.get_id() != actor.get_id():
                    total_force -= (
                        calc_repelling_force(
                            actor.position,
                            other_actor.position,
                            actor.get_comfort_zone(),
                            other_actor.get_comfort_zone(),
                        )
                        * actor.get_max_speed()
                    )

            # add repelling force towards obstacles
            for obstacle in self.__obstacles:
                total_force -= (
                    calc_repelling_force(
                        actor.position,
                        obstacle.position,
                        actor.get_comfort_zone(),
                        obstacle.get_radius(),
                    )
                    * obstacle.get_repelling_strength()
                )

            movements.append(total_force)
        return movements

    def simulate(self, step: float, step_size: float) -> np.ndarray:
        """
        simulates a social force model
        :param step_size: step size
        :param step: current step
        :return: recordings
        """

        movements = self.__move_all()
        # update positions
        i = 0
        for actor in self.__actors:
            # currently, euler is enforced
            actor.position = actor.position + movements[i] * step_size
            i += 1
        return np.array([step, self.__actors], dtype=object)
