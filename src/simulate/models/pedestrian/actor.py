"""
Actor
"""

from typing import List

import numpy as np

from .core import Distance, Identifier, Speed, Vec2D


class Actor:
    """
    Actor
    """

    def __init__(
        self,
        identifier: Identifier = 0,
        position: Vec2D = np.zeros(2, dtype=float),
        path: List[Vec2D] = None,
        arrival_tolerance: Distance = 0.1,
        max_speed: Speed = 1.3,
        comfort_zone: Distance = 0.5,
    ):
        """
        :param position: position [m]
        :param path:
        :param arrival_tolerance: minimum distance to destination, \
         so that an arrival is determined [m]
        :param max_speed: maximum speed [m/s]
        :param comfort_zone: comfort zone of the actor [m]
        """
        if path is None:
            path = [np.zeros(2, dtype=float)]

        self.__id = identifier
        self.position = position
        self.__path = path
        self.__current_edge = 0  # actor starts at edge 0 of the path
        self.__arrival_tolerance = arrival_tolerance
        self.__max_speed = max_speed
        self.__comfort_zone = comfort_zone

    def get_id(self) -> Identifier:
        """get id"""
        return self.__id

    def get_path(self) -> List[Vec2D]:
        """get path"""
        return self.__path

    def get_goal(self) -> Vec2D:
        """get goal"""
        return self.__path[self.__current_edge]

    def get_max_speed(self) -> Speed:
        """get max speed"""
        return self.__max_speed

    def get_comfort_zone(self) -> Distance:
        """get comfort zone"""
        return self.__comfort_zone

    def has_reached_goal(self) -> bool:
        """
        :return: true, if current goal is reached
        """
        goal = self.get_goal()
        goal_x = goal[0]
        goal_y = goal[1]
        pos_x = self.position[0]
        pos_y = self.position[1]

        # use manhattan distance
        arrived_x = abs(pos_x - goal_x) < self.__arrival_tolerance
        arrived_y = abs(pos_y - goal_y) < self.__arrival_tolerance

        return arrived_x and arrived_y

    def update_goal(self) -> None:
        """
        updates the goal
        """
        if self.__current_edge < (len(self.__path) - 1):
            self.__current_edge += 1

    def __str__(self) -> str:
        return f"{self.__id} {self.position} {self.get_goal()} {self.__max_speed}"
