"""
Obstacle
"""
import numpy as np

from simulate.models.pedestrian.core import Distance, Vec2D


class Obstacle:
    """
    Obstacle
    represents a circular obstacle
    """

    def __init__(
        self,
        position: Vec2D = np.zeros(2, dtype=float),
        radius: Distance = 0.5,
        repelling_strength: float = 2.0,
    ) -> None:
        """
        :param position: position
        :param radius: radius
        :param repelling_strength: repelling force multiplier
        """
        self.position = position
        self.__radius = radius
        self.__repelling_strength = repelling_strength

    def get_radius(self) -> Distance:
        """get radius"""
        return self.__radius

    def get_repelling_strength(self) -> float:
        """get repelling strength"""
        return self.__repelling_strength

    def __str__(self) -> str:
        return f"{self.position} {self.__radius}"
