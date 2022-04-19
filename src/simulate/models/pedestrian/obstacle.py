"""
Obstacle
"""
from typing import Tuple

from simulate.models.pedestrian.core import Position


class Obstacle:
    """
    Obstacle
    """

    def __init__(self, bbox: Tuple[Position]):
        """
        :param bbox: bounding box
        """
        self.bbox = bbox

    def __str__(self) -> str:
        return f"{self.bbox}"

    def collides(self):
        """
        :return: true, if an actor collides
        """
        print(self.bbox)
        return True
