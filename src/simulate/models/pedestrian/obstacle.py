from typing import Tuple

from simulate.models.pedestrian.core import Position


class Obstacle:

    def __init__(self, bbox: Tuple[Position]):
        self.__bbox = bbox

    def get_bbox(self):
        return self.__bbox

    def set_bbox(self, value):
        self.__bbox = value
