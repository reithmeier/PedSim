"""
Core

contains definition of core types
"""
import math

from numpy import ndarray

Speed: type = float  # m/s
Distance: type = float  # m

Vec2D: type = ndarray  # ndarray[float, float]

Identifier: type = int


def length(a: Vec2D) -> float:
    """
    length of a 2 sized vector
    :param a: vector
    :return: length
    """
    return math.sqrt(a[0] * a[0] + a[1] * a[1])


def normalize(a: Vec2D) -> Vec2D:
    """
    transforms a vector in a direction vector with length 1
    :param a: vector to normalize
    :return: vector with the direction of a with length 1
    """
    return a / length(a)
