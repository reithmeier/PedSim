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


def length(vec: Vec2D) -> float:
    """
    length of a 2 sized vector
    :param vec: vector
    :return: length
    """
    return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])


def normalize(vec: Vec2D) -> Vec2D:
    """
    transforms a vector in a direction vector with length 1
    :param vec: vector to normalize
    :return: vector with the direction of a with length 1
    """
    vector_length = length(vec)
    if vector_length == 0.0:
        raise ValueError(f"Vector {vec} has length of 0")
    return vec / vector_length
