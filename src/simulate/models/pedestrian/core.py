"""
Core

contains definition of core types
"""
import math

import numpy as np

# Type definitions
Speed: type = float  # m/s

Distance: type = float  # m

Vec2D: type = np.ndarray  # ndarray[float, float]

Identifier: type = int


# functions


def length(vec: Vec2D) -> Distance:
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


def random_vector() -> Vec2D:
    """
    :return: vector in random direction of length 1
    """
    return normalize(0.5 - np.random.rand(2))


def calc_repelling_force(
    position: Vec2D,
    other_position: Vec2D,
    comfort_zone: Distance,
    other_comfort_zone: Distance,
) -> Vec2D:
    """
    calculate the repelling force between 2 actors
    or between an actor and an obstacle
    :param position: position of the first actor
    :param other_position: position of the second actor or the obstacle
    :param comfort_zone: comfort zone of the first actor
    :param other_comfort_zone: comfort zone of the second actor
    :return: repelling force
    """
    # minimum acceptable distance
    min_distance: Distance = max(comfort_zone, other_comfort_zone)
    # calc repelling force vector
    repelling_force: Vec2D = other_position - position
    # calc distance between positions
    distance = length(repelling_force)

    if math.isclose(distance, 0.0):
        # both are on top of each other
        # force in random direction
        return random_vector()

    if distance < min_distance:
        # both are close to each other
        return (
            (repelling_force / distance)  # normalize
            # the closer, the bigger the force
            * ((min_distance - distance) / min_distance)
        )

    # both are far from each other
    # no force
    return np.zeros(2, dtype=float)
