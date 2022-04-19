"""
Core

contains definition of core types
"""
from typing import Tuple
import numpy as np
from numpy import ndarray

Speed: type = float  # m/s
Distance: type = float  # m

Position: type = ndarray  # ndarray[float, float]


def main():
    a = np.array([2., 2.])
    b = np.array([1., 1.])
    c = a + b
    print(c)


if __name__ == "__main__":
    main()
