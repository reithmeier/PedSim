"""
Various numerical integration methods
"""


def euler(current: float, delta: float, step_size: float) -> float:
    """
    This integrator uses the Euler Method perform numeric integration
    https://en.wikipedia.org/wiki/Euler_method
    :param step_size: step size
    :param current: current value
    :param delta: difference to add
    :return: current + delta * step_size
    """
    return current + delta * step_size
