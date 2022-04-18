"""
Various numerical integration methods
"""


def euler(current: float, diff_func: callable, step_size: float) -> float:
    """
    This integrator uses the Euler Method to perform numeric integration
    https://en.wikipedia.org/wiki/Euler_method
    q_i+1 = q_i + h * f(q_i)
    local error: step_size²
    global error: step_size
    :param step_size: step size
    :param current: current value
    :param diff_func: function to calculate difference to add from current value
    :return: integrated value using euler's method
    """
    return current + diff_func(current) * step_size


def heun(current: float, diff_func: callable, step_size: float) -> float:
    """
    This integrator uses the Heun Method to perform numeric integration
    https://en.wikipedia.org/wiki/Heun%27s_method
    q_i+1 = q_i + h/2 * (f(q_i) + f(q_i + h * f(q_i)))
    local error: step_size³
    global error: step_size²
    :param step_size: step size
    :param current: current value
    :param diff_func: function to calculate difference to add from current value
    :return: integrated value using heun's method
    """

    return current + step_size / 2 * (
        diff_func(current) + diff_func(current + step_size * diff_func(current))
    )


def runge_kutta(current: float, diff_func: callable, step_size: float) -> float:
    """
    This integrator uses the Runge Kutta Method to perform numeric integration
    https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

    k_1 = h * f(q_i)
    k_2 = h * f(q_i + 1/2 * k_1)
    k_3 = h * f(q_i + 1/2 * k_2)
    k_4 = h * f(q_i + k_3)
    q_i+1 = q_i + 1/6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)

    local error: step_size^5
    global error: step_size^4
    :param step_size: step size
    :param current: current value
    :param diff_func: function to calculate difference to add from current value
    :return: integrated value using the runge kutta method
    """
    k_1 = step_size * diff_func(current)
    k_2 = step_size * diff_func(current + 1 / 2 * k_1)
    k_3 = step_size * diff_func(current + 1 / 2 * k_2)
    k_4 = step_size * diff_func(current + k_3)

    return current + 1 / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
