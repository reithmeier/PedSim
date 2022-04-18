"""
Example - Logistic Growth Model
"""

import matplotlib.pyplot as plt

from simulate import Simulator
from simulate.integrators import integration_methods
from simulate.kernels import LogisticGrowthKernel


def main():
    """
    Simulate using Logistic Growth Kernel
    """
    kernel = LogisticGrowthKernel(
        start_value=1,
        alpha=0.2,
        beta=0.0001,
        integrator=integration_methods.euler,
    )
    sim = Simulator(kernel, step_size=0.01, max_steps=100)

    sim.run()
    result = sim.progress()

    labels = kernel.labels()

    steps = result[:, labels["step"]]
    value = result[:, labels["value"]]

    fig = plt.figure()
    axis = fig.add_subplot()
    axis.plot(steps, value, color="tab:blue")

    plt.show()


if __name__ == "__main__":
    main()
