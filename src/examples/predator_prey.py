"""
Example - Predator Prey Model
"""

import matplotlib.pyplot as plt

from simulate import Simulator
from simulate.integrators import integration_methods
from simulate.kernels import PredatorPreyKernel


def main():
    """
    Simulate using Predator Prey Kernel
    """
    kernel = PredatorPreyKernel(
        alpha=0.4,
        beta=0.008,
        gamma=0.3,
        delta=0.001,
        start_prey=500,
        start_predators=5,
        integrator=integration_methods.euler,
    )
    sim = Simulator(kernel, step_size=0.01, max_steps=100)

    sim.run()
    result = sim.progress()

    labels = kernel.labels()

    steps = result[:, labels["step"]]
    prey = result[:, labels["prey"]]
    predators = result[:, labels["predator"]]

    fig = plt.figure()
    axis = fig.add_subplot()
    axis.plot(steps, prey, color="tab:blue")
    axis.plot(steps, predators, color="tab:orange")

    plt.show()


if __name__ == "__main__":
    main()
