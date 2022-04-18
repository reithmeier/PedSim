"""
Example - SIR Model
"""

import matplotlib.pyplot as plt

from simulate import Simulator
from simulate.integrators import integration_methods
from simulate.kernels import SIRKernel


def main():
    """
    Simulate using SIR Kernel
    """
    kernel = SIRKernel(
        alpha=0.5,
        beta=0.1,
        population=1000,
        integrator=integration_methods.euler,
    )
    sim = Simulator(kernel, step_size=0.01, max_steps=100)

    sim.run()
    result = sim.progress()

    labels = kernel.labels()

    steps = result[:, labels["step"]]
    susceptible = result[:, labels["susceptible"]]
    infected = result[:, labels["infected"]]
    removed = result[:, labels["removed"]]

    fig = plt.figure()
    axis = fig.add_subplot()
    axis.plot(steps, susceptible, color="tab:blue")
    axis.plot(steps, infected, color="tab:red")
    axis.plot(steps, removed, color="tab:green")

    plt.show()


if __name__ == "__main__":
    main()
