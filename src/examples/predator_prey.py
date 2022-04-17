"""
Example - Predator Prey Model
"""
from simulate import Simulator
from simulate.kernels import PredatorPreyKernel

import matplotlib.pyplot as plt


def main():
    """
    Simulate using default Predator Prey Kernel
    """
    predator_prey = PredatorPreyKernel()
    sim = Simulator(predator_prey.callback, t_step=0.01, t_max=100)

    sim.run()
    result = sim.progress()

    steps = result[:, 0]
    prey = result[:, 1]
    predators = result[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(steps, prey, color="tab:blue")
    ax.plot(steps, predators, color="tab:orange")

    plt.show()


if __name__ == "__main__":
    main()
