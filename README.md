# PedSim

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/reithmeier/PedSim/blob/main/LICENSE)
[![Python Tests](https://github.com/reithmeier/PedSim/workflows/Python%20Tests/badge.svg)](https://github.com/reithmeier/PedSim/actions/workflows/python-tests.yml)
[![CodeQL](https://github.com/reithmeier/PedSim/workflows/CodeQL/badge.svg)](https://github.com/reithmeier/PedSim/actions/workflows/codeql-analysis.yml)

## Description

Pedestrian Simulation in python.

### Features

Continuous Simulation using numeric integration. The simulation can be configured using different models and different
integration method.

Supported simulation models:

* Logistic growth model
* Predator prey model
* SIR model

Supported numeric integration methods:

* Euler's method
* Heun's method
* Runge's & Kutta's method
* Actor-based simulation
* Pedestrian simulation models
  * Social force model

### What is still missing?
* Pedestrian simulation models
  * Social force model
    * Repelling Obstacle Force

## Requirements

* python
* pip

## Usage

Install Requirements

````shell
pip install -r requirements.txt
````

Install this project

````shell
pip install -e .
````

Start tox

````shell
tox
````

## Getting Started

### Try the examples

#### Requirements

* jupyter notebook

#### View the examples

Use the jupyter notebook using

````shell
jupyter notebook
````

Example notebooks are located in `./examples`

### Run a simulation

````python
# import simulation packages
from simulate import Simulator
from simulate.integrators import integration_methods
from simulate.models import SIRModel

# initialize the model
model = SIRModel(
    alpha=0.5,
    beta=0.1,
    population=1000,
    # choose a integration method
    integrator=integration_methods.runge_kutta,
)

# initialize the simulator
sim = Simulator(model, step_size=0.01, max_steps=100)

# run the simulation
sim.run()
result = sim.progress()

# extract the result vectors using the labels dictionary
labels = model.labels()
steps = result[:, labels["step"]]
susceptible = result[:, labels["susceptible"]]
infected = result[:, labels["infected"]]
removed = result[:, labels["removed"]]
````

### Write your own simulation model

````python
import numpy as np
from simulate.models import Model

# inherit from Model
class MyModel(Model):

    # override __init__
    def __init__(
            self,
            integrator: callable,
            my_param: float,
            my_start_value: float
    ):
        # labels dictionary maps from label to position in the result of simulate()
        super().__init__(
            integrator=integrator, labels={"step": 0, "my_value": 1}
        )
        self.__my_param = my_param
        self.__my_value = my_start_value

    # override simulate
    def simulate(self, step: float, step_size: float) -> np.ndarray:
        # use the integrator to determine the next value
        self.__my_value = self._integrator(
            self.__my_value,
            # use a custom function to calculate the difference to the next value
            lambda my_value: self.__my_param * my_value,
            step_size,
        )

        # labels must match this return value
        return np.array([step, self.__my_value])
````



## License

PedSim is MIT licensed, as found in the [LICENSE](./LICENSE) file.

