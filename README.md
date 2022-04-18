# PythonArchetype

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/reithmeier/PedSim/blob/main/LICENSE)
[![Python Tests](https://github.com/reithmeier/PedSim/workflows/Python%20Tests/badge.svg)](https://github.com/reithmeier/PedSim/actions/workflows/python-tests.yml)
[![CodeQL](https://github.com/reithmeier/PedSim/workflows/CodeQL/badge.svg)](https://github.com/reithmeier/PedSim/actions/workflows/codeql-analysis.yml)


## Description

Archetype python project that contains the packages
* black
* pylint
* pytest
* coverage

in addition to a `tox.ini` configuration and a github CodeQL configuration.

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


## Try the examples

### Requirements

* jupyter notebook

### View the examples

Use the jupyter notebook using 
````shell
jupyter notebook
````
Example notebooks are located in `./examples`
