#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name="PedSim",
    version="0.0.1",
    description="Pedestrian Simulation",
    author="Lukas Reithmeier",
    url="https://github.com/reithmeier/PedSim",
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # Optional
)
