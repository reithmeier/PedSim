#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name="PythonArchetype",
    version="0.0.1",
    description="Empty Python Archetype Project with tox, pylint, black, pytest, coverage",
    author="Lukas Reithmeier",
    url="https://github.com/reithmeier/PythonArchetype",
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # Optional
)
