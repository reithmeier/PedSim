"""
Model Tests
"""
import numpy as np
import pytest

from simulate import integrators
from simulate.models import (LogisticGrowthModel, Model, PredatorPreyModel,
                             SIRModel)


def integrate(current: float, delta: float, step_size: float) -> float:
    """sample integration method"""
    return current + delta * step_size


def test_model_init():
    """are labels used?"""
    # given
    expected = {"a": 1}
    # when
    model = Model(integrator=integrate, labels={"a": 1})
    result = model.labels()
    # then
    assert result == expected


@pytest.mark.parametrize(
    "step, step_size",
    [
        (0.0, 1.0),
        (-1.0, -1.0),
        (12, 12.12),
    ],
)
def test_model(step, step_size):
    """Initialize a default Model"""
    # given
    model = Model(integrator=integrate)
    # when
    # then
    with pytest.raises(NotImplementedError):
        model.simulate(step, step_size)


def test_predator_prey_model_labels():
    """valid inputs"""
    # given
    expected = {"step": 0, "prey": 1, "predator": 2}
    model = PredatorPreyModel(integrator=integrators.euler)
    # when
    result = model.labels()
    # then
    assert result == expected


@pytest.mark.parametrize(
    "step, step_size, expected",
    [
        [1, 0.1, [1.0, 170.34, 40.08136]],
        [0, 0.0, [0.0, 170.0, 40.0]],
    ],
)
def test_predator_prey_model_callback(step, step_size, expected):
    """valid inputs"""
    # given
    model = PredatorPreyModel(integrators.euler)
    # when
    result = model.simulate(step, step_size)
    # then
    assert np.allclose(result, expected)


def test_logistic_growth_model_labels():
    """valid inputs"""
    # given
    expected = {"step": 0, "value": 1}
    model = LogisticGrowthModel(integrator=integrators.euler)
    # when
    result = model.labels()
    # then
    assert result == expected


@pytest.mark.parametrize(
    "step, step_size, expected",
    [
        [1, 1.0, [1.0, 1.19980676]],
        [0, 0.0, [0.0, 1.0]],
    ],
)
def test_logistic_growth_model_callback(step, step_size, expected):
    """valid inputs"""
    # given
    model = LogisticGrowthModel(
        alpha=1 / 5, beta=1 / 5175, start_value=1, integrator=integrators.euler
    )
    # when
    result = model.simulate(step, step_size)
    # then
    assert np.allclose(result, expected)


def test_sir_model_labels():
    """valid inputs"""
    # given
    expected = {"step": 0, "susceptible": 1, "infected": 2, "removed": 3}
    model = SIRModel(integrator=integrators.euler)
    # when
    result = model.labels()
    # then
    assert result == expected


@pytest.mark.parametrize(
    "step, step_size, expected",
    [
        [1, 0.1, [1.0, 9.98600400e02, 1.09944016e00, 3.29832048e-01]],
        [0, 0.0, [0.0, 999.0, 1.0, 0.0]],
    ],
)
def test_sir_model_callback(step, step_size, expected):
    """valid inputs"""
    # given
    model = SIRModel(alpha=4, beta=3, population=1000, integrator=integrators.euler)
    # when
    result = model.simulate(step, step_size)
    # then
    assert np.allclose(result, expected)
