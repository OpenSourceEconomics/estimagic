"""Test suite for linear trust-region subsolvers."""

import math

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.optimizers._pounders.linear_subsolvers import (
    LinearModel,
    improve_geomtery_trsbox_linear,
    minimize_trsbox_linear,
)


@pytest.mark.parametrize(
    "model_gradient, lower_bounds, upper_bounds, delta, expected",
    [
        (
            np.array([1.0, 0.0, 1.0]),
            -np.ones(3),
            np.ones(3),
            2.0,
            np.array([-1.0, 0.0, -1.0]),
        ),
        (
            np.array([0.00028774, 0.00763968, 0.01217268]),
            -np.ones(3),
            np.ones(3),
            9.5367431640625e-05,
            np.array([-1.90902854e-06, -5.06859218e-05, -8.07603861e-05]),
        ),
        (
            np.array([0.00028774, 0.00763968, 0.01217268]),
            np.array([0, -1, -1]),
            np.ones(3),
            0.1,
            np.array([0.0, -5.31586927e-02, -8.47003742e-02]),
        ),
        (
            np.arange(5) * 0.1,
            -np.ones(5),
            np.ones(5),
            0.1,
            np.array([0.0, -0.01825742, -0.03651484, -0.05477226, -0.07302967]),
        ),
        (
            np.arange(4, -1, -1) * 0.1,
            -np.ones(5),
            np.ones(5),
            0.1,
            np.array([-0.07302967, -0.05477226, -0.03651484, -0.01825742, 0]),
        ),
        (
            np.arange(5) * 0.1,
            np.array([-1, -1, 0, -1, -1]),
            np.array([1, 1, 0.2, 0.2, 1]),
            0.1,
            np.array([0.0, -1.96116135e-02, 0.0, -5.88348405e-02, -7.84464541e-02]),
        ),
        (
            np.arange(4, -1, -1) * 0.1,
            np.array([-1, -1, -1, -1, 0]),
            np.array([0.3, 0.3, 1, 1, 1]),
            0.1,
            np.array([-0.07302967, -0.05477226, -0.03651484, -0.01825742, 0.0]),
        ),
    ],
)
def test_trsbox_linear(model_gradient, lower_bounds, upper_bounds, delta, expected):
    linear_model = LinearModel(linear_terms=model_gradient)

    x_out = minimize_trsbox_linear(linear_model, lower_bounds, upper_bounds, delta)
    aaae(x_out, expected)


@pytest.mark.parametrize(
    "x_center, c_term, model_gradient, lower_bounds, upper_bounds, delta, expected",
    [
        (
            np.array([0.0, 0.0]),
            -1.0,
            np.array([1.0, -1.0]),
            np.array([-2.0, -2.0]),
            np.array([1.0, 2.0]),
            2.0,
            np.array([-math.sqrt(2.0), math.sqrt(2.0)]),
        ),
        (
            np.array([0.0, 0.0]),
            -1.0,
            np.array([1.0, -1.0]),
            np.array([-2.0, -2.0]),
            np.array([1.0, 2.0]),
            5.0,
            np.array([-2.0, 2.0]),
        ),
        (
            np.array([0.0, 0.0]) + 1,
            3.0,
            np.array([1.0, -1.0]),
            np.array([-2.0, -2.0]) + 1,
            np.array([1.0, 2.0]) + 1,
            5.0,
            np.array([1.0, -2.0]) + 1,
        ),
        (
            np.array([0.0, 0.0]),
            -1.0,
            np.array([-1.0, -1.0]),
            np.array([-2.0, -2.0]),
            np.array([0.1, 0.9]),
            math.sqrt(2.0),
            np.array([0.1, 0.9]),
        ),
        (
            np.array([0.0, 0.0, 0.0]),
            -1.0,
            np.array([-1.0, -1.0, -1.0]),
            np.array([-2.0, -2.0, -2.0]),
            np.array([0.9, 0.1, 5.0]),
            math.sqrt(3.0),
            np.array([0.9, 0.1, math.sqrt(3.0 - 0.81 - 0.01)]),
        ),
        (
            np.array([0.0, 0.0]),
            0.0,
            np.array([1e-15, -1.0]),
            np.array([-2.0, -2.0]),
            np.array([1.0, 2.0]),
            5.0,
            np.array([0.0, 2.0]),
        ),
        (
            np.array([0.0, 0.0]),
            0.0,
            np.array([1e-15, 0.0]),
            np.array([-2.0, -2.0]),
            np.array([1.0, 2.0]),
            5.0,
            np.array([0.0, 0.0]),
        ),
    ],
)
def test_trsbox_geometry(
    x_center,
    c_term,
    model_gradient,
    lower_bounds,
    upper_bounds,
    delta,
    expected,
):
    linear_model = LinearModel(intercept=c_term, linear_terms=model_gradient)

    x_out = improve_geomtery_trsbox_linear(
        x_center,
        linear_model,
        lower_bounds,
        upper_bounds,
        delta,
    )
    aaae(x_out, expected)
