import numpy as np
import optimagic as om
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.optimization.optimize import maximize, minimize
from optimagic.parameters.scaling import ScalingOptions


def test_minimize_with_scaling_options():
    got = minimize(
        fun=lambda x: x @ x,
        x0=np.arange(3),
        jac=lambda x: 2 * x,
        algorithm="scipy_lbfgsb",
        constraints=[{"selector": lambda x: x[2], "type": "fixed"}],
        scaling=ScalingOptions(method="start_values", magnitude=1.2),
    )
    aaae(got.x, np.array([0, 0, 2]))


def test_minimize_with_scaling_options_dict():
    got = minimize(
        fun=lambda x: x @ x,
        x0=np.arange(3),
        jac=lambda x: 2 * x,
        algorithm="scipy_lbfgsb",
        constraints=[{"selector": lambda x: x[2], "type": "fixed"}],
        scaling={"method": "start_values", "magnitude": 1.2},
    )
    aaae(got.x, np.array([0, 0, 2]))


def test_minimize_with_scaling_true():
    got = minimize(
        fun=lambda x: x @ x,
        x0=np.arange(3),
        jac=lambda x: 2 * x,
        algorithm="scipy_lbfgsb",
        constraints=[{"selector": lambda x: x[2], "type": "fixed"}],
        scaling=True,
    )
    aaae(got.x, np.array([0, 0, 2]))


def test_maximize_with_scaling_options():
    got = maximize(
        fun=lambda x: -x @ x,
        x0=np.arange(3),
        jac=lambda x: -2 * x,
        algorithm="scipy_lbfgsb",
        constraints=[{"selector": lambda x: x[2], "type": "fixed"}],
        scaling=ScalingOptions(method="start_values", magnitude=1.2),
    )
    aaae(got.x, np.array([0, 0, 2]))


def test_maximize_with_scaling_options_dict():
    got = maximize(
        fun=lambda x: -x @ x,
        x0=np.arange(3),
        jac=lambda x: -2 * x,
        algorithm="scipy_lbfgsb",
        constraints=[{"selector": lambda x: x[2], "type": "fixed"}],
        scaling={"method": "start_values", "magnitude": 1.2},
    )
    aaae(got.x, np.array([0, 0, 2]))


def test_maximize_with_scaling_true():
    got = maximize(
        fun=lambda x: -x @ x,
        x0=np.arange(3),
        jac=lambda x: -2 * x,
        algorithm="scipy_lbfgsb",
        constraints=[{"selector": lambda x: x[2], "type": "fixed"}],
        scaling=True,
    )
    aaae(got.x, np.array([0, 0, 2]))


def test_minimize_with_scaling_options_with_bounds():
    got = minimize(
        fun=lambda x: x @ x,
        x0=np.arange(3),
        bounds=om.Bounds(lower=np.array([-1, 0, 0]), upper=np.full(3, 5)),
        jac=lambda x: 2 * x,
        algorithm="scipy_lbfgsb",
        constraints=[{"selector": lambda x: x[2], "type": "fixed"}],
        scaling=ScalingOptions(method="bounds", magnitude=1),
    )
    aaae(got.x, np.array([0, 0, 2]))


def test_minimize_with_scaling_options_dict_with_bounds():
    got = minimize(
        fun=lambda x: x @ x,
        x0=np.arange(3),
        bounds=om.Bounds(lower=np.array([-1, 0, 0]), upper=np.full(3, 5)),
        jac=lambda x: 2 * x,
        algorithm="scipy_lbfgsb",
        constraints=[{"selector": lambda x: x[2], "type": "fixed"}],
        scaling={"method": "bounds", "magnitude": 1},
    )
    aaae(got.x, np.array([0, 0, 2]))


def test_minimize_with_scaling_true_with_bounds():
    got = minimize(
        fun=lambda x: x @ x,
        x0=np.arange(3),
        bounds=om.Bounds(lower=np.array([-1, 0, 0]), upper=np.full(3, 5)),
        jac=lambda x: 2 * x,
        algorithm="scipy_lbfgsb",
        constraints=[{"selector": lambda x: x[2], "type": "fixed"}],
        scaling=True,
    )
    aaae(got.x, np.array([0, 0, 2]))
