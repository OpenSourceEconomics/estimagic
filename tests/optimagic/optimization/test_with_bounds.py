import numpy as np
from optimagic.optimization.optimize import maximize, minimize
from scipy.optimize import Bounds as ScipyBounds


def test_minimize_with_scipy_bounds():
    minimize(
        lambda x: x @ x,
        np.arange(3),
        bounds=ScipyBounds(np.full(3, -1), np.full(3, 5)),
        algorithm="scipy_lbfgsb",
    )


def test_minimize_with_sequence_bounds():
    minimize(
        lambda x: x @ x,
        np.arange(3),
        bounds=[(-1, 5)] * 3,
        algorithm="scipy_lbfgsb",
    )


def test_maximize_with_scipy_bounds():
    maximize(
        lambda x: -x @ x,
        np.arange(3),
        bounds=ScipyBounds(np.full(3, -1), np.full(3, 5)),
        algorithm="scipy_lbfgsb",
    )


def test_maximize_with_sequence_bounds():
    maximize(
        lambda x: -x @ x,
        np.arange(3),
        bounds=[(-1, 5)] * 3,
        algorithm="scipy_lbfgsb",
    )
