import numpy as np
import pytest
from optimagic.optimizers.neldermead import (
    _gao_han,
    _init_algo_params,
    _init_simplex,
    _nash,
    _pfeffer,
    _varadhan_borchers,
    neldermead_parallel,
)


# function to test
def sphere(x, *args, **kwargs):  # noqa: ARG001
    return (x**2).sum()


# unit tests
def test_init_algo_params():
    # test setting
    j = 2
    adaptive = True

    # outcome
    result = _init_algo_params(adaptive, j)

    # expected outcome
    expected = (1, 2, 0.5, 0.5)

    assert result == expected


def test_init_simplex():
    # test setting
    x = np.array([1, 2, 3])

    # outcome
    result = _init_simplex(x)

    # expected outcome
    expected = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])

    assert (result == expected).all()


def test_pfeffer():
    # test setting
    x = np.array([1, 0, 1])

    # outcome
    result = _pfeffer(x)

    # expected outcome
    expected = np.array([[1, 0, 1], [1.05, 0, 1], [1, 0.00025, 1], [1, 0, 1.05]])

    assert (result == expected).all()


def test_nash():
    # test setting
    x = np.array([1, 0, 1])

    # outcome
    result = _nash(x)

    # expected outcome
    expected = np.array([[1, 0, 1], [1.1, 0, 1], [1, 0.1, 1], [1, 0, 1.1]])

    assert (result == expected).all()


def test_gao_han():
    # test setting
    x = np.array([1, 0, 1])

    # outcome
    result = _gao_han(x)

    # expected outcome
    expected = np.array([[0.66667, -0.33333, 0.66667], [2, 0, 1], [1, 1, 1], [1, 0, 2]])

    np.testing.assert_allclose(result, expected, atol=1e-3)


def test_varadhan_borchers():
    # test setting
    x = np.array([1, 0, 1])

    # outcome
    result = _varadhan_borchers(x)

    # expected outcome
    expected = np.array(
        [
            [1, 0, 1],
            [2.3333, 0.3333, 1.3333],
            [1.3333, 1.3333, 1.3333],
            [1.3333, 0.3333, 2.3333],
        ]
    )

    np.testing.assert_allclose(result, expected, atol=1e-3)


# general parameter test
test_cases = [
    {},
    {"adaptive": False},
    {"init_simplex_method": "nash"},
    {"init_simplex_method": "pfeffer"},
    {"init_simplex_method": "varadhan_borchers"},
]


@pytest.mark.parametrize("algo_options", test_cases)
def test_neldermead_correct_algo_options(algo_options):
    res = neldermead_parallel(
        criterion=sphere,
        x=np.array([1, -5, 3]),
        **algo_options,
    )
    np.testing.assert_allclose(res["solution_x"], np.zeros(3), atol=5e-4)


# test if maximum number of iterations works
def test_fides_stop_after_one_iteration():
    res = neldermead_parallel(
        criterion=sphere,
        x=np.array([1, -5, 3]),
        stopping_maxiter=1,
    )
    assert not res["success"]
    assert res["n_iterations"] == 1
