import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from optimagic.exceptions import InvalidBoundsError
from optimagic.parameters.bounds import Bounds, get_internal_bounds, pre_process_bounds


@pytest.fixture()
def pytree_params():
    pytree_params = {
        "delta": 0.95,
        "utility": pd.DataFrame(
            [[0.5, 0]] * 3, index=["a", "b", "c"], columns=["value", "lower_bound"]
        ),
        "probs": np.array([[0.8, 0.2], [0.3, 0.7]]),
    }
    return pytree_params


@pytest.fixture()
def array_params():
    return np.arange(2)


def test_pre_process_bounds_trivial_case():
    got = pre_process_bounds(Bounds(lower=[0], upper=[1]))
    expected = Bounds(lower=[0], upper=[1])
    assert got == expected


def test_pre_process_bounds_none_case():
    assert pre_process_bounds(None) is None


def test_pre_process_bounds_sequence():
    got = pre_process_bounds([(0, 1), (None, 1)])
    expected = Bounds(lower=[0, -np.inf], upper=[1, 1])
    assert_array_equal(got.lower, expected.lower)
    assert_array_equal(got.upper, expected.upper)


def test_pre_process_bounds_invalid_type():
    with pytest.raises(InvalidBoundsError):
        pre_process_bounds(1)


def test_get_bounds_subdataframe(pytree_params):
    upper_bounds = {
        "utility": pd.DataFrame([[2]] * 2, index=["b", "c"], columns=["value"]),
    }
    lower_bounds = {
        "delta": 0,
        "utility": pd.DataFrame([[1]] * 2, index=["a", "b"], columns=["value"]),
    }

    bounds = Bounds(lower=lower_bounds, upper=upper_bounds)

    lb, ub = get_internal_bounds(pytree_params, bounds=bounds)

    assert np.all(lb[1:3] == np.ones(2))
    assert np.all(ub[2:4] == 2 * np.ones(2))


TEST_CASES = [
    Bounds(lower={"delta": [0, -1]}, upper=None),
    Bounds(lower={"probs": 1}, upper=None),
    Bounds(lower={"probs": np.array([0, 1])}, upper=None),  # wrong size lower bounds
    Bounds(lower=None, upper={"probs": np.array([0, 1])}),  # wrong size upper bounds
]


@pytest.mark.parametrize("bounds", TEST_CASES)
def test_get_bounds_error(pytree_params, bounds):
    with pytest.raises(InvalidBoundsError):
        get_internal_bounds(pytree_params, bounds=bounds)


def test_get_bounds_no_arguments(pytree_params):
    got_lower, got_upper = get_internal_bounds(pytree_params)

    expected_lower = np.array([-np.inf] + 3 * [0] + 4 * [-np.inf])
    expected_upper = np.full(8, np.inf)

    assert_array_equal(got_lower, expected_lower)
    assert_array_equal(got_upper, expected_upper)


def test_get_bounds_with_lower_bounds(pytree_params):
    lower_bounds = {"delta": 0.1}

    bounds = Bounds(lower=lower_bounds)

    got_lower, got_upper = get_internal_bounds(pytree_params, bounds=bounds)

    expected_lower = np.array([0.1] + 3 * [0] + 4 * [-np.inf])
    expected_upper = np.full(8, np.inf)

    assert_array_equal(got_lower, expected_lower)
    assert_array_equal(got_upper, expected_upper)


def test_get_bounds_with_upper_bounds(pytree_params):
    upper_bounds = {
        "utility": pd.DataFrame([[1]] * 3, index=["a", "b", "c"], columns=["value"]),
    }
    bounds = Bounds(upper=upper_bounds)
    got_lower, got_upper = get_internal_bounds(pytree_params, bounds=bounds)

    expected_lower = np.array([-np.inf] + 3 * [0] + 4 * [-np.inf])
    expected_upper = np.array([np.inf] + 3 * [1] + 4 * [np.inf])

    assert_array_equal(got_lower, expected_lower)
    assert_array_equal(got_upper, expected_upper)


def test_get_bounds_numpy(array_params):
    got_lower, got_upper = get_internal_bounds(array_params)

    expected = np.array([np.inf, np.inf])

    assert_array_equal(got_lower, -expected)
    assert_array_equal(got_upper, expected)


def test_get_bounds_numpy_error(array_params):
    # lower bounds larger than upper bounds
    bounds = Bounds(lower=np.ones_like(array_params), upper=np.zeros_like(array_params))
    with pytest.raises(InvalidBoundsError):
        get_internal_bounds(
            array_params,
            bounds=bounds,
        )
