import numpy as np
import pytest
from optimagic.exceptions import InvalidParamsError
from optimagic.parameters.check_constraints import _iloc
from optimagic.parameters.constraint_tools import check_constraints


def test_iloc():
    dictionary = {
        "index": np.array(["a", "b", "c"]),
        "lower_bounds": np.array([0, 0, 0]),
        "upper_bounds": np.array([1, 1, 1]),
        "is_fixed_to_value": np.array([False, False, True]),
    }
    position = [0, 2]
    expected_result = {
        "index": np.array(["a", "c"]),
        "lower_bounds": np.array([0, 0]),
        "upper_bounds": np.array([1, 1]),
        "is_fixed_to_value": np.array([False, True]),
    }
    result = _iloc(dictionary, position)
    assert len(result) == len(expected_result)
    for k, v in expected_result.items():
        assert k in result
        assert np.array_equal(result[k], v)


def test_check_constraints_are_satisfied_type_equality():
    with pytest.raises(InvalidParamsError):
        check_constraints(
            params=np.array([1, 2, 3]), constraints={"type": "equality", "loc": [0, 1]}
        )


def test_check_constraints_are_satisfied_type_increasing():
    with pytest.raises(InvalidParamsError):
        check_constraints(
            params=np.array([1, 2, 3, 2, 4]),
            constraints={"type": "increasing", "loc": [1, 2, 3]},
        )


def test_check_constraints_are_satisfied_type_decreasing():
    with pytest.raises(InvalidParamsError):
        check_constraints(
            params=np.array([1, 2, 3, 2, 4]),
            constraints={"type": "decreasing", "loc": [0, 1, 3]},
        )


def test_check_constraints_are_satisfied_type_pairwise_equality():
    with pytest.raises(InvalidParamsError):
        check_constraints(
            params=np.array([1, 2, 3, 3, 4]),
            constraints={"type": "pairwise_equality", "locs": [[0, 4], [3, 2]]},
        )


def test_check_constraints_are_satisfied_type_probability():
    with pytest.raises(InvalidParamsError):
        check_constraints(
            params=np.array([0.10, 0.25, 0.50, 1, 0.7]),
            constraints={"type": "probability", "loc": [0, 1, 2, 4]},
        )


def test_check_constraints_are_satisfied_type_linear_lower_bound():
    with pytest.raises(InvalidParamsError):
        check_constraints(
            params=np.ones(5),
            constraints={
                "type": "linear",
                "loc": [0, 2, 3, 4],
                "lower_bound": 1.1,
                "weights": 0.25,
            },
        )


def test_check_constraints_are_satisfied_type_linear_upper_bound():
    with pytest.raises(InvalidParamsError):
        check_constraints(
            params=np.ones(5),
            constraints={
                "type": "linear",
                "loc": [0, 2, 3, 4],
                "upper_bound": 0.9,
                "weights": 0.25,
            },
        )


def test_check_constraints_are_satisfied_type_linear_value():
    with pytest.raises(InvalidParamsError):
        check_constraints(
            params=np.ones(5),
            constraints={
                "type": "linear",
                "loc": [0, 2, 3, 4],
                "value": 2,
                "weights": 0.25,
            },
        )


def test_check_constraints_are_satisfied_type_covariance():
    with pytest.raises(InvalidParamsError):
        check_constraints(
            params=[1, 1, 1, -1, 1, -1],
            constraints={
                "type": "covariance",
                # "loc": [0, 1, 2],
                "selector": lambda params: params,
            },
        )


def test_check_constraints_are_satisfied_type_sdcorr():
    with pytest.raises(InvalidParamsError):
        check_constraints(
            params=[1, 1, 1, -1, 1, 1],
            constraints={
                "type": "sdcorr",
                # "loc": [0, 1, 2],
                "selector": lambda params: params,
            },
        )


# to ignore as per email?
# def test_check_constraints_are_satisfied_type_nonlinear():
