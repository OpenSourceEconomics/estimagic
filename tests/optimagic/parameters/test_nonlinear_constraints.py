import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from optimagic.differentiation.numdiff_options import NumdiffOptions
from optimagic.exceptions import InvalidConstraintError
from optimagic.parameters.nonlinear_constraints import (
    _check_validity_and_return_evaluation,
    _get_components,
    _get_selection_indices,
    _get_transformation,
    _get_transformation_type,
    _process_selector,
    _vector_to_list_of_scalar,
    equality_as_inequality_constraints,
    process_nonlinear_constraints,
    vector_as_list_of_scalar_constraints,
)
from optimagic.parameters.tree_registry import get_registry
from pandas.testing import assert_frame_equal
from pybaum import tree_just_flatten


@dataclass
class Converter:
    def params_from_internal(self, x):
        return x

    def params_to_internal(self, params):
        registry = get_registry(extended=True)
        return np.array(tree_just_flatten(params, registry=registry))


# ======================================================================================
# _get_transformation_type
# ======================================================================================
TEST_CASES = [
    (0, np.inf, "identity"),  # (lower_bounds, upper_bounds, expected)
    (-1, 2, "stack"),
    (np.zeros(3), np.ones(3), "stack"),
    (np.zeros(3), np.tile(np.inf, 3), "identity"),
    (np.array([1, 2]), np.tile(np.inf, 2), "subtract_lb"),
]


@pytest.mark.parametrize("lower_bounds, upper_bounds, expected", TEST_CASES)
def test_get_transformation_type(lower_bounds, upper_bounds, expected):
    got = _get_transformation_type(lower_bounds, upper_bounds)
    assert got == expected


# ======================================================================================
# _get_transformation
# ======================================================================================

TEST_CASES = [
    #  (lower_bounds, upper_bounds, case, expected)  # noqa: ERA001
    (0, 0, "func", {"name": "stack", "out": np.array([1, -1])}),
    (1, 1, "func", {"name": "stack", "out": np.array([0, 0])}),
    (0, 0, "derivative", {"name": "stack", "out": np.array([1, -1])}),
    (1, 1, "derivative", {"name": "stack", "out": np.array([1, -1])}),
    (1, np.inf, "func", {"name": "subtract_lb", "out": np.array([0])}),
    (0, np.inf, "derivative", {"name": "identity", "out": np.array([1])}),
]


@pytest.mark.parametrize("lower_bounds, upper_bounds, case, expected", TEST_CASES)
def test_get_positivity_transform(lower_bounds, upper_bounds, case, expected):
    transform = _get_transformation(lower_bounds, upper_bounds)

    got = transform[case](np.array([1]))
    assert np.all(got == expected["out"])
    assert transform["name"] == expected["name"]


# ======================================================================================
# _get_selection_indices
# ======================================================================================


def test_get_selection_indices():
    params = {"a": [0, 1, 2], "b": [3, 4, 5]}
    selector = lambda p: p["a"]

    expected = np.array([0, 1, 2], dtype=int)
    got_index, got_n_params = _get_selection_indices(params, selector)

    assert got_n_params == 6
    assert_array_equal(got_index, expected)


# ======================================================================================
# _process_selector
# ======================================================================================
TEST_CASES = [
    ({"selector": lambda x: x**2}, 10, 100),  # (constraint, params, expected)
    ({"loc": "a"}, pd.Series([0, 1], index=["a", "b"]), 0),
    (
        {"query": "a == 1"},
        pd.DataFrame([[1], [0]], columns=["a"]),
        pd.DataFrame([[1]], columns=["a"]),
    ),
]


@pytest.mark.parametrize("constraint, params, expected", TEST_CASES)
def test_process_selector(constraint, params, expected):
    _selector = _process_selector(constraint)
    got = _selector(params)

    if isinstance(got, pd.DataFrame):
        assert_frame_equal(got, expected)
    else:
        assert got == expected


# ======================================================================================
# _check_validity_nonlinear_constraint
# ======================================================================================
TEST_CASES = [
    {},  # no fun
    {"func": 10},  # non-callable fun
    {"func": lambda x: x, "derivative": 10},  # non-callable jac
    {"func": lambda x: x},  # no bounds at all
    {
        "func": lambda x: x,
        "value": 1,
        "lower_bounds": 1,
    },  # cannot have value and bounds
    {
        "func": lambda x: x,
        "value": 1,
        "upper_bounds": 1,
    },  # cannot have value and bounds
    {"func": lambda x: x},  # needs to have at least one bound
    {"func": lambda x: x, "lower_bounds": 1, "upper_bounds": 0},
    {"func": lambda x: x, "selector": 10},
    {"func": lambda x: x, "loc": 10},
    {"func": lambda x: x, "query": 10},
]

TEST_CASES = list(
    itertools.product(TEST_CASES, [np.arange(3), pd.DataFrame({"a": [0, 1, 2]})])
)


@pytest.mark.parametrize("constraint, params", TEST_CASES)
def test_check_validity_nonlinear_constraint(constraint, params):
    with pytest.raises(InvalidConstraintError):
        _check_validity_and_return_evaluation(constraint, params, skip_checks=False)


def test_check_validity_nonlinear_constraint_correct_example():
    constr = {
        "func": lambda x: x,
        "derivative": lambda x: np.ones_like(x),
        "lower_bounds": np.arange(4),
        "selector": lambda x: x[:1],
    }
    _check_validity_and_return_evaluation(
        constr, params=np.arange(4), skip_checks=False
    )


# ======================================================================================
# equality_as_inequality_constraints
# ======================================================================================
TEST_CASES = [
    (
        [
            {
                "type": "ineq",
                "fun": lambda x: np.array([x]),
                "jac": lambda x: np.array([[1]]),  # noqa: ARG005
                "n_constr": 1,
            }
        ],  # constraints
        "same",  # expected
    ),
    (
        [
            {
                "type": "ineq",
                "fun": lambda x: np.array([x]),
                "jac": lambda x: np.array([[1]]),  # noqa: ARG005
                "n_constr": 1,
            }
        ],  # constraints
        [
            {
                "type": "eq",
                "fun": lambda x: np.array([x, -x]).reshape(-1, 1),
                "jac": lambda x: np.array([[1], [-1]]),  # noqa: ARG005
                "n_constr": 1,
            }
        ],  # expected
    ),
]


@pytest.mark.parametrize("constraints, expected", TEST_CASES)
def test_equality_as_inequality_constraints(constraints, expected):
    got = equality_as_inequality_constraints(constraints)
    if expected == "same":
        assert got == constraints

    for g, c in zip(got, constraints, strict=False):
        if c["type"] == "eq":
            assert g["n_constr"] == 2 * c["n_constr"]
        assert g["type"] == "ineq"


# ======================================================================================
# process_nonlinear_constraints
# ======================================================================================


def test_process_nonlinear_constraints():
    nonlinear_constraints = [
        {"type": "nonlinear", "func": lambda x: np.dot(x, x), "value": 1},
        {
            "type": "nonlinear",
            "func": lambda x: x,
            "lower_bounds": -1,
            "upper_bounds": 2,
        },
    ]

    params = np.array([1.0])

    converter = Converter()

    numdiff_options = NumdiffOptions()

    got = process_nonlinear_constraints(
        nonlinear_constraints,
        params=params,
        bounds=None,
        converter=converter,
        numdiff_options=numdiff_options,
        skip_checks=False,
    )

    expected = [
        {"type": "eq", "fun": lambda x: np.dot(x, x) - 1.0, "n_constr": 1},
        {
            "type": "ineq",
            "fun": lambda x: np.concatenate((x + 1.0, 2.0 - x), axis=0),
            "n_constr": 2,
        },
    ]

    for g, e in zip(got, expected, strict=False):
        assert g["type"] == e["type"]
        assert g["n_constr"] == e["n_constr"]
        for value in [0.1, 0.2, 1.2, -2.0]:
            x = np.array([value])
            assert_array_equal(g["fun"](x), e["fun"](x))
        assert "jac" in g
        assert "tol" in g


# ======================================================================================
# vector_as_list_of_scalar_constraints
# ======================================================================================


def test_get_components():
    fun = lambda x: np.array([x[0], 2 * x[1]])
    jac = lambda x: np.array([[1, 0], [0, 2]])  # noqa: ARG005

    fun_component, jac_component = _get_components(fun, jac, idx=1)

    x = np.array([0, 3])
    assert fun_component(x) == 6
    assert_array_equal(jac_component(x), np.array([0, 2]))


def test_vector_to_list_of_scalar():
    constr = {
        "fun": lambda x: x,
        "jac": lambda x: np.eye(len(x)),
        "n_constr": 2,
    }
    got = _vector_to_list_of_scalar(constr)
    for got_constr in got:
        assert got_constr["n_constr"] == 1
    for i in range(2):
        assert got[i]["fun"](np.arange(2)) == i
        assert_array_equal(got[i]["jac"](np.arange(2)), np.eye(2)[i])


def test_vector_as_list_of_scalar_constraints():
    constr = {
        "fun": lambda x: x,
        "jac": lambda x: np.eye(len(x)),
        "n_constr": 2,
    }
    constraints = [constr, constr]
    got = vector_as_list_of_scalar_constraints(constraints)
    assert len(got) == 4
