import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from optimagic import second_derivative
from optimagic.parameters.block_trees import (
    block_tree_to_hessian,
    block_tree_to_matrix,
    hessian_to_block_tree,
    matrix_to_block_tree,
)
from optimagic.parameters.tree_registry import get_registry
from pybaum import tree_equal
from pybaum import tree_just_flatten as tree_leaves


def test_matrix_to_block_tree_array_and_scalar():
    t = {"a": 1.0, "b": np.arange(2)}
    calculated = matrix_to_block_tree(np.arange(9).reshape(3, 3), t, t)

    expected = {
        "a": {"a": np.array(0), "b": np.array([1, 2])},
        "b": {"a": np.array([3, 6]), "b": np.array([[4, 5], [7, 8]])},
    }

    assert _tree_equal_up_to_dtype(calculated, expected)


def test_matrix_to_block_tree_only_params_dfs():
    tree = {
        "a": pd.DataFrame(index=["a", "b"]).assign(value=[1, 2]),
        "b": pd.DataFrame(index=["j", "k", "l"]).assign(value=[3, 4, 5]),
    }

    calculated = matrix_to_block_tree(np.arange(25).reshape(5, 5), tree, tree)

    expected = {
        "a": {
            "a": pd.DataFrame([[0, 1], [5, 6]], columns=["a", "b"], index=["a", "b"]),
            "b": pd.DataFrame(
                [[2, 3, 4], [7, 8, 9]], columns=["j", "k", "l"], index=["a", "b"]
            ),
        },
        "b": {
            "a": pd.DataFrame(
                [[10, 11], [15, 16], [20, 21]],
                index=["j", "k", "l"],
                columns=["a", "b"],
            ),
            "b": pd.DataFrame(
                [[12, 13, 14], [17, 18, 19], [22, 23, 24]],
                index=["j", "k", "l"],
                columns=["j", "k", "l"],
            ),
        },
    }

    assert _tree_equal_up_to_dtype(calculated, expected)


def test_matrix_to_block_tree_single_element():
    tree1 = {"a": 0}
    tree2 = {"b": 1, "c": 2}

    block_tree = {"a": {"b": 0, "c": 1}}
    matrix = np.array([[0, 1]])

    calculated = matrix_to_block_tree(matrix, tree1, tree2)
    assert tree_equal(block_tree, calculated)


# one params df (make sure we don't get a list back)
# dataframe and scalar
# tests against jax


def test_block_tree_to_matrix_array_and_scalar():
    t1 = {"c": np.arange(3), "d": (2.0, 1)}
    t2 = {"a": 1.0, "b": np.arange(2)}

    expected = np.arange(15).reshape(5, 3)

    block_tree = {
        "c": {"a": np.array([0, 3, 6]), "b": np.array([[1, 2], [4, 5], [7, 8]])},
        "d": (
            {"a": np.array(9), "b": np.array([10, 11])},
            {"a": np.array(12), "b": np.array([13, 14])},
        ),
    }

    calculated = block_tree_to_matrix(block_tree, t1, t2)
    assert_array_equal(expected, calculated)


def test_block_tree_to_matrix_only_params_dfs():
    expected = np.arange(25).reshape(5, 5)

    tree = {
        "a": pd.DataFrame(index=["a", "b"]).assign(value=[1, 2]),
        "b": pd.DataFrame(index=["j", "k", "l"]).assign(value=[3, 4, 5]),
    }
    block_tree = {
        "a": {
            "a": pd.DataFrame([[0, 1], [5, 6]], columns=["a", "b"], index=["a", "b"]),
            "b": pd.DataFrame(
                [[2, 3, 4], [7, 8, 9]], columns=["j", "k", "l"], index=["a", "b"]
            ),
        },
        "b": {
            "a": pd.DataFrame(
                [[10, 11], [15, 16], [20, 21]],
                index=["j", "k", "l"],
                columns=["a", "b"],
            ),
            "b": pd.DataFrame(
                [[12, 13, 14], [17, 18, 19], [22, 23, 24]],
                index=["j", "k", "l"],
                columns=["j", "k", "l"],
            ),
        },
    }

    calculated = block_tree_to_matrix(block_tree, tree, tree)
    assert_array_equal(expected, calculated)


def test_block_tree_to_hessian_bijection():
    params = {"a": np.arange(4), "b": [{"c": (1, 2), "d": np.array([5, 6])}]}
    f_tree = {"e": np.arange(3), "f": (5, 6, [7, 8, {"g": 1.0}])}

    registry = get_registry(extended=True)
    n_p = len(tree_leaves(params, registry=registry))
    n_f = len(tree_leaves(f_tree, registry=registry))

    expected = np.arange(n_f * n_p**2).reshape(n_f, n_p, n_p)
    block_hessian = hessian_to_block_tree(expected, f_tree, params)
    got = block_tree_to_hessian(block_hessian, f_tree, params)
    assert_array_equal(expected, got)


def test_hessian_to_block_tree_bijection():
    params = {"a": np.arange(4), "b": [{"c": (1, 2), "d": np.array([5, 6])}]}

    def func(params):
        return {"e": params["a"] ** 3, "f": (params["b"][0]["c"][1] / 0.5)}

    expected = second_derivative(func, params)["derivative"]
    hessian = block_tree_to_hessian(expected, func(params), params)
    got = hessian_to_block_tree(hessian, func(params), params)
    _tree_equal_up_to_dtype(expected, got)


def test_block_tree_to_matrix_valueerror():
    # test that value error is raised when dimensions don't match
    inner = {"a": 1, "b": 1}
    outer = 1
    block_tree = {"a": 1}  # should have same structure as inner
    with pytest.raises(ValueError):
        block_tree_to_matrix(block_tree, inner, outer)


def _tree_equal_up_to_dtype(left, right):
    # does not compare dtypes for pandas.DataFrame
    return tree_equal(left, right, equality_checkers={pd.DataFrame: _frame_equal})


def _frame_equal(left, right):
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
        return True
    except AssertionError:
        return False
