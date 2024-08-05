import numpy as np
import pandas as pd
import pytest
from optimagic.parameters.tree_registry import get_registry
from pandas.testing import assert_frame_equal
from pybaum import leaf_names, tree_flatten, tree_unflatten


@pytest.fixture()
def value_df():
    df = pd.DataFrame(
        np.arange(6).reshape(3, 2),
        columns=["a", "value"],
        index=["alpha", "beta", "gamma"],
    )
    return df


@pytest.fixture()
def other_df():
    df = pd.DataFrame(index=["alpha", "beta", "gamma"])
    df["b"] = np.arange(3).astype(np.int16)
    df["c"] = 3.14
    return df


def test_flatten_df_with_value_column(value_df):
    registry = get_registry(extended=True)
    flat, _ = tree_flatten(value_df, registry=registry)
    assert flat == [1, 3, 5]


def test_unflatten_df_with_value_column(value_df):
    registry = get_registry(extended=True)
    _, treedef = tree_flatten(value_df, registry=registry)
    unflat = tree_unflatten(treedef, [10, 11, 12], registry=registry)
    assert unflat.equals(value_df.assign(value=[10, 11, 12]))


def test_leaf_names_df_with_value_column(value_df):
    registry = get_registry(extended=True)
    names = leaf_names(value_df, registry=registry)
    assert names == ["alpha", "beta", "gamma"]


def test_flatten_partially_numeric_df(other_df):
    registry = get_registry(extended=True)
    flat, _ = tree_flatten(other_df, registry=registry)
    assert flat == [0, 3.14, 1, 3.14, 2, 3.14]


def test_unflatten_partially_numeric_df(other_df):
    registry = get_registry(extended=True)
    _, treedef = tree_flatten(other_df, registry=registry)
    unflat = tree_unflatten(treedef, [1, 2, 3, 4, 5, 6], registry=registry)
    other_df = other_df.assign(b=[1, 3, 5], c=[2, 4, 6])
    assert_frame_equal(unflat, other_df, check_dtype=False)


def test_leaf_names_partially_numeric_df(other_df):
    registry = get_registry(extended=True)
    names = leaf_names(other_df, registry=registry)
    assert names == ["alpha_b", "alpha_c", "beta_b", "beta_c", "gamma_b", "gamma_c"]
