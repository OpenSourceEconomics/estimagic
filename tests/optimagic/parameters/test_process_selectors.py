import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal as aae
from optimagic.exceptions import InvalidConstraintError
from optimagic.parameters.process_selectors import process_selectors
from optimagic.parameters.tree_conversion import TreeConverter
from optimagic.parameters.tree_registry import get_registry
from pybaum import tree_flatten, tree_just_flatten, tree_unflatten


@pytest.mark.parametrize("constraints", [None, []])
def test_process_selectors_no_constraint(constraints):
    calculated = process_selectors(
        constraints=constraints,
        params=np.arange(5),
        tree_converter=None,
        param_names=list("abcde"),
    )

    assert calculated == []


@pytest.fixture()
def tree_params():
    df = pd.DataFrame({"value": [3, 4], "lower_bound": [0, 0]}, index=["c", "d"])
    params = ([0, np.array([1, 2]), {"a": df, "b": 5}], 6)
    return params


@pytest.fixture()
def tree_params_converter(tree_params):
    registry = get_registry(extended=True)
    _, treedef = tree_flatten(tree_params, registry=registry)

    converter = TreeConverter(
        params_flatten=lambda params: np.array(
            tree_just_flatten(params, registry=registry)
        ),
        params_unflatten=lambda x: tree_unflatten(
            treedef, x.tolist(), registry=registry
        ),
        func_flatten=None,
        derivative_flatten=None,
    )
    return converter


@pytest.fixture()
def np_params_converter():
    converter = TreeConverter(
        lambda x: x,
        lambda x: x,
        lambda x: x,
        lambda x: x,
    )
    return converter


@pytest.fixture()
def df_params():
    df = pd.DataFrame({"value": np.arange(6) + 10}, index=list("abcdef"))
    df.index.name = "name"
    return df


@pytest.fixture()
def df_params_converter(df_params):
    converter = TreeConverter(
        lambda x: x["value"].to_numpy(),
        lambda x: df_params.assign(value=x),
        None,
        None,
    )
    return converter


def test_process_selectors_tree_selector(tree_params, tree_params_converter):
    calculated = process_selectors(
        constraints=[{"type": "equality", "selector": lambda x: x[1]}],
        params=tree_params,
        tree_converter=tree_params_converter,
        param_names=list("abcdefg"),
    )
    aae(calculated[0]["index"], np.array([6]))


def test_process_selectors_tree_selectors(tree_params, tree_params_converter):
    constraints = [
        {
            "type": "pairwise_equality",
            "selectors": [lambda x: x[1], lambda x: x[0][1][0]],
        }
    ]
    calculated = process_selectors(
        constraints=constraints,
        params=tree_params,
        tree_converter=tree_params_converter,
        param_names=list("abcdefg"),
    )
    aae(calculated[0]["indices"][0], np.array([6]))
    aae(calculated[0]["indices"][1], np.array([1]))


def test_process_selectors_numpy_array_loc(np_params_converter):
    calculated = process_selectors(
        constraints=[{"type": "equality", "loc": [1, 4]}],
        params=np.arange(6) + 10,
        tree_converter=np_params_converter,
        param_names=list("abcdefg"),
    )

    aae(calculated[0]["index"], np.array([1, 4]))


def test_process_selectors_numpy_array_locs(np_params_converter):
    constraints = [
        {
            "type": "pairwise_equality",
            "locs": [[1, 4], [0, 3]],
        }
    ]
    calculated = process_selectors(
        constraints=constraints,
        params=np.arange(6) + 10,
        tree_converter=np_params_converter,
        param_names=list("abcdefg"),
    )

    aae(calculated[0]["indices"][0], np.array([1, 4]))
    aae(calculated[0]["indices"][1], np.array([0, 3]))


def test_process_selectors_dataframe_loc(df_params, df_params_converter):
    constraints = [{"type": "equality", "loc": ["b", "e"]}]

    calculated = process_selectors(
        constraints=constraints,
        params=df_params,
        tree_converter=df_params_converter,
        param_names=list("abcdefg"),
    )

    aae(calculated[0]["index"], np.array([1, 4]))


def test_process_selectors_dataframe_query(df_params, df_params_converter):
    q = "name == 'b' | name == 'e'"
    constraints = [{"type": "equality", "query": q}]

    calculated = process_selectors(
        constraints=constraints,
        params=df_params,
        tree_converter=df_params_converter,
        param_names=list("abcdefg"),
    )

    aae(calculated[0]["index"], np.array([1, 4]))


def test_process_selectors_dataframe_locs(df_params, df_params_converter):
    constraints = [{"type": "pairwise_equality", "locs": [["b", "e"], ["a", "d"]]}]

    calculated = process_selectors(
        constraints=constraints,
        params=df_params,
        tree_converter=df_params_converter,
        param_names=list("abcdefg"),
    )

    aae(calculated[0]["indices"][0], np.array([1, 4]))
    aae(calculated[0]["indices"][1], np.array([0, 3]))


def test_process_selectors_dataframe_queries(df_params, df_params_converter):
    queries = ["name == 'b' | name == 'e'", "name == 'a' | name == 'd'"]
    constraints = [{"type": "pairwise_equality", "queries": queries}]

    calculated = process_selectors(
        constraints=constraints,
        params=df_params,
        tree_converter=df_params_converter,
        param_names=list("abcdefg"),
    )

    aae(calculated[0]["indices"][0], np.array([1, 4]))
    aae(calculated[0]["indices"][1], np.array([0, 3]))


@pytest.mark.parametrize("field", ["selectors", "queries", "query", "locs"])
def test_process_selectors_numpy_array_invalid_fields(field, np_params_converter):
    with pytest.raises(InvalidConstraintError):
        process_selectors(
            constraints=[{"type": "equality", field: None}],
            params=np.arange(6),
            tree_converter=np_params_converter,
            param_names=list("abcdefg"),
        )


@pytest.mark.parametrize("field", ["selectors", "queries", "locs"])
def test_process_selectors_dataframe_invalid_fields(
    field, df_params, df_params_converter
):
    with pytest.raises(InvalidConstraintError):
        process_selectors(
            constraints=[{"type": "equality", field: None}],
            params=df_params,
            tree_converter=df_params_converter,
            param_names=list("abcdefg"),
        )


@pytest.mark.parametrize("field", ["selectors", "queries", "query", "locs", "loc"])
def test_process_selectors_tree_invalid_fields(
    field, tree_params, tree_params_converter
):
    with pytest.raises(InvalidConstraintError):
        process_selectors(
            constraints=[{"type": "equality", field: None}],
            params=tree_params,
            tree_converter=tree_params_converter,
            param_names=list("abcdefg"),
        )


def test_process_selectors_duplicates(np_params_converter):
    constraints = [
        {
            "type": "pairwise_equality",
            "locs": [[1, 4], [0, 0]],
        }
    ]
    with pytest.raises(InvalidConstraintError):
        process_selectors(
            constraints=constraints,
            params=np.arange(6) + 10,
            tree_converter=np_params_converter,
            param_names=list("abcdefg"),
        )


def test_process_selectors_differen_length_in_multiple_selectors(np_params_converter):
    constraints = [
        {
            "type": "pairwise_equality",
            "locs": [[1, 4], [0, 3, 5]],
        }
    ]
    with pytest.raises(InvalidConstraintError):
        process_selectors(
            constraints=constraints,
            params=np.arange(6) + 10,
            tree_converter=np_params_converter,
            param_names=list("abcdefg"),
        )
