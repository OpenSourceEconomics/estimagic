from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
from estimagic.shared_covs import (
    _to_numpy,
    calculate_estimation_summary,
    get_derivative_case,
    process_pandas_arguments,
    transform_covariance,
    transform_free_cov_to_cov,
    transform_free_values_to_params_tree,
)
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.parameters.tree_registry import get_registry
from optimagic.utilities import get_rng
from pybaum import leaf_names, tree_equal


@pytest.fixture()
def inputs():
    jac = pd.DataFrame(np.ones((5, 3)), columns=["a", "b", "c"])
    hess = pd.DataFrame(np.eye(3) / 2, columns=list("abc"), index=list("abc"))
    weights = pd.DataFrame(np.eye(5))
    moments_cov = 1 / weights
    out = {"jac": jac, "hess": hess, "weights": weights, "moments_cov": moments_cov}
    return out


def test_process_pandas_arguments_all_pd(inputs):
    *arrays, names = process_pandas_arguments(**inputs)
    for arr in arrays:
        assert isinstance(arr, np.ndarray)

    expected_names = {"moments": list(range(5)), "params": ["a", "b", "c"]}

    for key, value in expected_names.items():
        assert names[key].tolist() == value


def test_process_pandas_arguments_incompatible_names(inputs):
    inputs["jac"].columns = ["c", "d", "e"]

    with pytest.raises(ValueError):
        process_pandas_arguments(**inputs)


def _from_internal(x, return_type="flat"):  # noqa: ARG001
    return x


class FakeConverter(NamedTuple):
    has_transforming_constraints: bool = True
    params_from_internal: callable = _from_internal


class FakeInternalParams(NamedTuple):
    values: np.ndarray = np.arange(2)
    lower_bounds: np.ndarray = np.full(2, -np.inf)
    upper_bounds: np.ndarray = np.full(2, np.inf)
    free_mask: np.ndarray = np.array([True, True])


def test_transform_covariance_no_bounds():
    internal_cov = np.eye(2)

    converter = FakeConverter()
    internal_params = FakeInternalParams()

    got = transform_covariance(
        internal_params=internal_params,
        internal_cov=internal_cov,
        converter=converter,
        rng=get_rng(seed=5687),
        n_samples=100,
        bounds_handling="ignore",
    )

    expected_sample = get_rng(seed=5687).multivariate_normal(
        np.arange(2), np.eye(2), 100
    )
    expected = np.cov(expected_sample, rowvar=False)

    aaae(got, expected)


def test_transform_covariance_with_clipping():
    rng = get_rng(seed=1234)

    internal_cov = np.eye(2)

    converter = FakeConverter()
    internal_params = FakeInternalParams(
        lower_bounds=np.ones(2), upper_bounds=np.ones(2)
    )

    got = transform_covariance(
        internal_params=internal_params,
        internal_cov=internal_cov,
        converter=converter,
        rng=rng,
        n_samples=100,
        bounds_handling="clip",
    )

    expected = np.zeros((2, 2))

    aaae(got, expected)


def test_transform_covariance_invalid_bounds():
    rng = get_rng(seed=1234)

    internal_cov = np.eye(2)

    converter = FakeConverter()
    internal_params = FakeInternalParams(
        lower_bounds=np.ones(2), upper_bounds=np.ones(2)
    )

    with pytest.raises(ValueError):
        transform_covariance(
            internal_params=internal_params,
            internal_cov=internal_cov,
            converter=converter,
            rng=rng,
            n_samples=10,
            bounds_handling="raise",
        )


class FakeFreeParams(NamedTuple):
    free_mask: np.ndarray = np.array([True, False, True])
    all_names: list = ["a", "b", "c"]
    free_names: list = ["a", "c"]


def test_transform_free_cov_to_cov_pytree():
    got = transform_free_cov_to_cov(
        free_cov=np.eye(2),
        free_params=FakeFreeParams(),
        params={"a": 1, "b": 2, "c": 3},
        return_type="pytree",
    )

    assert got["a"]["a"] == 1
    assert got["c"]["c"] == 1
    assert got["a"]["c"] == 0
    assert got["c"]["a"] == 0
    assert np.isnan(got["a"]["b"])


def test_transform_free_cov_to_cov_array():
    got = transform_free_cov_to_cov(
        free_cov=np.eye(2),
        free_params=FakeFreeParams(),
        params={"a": 1, "b": 2, "c": 3},
        return_type="array",
    )

    expected = np.array([[1, np.nan, 0], [np.nan, np.nan, np.nan], [0, np.nan, 1]])

    assert np.array_equal(got, expected, equal_nan=True)


def test_transform_free_cov_to_cov_dataframe():
    got = transform_free_cov_to_cov(
        free_cov=np.eye(2),
        free_params=FakeFreeParams(),
        params={"a": 1, "b": 2, "c": 3},
        return_type="dataframe",
    )

    expected = np.array([[1, np.nan, 0], [np.nan, np.nan, np.nan], [0, np.nan, 1]])

    assert np.array_equal(got.to_numpy(), expected, equal_nan=True)
    assert isinstance(got, pd.DataFrame)
    assert list(got.columns) == list("abc")
    assert list(got.index) == list("abc")


def test_transform_free_cov_to_cov_invalid():
    with pytest.raises(ValueError):
        transform_free_cov_to_cov(
            free_cov=np.eye(2),
            free_params=FakeFreeParams(),
            params={"a": 1, "b": 2, "c": 3},
            return_type="bla",
        )


def test_transform_free_values_to_params_tree():
    got = transform_free_values_to_params_tree(
        values=np.array([10, 11]),
        free_params=FakeFreeParams(),
        params={"a": 1, "b": 2, "c": 3},
    )

    assert got["a"] == 10
    assert got["c"] == 11
    assert np.isnan(got["b"])


def test_get_derivative_case():
    assert get_derivative_case(lambda x: True) == "closed-form"  # noqa: ARG005
    assert get_derivative_case(False) == "skip"
    assert get_derivative_case(None) == "numerical"


def test_to_numpy_invalid():
    with pytest.raises(TypeError):
        _to_numpy(15)


def test_calculate_estimation_summary():
    # input data
    summary_data = {
        "value": {
            "a": pd.Series([0], index=["i"]),
            "b": pd.DataFrame({"c1": [1], "c2": [2]}),
        },
        "standard_error": {
            "a": pd.Series([0.1], index=["i"]),
            "b": pd.DataFrame({"c1": [0.2], "c2": [0.3]}),
        },
        "ci_lower": {
            "a": pd.Series([-0.2], index=["i"]),
            "b": pd.DataFrame({"c1": [-0.4], "c2": [-0.6]}),
        },
        "ci_upper": {
            "a": pd.Series([0.2], index=["i"]),
            "b": pd.DataFrame({"c1": [0.4], "c2": [0.6]}),
        },
        "p_value": {
            "a": pd.Series([0.001], index=["i"]),
            "b": pd.DataFrame({"c1": [0.2], "c2": [0.07]}),
        },
        "free": np.array([True, True, True]),
    }

    registry = get_registry(extended=True)
    names = leaf_names(summary_data["value"], registry=registry)
    free_names = names

    # function call
    summary = calculate_estimation_summary(summary_data, names, free_names)

    # expectations
    expectation = {
        "a": pd.DataFrame(
            {
                "value": 0,
                "standard_error": 0.1,
                "ci_lower": -0.2,
                "ci_upper": 0.2,
                "p_value": 0.001,
                "free": True,
                "stars": "***",
            },
            index=["i"],
        ),
        "b": pd.DataFrame(
            {
                "value": [1, 2],
                "standard_error": [0.2, 0.3],
                "ci_lower": [-0.4, -0.6],
                "ci_upper": [0.4, 0.6],
                "p_value": [0.2, 0.7],
                "free": [True, True],
                "stars": ["", "*"],
            },
            index=pd.MultiIndex.from_tuples([(0, "c1"), (0, "c2")]),
        ),
    }

    tree_equal(summary, expectation)
