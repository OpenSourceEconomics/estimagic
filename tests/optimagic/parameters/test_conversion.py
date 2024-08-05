import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.parameters.bounds import Bounds
from optimagic.parameters.conversion import (
    _is_fast_deriv_eval,
    _is_fast_func_eval,
    _is_fast_path,
    get_converter,
)
from optimagic.parameters.scaling import ScalingOptions


def test_get_converter_fast_case():
    converter, internal = get_converter(
        params=np.arange(3),
        constraints=None,
        bounds=None,
        func_eval=3,
        derivative_eval=2 * np.arange(3),
        primary_key="value",
    )

    aaae(internal.values, np.arange(3))
    aaae(internal.lower_bounds, np.full(3, -np.inf))
    aaae(internal.upper_bounds, np.full(3, np.inf))

    aaae(converter.params_to_internal(np.arange(3)), np.arange(3))
    aaae(converter.params_from_internal(np.arange(3)), np.arange(3))
    aaae(
        converter.derivative_to_internal(2 * np.arange(3), np.arange(3)),
        2 * np.arange(3),
    )
    aaae(converter.func_to_internal(3), 3)


def test_get_converter_with_constraints_and_bounds():
    bounds = Bounds(
        lower=np.array([-1, -np.inf, -np.inf]),
        upper=np.array([np.inf, 10, np.inf]),
    )
    converter, internal = get_converter(
        params=np.arange(3),
        constraints=[{"loc": 2, "type": "fixed"}],
        bounds=bounds,
        func_eval=3,
        derivative_eval=2 * np.arange(3),
        primary_key="value",
    )

    aaae(internal.values, np.arange(2))
    aaae(internal.lower_bounds, np.array([-1, -np.inf]))
    aaae(internal.upper_bounds, np.array([np.inf, 10]))

    aaae(converter.params_to_internal(np.arange(3)), np.arange(2))
    aaae(converter.params_from_internal(np.arange(2)), np.arange(3))
    aaae(
        converter.derivative_to_internal(2 * np.arange(3), np.arange(2)),
        2 * np.arange(2),
    )
    aaae(converter.func_to_internal(3), 3)


def test_get_converter_with_scaling():
    bounds = Bounds(
        lower=np.arange(3) - 1,
        upper=np.arange(3) + 1,
    )
    converter, internal = get_converter(
        params=np.arange(3),
        constraints=None,
        bounds=bounds,
        func_eval=3,
        derivative_eval=2 * np.arange(3),
        primary_key="value",
        scaling=ScalingOptions(method="start_values", clipping_value=0.5),
    )

    aaae(internal.values, np.array([0, 1, 1]))
    aaae(internal.lower_bounds, np.array([-2, 0, 0.5]))
    aaae(internal.upper_bounds, np.array([2, 2, 1.5]))

    aaae(converter.params_to_internal(np.arange(3)), np.array([0, 1, 1]))
    aaae(converter.params_from_internal(np.array([0, 1, 1])), np.arange(3))
    aaae(
        converter.derivative_to_internal(2 * np.arange(3), np.arange(3)),
        np.array([0, 2, 8]),
    )
    aaae(converter.func_to_internal(3), 3)


def test_get_converter_with_trees():
    params = {"a": 0, "b": 1, "c": 2}
    converter, internal = get_converter(
        params=params,
        constraints=None,
        bounds=None,
        func_eval={"contributions": {"d": 1, "e": 2}},
        derivative_eval={"a": 0, "b": 2, "c": 4},
        primary_key="value",
    )

    aaae(internal.values, np.arange(3))
    aaae(internal.lower_bounds, np.full(3, -np.inf))
    aaae(internal.upper_bounds, np.full(3, np.inf))

    aaae(converter.params_to_internal(params), np.arange(3))
    assert converter.params_from_internal(np.arange(3)) == params
    aaae(
        converter.derivative_to_internal(params, np.arange(3)),
        np.arange(3),
    )
    aaae(converter.func_to_internal({"contributions": {"d": 1, "e": 2}}), 3)


@pytest.fixture()
def fast_kwargs():
    kwargs = {
        "params": np.arange(3),
        "constraints": None,
        "func_eval": 3,
        "primary_key": "value",
        "scaling": None,
        "derivative_eval": np.arange(3),
        "add_soft_bounds": False,
    }
    return kwargs


STILL_FAST = [
    ("params", np.arange(3)),
    ("constraints", []),
    ("func_eval", {"value": 3}),
    ("derivative_eval", {"value": np.arange(3)}),
]


@pytest.mark.parametrize("name, value", STILL_FAST)
def test_is_fast_path_when_true(fast_kwargs, name, value):
    kwargs = fast_kwargs.copy()
    kwargs[name] = value
    assert _is_fast_path(**kwargs)


SLOW = [
    ("params", {"a": 1}),
    ("params", np.arange(4).reshape(2, 2)),
    ("constraints", [{}]),
    ("func_eval", np.array([1])),
    ("func_eval", {"a": 1}),
    ("scaling", ScalingOptions()),
    ("derivative_eval", {"bla": 3}),
    ("derivative_eval", np.arange(3).reshape(1, 3)),
    ("add_soft_bounds", True),
]


@pytest.mark.parametrize("name, value", SLOW)
def test_is_fast_path_when_false(fast_kwargs, name, value):
    kwargs = fast_kwargs.copy()
    kwargs[name] = value
    assert not _is_fast_path(**kwargs)


FAST_EVAL_CASES = [
    ("contributions", np.arange(3)),
    ("contributions", {"contributions": np.arange(3)}),
    ("root_contributions", np.arange(3)),
    ("root_contributions", {"root_contributions": np.arange(3)}),
]


@pytest.mark.parametrize("key, f", FAST_EVAL_CASES)
def test_is_fast_func_eval_true(key, f):
    assert _is_fast_func_eval(f, key)


helper = np.arange(6).reshape(3, 2)

FAST_DERIV_CASES = [
    ("contributions", helper),
    ("contributions", {"contributions": helper}),
    ("root_contributions", helper),
    ("root_contributions", {"root_contributions": helper}),
    ("value", None),
    ("contributions", None),
    ("root_contributions", None),
]


@pytest.mark.parametrize("key, f", FAST_DERIV_CASES)
def test_is_fast_deriv_eval_true(key, f):
    assert _is_fast_deriv_eval(f, key)


SLOW_EVAL_CASES = [
    ("contributions", {"a": 1, "b": 2, "c": 3}),
    ("contributions", {"contributions": {"a": 1, "b": 2, "c": 3}}),
    ("root_contributions", {"a": 1, "b": 2, "c": 3}),
    ("root_contributions", {"root_contributions": {"a": 1, "b": 2, "c": 3}}),
]


@pytest.mark.parametrize("key, f", SLOW_EVAL_CASES)
def test_is_fast_func_eval_false(key, f):
    assert not _is_fast_func_eval(f, key)


SLOW_DERIV_CASES = [
    ("contributions", np.arange(8).reshape(2, 2, 2)),
    ("contributions", {"contributions": np.arange(8).reshape(2, 2, 2)}),
    ("root_contributions", np.arange(8).reshape(2, 2, 2)),
    ("root_contributions", {"root_contributions": np.arange(8).reshape(2, 2, 2)}),
]


@pytest.mark.parametrize("key, f", SLOW_DERIV_CASES)
def test_is_fast_deriv_eval_false(key, f):
    assert not _is_fast_deriv_eval(f, key)
