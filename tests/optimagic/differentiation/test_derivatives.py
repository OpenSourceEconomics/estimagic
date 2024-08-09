from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import get_type_hints

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.differentiation.derivatives import (
    Evals,
    NumdiffResult,
    _consolidate_one_step_derivatives,
    _convert_evaluation_data_to_frame,
    _convert_richardson_candidates_to_frame,
    _is_scalar_nan,
    _nan_skipping_batch_evaluator,
    _reshape_cross_step_evals,
    _reshape_one_step_evals,
    _reshape_two_step_evals,
    _select_minimizer_along_axis,
    first_derivative,
    second_derivative,
)
from optimagic.differentiation.generate_steps import Steps
from optimagic.examples.numdiff_functions import (
    logit_loglike,
    logit_loglike_gradient,
    logit_loglike_hessian,
    logit_loglikeobs,
    logit_loglikeobs_jacobian,
)
from optimagic.parameters.bounds import Bounds
from pandas.testing import assert_frame_equal
from scipy.optimize._numdiff import approx_derivative


@pytest.fixture()
def binary_choice_inputs():
    fix_path = Path(__file__).resolve().parent / "binary_choice_inputs.pickle"
    inputs = pd.read_pickle(fix_path)
    return inputs


methods = ["forward", "backward", "central"]
methods_second_derivative = ["forward", "backward", "central_average", "central_cross"]


@pytest.mark.parametrize("method", methods)
def test_first_derivative_jacobian(binary_choice_inputs, method):
    fix = binary_choice_inputs
    func = partial(logit_loglikeobs, y=fix["y"], x=fix["x"])

    bounds = Bounds(
        lower=np.full(fix["params_np"].shape, -np.inf),
        upper=np.full(fix["params_np"].shape, np.inf),
    )

    calculated = first_derivative(
        func=func,
        method=method,
        params=fix["params_np"],
        step_size=None,
        bounds=bounds,
        min_steps=1e-8,
        f0=func(fix["params_np"]),
        n_cores=1,
    )

    expected = logit_loglikeobs_jacobian(fix["params_np"], fix["y"], fix["x"])

    aaae(calculated.derivative, expected, decimal=6)


def test_first_derivative_jacobian_works_at_defaults(binary_choice_inputs):
    fix = binary_choice_inputs
    func = partial(logit_loglikeobs, y=fix["y"], x=fix["x"])
    calculated = first_derivative(func=func, params=fix["params_np"], n_cores=1)
    expected = logit_loglikeobs_jacobian(fix["params_np"], fix["y"], fix["x"])
    aaae(calculated.derivative, expected, decimal=6)


@pytest.mark.parametrize("method", methods)
def test_first_derivative_gradient(binary_choice_inputs, method):
    fix = binary_choice_inputs
    func = partial(logit_loglike, y=fix["y"], x=fix["x"])

    calculated = first_derivative(
        func=func,
        method=method,
        params=fix["params_np"],
        f0=func(fix["params_np"]),
        n_cores=1,
    )

    expected = logit_loglike_gradient(fix["params_np"], fix["y"], fix["x"])

    aaae(calculated.derivative, expected, decimal=4)


@pytest.mark.parametrize("method", methods_second_derivative)
def test_second_derivative_hessian(binary_choice_inputs, method):
    fix = binary_choice_inputs
    func = partial(logit_loglike, y=fix["y"], x=fix["x"])

    calculated = second_derivative(
        func=func,
        method=method,
        params=fix["params_np"],
        f0=func(fix["params_np"]),
        n_cores=1,
    )

    expected = logit_loglike_hessian(fix["params_np"], fix["y"], fix["x"])

    assert np.max(np.abs(calculated.derivative - expected)) < 1.5 * 10 ** (-2)
    assert np.mean(np.abs(calculated.derivative - expected)) < 1.5 * 10 ** (-3)


@pytest.mark.parametrize("method", methods)
def test_first_derivative_scalar(method):  # noqa: ARG001
    def f(x):
        return x**2

    calculated = first_derivative(f, 3.0, n_cores=1)
    expected = 6.0
    assert calculated.derivative == expected


@pytest.mark.parametrize("method", methods_second_derivative)
def test_second_derivative_scalar(method):  # noqa: ARG001
    def f(x):
        return x**2

    calculated = second_derivative(f, 3.0, n_cores=1)
    expected = 2.0

    assert np.abs(calculated.derivative - expected) < 1.5 * 10 ** (-6)


def test_nan_skipping_batch_evaluator():
    arglist = [np.nan, np.ones(2), np.array([3, 4]), np.nan, np.array([1, 2])]
    expected = [
        np.full(2, np.nan),
        np.ones(2),
        np.array([9, 16]),
        np.full(2, np.nan),
        np.array([1, 4]),
    ]
    calculated = _nan_skipping_batch_evaluator(
        func=lambda x: x**2,
        arguments=arglist,
        n_cores=1,
        error_handling="continue",
        batch_evaluator="joblib",
    )
    for arr_calc, arr_exp in zip(calculated, expected, strict=False):
        if np.isnan(arr_exp).all():
            assert np.isnan(arr_calc).all()
        else:
            aaae(arr_calc, arr_exp)


def test_consolidate_one_step_derivatives():
    forward = np.ones((1, 4, 3))
    forward[:, :, 0] = np.nan
    backward = np.zeros_like(forward)

    calculated = _consolidate_one_step_derivatives(
        {"forward": forward, "backward": backward}, ["forward", "backward"]
    )
    expected = np.array([[0, 1, 1]] * 4)
    aaae(calculated, expected)


@pytest.fixture()
def example_function_gradient_fixtures():
    def f(x):
        """F:R^3 -> R."""
        x1, x2, x3 = x[0], x[1], x[2]
        y1 = np.sin(x1) + np.cos(x2) + x3 - x3
        return y1

    def fprime(x):
        """Gradient(f)(x):R^3 -> R^3."""
        x1, x2, x3 = x[0], x[1], x[2]
        grad = np.array([np.cos(x1), -np.sin(x2), x3 - x3])
        return grad

    return {"func": f, "func_prime": fprime}


@pytest.fixture()
def example_function_jacobian_fixtures():
    def f(x):
        """F:R^3 -> R^2."""
        x1, x2, x3 = x[0], x[1], x[2]
        y1, y2 = np.sin(x1) + np.cos(x2), np.exp(x3)
        return np.array([y1, y2])

    def fprime(x):
        """Jacobian(f)(x):R^3 -> R^(2x3)"""
        x1, x2, x3 = x[0], x[1], x[2]
        jac = np.array([[np.cos(x1), -np.sin(x2), 0], [0, 0, np.exp(x3)]])
        return jac

    return {"func": f, "func_prime": fprime}


@pytest.mark.filterwarnings("ignore:The `n_steps` argument")
def test_first_derivative_gradient_richardson(example_function_gradient_fixtures):
    f = example_function_gradient_fixtures["func"]
    fprime = example_function_gradient_fixtures["func_prime"]

    true_fprime = fprime(np.ones(3))
    scipy_fprime = approx_derivative(f, np.ones(3))

    our_fprime = first_derivative(f, np.ones(3), n_steps=3, method="central", n_cores=1)

    aaae(scipy_fprime, our_fprime.derivative)
    aaae(true_fprime, our_fprime.derivative)


@pytest.mark.filterwarnings("ignore:The `n_steps` argument")
def test_first_derivative_jacobian_richardson(example_function_jacobian_fixtures):
    f = example_function_jacobian_fixtures["func"]
    fprime = example_function_jacobian_fixtures["func_prime"]

    true_fprime = fprime(np.ones(3))
    scipy_fprime = approx_derivative(f, np.ones(3))

    our_fprime = first_derivative(f, np.ones(3), n_steps=3, method="central", n_cores=1)

    aaae(scipy_fprime, our_fprime.derivative)
    aaae(true_fprime, our_fprime.derivative)


def test_convert_evaluation_data_to_frame():
    arr = np.arange(4).reshape(2, 2)
    arr2 = arr.reshape(2, 1, 2)
    steps = Steps(pos=arr, neg=-arr)
    evals = Evals(pos=arr2, neg=-arr2)
    expected = [
        [1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 1],
        [1, 1, 0, 0, 2, 2],
        [1, 1, 1, 0, 3, 3],
        [-1, 0, 0, 0, 0, 0],
        [-1, 0, 1, 0, 1, -1],
        [-1, 1, 0, 0, 2, -2],
        [-1, 1, 1, 0, 3, -3],
    ]
    expected = pd.DataFrame(
        expected, columns=["sign", "step_number", "dim_x", "dim_f", "step", "eval"]
    )
    got = _convert_evaluation_data_to_frame(steps, evals)
    assert_frame_equal(expected, got.reset_index(), check_dtype=False)


def test__convert_richardson_candidates_to_frame():
    jac = {
        "forward1": np.array([[0, 1], [2, 3]]),
        "forward2": np.array([[0.5, 1], [2, 3]]),
    }
    err = {
        "forward1": np.array([[0, 0], [0, 1]]),
        "forward2": np.array([[1, 0], [0, 0]]),
    }
    expected = [
        ["forward", 1, 0, 0, 0, 0],
        ["forward", 1, 1, 0, 1, 0],
        ["forward", 1, 0, 1, 2, 0],
        ["forward", 1, 1, 1, 3, 1],
        ["forward", 2, 0, 0, 0.5, 1],
        ["forward", 2, 1, 0, 1, 0],
        ["forward", 2, 0, 1, 2, 0],
        ["forward", 2, 1, 1, 3, 0],
    ]
    expected = pd.DataFrame(
        expected, columns=["method", "num_term", "dim_x", "dim_f", "der", "err"]
    )
    expected = expected.set_index(["method", "num_term", "dim_x", "dim_f"])
    got = _convert_richardson_candidates_to_frame(jac, err)
    assert_frame_equal(got, expected, check_dtype=False, check_index_type=False)


def test__select_minimizer_along_axis():
    der = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
    err = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])
    expected = (np.array([[0, 5], [6, 3]]), np.array([[0, 0], [0, 0]]))
    got = _select_minimizer_along_axis(der, err)
    aaae(expected, got)


def test_reshape_one_step_evals():
    n_steps, dim_f, dim_x = 2, 3, 4
    raw_evals_one_step = np.arange(2 * n_steps * dim_f * dim_x)

    pos_expected = np.array(
        [
            [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]],
            [[12, 15, 18, 21], [13, 16, 19, 22], [14, 17, 20, 23]],
        ]
    )
    neg_expected = np.array(
        [
            [[24, 27, 30, 33], [25, 28, 31, 34], [26, 29, 32, 35]],
            [[36, 39, 42, 45], [37, 40, 43, 46], [38, 41, 44, 47]],
        ]
    )

    got = _reshape_one_step_evals(raw_evals_one_step, n_steps, dim_x)
    assert np.all(got.pos == pos_expected)
    assert np.all(got.neg == neg_expected)


def test_reshape_two_step_evals():
    n_steps, dim_x, dim_f = 1, 2, 2
    raw_evals_two_step = np.arange(2 * n_steps * dim_f * dim_x * dim_x)

    pos_expected = np.array([[[[0, 2], [2, 6]], [[1, 3], [3, 7]]]])
    neg_expected = np.array([[[[8, 10], [10, 14]], [[9, 11], [11, 15]]]])

    got = _reshape_two_step_evals(raw_evals_two_step, n_steps, dim_x)
    assert np.all(got.pos == pos_expected)
    assert np.all(got.neg == neg_expected)


def test_reshape_cross_step_evals():
    n_steps = 1
    dim_x = 2
    dim_f = 2
    f0 = np.array([-1000, 1000])

    raw_evals_cross_step = np.arange(2 * n_steps * dim_f * dim_x * dim_x)

    expected_pos = np.array([[[[-1000, 2], [10, -1000]], [[1000, 3], [11, 1000]]]])
    expected_neg = expected_pos.swapaxes(2, 3)

    got = _reshape_cross_step_evals(raw_evals_cross_step, n_steps, dim_x, f0)
    assert np.all(got.pos == expected_pos)
    assert np.all(got.neg == expected_neg)


def test_is_scalar_nan():
    assert _is_scalar_nan(np.nan)
    assert not _is_scalar_nan(1.0)
    assert not _is_scalar_nan(np.array([np.nan]))


@dataclass
class MyOutput:
    value: float
    message: str


def test_first_derivative_with_unpacking():
    def f(x):
        return MyOutput(x @ x, "success")

    got = first_derivative(
        func=f,
        params=np.ones(3),
        unpacker=lambda out: out.value,
    )

    assert isinstance(got.func_value, MyOutput)
    aaae(got.derivative, np.ones(3) * 2)


def test_second_derivative_with_unpacking():
    def f(x):
        return MyOutput(x @ x, "success")

    got = second_derivative(
        func=f,
        params=np.ones(3),
        unpacker=lambda out: out.value,
    )

    assert isinstance(got.func_value, MyOutput)
    aaae(got.derivative, np.eye(3) * 2, decimal=4)


@pytest.mark.filterwarnings("ignore:The dictionary access for")
def test_numdiff_result_getitem():
    res = NumdiffResult(
        derivative=1,
        func_value=2,
        _func_evals=pd.DataFrame([0, 1]),
        _derivative_candidates=pd.DataFrame([2, 3]),
    )
    assert res["derivative"] == res.derivative
    assert res["func_value"] == res.func_value
    assert_frame_equal(res["_func_evals"], res._func_evals)
    assert_frame_equal(res["_derivative_candidates"], res._derivative_candidates)


def test_first_and_second_derivative_have_same_type_hints():
    # exclude method from comparison, as the argument options differ here
    exclude = ["method"]
    first_hints = {
        k: v for k, v in get_type_hints(first_derivative).items() if k not in exclude
    }
    second_hints = {
        k: v for k, v in get_type_hints(second_derivative).items() if k not in exclude
    }
    assert first_hints == second_hints
