import itertools

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.decorators import AlgoInfo
from optimagic.differentiation.numdiff_options import NumdiffOptions
from optimagic.examples.criterion_functions import (
    sos_gradient,
)
from optimagic.optimization.fun_value import (
    LeastSquaresFunctionValue,
    ScalarFunctionValue,
)
from optimagic.optimization.internal_criterion_template import (
    internal_criterion_and_derivative_template,
)
from optimagic.parameters.bounds import Bounds
from optimagic.parameters.conversion import get_converter


def reparametrize_from_internal(x):
    res = pd.DataFrame()
    res["value"] = x
    return res


def convert_derivative(external_derivative, internal_values):  # noqa: ARG001
    return external_derivative


@pytest.fixture()
def base_inputs():
    x = np.arange(5).astype(float)
    params = pd.DataFrame(data=np.arange(5).reshape(-1, 1), columns=["value"])
    inputs = {
        "x": x,
        "params": params,
        "algo_info": AlgoInfo(
            name="my_algorithm",
            primary_criterion_entry="value",
            needs_scaling=False,
            is_available=True,
            parallelizes=False,
            arguments=[],
        ),
        "bounds": Bounds(lower=np.zeros(5), upper=np.ones(5)),
        "error_handling": "raise",
        "numdiff_options": NumdiffOptions(),
        "logging": False,
        "database": None,
        "error_penalty_func": None,
        "fixed_log_data": {"stage": "optimization", "substage": 0},
    }
    return inputs


directions = ["maximize", "minimize"]


def sos_ls(params):
    x = params["value"].to_numpy()
    return LeastSquaresFunctionValue(value=x)


def sos_scalar(params):
    x = params["value"].to_numpy()
    return ScalarFunctionValue(value=x @ x)


def sos_ls_pd(params):
    return LeastSquaresFunctionValue(value=params["value"])


def sos_fun_and_gradient(params):
    x = params["value"].to_numpy()
    grad = params.copy()
    grad["value"] = 2 * x
    return ScalarFunctionValue(value=x @ x), grad


crits = [sos_ls, sos_ls_pd, sos_scalar]
derivs = [sos_gradient, None]
crits_and_derivs = [sos_fun_and_gradient, None]

test_cases = list(itertools.product(directions, crits, derivs, crits_and_derivs))


@pytest.mark.parametrize("direction, crit, deriv, crit_and_deriv", test_cases)
def test_criterion_and_derivative_template(
    base_inputs, direction, crit, deriv, crit_and_deriv
):
    converter, _ = get_converter(
        params=base_inputs["params"],
        constraints=None,
        bounds=None,
        func_eval=crit(base_inputs["params"]),
        primary_key="value",
        derivative_eval=None,
    )
    inputs = {k: v for k, v in base_inputs.items() if k != "params"}
    inputs["converter"] = converter

    crit = crit if (deriv, crit_and_deriv) == (None, None) else crit

    inputs["criterion"] = crit
    inputs["derivative"] = deriv
    inputs["criterion_and_derivative"] = crit_and_deriv
    inputs["direction"] = direction

    calc_criterion, calc_derivative = internal_criterion_and_derivative_template(
        task="criterion_and_derivative", **inputs
    )

    calc_criterion2 = internal_criterion_and_derivative_template(
        task="criterion", **inputs
    )

    calc_derivative2 = internal_criterion_and_derivative_template(
        task="derivative", **inputs
    )

    if direction == "minimize":
        for c in calc_criterion, calc_criterion2:
            assert c == 30

        for d in calc_derivative, calc_derivative2:
            aaae(d, 2 * np.arange(5))
    else:
        for c in calc_criterion, calc_criterion2:
            assert c == -30

        for d in calc_derivative, calc_derivative2:
            aaae(d, -2 * np.arange(5))


@pytest.mark.parametrize("direction", directions)
def test_internal_criterion_with_penalty(base_inputs, direction):
    converter, _ = get_converter(
        params=base_inputs["params"],
        constraints=None,
        bounds=None,
        func_eval=sos_scalar(base_inputs["params"]),
        primary_key="value",
        derivative_eval=None,
    )
    inputs = {k: v for k, v in base_inputs.items() if k != "params"}

    inputs["converter"] = converter

    def raising_crit_and_deriv(x):  # noqa: ARG001
        raise ValueError()

    inputs["error_handling"] = "continue"
    inputs["x"] = inputs["x"] + 10
    inputs["criterion"] = sos_scalar
    inputs["derivative"] = sos_gradient
    inputs["criterion_and_derivative"] = raising_crit_and_deriv
    inputs["direction"] = direction
    inputs["error_penalty_func"] = lambda x, task: (ScalarFunctionValue(42), 52)  # noqa: ARG005

    with pytest.warns():
        calc_criterion, calc_derivative = internal_criterion_and_derivative_template(
            task="criterion_and_derivative", **inputs
        )

    expected_crit = 42
    expected_grad = 52

    if direction == "minimize":
        assert calc_criterion == expected_crit
        aaae(calc_derivative, expected_grad)

    else:
        assert calc_criterion == -expected_crit
        aaae(calc_derivative, -expected_grad)
