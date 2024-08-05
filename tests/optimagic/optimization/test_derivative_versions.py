"""Test different versions of specifying a criterion functions.

Here we want to take:
- Few representative algorithms (derivative based, derivative free, least squares)
- One basic criterion function (sum of squares)
- Many ways of specifying derivatives or not.
- Maximize and minimize since this might make a difference

"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.decorators import switch_sign
from optimagic.examples.criterion_functions import (
    sos_criterion_and_gradient,
    sos_criterion_and_jacobian,
    sos_dict_criterion,
    sos_dict_derivative,
    sos_gradient,
    sos_jacobian,
    sos_pandas_gradient,
    sos_pandas_jacobian,
)
from optimagic.optimization.optimize import maximize, minimize

algorithms = ["scipy_lbfgsb", "scipy_neldermead", "scipy_ls_dogbox"]

ls_algorithms = {"scipy_ls_dogbox"}


scalar_derivatives = [None, sos_gradient, sos_pandas_gradient]

scalar_criterion_and_derivtives = [None, sos_criterion_and_gradient]

ls_derivatives = [None, sos_jacobian, sos_pandas_jacobian]


ls_criterion_and_derivatives = [sos_criterion_and_jacobian]

dict_criterion_and_derivatives = [
    None,
]

valid_cases = []
invalid_cases = []
for algo in algorithms:
    if algo in ls_algorithms:
        for deriv in ls_derivatives:
            for crit_and_deriv in ls_criterion_and_derivatives:
                valid_cases.append(("minimize", algo, deriv, crit_and_deriv))
                invalid_cases.append(("maximize", algo, deriv, crit_and_deriv))
    else:
        for deriv in scalar_derivatives:
            for crit_and_deriv in scalar_criterion_and_derivtives:
                for direction in ["minimize", "maximize"]:
                    valid_cases.append((direction, algo, deriv, crit_and_deriv))


@pytest.mark.parametrize(
    "direction, algorithm, derivative, criterion_and_derivative", valid_cases
)
def test_valid_derivative_versions(
    direction, algorithm, derivative, criterion_and_derivative
):
    start_params = pd.DataFrame()
    start_params["value"] = [1, 2, 3]

    if direction == "minimize":
        res = minimize(
            fun=sos_dict_criterion,
            params=start_params,
            algorithm=algorithm,
            jac=derivative,
            fun_and_jac=criterion_and_derivative,
            error_handling="raise",
        )
    else:
        deriv = derivative if derivative is None else switch_sign(derivative)
        crit_and_deriv = (
            criterion_and_derivative
            if criterion_and_derivative is None
            else switch_sign(criterion_and_derivative)
        )
        res = maximize(
            fun=switch_sign(sos_dict_criterion),
            params=start_params,
            algorithm=algorithm,
            jac=deriv,
            fun_and_jac=crit_and_deriv,
            error_handling="raise",
        )

    aaae(res.params["value"].to_numpy(), np.zeros(3), decimal=4)


@pytest.mark.parametrize(
    "direction, algorithm, derivative, criterion_and_derivative", invalid_cases
)
def test_invalid_derivative_versions(
    direction, algorithm, derivative, criterion_and_derivative
):
    start_params = pd.DataFrame()
    start_params["value"] = [1, 2, 3]

    if direction == "minimize":
        with pytest.raises(ValueError):
            minimize(
                fun=sos_dict_criterion,
                params=start_params,
                algorithm=algorithm,
                jac=derivative,
                fun_and_jac=criterion_and_derivative,
            )
    else:
        deriv = derivative if derivative is None else switch_sign(derivative)
        crit_and_deriv = (
            criterion_and_derivative
            if criterion_and_derivative is None
            else switch_sign(criterion_and_derivative)
        )
        with pytest.raises(ValueError):
            maximize(
                fun=switch_sign(sos_dict_criterion),
                params=start_params,
                algorithm=algorithm,
                jac=deriv,
                fun_and_jac=crit_and_deriv,
            )


def test_dict_derivative():
    start_params = pd.DataFrame()
    start_params["value"] = [1, 2, 3]

    res = minimize(
        fun=sos_dict_criterion,
        params=start_params,
        algorithm="scipy_lbfgsb",
        jac=sos_dict_derivative,
    )

    aaae(res.params["value"].to_numpy(), np.zeros(3))
