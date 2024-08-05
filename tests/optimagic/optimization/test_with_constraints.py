"""Test many different criterion functions and many sets of constraints.

- only minimize
- only gradient based algorithms scipy_lbfgsb (scalar) and scipy_ls_dogbox (least
  squares)
- closed form and numerical derivatives

"""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.examples.criterion_functions import (
    rosenbrock_dict_criterion,
    rosenbrock_gradient,
    rotated_hyper_ellipsoid_dict_criterion,
    rotated_hyper_ellipsoid_gradient,
    sos_dict_criterion,
    sos_gradient,
    sos_jacobian,
    sos_ls_jacobian,
    trid_gradient,
    trid_scalar_criterion,
)
from optimagic.exceptions import InvalidConstraintError, InvalidParamsError
from optimagic.optimization.optimize import maximize, minimize
from optimagic.parameters.bounds import Bounds


def logit_loglike(params, y, x):
    """Log-likelihood function of a logit model.

    Args:
        params (pd.DataFrame): The index consists of the parameter names,
            the "value" column are the parameter values.
        y (np.array): 1d numpy array with the dependent variable
        x (np.array): 2d numpy array with the independent variables

    Returns:
        loglike (np.array): 1d numpy array with likelihood contribution  per individual

    """
    if isinstance(params, pd.DataFrame):
        p = params["value"].to_numpy()
    else:
        p = params
    q = 2 * y - 1
    contribs = np.log(1 / (1 + np.exp(-(q * np.dot(x, p)))))

    out = {"value": contribs.sum(), "contributions": contribs}

    return out


FUNC_INFO = {
    "sos": {
        "criterion": sos_dict_criterion,
        "gradient": sos_gradient,
        "jacobian": sos_jacobian,
        "ls_jacobian": sos_ls_jacobian,
        "default_result": np.zeros(3),
        "fixed_result": [1, 0, 0],
        "entries": ["value", "contributions", "root_contributions"],
        "linear_result": [0.8, 1.6, 0],
        "probability_result": [0.5, 0.5, 0],
    },
    "rotated_hyper_ellipsoid": {
        "criterion": rotated_hyper_ellipsoid_dict_criterion,
        "gradient": rotated_hyper_ellipsoid_gradient,
        "entries": ["value", "contributions", "root_contributions"],
        "default_result": np.zeros(3),
        "fixed_result": [1, 0, 0],
        "linear_result": [0.571428571, 1.714285714, 0],
        "probability_result": [0.4, 0.6, 0],
    },
    "rosenbrock": {
        "criterion": rosenbrock_dict_criterion,
        "gradient": rosenbrock_gradient,
        "entries": ["value", "contributions"],
        "default_result": np.ones(3),
        "linear_result": "unknown",
        "probability_result": "unknown",
    },
    "trid": {
        "criterion": trid_scalar_criterion,
        "gradient": trid_gradient,
        "entries": ["value"],
        "default_result": [3, 4, 3],
        "fixed_result": [1, 2.666666667, 2.333333333],
        "equality_result": [3, 3, 3],
        "pairwise_equality_result": [3.333333333, 3.333333333, 2.666666667],
        "increasing_result": [2.666666667, 3.3333333, 3.3333333],
        "decreasing_result": "unknown",
        "linear_result": [1.185185185, 1.4074074069999998, 1.703703704],
        "probability_result": [0.272727273, 0.727272727, 1.363636364],
        "covariance_result": "unknown",
        "sdcorr_result": "unknown",
    },
}


CONSTR_INFO = {
    "fixed": {"loc": [0], "type": "fixed", "value": 1},
    "equality": [{"loc": [0, 1, 2], "type": "equality"}],
    "pairwise_equality": {"locs": [0, 1], "type": "pairwise_equality"},
    "increasing": [{"loc": [1, 2], "type": "increasing"}],
    "decreasing": [{"loc": [0, 1], "type": "decreasing"}],
    "linear": [{"loc": [0, 1], "type": "linear", "value": 4, "weights": [1, 2]}],
    "probability": [{"loc": [0, 1], "type": "probability"}],
    "covariance": [{"loc": [0, 1, 2], "type": "covariance"}],
    "sdcorr": [{"loc": [0, 1, 2], "type": "sdcorr"}],
}


START_INFO = {
    "fixed": [1, 1.5, 4.5],
    "equality": [1, 1, 1],
    "pairwise_equality": [2, 2, 3],
    "increasing": [1, 2, 3],
    "decreasing": [3, 2, 1],
    "linear": [2, 1, 3],
    "probability": [0.8, 0.2, 3],
    "covariance": [2, 1, 2],
    "sdcorr": [2, 2, 0.5],
}

KNOWN_FAILURES = {
    ("rosenbrock", "equality"),
    ("rosenbrock", "decreasing"),  # imprecise
}

PARAMS_TYPES = ["numpy", "pandas"]

test_cases = []
for crit_name in FUNC_INFO:
    for constr_name in CONSTR_INFO:
        for ptype in PARAMS_TYPES:
            unknown_res = FUNC_INFO[crit_name].get(f"{constr_name}_result") == "unknown"
            known_failure = (crit_name, constr_name) in KNOWN_FAILURES
            if not any([unknown_res, known_failure]):
                for deriv in None, FUNC_INFO[crit_name]["gradient"]:
                    test_cases.append(
                        (crit_name, "scipy_lbfgsb", deriv, constr_name, ptype)
                    )

                if "root_contributions" in FUNC_INFO[crit_name]["entries"]:
                    for deriv in [FUNC_INFO[crit_name].get("ls_jacobian"), None]:
                        test_cases.append(
                            (crit_name, "scipy_ls_dogbox", deriv, constr_name, ptype)
                        )


@pytest.mark.parametrize(
    "criterion_name, algorithm, derivative, constraint_name, params_type",
    test_cases,
)
def test_constrained_minimization(
    criterion_name, algorithm, derivative, constraint_name, params_type
):
    constraints = CONSTR_INFO[constraint_name]
    criterion = FUNC_INFO[criterion_name]["criterion"]
    if params_type == "pandas":
        params = pd.Series(START_INFO[constraint_name], name="value").to_frame()
    else:
        params = np.array(START_INFO[constraint_name])

    res = minimize(
        fun=criterion,
        params=params,
        algorithm=algorithm,
        jac=derivative,
        constraints=constraints,
        algo_options={"convergence.ftol_rel": 1e-12},
    )

    if params_type == "pandas":
        calculated = res.params["value"].to_numpy()
    else:
        calculated = res.params

    expected = FUNC_INFO[criterion_name].get(
        f"{constraint_name}_result", FUNC_INFO[criterion_name]["default_result"]
    )

    aaae(calculated, expected, decimal=4)


def test_fix_that_differs_from_start_value_raises_an_error():
    with pytest.raises(InvalidParamsError):
        minimize(
            fun=lambda x: x @ x,
            params=np.arange(3),
            algorithm="scipy_lbfgsb",
            constraints=[{"loc": [1], "type": "fixed", "value": 10}],
        )


def test_three_independent_constraints():
    params = np.arange(10)
    params[0] = 2

    constraints = [
        {"loc": [0, 1, 2], "type": "covariance"},
        {"loc": [4, 5], "type": "fixed"},
        {"loc": [7, 8], "type": "linear", "value": 15, "weights": 1},
    ]

    res = minimize(
        fun=lambda x: x @ x,
        params=params,
        algorithm="scipy_lbfgsb",
        constraints=constraints,
        algo_options={"convergence.ftol_rel": 1e-12},
    )
    expected = np.array([0] * 4 + [4, 5] + [0] + [7.5] * 2 + [0])

    aaae(res.params, expected, decimal=4)


INVALID_CONSTRAINT_COMBIS = [
    [{"loc": [1, 0, 2], "type": "covariance"}, {"loc": [0, 1], "type": "probability"}],
    [
        {"loc": [6, 3, 5, 2, 1, 4], "type": "covariance"},
        {"loc": [0, 1, 2], "type": "increasing"},
    ],
]


@pytest.mark.parametrize("constraints", INVALID_CONSTRAINT_COMBIS)
def test_incompatible_constraints_raise_errors(constraints):
    params = np.arange(10)

    with pytest.raises(InvalidConstraintError):
        minimize(
            fun=lambda x: x @ x,
            params=params,
            algorithm="scipy_lbfgsb",
            constraints=constraints,
        )


def test_bug_from_copenhagen_presentation():
    # Make sure maximum of work hours is optimal
    def u(params):
        return params["work"]["hours"] ** 2

    start_params = {
        "work": {"hourly_wage": 25.5, "hours": 2_000},
        "time_budget": 24 * 7 * 365,
    }

    def return_all_but_working_hours(params):
        out = deepcopy(params)
        del out["work"]["hours"]
        return out

    res = maximize(
        fun=u,
        params=start_params,
        algorithm="scipy_lbfgsb",
        constraints=[
            {"selector": return_all_but_working_hours, "type": "fixed"},
            {
                "selector": lambda p: [p["work"]["hours"], p["time_budget"]],
                "type": "increasing",
            },
        ],
        bounds=Bounds(lower={"work": {"hours": 0}}),
    )

    assert np.allclose(res.params["work"]["hours"], start_params["time_budget"])


def test_constraint_inheritance():
    """Test that probability constraint applies both sets of parameters in a pairwise
    equality constraint, no matter to which set they were applied originally."""
    for loc in [[0, 1], [2, 3]]:
        res = minimize(
            fun=lambda x: x @ x,
            params=np.array([0.1, 0.9, 0.9, 0.1]),
            algorithm="scipy_lbfgsb",
            constraints=[
                {"locs": [[0, 1], [3, 2]], "type": "pairwise_equality"},
                {"loc": loc, "type": "probability"},
            ],
        )
        aaae(res.params, [0.5] * 4)


def test_invalid_start_params():
    def criterion(x):
        return np.dot(x, x)

    x = np.arange(3)

    with pytest.raises(InvalidParamsError):
        minimize(
            criterion,
            params=x,
            algorithm="scipy_lbfgsb",
            constraints=[{"loc": [1, 2], "type": "probability"}],
        )


def test_covariance_constraint_in_2_by_2_case():
    spector_data = sm.datasets.spector.load_pandas()
    spector_data.exog = sm.add_constant(spector_data.exog)
    x_df = sm.add_constant(spector_data.exog)

    start_params = np.array([-10, 2, 0.2, 2])
    kwargs = {"y": spector_data.endog, "x": x_df.to_numpy()}

    result = maximize(
        fun=logit_loglike,
        fun_kwargs=kwargs,
        params=start_params,
        algorithm="scipy_lbfgsb",
        constraints={"loc": [1, 2, 3], "type": "covariance"},
    )

    expected = np.array([-13.0213351, 2.82611417, 0.09515704, 2.37867869])
    aaae(result.params, expected, decimal=4)
