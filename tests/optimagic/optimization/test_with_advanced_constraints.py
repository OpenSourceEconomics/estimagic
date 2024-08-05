"""Tests using constraints with optional entries or combination of constraints.

- Only sum of squares
- Only scipy_lbfgsb
- Only minimize

"""

import itertools

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.examples.criterion_functions import sos_gradient, sos_scalar_criterion
from optimagic.optimization.optimize import minimize

CONSTR_INFO = {
    "cov_bounds_distance": [
        {
            "loc": [0, 1, 2, 3, 4, 5],
            "type": "covariance",
            "bounds_distance": 0.1,
            "robust_bounds": True,
        }
    ],
    "sdcorr_bounds_distance": [
        {
            "loc": [0, 1, 2, 3, 4, 5],
            "type": "sdcorr",
            "bounds_distance": 0.1,
            "robust_bounds": True,
        }
    ],
    "fixed_and_decreasing": [
        {"loc": [1, 2, 3, 4], "type": "decreasing"},
        {"loc": 2, "type": "fixed", "value": 4},
    ],
    "fixed_and_increasing": [
        {"loc": [0, 1, 2, 3], "type": "increasing"},
        {"loc": 2, "type": "fixed", "value": 3},
    ],
}


START_INFO = {
    "cov_bounds_distance": [1, 0.1, 2, 0.2, 0.3, 3],
    "sdcorr_bounds_distance": [1, 2, 3, 0.1, 0.2, 0.3],
    "fixed_and_decreasing": [1, 4, 4, 2, 1],
    "fixed_and_increasing": [1, 2, 3, 4, 1],
}

RES_INFO = {
    "cov_bounds_distance": [0.1, 0, 0.1, 0, 0, 0.1],
    "sdcorr_bounds_distance": [0.1, 0.1, 0.1, 0, 0, 0.0],
    "fixed_and_decreasing": [0, 4, 4, 0, 0],
    "fixed_and_increasing": [0, 0, 3, 3, 0],
}


derivatives = [sos_gradient, None]
constr_names = list(CONSTR_INFO.keys())


test_cases = list(itertools.product(derivatives, constr_names))


@pytest.mark.parametrize("derivative, constr_name", test_cases)
def test_with_covariance_constraint_bounds_distance(derivative, constr_name):
    params = pd.Series(START_INFO[constr_name], name="value").to_frame()

    res = minimize(
        fun=sos_scalar_criterion,
        params=params,
        algorithm="scipy_lbfgsb",
        jac=derivative,
        constraints=CONSTR_INFO[constr_name],
    )

    assert res.success, "scipy_lbfgsb did not converge."

    expected = np.array(RES_INFO[constr_name])
    aaae(res.params["value"].to_numpy(), expected, decimal=4)
