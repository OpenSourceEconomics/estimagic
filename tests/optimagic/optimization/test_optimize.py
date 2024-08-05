"""Tests for (almost) algorithm independent properties of maximize and minimize."""

import numpy as np
import pandas as pd
import pytest
from optimagic.examples.criterion_functions import sos_scalar_criterion
from optimagic.exceptions import InvalidFunctionError, InvalidKwargsError
from optimagic.optimization.optimize import maximize, minimize


def test_sign_is_switched_back_after_maximization():
    params = pd.DataFrame()
    params["value"] = [1, 2, 3]
    res = maximize(
        lambda params: 1 - params["value"] @ params["value"],
        params=params,
        algorithm="scipy_lbfgsb",
    )

    assert np.allclose(res.fun, 1)


def test_scipy_lbfgsb_actually_calls_criterion_and_derivative():
    params = pd.DataFrame(data=np.ones((10, 1)), columns=["value"])

    def raising_crit_and_deriv(params):  # noqa: ARG001
        raise NotImplementedError("This should not be called.")

    with pytest.raises(InvalidFunctionError, match="Error while evaluating"):
        minimize(
            fun=sos_scalar_criterion,
            params=params,
            algorithm="scipy_lbfgsb",
            fun_and_jac=raising_crit_and_deriv,
        )


def test_with_invalid_numdiff_options():
    with pytest.raises(InvalidKwargsError):
        minimize(
            fun=lambda x: x @ x,
            params=np.arange(5),
            algorithm="scipy_lbfgsb",
            numdiff_options={"bla": 15},
        )
