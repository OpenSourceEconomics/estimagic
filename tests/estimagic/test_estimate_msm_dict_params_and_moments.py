"""Most test exploit the special case where simulate_moments just returns parameters."""

import numpy as np
import pandas as pd
from estimagic.estimate_msm import estimate_msm
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.parameters.tree_registry import get_registry
from pybaum import tree_just_flatten


def test_estimate_msm_dict_params_and_moments():
    def simulate_moments(params):
        return {k * 2: v for k, v in params.items()}

    start_params = {"a": 3, "b": 2, "c": 1}

    expected_params = {"a": 0, "b": 0, "c": 0}

    empirical_moments = {"aa": 0, "bb": 0, "cc": 0}

    moments_cov = {
        "aa": {"aa": 1, "bb": 0, "cc": 0},
        "bb": {"aa": 0, "bb": 2, "cc": 0},
        "cc": {"aa": 0, "bb": 0, "cc": 3},
    }

    calculated = estimate_msm(
        simulate_moments=simulate_moments,
        empirical_moments=empirical_moments,
        moments_cov=moments_cov,
        params=start_params,
        optimize_options="scipy_lbfgsb",
    )

    # check that minimization works
    assert_almost_equal(calculated.params, expected_params)

    # this works only in the very special case with diagonal moments cov and
    # jac = identity matrix
    assert_almost_equal(calculated.cov(), moments_cov)

    assert_almost_equal(calculated.se(), {"a": 1, "b": np.sqrt(2), "c": np.sqrt(3)})

    # works only because parameter point estimates are exactly zero
    assert_almost_equal(calculated.p_values(), {"a": 1, "b": 1, "c": 1})

    expected_ci_upper = {"a": 1.95996398, "b": 2.77180765, "c": 3.3947572}
    expected_ci_lower = {k: -v for k, v in expected_ci_upper.items()}

    lower, upper = calculated.ci()
    assert_almost_equal(lower, expected_ci_lower)
    assert_almost_equal(upper, expected_ci_upper)

    assert_almost_equal(calculated.ci(), calculated._ci)
    assert_almost_equal(calculated.p_values(), calculated._p_values)
    assert_almost_equal(calculated.se(), calculated._se)
    assert_almost_equal(calculated.cov(), calculated._cov)

    summary = calculated.summary()
    summary_df = pd.concat(list(summary.values()))
    aaae(summary_df["value"], np.zeros(3))
    aaae(summary_df["p_value"], np.ones(3))
    assert summary_df["stars"].tolist() == [""] * 3

    expected_sensitivity_to_bias_dict = {
        "a": {"aa": -1.0, "bb": 0.0, "cc": 0.0},
        "b": {"aa": 0.0, "bb": -1.0, "cc": 0.0},
        "c": {"aa": 0.0, "bb": 0.0, "cc": -1.0},
    }

    assert_almost_equal(
        calculated.sensitivity("bias"), expected_sensitivity_to_bias_dict
    )

    expected_sensitivity_to_bias_arr = -np.eye(3)

    aaae(
        calculated.sensitivity("bias", return_type="array"),
        expected_sensitivity_to_bias_arr,
    )
    aaae(
        calculated.sensitivity("bias", return_type="dataframe").to_numpy(),
        expected_sensitivity_to_bias_arr,
    )

    expected_jacobian = {
        "a": {"aa": 1.0, "bb": 0.0, "cc": 0.0},
        "b": {"aa": 0.0, "bb": 1.0, "cc": 0.0},
        "c": {"aa": 0.0, "bb": 0.0, "cc": 1.0},
    }

    assert_almost_equal(calculated.jacobian, expected_jacobian)


def assert_almost_equal(x, y, decimal=6):
    if isinstance(x, np.ndarray):
        x_flat = x
        y_flat = y
    else:
        registry = get_registry(extended=True)
        x_flat = np.array(tree_just_flatten(x, registry=registry))
        y_flat = np.array(tree_just_flatten(x, registry=registry))

    aaae(x_flat, y_flat, decimal=decimal)
