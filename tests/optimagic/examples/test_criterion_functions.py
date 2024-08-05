import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from optimagic.examples.criterion_functions import (
    rosenbrock_criterion_and_gradient,
    rosenbrock_dict_criterion,
    rosenbrock_gradient,
    rosenbrock_scalar_criterion,
    rotated_hyper_ellipsoid_criterion_and_gradient,
    rotated_hyper_ellipsoid_dict_criterion,
    rotated_hyper_ellipsoid_gradient,
    rotated_hyper_ellipsoid_scalar_criterion,
    trid_criterion_and_gradient,
    trid_dict_criterion,
    trid_gradient,
    trid_scalar_criterion,
)


# Fix input params to test every criterion function
@pytest.fixture()
def input_params():
    params = pd.DataFrame({"value": [9, 9, 6, 7, -5]})
    return params


# Define dicts containing functions to be tested
scalar_criterion = {
    "trid": trid_scalar_criterion,
    "rhe": rotated_hyper_ellipsoid_scalar_criterion,
    "rosenbrock": rosenbrock_scalar_criterion,
}

criterion_gradient = {
    "trid": trid_gradient,
    "rhe": rotated_hyper_ellipsoid_gradient,
    "rosenbrock": rosenbrock_gradient,
}

criterion_and_gradient = {
    "trid": trid_criterion_and_gradient,
    "rhe": rotated_hyper_ellipsoid_criterion_and_gradient,
    "rosenbrock": rosenbrock_criterion_and_gradient,
}

dict_criterion = {
    "trid": trid_dict_criterion,
    "rhe": rotated_hyper_ellipsoid_dict_criterion,
    "rosenbrock": rosenbrock_dict_criterion,
}

# Define dicts containing expected outputs
criterion_output = {"trid": 83, "rhe": 960, "rosenbrock": 1456789}

gradient_output = {
    "trid": np.array([7, 1, -6, 11, -19]),
    "rhe": np.array([90, 72, 36, 28, -10]),
    "rosenbrock": np.array([259216, 255616, 54610, 145412, -10800]),
}

dict_criterion_output = {
    "trid": {"value": 83},
    "rhe": {
        "value": 960,
        "contributions": np.array([81, 162, 198, 247, 272]),
        "root_contributions": np.array(
            [9, 12.72792206, 14.07124728, 15.71623365, 16.4924225]
        ),
    },
    "rosenbrock": {
        "value": 1456789,
        "contributions": np.array([518464, 562564, 84125, 291636, 0]),
        "root_contributions": np.array(
            [720.04444307, 750.04266545, 290.04310025, 540.0333323, 0]
        ),
    },
}

crit_list = ["trid", "rhe", "rosenbrock"]


@pytest.mark.parametrize("crit", crit_list)
def test_scalar_criterion(input_params, crit):
    out = scalar_criterion[crit](input_params)
    assert out == criterion_output[crit]


@pytest.mark.parametrize("crit", crit_list)
def test_criterion_gradient(input_params, crit):
    out = criterion_gradient[crit](input_params)
    assert isinstance(out, np.ndarray)
    assert_array_equal(out, gradient_output[crit])


@pytest.mark.parametrize("crit", crit_list)
def test_criterion_and_gradient(input_params, crit):
    out = criterion_and_gradient[crit](input_params)
    expected = criterion_output[crit], gradient_output[crit]
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0] == expected[0]
    assert isinstance(out[1], np.ndarray)
    assert_array_equal(out[1], expected[1])


@pytest.mark.parametrize("crit", crit_list)
def test_dict_criterion(input_params, crit):
    out_dict = dict_criterion[crit](input_params)
    expected_dict = dict_criterion_output[crit]
    assert isinstance(out_dict, dict)
    assert len(out_dict) == len(expected_dict)
    if crit != "trid":
        assert isinstance(out_dict["contributions"], np.ndarray)
        assert isinstance(out_dict["root_contributions"], np.ndarray)

    for key in expected_dict:
        assert_allclose(out_dict[key], expected_dict[key])
