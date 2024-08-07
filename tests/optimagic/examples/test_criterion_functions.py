import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from numpy.testing import assert_array_equal
from optimagic.examples.criterion_functions import (
    rhe_fun_and_gradient,
    rhe_function_value,
    rhe_gradient,
    rhe_scalar,
    rosenbrock_fun_and_gradient,
    rosenbrock_function_value,
    rosenbrock_gradient,
    rosenbrock_scalar,
    sos_fun_and_gradient,
    sos_gradient,
    sos_likelihood_fun_and_jac,
    sos_likelihood_jacobian,
    sos_ls,
    sos_ls_fun_and_jac,
    sos_ls_jacobian,
    sos_ls_with_pd_objects,
    sos_pandas_gradient,
    sos_pandas_likelihood_jacobian,
    sos_pandas_ls_jacobian,
    sos_scalar,
    trid_fun_and_gradient,
    trid_gradient,
    trid_scalar,
)
from optimagic.optimization.fun_value import FunctionValue


@pytest.fixture()
def input_params():
    params = pd.DataFrame({"value": [9, 9, 6, 7, -5]})
    return params


def test_trid_scalar(input_params):
    got = trid_scalar(input_params)
    assert got == 83


def test_trid_gradient(input_params):
    got = trid_gradient(input_params)
    assert_array_equal(got, np.array([7, 1, -6, 11, -19]))


def test_trid_fun_and_gradient(input_params):
    got = trid_fun_and_gradient(input_params)
    assert_array_equal(got[1], np.array([7, 1, -6, 11, -19]))


def test_rhe_scalar(input_params):
    got = rhe_scalar(input_params)
    assert got == 960


def test_rhe_gradient(input_params):
    got = rhe_gradient(input_params)
    assert_array_equal(got, np.array([90, 72, 36, 28, -10]))


def test_rhe_fun_and_gradient(input_params):
    got = rhe_fun_and_gradient(input_params)
    assert_array_equal(got[1], np.array([90, 72, 36, 28, -10]))


def test_rosenbrock_scalar(input_params):
    got = rosenbrock_scalar(input_params)
    assert got == 1456789


def test_rosenbrock_gradient(input_params):
    got = rosenbrock_gradient(input_params)
    assert_array_equal(got, np.array([259216, 255616, 54610, 145412, -10800]))


def test_rosenbrock_fun_and_gradient(input_params):
    got = rosenbrock_fun_and_gradient(input_params)
    assert_array_equal(got[1], np.array([259216, 255616, 54610, 145412, -10800]))


def test_rhe_function_value(input_params):
    got = rhe_function_value(input_params)
    assert isinstance(got, FunctionValue)
    expected = np.array([9, 12.72792206, 14.07124728, 15.71623365, 16.4924225])
    aaae(got.value, expected)


def test_rosenbrock_function_value(input_params):
    got = rosenbrock_function_value(input_params)
    assert isinstance(got, FunctionValue)
    expected = np.array([720.04444307, 750.04266545, 290.04310025, 540.0333323, 0])
    aaae(got.value, expected)


def test_sos_ls():
    got = sos_ls({"a": 1, "b": 2})
    aaae(got, np.array([1, 2.0]))


def test_sos_ls_with_pd_objects():
    got = sos_ls_with_pd_objects({"a": 1, "b": 2})
    assert isinstance(got, pd.Series)
    aaae(got.to_numpy(), np.array([1, 2.0]))


def test_sos_scalar():
    got = sos_scalar({"a": 1, "b": 2})
    assert got == 5


def test_sos_gradient():
    got = sos_gradient({"a": 1, "b": 2})
    assert_array_equal(got, np.array([2, 4]))


def test_sos_likelihood_jacobian():
    got = sos_likelihood_jacobian({"a": 1, "b": 2})
    assert_array_equal(got, np.array([[2, 0], [0, 4]]))


def test_sos_ls_jacobian():
    got = sos_ls_jacobian({"a": 1, "b": 2})
    assert_array_equal(got, np.eye(2))


def test_sos_pandas_gradient():
    got = sos_pandas_gradient({"a": 1, "b": 2})
    assert isinstance(got, pd.Series)
    aaae(got.to_numpy(), np.array([2, 4]))


def test_sos_pandas_likelihood_jacobian():
    got = sos_pandas_likelihood_jacobian({"a": 1, "b": 2})
    assert isinstance(got, pd.DataFrame)
    aaae(got.to_numpy(), np.array([[2, 0], [0, 4]]))


def test_sos_pandas_ls_jacobian():
    got = sos_pandas_ls_jacobian({"a": 1, "b": 2})
    assert isinstance(got, pd.DataFrame)
    aaae(got.to_numpy(), np.eye(2))


def test_sos_fun_and_gradient():
    got_val, got_grad = sos_fun_and_gradient({"a": 1, "b": 2})
    assert got_val == 5
    assert_array_equal(got_grad, np.array([2, 4]))


def test_sos_likelihood_fun_and_jac():
    got_val, got_jac = sos_likelihood_fun_and_jac({"a": 1, "b": 2})
    aaae(got_val, np.array([1, 4]))
    assert_array_equal(got_jac, np.array([[2, 0], [0, 4]]))


def test_sos_ls_fun_and_jac():
    got_val, got_jac = sos_ls_fun_and_jac({"a": 1, "b": 2})
    aaae(got_val, np.array([1, 2]))
    assert_array_equal(got_jac, np.eye(2))
