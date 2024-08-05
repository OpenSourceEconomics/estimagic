"""Compare first and second derivative behavior to that of jax.

This test module only runs if jax is installed.

"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.config import IS_JAX_INSTALLED
from optimagic.differentiation.derivatives import first_derivative, second_derivative
from pybaum import tree_equal

if not IS_JAX_INSTALLED:
    pytestmark = pytest.mark.skip(reason="jax is not installed.")
else:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)


# arrays have to be equal up to 5 decimals
DECIMALS = 5


def _tree_equal_numpy_leaves(tree1, tree2):
    equality_checkers = {np.ndarray: lambda x, y: aaae(x, y, decimal=DECIMALS)}
    tree_equal(tree1, tree2, equality_checkers=equality_checkers)


def _compute_testable_optimagic_and_jax_derivatives(func, params, func_jax=None):
    """Computes first and second derivative using optimagic and jax.

    Then converts leaves of jax output to numpy so that we can use numpy.testing. For
    higher dimensional output we need to define two function, one with numpy array
    output and one with jax.numpy array output.

    """
    func_jax = func if func_jax is None else func_jax

    optimagic_jac = first_derivative(func, params)["derivative"]
    jax_jac = jax.jacobian(func_jax)(params)

    optimagic_hess = second_derivative(func, params)["derivative"]
    jax_hess = jax.hessian(func_jax)(params)

    out = {
        "jac": {"optimagic": optimagic_jac, "jax": jax_jac},
        "hess": {"optimagic": optimagic_hess, "jax": jax_hess},
    }
    return out


@pytest.mark.jax()
def test_scalar_input_scalar_output():
    def func(params):
        return params**2

    params = 1.0

    result = _compute_testable_optimagic_and_jax_derivatives(func, params)
    _tree_equal_numpy_leaves(result["jac"]["optimagic"], result["jac"]["jax"])
    _tree_equal_numpy_leaves(result["hess"]["optimagic"], result["hess"]["jax"])


@pytest.mark.jax()
def test_array_input_scalar_output():
    def func(params):
        return params @ params

    params = np.array([1.0, 2, 3])

    result = _compute_testable_optimagic_and_jax_derivatives(func, params)
    _tree_equal_numpy_leaves(result["jac"]["optimagic"], result["jac"]["jax"])
    _tree_equal_numpy_leaves(result["hess"]["optimagic"], result["hess"]["jax"])


@pytest.mark.jax()
def test_dict_input_scalar_output():
    def func(params):
        return params["a"] * params["b"]

    params = {"a": 1.0, "b": 2.0}

    result = _compute_testable_optimagic_and_jax_derivatives(func, params)
    _tree_equal_numpy_leaves(result["jac"]["optimagic"], result["jac"]["jax"])
    _tree_equal_numpy_leaves(result["hess"]["optimagic"], result["hess"]["jax"])


@pytest.mark.jax()
def test_array_dict_input_scalar_output():
    def func(params):
        return params["a"].sum() * params["b"].prod()

    params = {
        "a": np.array([1.0, 2, 3]),
        "b": np.arange(9, dtype=np.float64).reshape(3, 3),
    }

    result = _compute_testable_optimagic_and_jax_derivatives(func, params)
    _tree_equal_numpy_leaves(result["jac"]["optimagic"], result["jac"]["jax"])
    _tree_equal_numpy_leaves(result["hess"]["optimagic"], result["hess"]["jax"])


@pytest.mark.jax()
def test_array_input_array_output():
    def func(params):
        return np.array([params.sum(), params.prod()])

    def func_jax(params):
        return jnp.array([params.sum(), params.prod()])

    params = np.array([1.0, 2, 3])

    result = _compute_testable_optimagic_and_jax_derivatives(func, params, func_jax)
    _tree_equal_numpy_leaves(result["jac"]["optimagic"], result["jac"]["jax"])
    _tree_equal_numpy_leaves(result["hess"]["optimagic"], result["hess"]["jax"])


@pytest.mark.jax()
def test_array_dict_input_array_output():
    def func(params):
        return params["b"] * np.array([params["a"].sum(), params["a"].prod()])

    def func_jax(params):
        return params["b"] * jnp.array([params["a"].sum(), params["a"].prod()])

    params = {"a": np.array([1.0, 2, 3]), "b": 2.0}

    result = _compute_testable_optimagic_and_jax_derivatives(func, params, func_jax)
    _tree_equal_numpy_leaves(result["jac"]["optimagic"], result["jac"]["jax"])
    _tree_equal_numpy_leaves(result["hess"]["optimagic"], result["hess"]["jax"])


@pytest.mark.jax()
def test_array_dict_input_dict_output():
    def func(params):
        value = params["b"] * np.array([params["a"].sum(), params["a"].prod()])
        return [value[0], {"c": 0.0, "d": value[1]}]

    def func_jax(params):
        value = params["b"] * jnp.array([params["a"].sum(), params["a"].prod()])
        return [value[0], {"c": 0.0, "d": value[1]}]

    params = {"a": np.array([1.0, 2, 3]), "b": 2.0}

    result = _compute_testable_optimagic_and_jax_derivatives(func, params, func_jax)
    _tree_equal_numpy_leaves(result["jac"]["optimagic"], result["jac"]["jax"])
    _tree_equal_numpy_leaves(result["hess"]["optimagic"], result["hess"]["jax"])
