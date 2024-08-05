import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.config import IS_JAX_INSTALLED
from optimagic.optimization.optimize import minimize

if IS_JAX_INSTALLED:
    import jax
    import jax.numpy as jnp


@pytest.mark.skipif(not IS_JAX_INSTALLED, reason="Needs jax.")
def test_scipy_conference_example():
    def criterion(x):
        first = (x["a"] - jnp.pi) ** 2
        second = jnp.linalg.norm(x["b"] - jnp.arange(3))
        third = jnp.linalg.norm(x["c"] - jnp.eye(2))
        return first + second + third

    start_params = {
        "a": 1.0,
        "b": jnp.ones(3).astype(float),
        "c": jnp.ones((2, 2)).astype(float),
    }

    gradient = jax.grad(criterion)

    res = minimize(
        fun=criterion,
        jac=gradient,
        params=start_params,
        algorithm="scipy_lbfgsb",
    )

    assert isinstance(res.params["b"], jnp.ndarray)
    aaae(res.params["b"], jnp.arange(3))
    aaae(res.params["c"], jnp.eye(2))
    assert np.allclose(res.params["a"], np.pi, atol=1e-4)


@pytest.mark.skipif(not IS_JAX_INSTALLED, reason="Needs jax.")
def test_params_is_jax_scalar():
    def criterion(x):
        return x**2

    res = minimize(
        fun=criterion,
        params=jnp.array(1.0),
        algorithm="scipy_lbfgsb",
        jac=jax.grad(criterion),
    )

    assert isinstance(res.params, jnp.ndarray)
    assert np.allclose(res.params, 0.0)


@pytest.mark.skipif(not IS_JAX_INSTALLED, reason="Needs jax.")
def params_is_1d_array():
    def criterion(x):
        return x @ x

    res = minimize(
        fun=criterion,
        params=jnp.arange(3),
        algorithm="scipy_lbfgsb",
        jac=jax.grad(criterion),
    )

    assert isinstance(res.params, jnp.ndarray)
    assert aaae(res.params, jnp.arange(3))


@pytest.mark.skipif(not IS_JAX_INSTALLED, reason="Needs jax.")
@pytest.mark.parametrize("algorithm", ["scipy_lbfgsb", "scipy_ls_lm"])
def test_dict_output_works(algorithm):
    def criterion(x):
        return {"root_contributions": x, "value": x @ x}

    def scalar_wrapper(x):
        return criterion(x)["value"]

    def ls_wrapper(x):
        return criterion(x)["root_contributions"]

    deriv_dict = {
        "value": jax.grad(scalar_wrapper),
        "root_contributions": jax.jacobian(ls_wrapper),
    }

    res = minimize(
        fun=criterion,
        params=jnp.array([1.0, 2.0, 3.0]),
        algorithm=algorithm,
        jac=deriv_dict,
    )

    assert isinstance(res.params, jnp.ndarray)
    aaae(res.params, np.zeros(3))


@pytest.mark.skipif(not IS_JAX_INSTALLED, reason="Needs jax.")
def test_least_squares_optimizer_pytree():
    def criterion(x):
        return {"root_contributions": x}

    def ls_wrapper(x):
        return criterion(x)["root_contributions"]

    params = {"a": 1.0, "b": 2.0, "c": jnp.array([1.0, 2.0])}
    jac = jax.jacobian(ls_wrapper)

    res = minimize(
        fun=criterion,
        params=params,
        algorithm="scipy_ls_lm",
        jac=jac,
    )

    assert isinstance(res.params, dict)
    assert np.allclose(res.params["a"], 0)
    assert np.allclose(res.params["b"], 0)
    aaae(res.params["c"], np.zeros(2))
