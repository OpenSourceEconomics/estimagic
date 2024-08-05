import numpy as np
import optimagic as om
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.exceptions import AliasError


def test_x0_works_in_minimize():
    res = om.minimize(
        fun=lambda x: x @ x,
        x0=np.arange(3),
        algorithm="scipy_lbfgsb",
    )
    aaae(res.params, np.zeros(3))


def test_x0_works_in_maximize():
    res = om.maximize(
        fun=lambda x: -x @ x,
        x0=np.arange(3),
        algorithm="scipy_lbfgsb",
    )
    aaae(res.params, np.zeros(3))


def test_x0_and_params_do_not_work_together_in_minimize():
    with pytest.raises(AliasError, match="x0 is an alias"):
        om.minimize(
            fun=lambda x: x @ x,
            x0=np.arange(3),
            params=np.arange(3),
            algorithm="scipy_lbfgsb",
        )


def test_x0_and_params_do_not_work_together_in_maximize():
    with pytest.raises(AliasError, match="x0 is an alias"):
        om.maximize(
            fun=lambda x: -x @ x,
            x0=np.arange(3),
            params=np.arange(3),
            algorithm="scipy_lbfgsb",
        )


METHODS = [
    "Nelder-Mead",
    "Powell",
    "CG",
    "BFGS",
    "Newton-CG",
    "L-BFGS-B",
    "TNC",
    "COBYLA",
    "SLSQP",
    "trust-constr",
]


@pytest.mark.parametrize("method", METHODS)
def test_method_works_in_minimize(method):
    res = om.minimize(
        fun=lambda x: x @ x,
        x0=np.arange(3),
        method="L-BFGS-B",
    )
    aaae(res.params, np.zeros(3))


@pytest.mark.parametrize("method", METHODS)
def test_method_works_in_maximize(method):
    res = om.maximize(
        fun=lambda x: -x @ x,
        x0=np.arange(3),
        method="L-BFGS-B",
    )
    aaae(res.params, np.zeros(3))


def test_method_and_algorithm_do_not_work_together_in_minimize():
    with pytest.raises(AliasError, match="method is an alias"):
        om.minimize(
            fun=lambda x: x @ x,
            x0=np.arange(3),
            algorithm="scipy_lbfgsb",
            method="L-BFGS-B",
        )


def test_method_and_algorithm_do_not_work_together_in_maximize():
    with pytest.raises(AliasError, match="method is an alias"):
        om.maximize(
            fun=lambda x: -x @ x,
            x0=np.arange(3),
            algorithm="scipy_lbfgsb",
            method="L-BFGS-B",
        )


def test_exception_for_hess():
    msg = "The hess argument is not yet supported"
    with pytest.raises(NotImplementedError, match=msg):
        om.minimize(
            fun=lambda x: x @ x,
            x0=np.arange(3),
            algorithm="scipy_lbfgsb",
            hess=lambda x: np.eye(len(x)),
        )


def test_exception_for_hessp():
    msg = "The hessp argument is not yet supported"
    with pytest.raises(NotImplementedError, match=msg):
        om.minimize(
            fun=lambda x: x @ x,
            x0=np.arange(3),
            algorithm="scipy_lbfgsb",
            hessp=lambda x, p: np.eye(len(x)) @ p,
        )


def test_exception_for_callback():
    msg = "The callback argument is not yet supported"
    with pytest.raises(NotImplementedError, match=msg):
        om.minimize(
            fun=lambda x: x @ x,
            x0=np.arange(3),
            algorithm="scipy_lbfgsb",
            callback=lambda x: print(x),
        )


def test_exception_for_options():
    msg = "The options argument is not supported"
    with pytest.raises(NotImplementedError, match=msg):
        om.minimize(
            fun=lambda x: x @ x,
            x0=np.arange(3),
            algorithm="scipy_lbfgsb",
            options={"maxiter": 100},
        )


def test_exception_for_tol():
    msg = "The tol argument is not supported"
    with pytest.raises(NotImplementedError, match=msg):
        om.minimize(
            fun=lambda x: x @ x,
            x0=np.arange(3),
            algorithm="scipy_lbfgsb",
            tol=1e-6,
        )


def test_args_works_in_minimize():
    res = om.minimize(
        fun=lambda x, a: ((x - a) ** 2).sum(),
        x0=np.arange(3),
        args=(1,),
        algorithm="scipy_lbfgsb",
    )
    aaae(res.params, np.ones(3))


def test_args_works_in_maximize():
    res = om.maximize(
        fun=lambda x, a: -((x - a) ** 2).sum(),
        x0=np.arange(3),
        args=(1,),
        algorithm="scipy_lbfgsb",
    )
    aaae(res.params, np.ones(3))


def test_args_does_not_work_with_together_with_any_kwargs():
    with pytest.raises(AliasError, match="args is an alternative"):
        om.minimize(
            fun=lambda x, a: ((x - a) ** 2).sum(),
            params=np.arange(3),
            algorithm="scipy_lbfgsb",
            args=(1,),
            fun_kwargs={"a": 1},
        )


def test_jac_equal_true_works_in_minimize():
    res = om.minimize(
        fun=lambda x: (x @ x, 2 * x),
        params=np.arange(3),
        algorithm="scipy_lbfgsb",
        jac=True,
    )
    aaae(res.params, np.zeros(3))


def test_jac_equal_true_works_in_maximize():
    res = om.maximize(
        fun=lambda x: (-x @ x, -2 * x),
        params=np.arange(3),
        algorithm="scipy_lbfgsb",
        jac=True,
    )
    aaae(res.params, np.zeros(3))
