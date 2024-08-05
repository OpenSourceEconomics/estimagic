from itertools import product

import numpy as np
import optimagic as om
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.decorators import switch_sign
from optimagic.examples.criterion_functions import (
    sos_dict_criterion,
    sos_scalar_criterion,
)
from optimagic.logging.load_database import load_database
from optimagic.logging.read_from_database import read_new_rows
from optimagic.logging.read_log import read_steps_table
from optimagic.optimization.optimize import maximize, minimize
from optimagic.optimization.optimize_result import OptimizeResult
from optimagic.parameters.bounds import Bounds

criteria = [sos_scalar_criterion, sos_dict_criterion]


@pytest.fixture()
def params():
    params = pd.DataFrame()
    params["value"] = np.arange(4)
    params["soft_lower_bound"] = [-5] * 4
    params["soft_upper_bound"] = [10] * 4
    return params


test_cases = product(criteria, ["maximize", "minimize"])


@pytest.mark.parametrize("criterion, direction", test_cases)
def test_multistart_minimize_with_sum_of_squares_at_defaults(
    criterion, direction, params
):
    if direction == "minimize":
        res = minimize(
            fun=criterion,
            params=params,
            algorithm="scipy_lbfgsb",
            multistart=True,
        )
    else:
        res = maximize(
            fun=switch_sign(sos_dict_criterion),
            params=params,
            algorithm="scipy_lbfgsb",
            multistart=True,
        )

    assert hasattr(res, "multistart_info")
    ms_info = res.multistart_info
    assert len(ms_info.exploration_sample) == 400
    assert len(ms_info.exploration_results) == 400
    assert all(isinstance(entry, float) for entry in ms_info.exploration_results)
    assert all(isinstance(entry, OptimizeResult) for entry in ms_info.local_optima)
    assert all(isinstance(entry, pd.DataFrame) for entry in ms_info.start_parameters)
    assert np.allclose(res.fun, 0)
    aaae(res.params["value"], np.zeros(4))


def test_multistart_with_existing_sample(params):
    sample = [params.assign(value=x) for x in np.arange(20).reshape(5, 4) / 10]
    options = om.MultistartOptions(sample=sample)

    res = minimize(
        fun=sos_dict_criterion,
        params=params,
        algorithm="scipy_lbfgsb",
        multistart=options,
    )

    assert all(
        got.equals(expected)
        for expected, got in zip(
            sample, res.multistart_info.exploration_sample, strict=False
        )
    )


def test_convergence_via_max_discoveries_works(params):
    options = om.MultistartOptions(
        convergence_xtol_rel=np.inf,
        convergence_max_discoveries=2,
    )

    res = maximize(
        fun=switch_sign(sos_dict_criterion),
        params=params,
        algorithm="scipy_lbfgsb",
        multistart=options,
    )

    assert len(res.multistart_info.local_optima) == 2


def test_steps_are_logged_as_skipped_if_convergence(params):
    options = om.MultistartOptions(
        n_samples=10 * len(params),
        convergence_xtol_rel=np.inf,
        convergence_max_discoveries=2,
    )

    minimize(
        fun=sos_dict_criterion,
        params=params,
        algorithm="scipy_lbfgsb",
        multistart=options,
        logging="logging.db",
    )

    steps_table = read_steps_table("logging.db")
    expected_status = ["complete", "complete", "complete", "skipped", "skipped"]
    assert steps_table["status"].tolist() == expected_status


def test_all_steps_occur_in_optimization_iterations_if_no_convergence(params):
    options = om.MultistartOptions(
        convergence_max_discoveries=np.inf,
        n_samples=10 * len(params),
    )

    minimize(
        fun=sos_dict_criterion,
        params=params,
        algorithm="scipy_lbfgsb",
        multistart=options,
        logging="logging.db",
    )

    database = load_database(path_or_database="logging.db")
    iterations, _ = read_new_rows(
        database=database,
        table_name="optimization_iterations",
        last_retrieved=0,
        return_type="dict_of_lists",
    )

    present_steps = set(iterations["step"])

    assert present_steps == {1, 2, 3, 4, 5}


def test_with_non_transforming_constraints(params):
    res = minimize(
        fun=sos_dict_criterion,
        params=params,
        constraints=[{"loc": [0, 1], "type": "fixed", "value": [0, 1]}],
        algorithm="scipy_lbfgsb",
        multistart=True,
    )

    aaae(res.params["value"].to_numpy(), np.array([0, 1, 0, 0]))


def test_error_is_raised_with_transforming_constraints(params):
    with pytest.raises(NotImplementedError):
        minimize(
            fun=sos_dict_criterion,
            params=params,
            constraints=[{"loc": [0, 1], "type": "probability"}],
            algorithm="scipy_lbfgsb",
            multistart=True,
        )


def test_multistart_with_numpy_params():
    res = minimize(
        fun=lambda params: params @ params,
        params=np.arange(5),
        algorithm="scipy_lbfgsb",
        bounds=Bounds(soft_lower=np.full(5, -10), soft_upper=np.full(5, 10)),
        multistart=True,
    )

    aaae(res.params, np.zeros(5))


def test_multistart_with_rng_seed():
    rng = np.random.default_rng(12345)

    res = minimize(
        fun=lambda params: params @ params,
        params=np.arange(5),
        algorithm="scipy_lbfgsb",
        bounds=Bounds(soft_lower=np.full(5, -10), soft_upper=np.full(5, 10)),
        multistart=om.MultistartOptions(seed=rng),
    )

    aaae(res.params, np.zeros(5))


def test_with_invalid_bounds():
    with pytest.raises(ValueError):
        minimize(
            fun=lambda x: x @ x,
            params=np.arange(5),
            algorithm="scipy_neldermead",
            multistart=True,
        )


def test_with_scaling():
    def _crit(params):
        x = params - np.arange(len(params))
        return x @ x

    res = minimize(
        fun=_crit,
        params=np.full(5, 10),
        bounds=Bounds(soft_lower=np.full(5, -1), soft_upper=np.full(5, 11)),
        algorithm="scipy_lbfgsb",
        multistart=True,
    )

    aaae(res.params, np.arange(5))


def test_with_ackley():
    def ackley(x):
        out = (
            -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
            - np.exp(np.mean(np.cos(2 * np.pi * x)))
            + 20
            + np.exp(1)
        )
        return out

    dim = 5

    kwargs = {
        "fun": ackley,
        "params": np.full(dim, -10),
        "bounds": Bounds(lower=np.full(dim, -32), upper=np.full(dim, 32)),
        "algo_options": {"stopping.maxfun": 1000},
    }

    minimize(
        **kwargs,
        algorithm="scipy_lbfgsb",
        multistart=om.MultistartOptions(
            n_samples=200,
            stopping_maxopt=20,
            convergence_max_discoveries=10,
        ),
    )


def test_multistart_with_least_squares_optimizers():
    est = minimize(
        fun=sos_dict_criterion,
        params=np.array([-1, 1.0]),
        bounds=Bounds(soft_lower=np.full(2, -10), soft_upper=np.full(2, 10)),
        algorithm="scipy_ls_trf",
        multistart=om.MultistartOptions(n_samples=3, stopping_maxopt=3),
    )

    aaae(est.params, np.zeros(2))


def test_with_ackley_using_dict_options():
    def ackley(x):
        out = (
            -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
            - np.exp(np.mean(np.cos(2 * np.pi * x)))
            + 20
            + np.exp(1)
        )
        return out

    dim = 5

    kwargs = {
        "fun": ackley,
        "params": np.full(dim, -10),
        "bounds": Bounds(lower=np.full(dim, -32), upper=np.full(dim, 32)),
        "algo_options": {"stopping.maxfun": 1000},
    }

    minimize(
        **kwargs,
        algorithm="scipy_lbfgsb",
        multistart={
            "n_samples": 200,
            "stopping_maxopt": 20,
            "convergence_max_discoveries": 10,
        },
    )
