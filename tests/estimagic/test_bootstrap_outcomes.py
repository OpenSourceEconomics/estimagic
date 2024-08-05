import functools

import numpy as np
import pandas as pd
import pytest
from estimagic.bootstrap_outcomes import (
    _get_bootstrap_outcomes_from_indices,
    get_bootstrap_outcomes,
)
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.batch_evaluators import joblib_batch_evaluator
from optimagic.utilities import get_rng


@pytest.fixture()
def data():
    df = pd.DataFrame([[1, 10], [2, 7], [3, 6], [4, 5]], columns=["x1", "x2"])
    return df


def _mean_return_series(data):
    out = np.mean(data, axis=0)
    return out


def _mean_return_dict(data):
    out = np.mean(data, axis=0)
    return out.to_dict()


def _mean_return_array(data):
    out = np.mean(data, axis=0).to_numpy()
    return out


@pytest.mark.parametrize(
    "outcome",
    [
        (functools.partial(np.mean, axis=0)),
        (_mean_return_series),
        (_mean_return_dict),
        (_mean_return_array),
    ],
)
def test_get_bootstrap_estimates_runs(outcome, data):
    rng = get_rng(seed=1234)
    get_bootstrap_outcomes(
        data=data,
        outcome=outcome,
        rng=rng,
        n_draws=5,
    )


def test_bootstrap_estimates_from_indices_without_errors(data):
    calculated = _get_bootstrap_outcomes_from_indices(
        indices=[np.array([1, 3]), np.array([0, 2])],
        data=data,
        outcome=functools.partial(np.mean, axis=0),
        n_cores=1,
        error_handling="raise",
        batch_evaluator=joblib_batch_evaluator,
    )

    expected = [[3.0, 6.0], [2, 8]]
    aaae(calculated, expected)


def test_get_bootstrap_estimates_with_error_and_raise(data):
    rng = get_rng(seed=1234)

    def _raise_assertion_error(data):  # noqa: ARG001
        raise AssertionError()

    with pytest.raises(AssertionError):
        get_bootstrap_outcomes(
            data=data,
            outcome=_raise_assertion_error,
            rng=rng,
            n_draws=2,
            error_handling="raise",
        )


def test_get_bootstrap_estimates_with_all_errors_and_continue(data):
    rng = get_rng(seed=1234)

    def _raise_assertion_error(data):  # noqa: ARG001
        raise AssertionError()

    with pytest.warns(UserWarning):
        with pytest.raises(RuntimeError):
            get_bootstrap_outcomes(
                data=data,
                outcome=_raise_assertion_error,
                rng=rng,
                n_draws=2,
                error_handling="continue",
            )


def test_get_bootstrap_estimates_with_some_errors_and_continue(data):
    rng = get_rng(seed=1234)

    def _raise_assertion_error_sometimes(data):
        assert rng.uniform() > 0.5
        return data.mean()

    with pytest.warns(UserWarning):
        res_flat = get_bootstrap_outcomes(
            data=data,
            outcome=_raise_assertion_error_sometimes,
            rng=rng,
            n_draws=100,
            error_handling="continue",
        )

    assert 30 <= len(res_flat) <= 70
