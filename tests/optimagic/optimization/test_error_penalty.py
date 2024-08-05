import functools

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.differentiation.derivatives import first_derivative
from optimagic.optimization.error_penalty import (
    _penalty_contributions,
    _penalty_contributions_derivative,
    _penalty_root_contributions,
    _penalty_root_contributions_derivative,
    _penalty_value,
    _penalty_value_derivative,
    get_error_penalty_function,
)
from optimagic.utilities import get_rng


@pytest.mark.parametrize("seed", range(10))
def test_penalty_aggregations(seed):
    rng = get_rng(seed)
    x = rng.uniform(size=5)
    x0 = rng.uniform(size=5)
    slope = 0.3
    constant = 3
    dim_out = 10

    scalar = _penalty_value(x, constant, slope, x0)
    contribs = _penalty_contributions(x, constant, slope, x0, dim_out)
    root_contribs = _penalty_root_contributions(x, constant, slope, x0, dim_out)

    assert np.isclose(scalar, contribs.sum())
    assert np.isclose(scalar, (root_contribs**2).sum())


pairs = [
    (_penalty_value, _penalty_value_derivative),
    (_penalty_contributions, _penalty_contributions_derivative),
    (_penalty_root_contributions, _penalty_root_contributions_derivative),
]


@pytest.mark.parametrize("func, deriv", pairs)
def test_penalty_derivatives(func, deriv):
    rng = get_rng(seed=5)
    x = rng.uniform(size=5)
    x0 = rng.uniform(size=5)
    slope = 0.3
    constant = 3
    dim_out = 8

    calculated = deriv(x, constant, slope, x0, dim_out)

    partialed = functools.partial(
        func, constant=constant, slope=slope, x0=x0, dim_out=dim_out
    )
    expected = first_derivative(partialed, x)

    aaae(calculated, expected["derivative"])


@pytest.mark.parametrize("seed", range(10))
def test_penalty_aggregations_via_get_error_penalty(seed):
    rng = get_rng(seed)
    x = rng.uniform(size=5)
    x0 = rng.uniform(size=5)
    slope = 0.3
    constant = 3

    scalar_func = get_error_penalty_function(
        error_handling="continue",
        start_x=x0,
        start_criterion=3,
        error_penalty={"slope": slope, "constant": constant},
        primary_key="value",
        direction="minimize",
    )

    contribs_func = get_error_penalty_function(
        error_handling="continue",
        start_x=x0,
        start_criterion=np.ones(10),
        error_penalty={"slope": slope, "constant": constant},
        primary_key="contributions",
        direction="minimize",
    )

    root_contribs_func = get_error_penalty_function(
        error_handling="continue",
        start_x=x0,
        start_criterion=np.ones(10),
        error_penalty={"slope": slope, "constant": constant},
        primary_key="root_contributions",
        direction="minimize",
    )

    scalar = scalar_func(x, task="criterion")
    contribs = contribs_func(x, task="criterion")
    root_contribs = root_contribs_func(x, task="criterion")

    assert np.isclose(scalar, contribs.sum())
    assert np.isclose(scalar, (root_contribs**2).sum())
