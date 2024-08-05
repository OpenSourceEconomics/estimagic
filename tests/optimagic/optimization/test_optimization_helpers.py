"""Test optimization helper functions."""

import numpy as np
import pytest
from optimagic.optimizers.pygmo_optimizers import (
    _check_that_every_param_is_bounded,
    _convert_str_to_int,
    _replace_none,
    get_population_size,
)


def test_check_that_every_param_is_bounded():
    _check_that_every_param_is_bounded([1, 2, 3], [4, 5, 6])
    with pytest.raises(AssertionError):
        _check_that_every_param_is_bounded([-np.inf, 2, 3], [4, 5, 6])
    with pytest.raises(AssertionError):
        _check_that_every_param_is_bounded([np.nan, 2, 3], [4, 5, 6])
    with pytest.raises(AssertionError):
        _check_that_every_param_is_bounded([np.inf, 2, 3], [4, 5, 6])


test_cases = [
    # popsize, x, lower_bound, expected
    (55.3, None, None, 55),
    (None, np.ones(5), 500, 500),
    (None, np.ones(5), 4, 60),
]


@pytest.mark.parametrize("popsize, x, lower_bound, expected", test_cases)
def test_determine_population_size(popsize, x, lower_bound, expected):
    res = get_population_size(population_size=popsize, x=x, lower_bound=lower_bound)
    assert res == expected


def test_convert_str_to_int():
    d = {"a": 1, "b": 3}
    assert _convert_str_to_int(d, "a") == 1
    assert _convert_str_to_int(d, 1) == 1
    with pytest.raises(ValueError):
        _convert_str_to_int(d, 5)
    with pytest.raises(ValueError):
        _convert_str_to_int(d, "hello")


def test_replace_none():
    assert _replace_none(None, 5) == 5
    assert _replace_none("a", 10) == "a"
