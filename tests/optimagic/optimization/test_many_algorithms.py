"""Test all available algorithms on a simple sum of squares function.

- only minimize
- only numerical derivative

"""

import inspect
import sys

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.algorithms import AVAILABLE_ALGORITHMS, GLOBAL_ALGORITHMS
from optimagic.optimization.optimize import minimize
from optimagic.parameters.bounds import Bounds

LOCAL_ALGORITHMS = {
    key: value
    for key, value in AVAILABLE_ALGORITHMS.items()
    if key not in GLOBAL_ALGORITHMS and key != "bhhh"
}

GLOBAL_ALGORITHMS_AVAILABLE = [
    name for name in AVAILABLE_ALGORITHMS if name in GLOBAL_ALGORITHMS
]

BOUNDED_ALGORITHMS = []
for name, func in LOCAL_ALGORITHMS.items():
    arguments = list(inspect.signature(func).parameters)
    if "lower_bounds" in arguments and "upper_bounds" in arguments:
        BOUNDED_ALGORITHMS.append(name)


def sos(x):
    contribs = x**2
    return {"value": contribs.sum(), "contributions": contribs, "root_contributions": x}


@pytest.mark.parametrize("algorithm", LOCAL_ALGORITHMS)
def test_algorithm_on_sum_of_squares(algorithm):
    res = minimize(
        fun=sos,
        params=np.arange(3),
        algorithm=algorithm,
        collect_history=True,
        skip_checks=True,
    )
    assert res.success in [True, None]
    aaae(res.params, np.zeros(3), decimal=4)


@pytest.mark.parametrize("algorithm", BOUNDED_ALGORITHMS)
def test_algorithm_on_sum_of_squares_with_binding_bounds(algorithm):
    res = minimize(
        fun=sos,
        params=np.array([3, 2, -3]),
        bounds=Bounds(
            lower=np.array([1, -np.inf, -np.inf]), upper=np.array([np.inf, np.inf, -1])
        ),
        algorithm=algorithm,
        collect_history=True,
        skip_checks=True,
    )
    assert res.success in [True, None]
    decimal = 3
    aaae(res.params, np.array([1, 0, -1]), decimal=decimal)


skip_msg = (
    "The very slow tests of global algorithms are only run on linux which always "
    "runs much faster in continuous integration."
)


@pytest.mark.skipif(sys.platform != "linux", reason=skip_msg)
@pytest.mark.parametrize("algorithm", GLOBAL_ALGORITHMS_AVAILABLE)
def test_global_algorithms_on_sum_of_squares(algorithm):
    res = minimize(
        fun=sos,
        params=np.array([0.35, 0.35]),
        bounds=Bounds(lower=np.array([0.2, -0.5]), upper=np.array([1, 0.5])),
        algorithm=algorithm,
        collect_history=False,
        skip_checks=True,
    )
    assert res.success in [True, None]
    aaae(res.params, np.array([0.2, 0]), decimal=1)
