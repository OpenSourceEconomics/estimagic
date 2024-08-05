from itertools import product

import numpy as np
import pytest
from optimagic.benchmarking.get_benchmark_problems import (
    _step_func,
    get_benchmark_problems,
)

PARMETRIZATION = []
for name in ["more_wild", "cartis_roberts", "example", "estimagic"]:
    for additive, multiplicative, scaling in product([False, True], repeat=3):
        PARMETRIZATION.append((name, additive, multiplicative, scaling))


@pytest.mark.parametrize(
    "name, additive_noise, multiplicative_noise, scaling", PARMETRIZATION
)
def test_get_problems(name, additive_noise, multiplicative_noise, scaling):
    is_noisy = any((additive_noise, multiplicative_noise))
    problems = get_benchmark_problems(
        name=name,
        additive_noise=additive_noise,
        multiplicative_noise=multiplicative_noise,
        scaling=scaling,
    )
    first_name = list(problems)[0]
    first = problems[first_name]
    func = first["inputs"]["fun"]
    params = first["inputs"]["params"]

    first_eval = func(params)["value"]
    second_eval = func(params)["value"]

    if is_noisy:
        assert first_eval != second_eval
    else:
        assert first_eval == second_eval

    for problem in problems.values():
        assert isinstance(problem["inputs"]["params"], np.ndarray)
        assert isinstance(problem["solution"]["params"], np.ndarray)


def test_step_func():
    p = np.array([0.0001, 0.0002])
    got = _step_func(p, lambda x: x @ x)
    assert np.allclose(got, 0)
    assert not np.allclose(p @ p, 0)
