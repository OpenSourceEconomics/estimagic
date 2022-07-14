from itertools import product

import numpy as np
import pytest
from estimagic import get_benchmark_problems

TEST_CASES = []
for name in [
    "more_wild",
    "cartis_roberts",
    "example",
    "estimagic",
    "scalar_functions",
    "scalar_functions_extra",
]:
    for additive, multiplicative, scaling in product([False, True], repeat=3):
        TEST_CASES.append((name, additive, multiplicative, scaling))


@pytest.mark.parametrize(
    "name, additive_noise, multiplicative_noise, scaling", TEST_CASES
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
    func = first["inputs"]["criterion"]
    params = first["inputs"]["params"]

    if name in ("scalar_functions", "scalar_functions_extra"):
        first_eval = func(params)
        second_eval = func(params)
    else:
        first_eval = func(params)["value"]
        second_eval = func(params)["value"]

    if is_noisy:
        assert first_eval != second_eval
    else:
        assert first_eval == second_eval

    for problem in problems.values():
        assert isinstance(problem["inputs"]["params"], np.ndarray)
        assert isinstance(problem["solution"]["params"], np.ndarray)
