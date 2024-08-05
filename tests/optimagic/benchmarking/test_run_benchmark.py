from functools import partial

import numpy as np
import pandas as pd
import pytest
from optimagic import get_benchmark_problems
from optimagic.benchmarking.more_wild import (
    linear_full_rank,
    linear_full_rank_solution_x,
    rosenbrock,
)
from optimagic.benchmarking.run_benchmark import run_benchmark

# ======================================================================================
# Internal benchmark suite
# ======================================================================================


def test_run_benchmark_dict_options():
    all_problems = get_benchmark_problems("more_wild")
    first_two_names = list(all_problems)[:2]
    first_two = {name: all_problems[name] for name in first_two_names}

    optimize_options = {
        "default_lbfgsb": "scipy_lbfgsb",
        "tuned_lbfgsb": {
            "algorithm": "scipy_lbfgsb",
            "algo_options": {"convergence.relative_criterion_tolerance": 1e-10},
        },
    }

    result = run_benchmark(
        problems=first_two,
        optimize_options=optimize_options,
    )

    expected_keys = {
        ("linear_full_rank_good_start", "default_lbfgsb"),
        ("linear_full_rank_bad_start", "default_lbfgsb"),
        ("linear_full_rank_good_start", "tuned_lbfgsb"),
        ("linear_full_rank_bad_start", "tuned_lbfgsb"),
    }
    assert set(result) == expected_keys


def test_run_benchmark_list_options():
    all_problems = get_benchmark_problems("example")
    first_two_names = list(all_problems)[:2]
    first_two = {name: all_problems[name] for name in first_two_names}
    optimize_options = ["scipy_lbfgsb", "scipy_neldermead"]

    result = run_benchmark(
        problems=first_two,
        optimize_options=optimize_options,
    )

    expected_keys = {
        ("helical_valley_good_start", "scipy_lbfgsb"),
        ("rosenbrock_good_start", "scipy_lbfgsb"),
        ("helical_valley_good_start", "scipy_neldermead"),
        ("rosenbrock_good_start", "scipy_neldermead"),
    }
    assert set(result) == expected_keys


def test_run_benchmark_failing():
    all_problems = get_benchmark_problems("more_wild")
    failing_name = "jennrich_sampson"
    failing = {failing_name: all_problems[failing_name]}

    optimize_options = ["scipy_lbfgsb"]

    with pytest.warns():
        result = run_benchmark(problems=failing, optimize_options=optimize_options)

    key = (failing_name, "scipy_lbfgsb")
    assert isinstance(result[key]["solution"], str)


# ======================================================================================
# Custom benchmark functions
# ======================================================================================


def get_benchmark_problems_custom(raw_problems, internal_criterion):
    problems = {}
    for name, specification in raw_problems.items():
        inputs = _create_problem_inputs_custom(specification, internal_criterion)

        problems[name] = {
            "inputs": inputs,
            "solution": _create_problem_solution_custom(specification),
            "info": specification.get("info", {}),
            "noise_free_fun": partial(
                internal_criterion, criterion=specification["fun"]
            ),
            "noisy": False,
        }

    return problems


def _create_problem_inputs_custom(specification, internal_criterion_func):
    _criterion = partial(
        internal_criterion_func,
        criterion=specification["fun"],
    )

    _params = specification["start_x"]
    inputs = {"fun": _criterion, "params": _params}

    return inputs


def _create_problem_solution_custom(specification):
    _solution_x = specification.get("solution_x")
    if _solution_x is None:
        _solution_x = specification["start_x"] * np.nan

    _value = specification["solution_criterion"]

    solution = {
        "params": _solution_x,
        "value": _value,
    }
    return solution


def _internal_criterion_pandas(params, criterion):
    x = params["value"].to_numpy()
    critval = criterion(x)

    out = {
        "root_contributions": critval,
        "value": critval @ critval,
    }

    return out


def _internal_criterion_dict(params, criterion):
    if "b" in params:
        x = np.array([params["a"], *params["b"].flatten().tolist()])
    else:
        x = params["a"]
    critval = criterion(x)

    out = {
        "root_contributions": critval,
        "value": critval @ critval,
    }

    return out


problems_pandas_input = {
    "linear_full_rank_good_start": {
        "fun": partial(linear_full_rank, dim_out=45),
        "start_x": pd.DataFrame(np.ones(9), columns=["value"]),
        "solution_x": pd.DataFrame(linear_full_rank_solution_x, columns=["value"]),
        "start_criterion": 72,
        "solution_criterion": 36,
    },
    "rosenbrock_good_start": {
        "fun": rosenbrock,
        "start_x": pd.DataFrame([-1.2, 1], columns=["value"]),
        "solution_x": pd.DataFrame(np.ones(2), columns=["value"]),
        "start_criterion": 24.2,
        "solution_criterion": 0,
    },
}


prolems_dict_input = {
    "linear_full_rank_good_start": {
        "fun": partial(linear_full_rank, dim_out=45),
        "start_x": {"a": 1, "b": np.ones((2, 2, 2))},
        "solution_x": {
            "a": linear_full_rank_solution_x[0],
            "b": np.array(linear_full_rank_solution_x[1:]).reshape(2, 2, 2),
        },
        "start_criterion": 72,
        "solution_criterion": 36,
    },
    "rosenbrock_good_start": {
        "fun": rosenbrock,
        "start_x": {"a": np.array([-1.2, 1])},
        "solution_x": {"a": np.ones(2)},
        "start_criterion": 24.2,
        "solution_criterion": 0,
    },
}

TEST_CASES = [
    (problems_pandas_input, _internal_criterion_pandas),
    (prolems_dict_input, _internal_criterion_dict),
]


@pytest.mark.parametrize("raw_problems, internal_criterion_func", TEST_CASES)
def test_custom_benchmarks(raw_problems, internal_criterion_func):
    problems = get_benchmark_problems_custom(raw_problems, internal_criterion_func)

    optimize_options = ["scipy_lbfgsb"]
    result = run_benchmark(problems=problems, optimize_options=optimize_options)

    expected_keys = {
        ("linear_full_rank_good_start", "scipy_lbfgsb"),
        ("rosenbrock_good_start", "scipy_lbfgsb"),
    }
    assert set(result) == expected_keys
