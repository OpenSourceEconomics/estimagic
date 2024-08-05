import numpy as np
import pandas as pd
import pytest
from optimagic import get_benchmark_problems
from optimagic.benchmarking.run_benchmark import run_benchmark
from optimagic.visualization.profile_plot import (
    _determine_alpha_grid,
    _find_switch_points,
    create_solution_times,
    profile_plot,
)


@pytest.fixture()
def performance_ratios():
    df = pd.DataFrame(
        data={"algo1": [1.0, 1.0, 4.0], "algo2": [1.5, np.inf, 1.0]},
        index=["prob1", "prob2", "prob3"],
    )
    return df


def test_find_switch_points(performance_ratios):
    res = _find_switch_points(performance_ratios)
    expected = np.array([1.0, 1.5, 4.0])
    np.testing.assert_array_almost_equal(res, expected)


def test_determine_alpha_grid(performance_ratios):
    res = _determine_alpha_grid(performance_ratios)
    expected = np.array([1.0 + 1e-10, 1.25, 1.5, 2.75, 4.0, 4.0 * 1.025, 4.0 * 1.05])
    np.testing.assert_array_almost_equal(res, expected)


def test_create_solution_times_n_evaluations():
    df = pd.DataFrame(
        columns=["problem", "algorithm", "n_evaluations"],
        data=[
            ["prob1", "algo1", 0],
            ["prob1", "algo1", 1],
            ["prob1", "algo2", 2],
            ["prob1", "algo2", 3],
            ["prob2", "algo1", 5],
            ["prob2", "algo2", 0],
            ["prob2", "algo2", 1],
        ],
    )
    info = pd.DataFrame(
        {
            "algo1": [True, True],
            "algo2": [True, False],
        },
        index=["prob1", "prob2"],
    )
    expected = pd.DataFrame(
        {
            "algo1": [1.0, 5],
            "algo2": [3.0, np.inf],
        },
        index=pd.Index(["prob1", "prob2"], name="problem"),
    )
    expected.columns.name = "algorithm"

    res = create_solution_times(
        df=df, runtime_measure="n_evaluations", converged_info=info
    )
    pd.testing.assert_frame_equal(res, expected)


def test_create_solution_times_n_batches():
    df = pd.DataFrame(
        columns=["problem", "algorithm", "n_batches"],
        data=[
            ["prob1", "algo1", 0],
            ["prob1", "algo1", 1],
            ["prob1", "algo2", 2],
            ["prob1", "algo2", 2],
            ["prob2", "algo1", 1],
            ["prob2", "algo2", 0],
            ["prob2", "algo2", 0],
        ],
    )
    info = pd.DataFrame(
        {
            "algo1": [True, True],
            "algo2": [True, False],
        },
        index=["prob1", "prob2"],
    )
    expected = pd.DataFrame(
        {
            "algo1": [1.0, 1],
            "algo2": [2.0, np.inf],
        },
        index=pd.Index(["prob1", "prob2"], name="problem"),
    )
    expected.columns.name = "algorithm"

    res = create_solution_times(df=df, runtime_measure="n_batches", converged_info=info)
    pd.testing.assert_frame_equal(res, expected)


def test_create_solution_times_walltime():
    df = pd.DataFrame(
        columns=["problem", "algorithm", "n_evaluations", "walltime"],
        data=[
            ["prob1", "algo1", 0, 0],
            ["prob1", "algo1", 1, 1],
            ["prob1", "algo2", 2, 2],
            ["prob1", "algo2", 3, 3],
            ["prob2", "algo1", 5, 5],
            ["prob2", "algo2", 0, 0],
            ["prob2", "algo2", 1, 1],
        ],
    )
    info = pd.DataFrame(
        {
            "algo1": [True, True],
            "algo2": [True, False],
        },
        index=["prob1", "prob2"],
    )
    expected = pd.DataFrame(
        {
            "algo1": [1.0, 5],
            "algo2": [3.0, np.inf],
        },
        index=pd.Index(["prob1", "prob2"], name="problem"),
    )
    expected.columns.name = "algorithm"

    res = create_solution_times(df=df, runtime_measure="walltime", converged_info=info)
    pd.testing.assert_frame_equal(res, expected)


# integration test to make sure non default argument do not throw Errors
profile_options = [
    {"runtime_measure": "walltime"},
    {"runtime_measure": "n_batches"},
    {"stopping_criterion": "x_or_y"},
]


@pytest.mark.parametrize("options", profile_options)
def test_profile_plot_options(options):
    problems = get_benchmark_problems("example")
    stop_after_10 = {
        "stopping_max_criterion_evaluations": 10,
        "stopping_max_iterations": 10,
    }
    optimizers = {
        "lbfgsb": {"algorithm": "scipy_lbfgsb", "algo_options": stop_after_10},
        "neldermead": {
            "algorithm": "scipy_neldermead",
            "algo_options": stop_after_10,
        },
    }
    results = run_benchmark(
        problems,
        optimizers,
        n_cores=1,  # must be 1 for the test to work
    )

    profile_plot(problems=problems, results=results, **options)
