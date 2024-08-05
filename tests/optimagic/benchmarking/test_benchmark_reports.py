from itertools import product

import numpy as np
import pytest
from optimagic import (
    OptimizeResult,
    convergence_report,
    get_benchmark_problems,
    rank_report,
    traceback_report,
)


@pytest.fixture
def benchmark_example():
    all_problems = get_benchmark_problems("example")
    problems = {
        k: v
        for k, v in all_problems.items()
        if k in ["bard_good_start", "box_3d", "rosenbrock_good_start"]
    }
    _stop_after_10 = {
        "stopping_max_criterion_evaluations": 10,
        "stopping_max_iterations": 10,
    }
    optimizers = {
        "lbfgsb": {"algorithm": "scipy_lbfgsb", "algo_options": _stop_after_10},
        "nm": {"algorithm": "scipy_neldermead", "algo_options": _stop_after_10},
    }

    results = {
        ("bard_good_start", "lbfgsb"): {
            "params_history": [
                [1.0, 1.0, 1.0],
                [0.48286315298120086, 1.6129119244711858, 1.5974181569859445],
                [0.09754340799557773, 1.7558262514618663, 1.7403560082627973],
            ],
            "criterion_history": np.array(
                [
                    4.16816959e01,
                    3.20813118e00,
                    9.97263708e-03,
                ]
            ),
            "time_history": [
                0.0,
                0.0003762839987757616,
                0.0007037959985609632,
            ],
            "batches_history": [0, 1, 2],
            "solution": OptimizeResult,  # success
        },
        ("box_3d", "lbfgsb"): {
            "params_history": [
                [0.0, 10.0, 20.0],
                [-0.6579976970071755, 10.014197643614924, 19.247113914560085],
                [-3.2899884850358774, 10.070988218074623, 16.235569572800433],
            ],
            "criterion_history": np.array(
                [
                    1.03115381e03,
                    8.73640769e02,
                    9.35093416e02,
                ]
            ),
            "time_history": [
                0.0,
                0.000555748996703187,
                0.0009771709992492106,
            ],
            "batches_history": [0, 1, 2],
            "solution": OptimizeResult,  # failed
        },
        ("rosenbrock_good_start", "lbfgsb"): {
            "params_history": [
                [-1.2, 1.0],
                [0.0, 0.0],
            ],
            "criterion_history": np.array([1.795769e6, 1e3]),
            "time_history": [
                0.0,
                5.73799989069812e-04,
            ],
            "batches_history": [0, 1],
            "solution": "lbfgsb traceback",  # error
        },
        ("bard_good_start", "nm"): {
            "params_history": [
                [1.0, 1.0, 1.0],
                [1.05, 1.0, 1.0],
                [0.7999999999999998, 1.1999999999999993, 1.0499999999999994],
                [0.08241056, 1.13303608, 2.34369519],
            ],
            "criterion_history": np.array(
                [
                    41.68169586,
                    43.90748158,
                    23.92563745,
                    0.00821487730657897,
                ]
            ),
            "time_history": [
                0.0,
                3.603900040616281e-05,
                0.0004506860022956971,
                0.00015319500016630627,
            ],
            "batches_history": [0, 1, 2, 4],
            "solution": OptimizeResult,  # success
        },
        ("box_3d", "nm"): {
            "params_history": [
                [0.0, 10.0, 20.0],
                [0.025, 10.0, 20.0],
                [0.0, 10.5, 20.0],
            ],
            "criterion_history": np.array(
                [1031.15381061, 1031.17836473, 1030.15033678]
            ),
            "time_history": [
                0.0,
                5.73799989069812e-05,
                0.00010679600018193014,
            ],
            "batches_history": [0, 1, 2],
            "solution": "some traceback",  # error
        },
        ("rosenbrock_good_start", "nm"): {
            "params_history": [
                [-1.2, 1.0],
                [0.0, 0.0],
            ],
            "criterion_history": np.array([1.795769e6, 1e3]),
            "time_history": [
                0.0,
                5.73799989069812e-04,
            ],
            "batches_history": [0, 1],
            "solution": "another traceback",  # error
        },
    }

    return problems, optimizers, results


# ====================================================================================
# Convergence report
# ====================================================================================

keys = ["stopping_criterion"]
stopping_criterion = ["x_and_y", "x_or_y", "x", "y"]
x_precision = [1e-4, 1e-6]
y_precision = [1e-4, 1e-6]
CONVERGENCE_REPORT_OPTIONS = [
    dict(zip(keys, value, strict=False))
    for value in product(stopping_criterion, x_precision, y_precision)
]


@pytest.mark.parametrize("options", CONVERGENCE_REPORT_OPTIONS)
def test_convergence_report(options, benchmark_example):
    problems, optimizers, results = benchmark_example

    df = convergence_report(problems=problems, results=results, **options)

    expected_columns = list(optimizers.keys()) + ["dimensionality"]
    assert df.shape == (len(problems), len(expected_columns))
    assert set(df.columns) == set(expected_columns)

    assert df["lbfgsb"].loc["box_3d"] == "failed"
    assert df["nm"].loc["box_3d"] == "error"


# ====================================================================================
# Rank report
# ====================================================================================

keys = ["runtime_measure", "stopping_criterion"]
runtime_measure = ["n_evaluations", "walltime", "n_batches"]
RANK_REPORT_OPTIONS = [
    dict(zip(keys, value, strict=False))
    for value in product(runtime_measure, stopping_criterion)
]


@pytest.mark.parametrize("options", RANK_REPORT_OPTIONS)
def test_rank_report(options, benchmark_example):
    problems, optimizers, results = benchmark_example

    df = rank_report(problems=problems, results=results, **options)

    assert df.shape == (len(problems), len(optimizers) + 1)  # +1 for dimensionality
    assert set(df.columns) == set(optimizers.keys()) | {"dimensionality"}

    assert df["lbfgsb"].loc["box_3d"] == "failed"
    assert df["nm"].loc["box_3d"] == "error"


# ====================================================================================
# Traceback report
# ====================================================================================


@pytest.mark.parametrize("return_type", ["text", "markdown", "dict", "dataframe"])
def test_traceback_report(return_type, benchmark_example):
    problems, optimizers, results = benchmark_example
    n_failed_problems = 3

    report = traceback_report(
        problems=problems, results=results, return_type=return_type
    )

    if return_type in ["text", "dict"]:
        assert len(report) == n_failed_problems

    elif return_type == "markdown":
        for algorithm_name in optimizers:
            assert algorithm_name in report

    elif return_type == "dataframe":
        assert report.shape == (n_failed_problems, 2)
        assert list(report.index.names) == ["algorithm", "problem"]
