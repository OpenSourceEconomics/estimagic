from itertools import product

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.optimization.multistart import (
    _draw_exploration_sample,
    get_batched_optimization_sample,
    run_explorations,
    update_convergence_state,
)


@pytest.fixture()
def params():
    df = pd.DataFrame(index=["a", "b", "c"])
    df["value"] = [0, 1, 2.0]
    df["soft_lower_bound"] = [-1, 0, np.nan]
    df["upper_bound"] = [2, 2, np.nan]
    return df


@pytest.fixture()
def constraints():
    return [{"type": "fixed", "loc": "c", "value": 2}]


dim = 2
distributions = ["uniform", "triangular"]
rules = ["sobol", "halton", "latin_hypercube", "random"]
lower = [np.zeros(dim), np.ones(dim) * 0.5, -np.ones(dim)]
upper = [np.ones(dim), np.ones(dim) * 0.75, np.ones(dim) * 2]
test_cases = list(product(distributions, rules, lower, upper))


@pytest.mark.parametrize("dist, rule, lower, upper", test_cases)
def test_draw_exploration_sample(dist, rule, lower, upper):
    results = []

    for _ in range(2):
        results.append(
            _draw_exploration_sample(
                x=np.ones_like(lower) * 0.5,
                lower=lower,
                upper=upper,
                n_samples=3,
                distribution=dist,
                method=rule,
                seed=1234,
            )
        )

    aaae(results[0], results[1])
    calculated = results[0]
    assert calculated.shape == (3, 2)


def test_run_explorations():
    def _dummy(x, **kwargs):
        assert set(kwargs) == {
            "task",
            "algo_info",
            "error_handling",
            "fixed_log_data",
        }
        if x.sum() == 5:
            out = np.nan
        else:
            out = -x.sum()
        return out

    calculated = run_explorations(
        func=_dummy,
        primary_key="value",
        sample=np.arange(6).reshape(3, 2),
        batch_evaluator="joblib",
        n_cores=1,
        step_id=0,
        error_handling="raise",
    )

    exp_values = np.array([-9, -1])
    exp_sample = np.array([[4, 5], [0, 1]])

    aaae(calculated["sorted_values"], exp_values)
    aaae(calculated["sorted_sample"], exp_sample)


def test_get_batched_optimization_sample():
    calculated = get_batched_optimization_sample(
        sorted_sample=np.arange(12).reshape(6, 2),
        stopping_maxopt=5,
        batch_size=4,
    )
    expected = [[[0, 1], [2, 3], [4, 5], [6, 7]], [[8, 9]]]

    assert len(calculated[0]) == 4
    assert len(calculated[1]) == 1
    assert len(calculated) == 2

    for calc_batch, exp_batch in zip(calculated, expected, strict=False):
        assert isinstance(calc_batch, list)
        for calc_entry, exp_entry in zip(calc_batch, exp_batch, strict=False):
            assert isinstance(calc_entry, np.ndarray)
            assert calc_entry.tolist() == exp_entry


@pytest.fixture()
def current_state():
    state = {
        "best_x": np.ones(3),
        "best_y": 5,
        "best_res": None,
        "x_history": [np.arange(3) - 1e-20, np.ones(3)],
        "y_history": [6, 5],
        "result_history": [],
        "start_history": [],
    }

    return state


@pytest.fixture()
def starts():
    return [np.zeros(3)]


@pytest.fixture()
def results():
    return [{"solution_x": np.arange(3) + 1e-10, "solution_criterion": 4}]


def test_update_state_converged(current_state, starts, results):
    criteria = {
        "xtol": 1e-3,
        "max_discoveries": 2,
    }

    new_state, is_converged = update_convergence_state(
        current_state=current_state,
        starts=starts,
        results=results,
        convergence_criteria=criteria,
        primary_key="value",
    )

    aaae(new_state["best_x"], np.arange(3))
    assert new_state["best_y"] == 4
    assert new_state["y_history"] == [6, 5, 4]
    assert new_state["result_history"][0]["solution_criterion"] == 4
    aaae(new_state["start_history"][0], np.zeros(3))
    assert new_state["best_res"].keys() == results[0].keys()

    assert is_converged


def test_update_state_not_converged(current_state, starts, results):
    criteria = {
        "xtol": 1e-3,
        "max_discoveries": 5,
    }

    _, is_converged = update_convergence_state(
        current_state=current_state,
        starts=starts,
        results=results,
        convergence_criteria=criteria,
        primary_key="value",
    )

    assert not is_converged
