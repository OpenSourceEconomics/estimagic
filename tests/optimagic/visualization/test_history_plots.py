import itertools

import numpy as np
import optimagic as om
import pytest
from optimagic.optimization.optimize import minimize
from optimagic.parameters.bounds import Bounds
from optimagic.visualization.history_plots import criterion_plot, params_plot


@pytest.fixture()
def minimize_result():
    bounds = Bounds(soft_lower=np.full(5, -1), soft_upper=np.full(5, 6))
    out = {}
    for multistart in [True, False]:
        res = []
        for algorithm in ["scipy_neldermead", "scipy_lbfgsb"]:
            _res = minimize(
                fun=lambda x: x @ x,
                params=np.arange(5),
                algorithm=algorithm,
                bounds=bounds,
                multistart=om.MultistartOptions(
                    n_samples=1000, convergence_max_discoveries=5
                ),
            )
            res.append(_res)
        out[multistart] = res
    return out


# ======================================================================================
# Params plot
# ======================================================================================


TEST_CASES = list(
    itertools.product(
        [True, False],  # multistart
        [None, lambda x: x[:2]],  # selector
        [None, 50],  # max_evaluations
        [True, False],  # show_exploration
    )
)


@pytest.mark.parametrize(
    "multistart, selector, max_evaluations, show_exploration", TEST_CASES
)
def test_params_plot_multistart(
    minimize_result, multistart, selector, max_evaluations, show_exploration
):
    for _res in minimize_result[multistart]:
        params_plot(
            _res,
            selector=selector,
            max_evaluations=max_evaluations,
            show_exploration=show_exploration,
        )


# ======================================================================================
# Test criterion plot
# ======================================================================================


TEST_CASES = list(itertools.product([True, False], repeat=4))


@pytest.mark.parametrize(
    "multistart, monotone, stack_multistart, exploration", TEST_CASES
)
def test_criterion_plot_list_input(
    minimize_result, multistart, monotone, stack_multistart, exploration
):
    res = minimize_result[multistart]

    criterion_plot(
        res,
        monotone=monotone,
        stack_multistart=stack_multistart,
        show_exploration=exploration,
    )


def test_criterion_plot_name_input(minimize_result):
    result = minimize_result[False]
    criterion_plot(result[0], names="neldermead", palette="blue")


def test_criterion_plot_wrong_results():
    with pytest.raises(TypeError):
        criterion_plot([10, np.array([1, 2, 3])])


def test_criterion_plot_different_input_types():
    bounds = Bounds(soft_lower=np.full(5, -1), soft_upper=np.full(5, 6))
    # logged result
    minimize(
        fun=lambda x: x @ x,
        params=np.arange(5),
        algorithm="scipy_lbfgsb",
        bounds=bounds,
        multistart=om.MultistartOptions(n_samples=1000, convergence_max_discoveries=5),
        log_options={"fast_logging": True},
        logging="test.db",
    )

    res = minimize(
        fun=lambda x: x @ x,
        params=np.arange(5),
        algorithm="scipy_lbfgsb",
        bounds=bounds,
        multistart=om.MultistartOptions(n_samples=1000, convergence_max_discoveries=5),
    )

    results = ["test.db", res]

    criterion_plot(results)
    criterion_plot(results, monotone=True)
    criterion_plot(results, stack_multistart=True)
    criterion_plot(results, monotone=True, stack_multistart=True)
    criterion_plot(results, show_exploration=True)


def test_criterion_plot_wrong_inputs():
    with pytest.raises(ValueError):
        criterion_plot("bla", names=[1, 2])

    with pytest.raises(ValueError):
        criterion_plot(["bla", "bla"], names="blub")
