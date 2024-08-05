import numpy as np
import pandas as pd
import pytest
from optimagic.optimization.optimize_result import OptimizeResult, _create_stars
from optimagic.utilities import get_rng


@pytest.fixture()
def convergence_report():
    conv_report = pd.DataFrame(
        index=[
            "relative_criterion_change",
            "relative_params_change",
            "absolute_criterion_change",
            "absolute_params_change",
        ],
        columns=["one_step", "five_steps"],
    )
    u = get_rng(seed=0).uniform
    conv_report["one_step"] = [
        u(1e-12, 1e-10),
        u(1e-9, 1e-8),
        u(1e-7, 1e-6),
        u(1e-6, 1e-5),
    ]
    conv_report["five_steps"] = [1e-8, 1e-4, 1e-3, 100]
    return conv_report


@pytest.fixture()
def base_inputs():
    out = {
        "params": np.ones(3),
        "fun": 500,
        "start_fun": 1000,
        "start_params": np.full(3, 10),
        "direction": "minimize",
        "message": "OPTIMIZATION TERMINATED SUCCESSFULLY",
        "success": True,
        "n_fun_evals": 100,
        "n_jac_evals": 0,
        "n_iterations": 80,
        "history": {"criterion": list(range(10))},
        "algorithm": "scipy_lbfgsb",
        "n_free": 2,
    }
    return out


def test_optimize_result_runs(base_inputs, convergence_report):
    res = OptimizeResult(
        convergence_report=convergence_report,
        **base_inputs,
    )
    res.__repr__()


def test_create_stars():
    sr = pd.Series([1e-12, 1e-9, 1e-7, 1e-4, 1e-2])
    calculated = _create_stars(sr).tolist()
    expected = ["***", "** ", "*  ", "   ", "   "]
    assert calculated == expected


def test_to_pickle(base_inputs, convergence_report, tmp_path):
    res = OptimizeResult(
        convergence_report=convergence_report,
        **base_inputs,
    )
    res.to_pickle(tmp_path / "bla.pkl")


def test_dict_access(base_inputs):
    res = OptimizeResult(**base_inputs)
    assert res["fun"] == 500
    assert res["nfev"] == 100
