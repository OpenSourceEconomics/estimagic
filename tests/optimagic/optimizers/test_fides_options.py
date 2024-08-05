"""Test the different options of fides."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.config import IS_FIDES_INSTALLED

if IS_FIDES_INSTALLED:
    from fides.hessian_approximation import FX, SR1, Broyden
    from optimagic.optimizers.fides import fides
else:
    FX = lambda: None
    SR1 = lambda: None
    Broyden = lambda phi: None  # noqa: ARG005

test_cases_no_contribs_needed = [
    {},
    {"hessian_update_strategy": "bfgs"},
    {"hessian_update_strategy": "BFGS"},
    {"hessian_update_strategy": SR1()},
    {"hessian_update_strategy": Broyden(phi=0.5)},
    {"hessian_update_strategy": "sr1"},
    {"hessian_update_strategy": "DFP"},
    {"hessian_update_strategy": "bb"},
    {"convergence_ftol_rel": 1e-6},
    {"convergence_xtol_abs": 1e-6},
    {"convergence_gtol_abs": 1e-6},
    {"convergence_gtol_rel": 1e-6},
    {"stopping_maxiter": 100},
    {"stopping_max_seconds": 200},
    {"trustregion_initial_radius": 20, "trustregion_stepback_strategy": "truncate"},
    {"trustregion_subspace_dimension": "full"},
    {"trustregion_max_stepback_fraction": 0.8},
    {"trustregion_decrease_threshold": 0.4, "trustregion_decrease_factor": 0.2},
    {"trustregion_increase_threshold": 0.9, "trustregion_increase_factor": 4},
]


def criterion_and_derivative(x):
    return (x**2).sum(), 2 * x


@pytest.mark.skipif(not IS_FIDES_INSTALLED, reason="fides not installed.")
@pytest.mark.parametrize("algo_options", test_cases_no_contribs_needed)
def test_fides_correct_algo_options(algo_options):
    res = fides(
        criterion_and_derivative=criterion_and_derivative,
        x=np.array([1, -5, 3]),
        lower_bounds=np.array([-10, -10, -10]),
        upper_bounds=np.array([10, 10, 10]),
        **algo_options,
    )
    aaae(res["solution_x"], np.zeros(3), decimal=4)


test_cases_needing_contribs = [
    {"hessian_update_strategy": FX()},
    {"hessian_update_strategy": "ssm"},
    {"hessian_update_strategy": "TSSM"},
    {"hessian_update_strategy": "gnsbfgs"},
]


@pytest.mark.skipif(not IS_FIDES_INSTALLED, reason="fides not installed.")
@pytest.mark.parametrize("algo_options", test_cases_needing_contribs)
def test_fides_unimplemented_algo_options(algo_options):
    with pytest.raises(NotImplementedError):
        fides(
            criterion_and_derivative=criterion_and_derivative,
            x=np.array([1, -5, 3]),
            lower_bounds=np.array([-10, -10, -10]),
            upper_bounds=np.array([10, 10, 10]),
            **algo_options,
        )


@pytest.mark.skipif(not IS_FIDES_INSTALLED, reason="fides not installed.")
def test_fides_stop_after_one_iteration():
    res = fides(
        criterion_and_derivative=criterion_and_derivative,
        x=np.array([1, -5, 3]),
        lower_bounds=np.array([-10, -10, -10]),
        upper_bounds=np.array([10, 10, 10]),
        stopping_maxiter=1,
    )
    assert not res["success"]
    assert res["n_iterations"] == 1
