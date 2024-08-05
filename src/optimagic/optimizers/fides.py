"""Implement the fides optimizer."""

import logging

import numpy as np

from optimagic.config import IS_FIDES_INSTALLED
from optimagic.decorators import mark_minimizer
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import (
    CONVERGENCE_FTOL_ABS,
    CONVERGENCE_FTOL_REL,
    CONVERGENCE_GTOL_ABS,
    CONVERGENCE_GTOL_REL,
    CONVERGENCE_XTOL_ABS,
    STOPPING_MAXITER,
)

if IS_FIDES_INSTALLED:
    from fides import Optimizer, hessian_approximation


@mark_minimizer(
    name="fides",
    primary_criterion_entry="value",
    needs_scaling=False,
    is_available=IS_FIDES_INSTALLED,
)
def fides(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    hessian_update_strategy="bfgs",
    convergence_ftol_abs=CONVERGENCE_FTOL_ABS,
    convergence_ftol_rel=CONVERGENCE_FTOL_REL,
    convergence_xtol_abs=CONVERGENCE_XTOL_ABS,
    convergence_gtol_abs=CONVERGENCE_GTOL_ABS,
    convergence_gtol_rel=CONVERGENCE_GTOL_REL,
    stopping_maxiter=STOPPING_MAXITER,
    stopping_max_seconds=np.inf,
    trustregion_initial_radius=1.0,
    trustregion_stepback_strategy="truncate",
    trustregion_subspace_dimension="full",
    trustregion_max_stepback_fraction=0.95,
    trustregion_decrease_threshold=0.25,
    trustregion_increase_threshold=0.75,
    trustregion_decrease_factor=0.25,
    trustregion_increase_factor=2.0,
):
    """Minimize a scalar function using the Fides Optimizer.

    For details see
    :ref: `fides_algorithm`.

    """
    if not IS_FIDES_INSTALLED:
        raise NotInstalledError(
            "The 'fides' algorithm requires the fides package to be installed. "
            "You can install it with `pip install fides>=0.7.4`."
        )

    fides_options = {
        "delta_init": trustregion_initial_radius,
        "eta": trustregion_increase_threshold,
        "fatol": convergence_ftol_abs,
        "frtol": convergence_ftol_rel,
        "gamma1": trustregion_decrease_factor,
        "gamma2": trustregion_increase_factor,
        "gatol": convergence_gtol_abs,
        "grtol": convergence_gtol_rel,
        "maxiter": stopping_maxiter,
        "maxtime": stopping_max_seconds,
        "mu": trustregion_decrease_threshold,
        "stepback_strategy": trustregion_stepback_strategy,
        "subspace_solver": trustregion_subspace_dimension,
        "theta_max": trustregion_max_stepback_fraction,
        "xtol": convergence_xtol_abs,
    }

    hessian_instance = _create_hessian_updater_from_user_input(hessian_update_strategy)

    opt = Optimizer(
        fun=criterion_and_derivative,
        lb=lower_bounds,
        ub=upper_bounds,
        verbose=logging.ERROR,
        options=fides_options,
        funargs=None,
        hessian_update=hessian_instance,
        resfun=False,
    )
    raw_res = opt.minimize(x)
    res = _process_fides_res(raw_res, opt)
    return res


def _process_fides_res(raw_res, opt):
    """Create an optimagic results dictionary from the Fides output.

    Args:
        raw_res (tuple): Tuple containing the Fides result
        opt (fides.Optimizer): Fides Optimizer after minimize has been called on it.

    """
    fval, x, grad, hess = raw_res
    res = {
        "solution_criterion": fval,
        "solution_x": x,
        "solution_derivative": grad,
        "solution_hessian": hess,
        "success": opt.converged,
        "n_iterations": opt.iteration,
        "message": _process_exitflag(opt.exitflag),
    }
    return res


def _process_exitflag(exitflag):
    messages = {
        "DID_NOT_RUN": "The optimizer did not run",
        "MAXITER": "Reached maximum number of allowed iterations",
        "MAXTIME": "Expected to reach maximum allowed time in next iteration",
        "NOT_FINITE": "Encountered non-finite fval/grad/hess",
        "EXCEEDED_BOUNDARY": "Exceeded specified boundaries",
        "DELTA_TOO_SMALL": "Trust Region Radius too small to proceed",
        "FTOL": "Converged according to fval difference",
        "XTOL": "Converged according to x difference",
        "GTOL": "Converged according to gradient norm",
    }

    out = messages.get(exitflag.name)

    return out


def _create_hessian_updater_from_user_input(hessian_update_strategy):
    hessians_needing_residuals = (
        hessian_approximation.FX,
        hessian_approximation.SSM,
        hessian_approximation.TSSM,
        hessian_approximation.GNSBFGS,
    )
    unsupported_hess_msg = (
        f"{hessian_update_strategy} not supported because it requires "
        "residuals. Choose one of 'BB', 'BFGS', 'BG', 'DFP' or 'SR1' or pass "
        "an instance of the fides.hessian_approximation.HessianApproximation "
        "class."
    )

    if hessian_update_strategy in ("broyden", "Broyden", "BROYDEN"):
        raise ValueError(
            "You cannot use the Broyden update strategy without specifying the "
            "interpolation parameter phi. Import the Broyden class from "
            "`fides.hessian_approximation`, create an instance of it with your "
            "desired value of phi and pass this instance instead."
        )
    elif isinstance(hessian_update_strategy, str):
        if hessian_update_strategy.lower() in ["fx", "ssm", "tssm", "gnsbfgs"]:
            raise NotImplementedError(unsupported_hess_msg)
        else:
            hessian_name = hessian_update_strategy.upper()
            hessian_class = getattr(hessian_approximation, hessian_name)
            hessian_instance = hessian_class()
    elif isinstance(
        hessian_update_strategy, hessian_approximation.HessianApproximation
    ):
        hessian_instance = hessian_update_strategy
        if isinstance(hessian_instance, hessians_needing_residuals):
            raise NotImplementedError(unsupported_hess_msg)
    else:
        raise TypeError(
            "You must provide a hessian_update_strategy that is either a string or an "
            "instance of the fides.hessian_approximation.HessianApproximation class."
        )
    return hessian_instance
