"""This module implements the POUNDERs algorithm."""

import contextlib
import functools

import numpy as np

from optimagic.config import IS_PETSC4PY_INSTALLED
from optimagic.decorators import mark_minimizer
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import (
    CONVERGENCE_GTOL_ABS,
    CONVERGENCE_GTOL_REL,
    CONVERGENCE_GTOL_SCALED,
    STOPPING_MAXITER,
)
from optimagic.utilities import calculate_trustregion_initial_radius

with contextlib.suppress(ImportError):
    from petsc4py import PETSc


@mark_minimizer(
    name="tao_pounders",
    primary_criterion_entry="root_contributions",
    is_available=IS_PETSC4PY_INSTALLED,
    needs_scaling=True,
)
def tao_pounders(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_gtol_abs=CONVERGENCE_GTOL_ABS,
    convergence_gtol_rel=CONVERGENCE_GTOL_REL,
    convergence_gtol_scaled=CONVERGENCE_GTOL_SCALED,
    trustregion_initial_radius=None,
    stopping_maxiter=STOPPING_MAXITER,
):
    r"""Minimize a function using the POUNDERs algorithm.

    For details see
    :ref: `tao_algorithm`.

    """
    if not IS_PETSC4PY_INSTALLED:
        raise NotInstalledError(
            "The 'tao_pounders' algorithm requires the petsc4py package to be "
            "installed. If you are using Linux or MacOS, install the package with "
            "'conda install -c conda-forge petsc4py. The package is not available on "
            "Windows. Windows users can use optimagics 'pounders' algorithm instead."
        )

    first_eval = criterion(x)
    n_errors = len(first_eval)
    _x = _initialise_petsc_array(x)
    # We need to know the number of contributions of the criterion value to allocate the
    # array.
    residuals_out = _initialise_petsc_array(n_errors)

    # Create the solver object.
    tao = PETSc.TAO().create(PETSc.COMM_WORLD)

    # Set the solver type.
    tao.setType("pounders")

    tao.setFromOptions()

    def func_tao(tao, x, resid_out):  # noqa: ARG001
        """Evaluate objective and attach result to an petsc object f.

        This is required to use the pounders solver from tao.

        Args:
             tao: The tao object we created for the optimization task.
             x (PETSc.array): Current parameter values.
             f: Petsc object in which we save the current function value.

        """
        resid_out.array = criterion(x.array)

    # Set the procedure for calculating the objective. This part has to be changed if we
    # want more than pounders.
    tao.setResidual(func_tao, residuals_out)

    if trustregion_initial_radius is None:
        trustregion_initial_radius = calculate_trustregion_initial_radius(_x)
    elif trustregion_initial_radius <= 0:
        raise ValueError("The initial trust region radius must be > 0.")
    tao.setInitialTrustRegionRadius(trustregion_initial_radius)

    # Add bounds.
    lower_bounds = _initialise_petsc_array(lower_bounds)
    upper_bounds = _initialise_petsc_array(upper_bounds)
    tao.setVariableBounds(lower_bounds, upper_bounds)

    # Put the starting values into the container and pass them to the optimizer.
    tao.setInitial(_x)

    # Obtain tolerances for the convergence criteria. Since we can not create
    # scaled_gradient_tolerance manually we manually set absolute_gradient_tolerance and
    # or relative_gradient_tolerance to zero once a subset of these two is turned off
    # and scaled_gradient_tolerance is still turned on.
    default_gatol = convergence_gtol_abs if convergence_gtol_abs else -1
    default_gttol = convergence_gtol_scaled if convergence_gtol_scaled else -1
    default_grtol = convergence_gtol_rel if convergence_gtol_rel else -1
    # Set tolerances for default convergence tests.
    tao.setTolerances(
        gatol=default_gatol,
        grtol=default_grtol,
        gttol=default_gttol,
    )

    # Set user defined convergence tests. Beware that specifying multiple tests could
    # overwrite others or lead to unclear behavior.
    if stopping_maxiter is not None:
        tao.setConvergenceTest(functools.partial(_max_iters, stopping_maxiter))
    elif convergence_gtol_scaled is False and convergence_gtol_abs is False:
        tao.setConvergenceTest(functools.partial(_grtol_conv, convergence_gtol_rel))
    elif convergence_gtol_rel is False and convergence_gtol_scaled is False:
        tao.setConvergenceTest(functools.partial(_gatol_conv, convergence_gtol_abs))
    elif convergence_gtol_scaled is False:
        tao.setConvergenceTest(
            functools.partial(
                _grtol_gatol_conv,
                convergence_gtol_rel,
                convergence_gtol_abs,
            )
        )

    # Run the problem.
    tao.solve()

    results = _process_pounders_results(residuals_out, tao)

    # Destroy petsc objects for memory reasons.
    for obj in [tao, _x, residuals_out, lower_bounds, upper_bounds]:
        obj.destroy()

    return results


def _initialise_petsc_array(len_or_array):
    """Initialize an empty array or fill in provided values.

    Args:
        len_or_array (int or numpy.ndarray): If the value is an integer, allocate an
            empty array with the given length. If the value is an array, allocate an
            array of equal length and fill in the values.

    """
    length = len_or_array if isinstance(len_or_array, int) else len(len_or_array)

    array = PETSc.Vec().create(PETSc.COMM_WORLD)
    array.setSizes(length)
    array.setFromOptions()

    if isinstance(len_or_array, np.ndarray):
        array.array = len_or_array

    return array


def _max_iters(max_iterations, tao):
    if tao.getSolutionStatus()[0] < max_iterations:
        return 0
    elif tao.getSolutionStatus()[0] >= max_iterations:
        tao.setConvergedReason(8)


def _gatol_conv(absolute_gradient_tolerance, tao):
    if tao.getSolutionStatus()[2] >= absolute_gradient_tolerance:
        return 0
    elif tao.getSolutionStatus()[2] < absolute_gradient_tolerance:
        tao.setConvergedReason(3)


def _grtol_conv(relative_gradient_tolerance, tao):
    if (
        tao.getSolutionStatus()[2] / tao.getSolutionStatus()[1]
        >= relative_gradient_tolerance
    ):
        return 0
    elif (
        tao.getSolutionStatus()[2] / tao.getSolutionStatus()[1]
        < relative_gradient_tolerance
    ):
        tao.setConvergedReason(4)


def _grtol_gatol_conv(relative_gradient_tolerance, absolute_gradient_tolerance, tao):
    if (
        tao.getSolutionStatus()[2] / tao.getSolutionStatus()[1]
        >= relative_gradient_tolerance
    ):
        return 0
    elif (
        tao.getSolutionStatus()[2] / tao.getSolutionStatus()[1]
        < relative_gradient_tolerance
    ):
        tao.setConvergedReason(4)

    elif tao.getSolutionStatus()[2] < absolute_gradient_tolerance:
        tao.setConvergedReason(3)


def _translate_tao_convergence_reason(tao_resaon):
    mapping = {
        3: "absolute_gradient_tolerance below critical value",
        4: "relative_gradient_tolerance below critical value",
        5: "scaled_gradient_tolerance below critical value",
        6: "step size small",
        7: "objective below min value",
        8: "user defined",
        -2: "maxits reached",
        -4: "numerical problems",
        -5: "max funcevals reached",
        -6: "line search failure",
        -7: "trust region failure",
        -8: "user defined",
    }
    return mapping[tao_resaon]


def _process_pounders_results(residuals_out, tao):
    convergence_code = tao.getConvergedReason()
    convergence_reason = _translate_tao_convergence_reason(convergence_code)

    results = {
        "solution_x": tao.solution.array,
        "solution_criterion": tao.function,
        "solution_derivative": None,
        "solution_hessian": None,
        "n_fun_evals": tao.getIterationNumber(),
        "n_jac_evals": None,
        "n_iterations": None,
        "success": bool(convergence_code >= 0),
        "reached_convergence_criterion": (
            convergence_reason if convergence_code >= 0 else None
        ),
        "message": convergence_reason,
        # Further results.
        "solution_criterion_values": residuals_out.array,
        "gradient_norm": tao.gnorm,
        "criterion_norm": tao.cnorm,
        "convergence_code": convergence_code,
    }

    return results
