"""Implement scipy algorithms.

The following ``scipy`` algorithms are not supported because they
require the specification of the Hessian:

- dogleg
- trust-ncg
- trust-exact
- trust-krylov

The following arguments are not supported as part of ``algo_options``:

- ``disp``
    If set to True would print a convergence message.
    In optimagic it's always set to its default False.
    Refer to optimagic's result dictionary's "success" entry for the convergence
    message.
- ``return_all``
    If set to True, a list of the best solution at each iteration is returned.
    In optimagic it's always set to its default False.
- ``tol``
    This argument of minimize (not an options key) is passed as different types of
    tolerance (gradient, parameter or criterion, as well as relative or absolute)
    depending on the selected algorithm. We require the user to explicitely input
    the tolerance criteria or use our defaults instead.
- ``args``
    This argument of minimize (not an options key) is partialed into the function
    for the user. Specify ``criterion_kwargs`` in ``maximize`` or ``minimize`` to
    achieve the same behavior.
- ``callback``
    This argument would be called after each iteration and the algorithm would
    terminate if it returned True.

"""

import functools

import numpy as np
import scipy
from scipy.optimize import Bounds, NonlinearConstraint

from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.decorators import mark_minimizer
from optimagic.optimization.algo_options import (
    CONVERGENCE_FTOL_ABS,
    CONVERGENCE_FTOL_REL,
    CONVERGENCE_GTOL_ABS,
    CONVERGENCE_GTOL_REL,
    CONVERGENCE_SECOND_BEST_FTOL_ABS,
    CONVERGENCE_SECOND_BEST_XTOL_ABS,
    CONVERGENCE_XTOL_ABS,
    CONVERGENCE_XTOL_REL,
    LIMITED_MEMORY_STORAGE_LENGTH,
    MAX_LINE_SEARCH_STEPS,
    STOPPING_MAXFUN,
    STOPPING_MAXFUN_GLOBAL,
    STOPPING_MAXITER,
)
from optimagic.parameters.nonlinear_constraints import (
    equality_as_inequality_constraints,
    vector_as_list_of_scalar_constraints,
)
from optimagic.utilities import calculate_trustregion_initial_radius


@mark_minimizer(name="scipy_lbfgsb")
def scipy_lbfgsb(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_ftol_rel=CONVERGENCE_FTOL_REL,
    convergence_gtol_abs=CONVERGENCE_GTOL_ABS,
    stopping_maxfun=STOPPING_MAXFUN,
    stopping_maxiter=STOPPING_MAXITER,
    limited_memory_storage_length=LIMITED_MEMORY_STORAGE_LENGTH,
    max_line_search_steps=MAX_LINE_SEARCH_STEPS,
):
    """Minimize a scalar function of one or more variables using the L-BFGS-B algorithm.

    For details see
    :ref: `list_of_scipy_algorithms`.

    """
    options = {
        "maxcor": limited_memory_storage_length,
        "ftol": convergence_ftol_rel,
        "gtol": convergence_gtol_abs,
        "maxfun": stopping_maxfun,
        "maxiter": stopping_maxiter,
        "maxls": max_line_search_steps,
    }
    res = scipy.optimize.minimize(
        fun=criterion_and_derivative,
        x0=x,
        method="L-BFGS-B",
        jac=True,
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        options=options,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_slsqp")
def scipy_slsqp(
    criterion,
    derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    nonlinear_constraints=(),
    convergence_ftol_abs=CONVERGENCE_SECOND_BEST_FTOL_ABS,  # noqa: E501
    stopping_maxiter=STOPPING_MAXITER,
):
    """Minimize a scalar function of one or more variables using the SLSQP algorithm.

    For details see
    :ref: `list_of_scipy_algorithms`.

    """
    options = {
        "maxiter": stopping_maxiter,
        "ftol": convergence_ftol_abs,
    }

    res = scipy.optimize.minimize(
        fun=criterion,
        x0=x,
        method="SLSQP",
        jac=derivative,
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        constraints=nonlinear_constraints,
        options=options,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_neldermead", needs_scaling=True)
def scipy_neldermead(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    stopping_maxiter=STOPPING_MAXITER,
    stopping_maxfun=STOPPING_MAXFUN,
    convergence_ftol_abs=CONVERGENCE_SECOND_BEST_FTOL_ABS,  # noqa: E501
    convergence_xtol_abs=CONVERGENCE_SECOND_BEST_XTOL_ABS,  # noqa: E501
    adaptive=False,
):
    """Minimize a scalar function using the Nelder-Mead algorithm.

    For details see
    :ref: `list_of_scipy_algorithms`.

    """
    options = {
        "maxiter": stopping_maxiter,
        "maxfev": stopping_maxfun,
        # both tolerances seem to have to be fulfilled for Nelder-Mead to converge.
        # if not both are specified it does not converge in our tests.
        "xatol": convergence_xtol_abs,
        "fatol": convergence_ftol_abs,
        "adaptive": adaptive,
    }

    res = scipy.optimize.minimize(
        fun=criterion,
        x0=x,
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        method="Nelder-Mead",
        options=options,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_powell", needs_scaling=True)
def scipy_powell(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_xtol_rel=CONVERGENCE_XTOL_REL,
    convergence_ftol_rel=CONVERGENCE_FTOL_REL,
    stopping_maxfun=STOPPING_MAXFUN,
    stopping_maxiter=STOPPING_MAXITER,
):
    """Minimize a scalar function using the modified Powell method.

    For details see
    :ref: `list_of_scipy_algorithms`.

    """
    options = {
        "xtol": convergence_xtol_rel,
        "ftol": convergence_ftol_rel,
        "maxfev": stopping_maxfun,
        "maxiter": stopping_maxiter,
    }

    res = scipy.optimize.minimize(
        fun=criterion,
        x0=x,
        method="Powell",
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        options=options,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_bfgs")
def scipy_bfgs(
    criterion_and_derivative,
    x,
    *,
    convergence_gtol_abs=CONVERGENCE_GTOL_ABS,
    stopping_maxiter=STOPPING_MAXITER,
    norm=np.inf,
):
    """Minimize a scalar function of one or more variables using the BFGS algorithm.

    For details see
    :ref: `list_of_scipy_algorithms`.

    """
    options = {
        "gtol": convergence_gtol_abs,
        "maxiter": stopping_maxiter,
        "norm": norm,
    }

    res = scipy.optimize.minimize(
        fun=criterion_and_derivative,
        x0=x,
        method="BFGS",
        jac=True,
        options=options,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_conjugate_gradient")
def scipy_conjugate_gradient(
    criterion_and_derivative,
    x,
    *,
    convergence_gtol_abs=CONVERGENCE_GTOL_ABS,
    stopping_maxiter=STOPPING_MAXITER,
    norm=np.inf,
):
    """Minimize a function using a nonlinear conjugate gradient algorithm.

    For details see
    :ref: `list_of_scipy_algorithms`.

    """
    options = {
        "gtol": convergence_gtol_abs,
        "maxiter": stopping_maxiter,
        "norm": norm,
    }

    res = scipy.optimize.minimize(
        fun=criterion_and_derivative,
        x0=x,
        method="CG",
        jac=True,
        options=options,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_newton_cg")
def scipy_newton_cg(
    criterion_and_derivative,
    x,
    *,
    convergence_xtol_rel=CONVERGENCE_XTOL_REL,
    stopping_maxiter=STOPPING_MAXITER,
):
    """Minimize a scalar function using Newton's conjugate gradient algorithm.

    For details see
    :ref: `list_of_scipy_algorithms`.

    """
    options = {
        "xtol": convergence_xtol_rel,
        "maxiter": stopping_maxiter,
    }

    res = scipy.optimize.minimize(
        fun=criterion_and_derivative,
        x0=x,
        method="Newton-CG",
        jac=True,
        options=options,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_cobyla", needs_scaling=True)
def scipy_cobyla(
    criterion,
    x,
    *,
    nonlinear_constraints=(),
    stopping_maxiter=STOPPING_MAXITER,
    convergence_xtol_rel=CONVERGENCE_XTOL_REL,
    trustregion_initial_radius=None,
):
    """Minimize a scalar function of one or more variables using the COBYLA algorithm.

    For details see
    :ref: `list_of_scipy_algorithms`.

    """
    if trustregion_initial_radius is None:
        trustregion_initial_radius = calculate_trustregion_initial_radius(x)

    options = {"maxiter": stopping_maxiter, "rhobeg": trustregion_initial_radius}

    # cannot handle equality constraints
    nonlinear_constraints = equality_as_inequality_constraints(nonlinear_constraints)

    res = scipy.optimize.minimize(
        fun=criterion,
        x0=x,
        method="COBYLA",
        constraints=nonlinear_constraints,
        options=options,
        tol=convergence_xtol_rel,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_truncated_newton")
def scipy_truncated_newton(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    stopping_maxfun=STOPPING_MAXFUN,
    convergence_ftol_abs=CONVERGENCE_FTOL_ABS,
    convergence_xtol_abs=CONVERGENCE_XTOL_ABS,
    convergence_gtol_abs=CONVERGENCE_GTOL_ABS,
    func_min_estimate=0,
    max_hess_evaluations_per_iteration=-1,
    max_step_for_line_search=0,
    line_search_severity=-1,
    finitie_difference_precision=0,
    criterion_rescale_factor=-1,
):
    """Minimize a scalar function using truncated Newton algorithm.

    For details see
    :ref: `list_of_scipy_algorithms`.

    """
    options = {
        # scipy/optimize/tnc/tnc.c::809 and 844 show that ftol is the
        # absolute criterion tolerance
        "ftol": convergence_ftol_abs,
        # scipy/optimize/tnc/tnc.c::856 show sthat xtol is the absolute parameter
        # tolerance
        "xtol": convergence_xtol_abs,
        "gtol": convergence_gtol_abs,
        "maxfun": stopping_maxfun,
        "maxCGit": max_hess_evaluations_per_iteration,
        "stepmx": max_step_for_line_search,
        "minfev": func_min_estimate,
        "eta": line_search_severity,
        "accuracy": finitie_difference_precision,
        "rescale": criterion_rescale_factor,
    }

    res = scipy.optimize.minimize(
        fun=criterion_and_derivative,
        x0=x,
        method="TNC",
        jac=True,
        options=options,
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_trust_constr")
def scipy_trust_constr(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    nonlinear_constraints=(),
    convergence_gtol_abs=1e-08,
    convergence_xtol_rel=CONVERGENCE_XTOL_REL,
    stopping_maxiter=STOPPING_MAXITER,
    trustregion_initial_radius=None,
):
    """Minimize a scalar function of one or more variables subject to constraints.

    For details see
    :ref: `list_of_scipy_algorithms`.

    """
    if trustregion_initial_radius is None:
        trustregion_initial_radius = calculate_trustregion_initial_radius(x)

    options = {
        "gtol": convergence_gtol_abs,
        "maxiter": stopping_maxiter,
        "xtol": convergence_xtol_rel,
        "initial_tr_radius": trustregion_initial_radius,
    }

    # cannot handle equality constraints
    nonlinear_constraints = equality_as_inequality_constraints(nonlinear_constraints)

    res = scipy.optimize.minimize(
        fun=criterion_and_derivative,
        jac=True,
        x0=x,
        method="trust-constr",
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        constraints=_get_scipy_constraints(nonlinear_constraints),
        options=options,
    )

    return process_scipy_result(res)


def process_scipy_result(scipy_results_obj):
    # using get with defaults to access dict elements is just a safety measure
    raw_res = {**scipy_results_obj}
    processed = {
        "solution_x": raw_res.get("x"),
        "solution_criterion": raw_res.get("fun"),
        "solution_derivative": raw_res.get("jac"),
        "solution_hessian": raw_res.get("hess"),
        "n_fun_evals": raw_res.get("nfev"),
        "n_jac_evals": raw_res.get("njac") or raw_res.get("njev"),
        "n_iterations": raw_res.get("nit"),
        "success": raw_res.get("success"),
        "reached_convergence_criterion": None,
        "message": raw_res.get("message"),
    }
    return processed


def _get_scipy_bounds(lower_bounds, upper_bounds):
    return Bounds(lb=lower_bounds, ub=upper_bounds)


def _get_scipy_constraints(constraints):
    """Transform internal nonlinear constraints to scipy readable format.

    This format is currently only used by scipy_trust_constr.

    """
    scipy_constraints = [_internal_to_scipy_constraint(c) for c in constraints]
    return scipy_constraints


def _internal_to_scipy_constraint(c):
    new_constr = NonlinearConstraint(
        fun=c["fun"],
        lb=np.zeros(c["n_constr"]),
        ub=np.tile(np.inf, c["n_constr"]),
        jac=c["jac"],
    )
    return new_constr


def _scipy_least_squares(
    criterion,
    derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_ftol_rel=CONVERGENCE_FTOL_REL,
    convergence_gtol_rel=CONVERGENCE_GTOL_REL,
    stopping_maxfun=STOPPING_MAXFUN,
    relative_step_size_diff_approx=None,
    tr_solver=None,
    tr_solver_options=None,
    method="trf",
):
    """Internal function used by the scipy_ls_trf and scipy_ls_dogbox functions."""

    if method not in ["trf", "dogbox", "lm"]:
        raise ValueError(
            f"Method {method} is not supported within scipy_least_squares."
        )

    if tr_solver_options is None:
        tr_solver_options = {}

    res = scipy.optimize.least_squares(
        fun=criterion,
        x0=x,
        jac=derivative,
        # Don't use get_scipy_bounds, b.c. least_squares uses np.inf
        bounds=(lower_bounds, upper_bounds),
        max_nfev=stopping_maxfun,
        ftol=convergence_ftol_rel,
        gtol=convergence_gtol_rel,
        method=method,
        diff_step=relative_step_size_diff_approx,
        tr_solver=tr_solver,
        tr_options=tr_solver_options,
    )

    return process_scipy_result(res)


_scipy_ls_trf = functools.partial(_scipy_least_squares, method="trf")
scipy_ls_trf = mark_minimizer(
    _scipy_ls_trf, name="scipy_ls_trf", primary_criterion_entry="root_contributions"
)

_scipy_ls_dogbox = functools.partial(_scipy_least_squares, method="dogbox")
scipy_ls_dogbox = mark_minimizer(
    _scipy_ls_dogbox,
    name="scipy_ls_dogbox",
    primary_criterion_entry="root_contributions",
)


@mark_minimizer(name="scipy_ls_lm", primary_criterion_entry="root_contributions")
def scipy_ls_lm(
    criterion,
    derivative,
    x,
    *,
    convergence_ftol_rel=CONVERGENCE_FTOL_REL,
    convergence_gtol_rel=CONVERGENCE_GTOL_REL,
    stopping_maxfun=STOPPING_MAXFUN,
    relative_step_size_diff_approx=None,
    tr_solver=None,
    tr_solver_options=None,
):
    """Internal function used by the scipy_ls_trf and scipy_ls_dogbox functions."""

    if tr_solver_options is None:
        tr_solver_options = {}

    res = scipy.optimize.least_squares(
        fun=criterion,
        x0=x,
        jac=derivative,
        max_nfev=stopping_maxfun,
        ftol=convergence_ftol_rel,
        gtol=convergence_gtol_rel,
        method="lm",
        diff_step=relative_step_size_diff_approx,
        tr_solver=tr_solver,
        tr_options=tr_solver_options,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_basinhopping", is_global=True)
def scipy_basinhopping(
    criterion,
    derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    local_algorithm="L-BFGS-B",
    n_local_optimizations=100,
    temperature=1.0,
    stepsize=0.5,
    local_algo_options=None,
    take_step=None,
    accept_test=None,
    interval=50,
    convergence_n_unchanged_iterations=None,
    seed=None,
    target_accept_rate=0.5,
    stepwise_factor=0.9,
):
    """Find the global minimum of a function using the basin-hopping algorithm.

    For details see
    :ref: `list_of_scipy_algorithms`.

    """
    n_local_optimizations = max(1, n_local_optimizations - 1)
    local_algo_options = {} if local_algo_options is None else local_algo_options
    default_minimizer_kwargs = {
        "method": local_algorithm,
        "bounds": _get_scipy_bounds(lower_bounds, upper_bounds),
        "jac": derivative,
    }

    minimizer_kwargs = {**default_minimizer_kwargs, **local_algo_options}

    res = scipy.optimize.basinhopping(
        func=criterion,
        x0=x,
        minimizer_kwargs=minimizer_kwargs,
        niter=n_local_optimizations,
        T=temperature,
        stepsize=stepsize,
        take_step=take_step,
        accept_test=accept_test,
        interval=interval,
        niter_success=convergence_n_unchanged_iterations,
        seed=seed,
        target_accept_rate=target_accept_rate,
        stepwise_factor=stepwise_factor,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_brute", is_global=True, disable_history=True)
def scipy_brute(
    criterion,
    lower_bounds,
    upper_bounds,
    x,  # noqa: ARG001
    *,
    n_grid_points=20,
    polishing_function=None,
    n_cores=1,
    batch_evaluator="joblib",
):
    """Minimize a function over a given range by brute force.

    For details see :ref:`list_of_scipy_algorithms`.

    Lower and upper bounds are passed to range where each component
    of the ranges tuple must be either a “slice object” or a range tuple
    of the form (low, high). The program uses these to create the
    grid of points on which the objective function will be computed.

    full_output always true.
    polishing func none to stay inside bounds, bounds - ranges difference.

    """
    workers = _get_workers(n_cores, batch_evaluator)

    res = scipy.optimize.brute(
        func=criterion,
        ranges=tuple(map(tuple, np.column_stack([lower_bounds, upper_bounds]))),
        Ns=n_grid_points,
        full_output=True,
        finish=polishing_function,
        workers=workers,
    )
    out = {
        "solution_x": res[0],
        "solution_criterion": res[1],
        "n_fun_evals": res[2].size,
        "n_iterations": res[2].size,
        "success": True,
        "message": "brute force optimization terminated successfully",
    }

    return out


@mark_minimizer(
    name="scipy_differential_evolution", is_global=True, disable_history=True
)
def scipy_differential_evolution(
    criterion,
    lower_bounds,
    upper_bounds,
    x,  # noqa: ARG001
    nonlinear_constraints,
    *,
    strategy="best1bin",
    stopping_maxiter=STOPPING_MAXFUN_GLOBAL,
    population_size_multiplier=15,
    convergence_ftol_rel=0.01,
    mutation_constant=(0.5, 1),
    recombination_constant=0.7,
    seed=None,
    polish=True,
    sampling_method="latinhypercube",
    convergence_ftol_abs=CONVERGENCE_SECOND_BEST_FTOL_ABS,  # noqa: E501
    n_cores=1,
    batch_evaluator="joblib",
):
    """Finds the global minimum of a multivariate function.

    For details see
    :ref: `list_of_scipy_algorithms`.

    """
    workers = _get_workers(n_cores, batch_evaluator)
    res = scipy.optimize.differential_evolution(
        func=criterion,
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        strategy=strategy,
        maxiter=stopping_maxiter,
        popsize=population_size_multiplier,
        tol=convergence_ftol_rel,
        mutation=mutation_constant,
        recombination=recombination_constant,
        seed=seed,
        polish=polish,
        init=sampling_method,
        atol=convergence_ftol_abs,
        updating="deferred",
        workers=workers,
        constraints=_get_scipy_constraints(nonlinear_constraints),
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_shgo", is_global=True)
def scipy_shgo(
    criterion,
    lower_bounds,
    upper_bounds,
    x,  # noqa: ARG001
    derivative,
    nonlinear_constraints,
    *,
    local_algorithm="L-BFGS-B",
    local_algo_options=None,
    n_sampling_points=128,
    n_simplex_iterations=1,
    sampling_method="simplicial",
    max_sampling_evaluations=None,
    convergence_minimum_criterion_value=None,
    convergence_minimum_criterion_tolerance=1e-4,
    stopping_maxiter=None,
    stopping_maxfun=STOPPING_MAXFUN_GLOBAL,
    stopping_max_processing_time=None,
    minimum_homology_group_rank_differential=None,
    symmetry=False,
    minimize_every_iteration=True,
    max_local_minimizations_per_iteration=False,
    infinity_constraints=True,
):
    """Finds the global minimum of a function using SHG optimization.

    SHGO stands for “simplicial homology global optimization”.

    For details see

    :ref: `list_of_scipy_algorithms`.

    """
    if local_algorithm == "COBYLA":
        nonlinear_constraints = equality_as_inequality_constraints(
            nonlinear_constraints
        )

    nonlinear_constraints = vector_as_list_of_scalar_constraints(nonlinear_constraints)

    local_algo_options = {} if local_algo_options is None else local_algo_options
    default_minimizer_kwargs = {
        "method": local_algorithm,
        "bounds": _get_scipy_bounds(lower_bounds, upper_bounds),
        "jac": derivative,
    }

    minimizer_kwargs = {**default_minimizer_kwargs, **local_algo_options}
    options = {
        "maxfev": max_sampling_evaluations,
        "f_min": convergence_minimum_criterion_value,
        "f_tol": convergence_minimum_criterion_tolerance,
        "maxiter": stopping_maxiter,
        "maxev": stopping_maxfun,
        "maxtime": stopping_max_processing_time,
        "minhgrd": minimum_homology_group_rank_differential,
        "symmetry": symmetry,
        "jac": derivative,
        "minimize_every_iter": minimize_every_iteration,
        "local_iter": max_local_minimizations_per_iteration,
        "infty_constraints": infinity_constraints,
    }

    if any(options.values()) is False:
        options = None

    res = scipy.optimize.shgo(
        func=criterion,
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        constraints=nonlinear_constraints,
        minimizer_kwargs=minimizer_kwargs,
        n=n_sampling_points,
        iters=n_simplex_iterations,
        sampling_method=sampling_method,
        options=options,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_dual_annealing", is_global=True)
def scipy_dual_annealing(
    criterion,
    derivative,
    lower_bounds,
    upper_bounds,
    x,
    *,
    stopping_maxiter=STOPPING_MAXFUN_GLOBAL,
    local_algorithm="L-BFGS-B",
    local_algo_options=None,
    initial_temperature=5230.0,
    restart_temperature_ratio=2e-05,
    visit=2.62,
    accept=-5.0,
    stopping_maxfun=STOPPING_MAXFUN,
    seed=None,
    no_local_search=False,
):
    """Find the global minimum of a function using Dual Annealing.

    For details see
    :ref: `list_of_scipy_algorithms`.

    """
    local_algo_options = {} if local_algo_options is None else local_algo_options
    default_minimizer_kwargs = {
        "method": local_algorithm,
        "bounds": _get_scipy_bounds(lower_bounds, upper_bounds),
        "jac": derivative,
    }

    minimizer_kwargs = {**default_minimizer_kwargs, **local_algo_options}

    res = scipy.optimize.dual_annealing(
        func=criterion,
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        maxiter=stopping_maxiter,
        minimizer_kwargs=minimizer_kwargs,
        initial_temp=initial_temperature,
        restart_temp_ratio=restart_temperature_ratio,
        visit=visit,
        accept=accept,
        maxfun=stopping_maxfun,
        seed=seed,
        no_local_search=no_local_search,
        x0=x,
    )

    return process_scipy_result(res)


@mark_minimizer(name="scipy_direct", is_global=True)
def scipy_direct(
    criterion,
    lower_bounds,
    upper_bounds,
    x,  # noqa: ARG001
    *,
    convergence_ftol_rel=CONVERGENCE_FTOL_REL,
    stopping_maxfun=STOPPING_MAXFUN,
    stopping_maxiter=STOPPING_MAXFUN_GLOBAL,
    locally_biased=True,
    convergence_minimum_criterion_value=-np.inf,
    convergence_minimum_criterion_tolerance=1e-4,
    volume_hyperrectangle_tolerance=1e-16,
    length_hyperrectangle_tolerance=1e-6,
):
    """Finds the global minimum of a function using the DIRECT algorithm.

    For details see
    :ref: `list_of_scipy_algorithms`.

    """

    res = scipy.optimize.direct(
        func=criterion,
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        eps=convergence_ftol_rel,
        maxfun=stopping_maxfun,
        maxiter=stopping_maxiter,
        locally_biased=locally_biased,
        f_min=convergence_minimum_criterion_value,
        f_min_rtol=convergence_minimum_criterion_tolerance,
        vol_tol=volume_hyperrectangle_tolerance,
        len_tol=length_hyperrectangle_tolerance,
    )

    return process_scipy_result(res)


def _get_workers(n_cores, batch_evaluator):
    batch_evaluator = process_batch_evaluator(batch_evaluator)
    out = functools.partial(
        batch_evaluator,
        n_cores=n_cores,
        error_handling="raise",
    )
    return out
