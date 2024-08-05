"""Implement `nlopt` algorithms.

The documentation is heavily based on (nlopt documentation)[nlopt.readthedocs.io].

"""

import numpy as np

from optimagic.config import IS_NLOPT_INSTALLED
from optimagic.decorators import mark_minimizer
from optimagic.optimization.algo_options import (
    CONVERGENCE_FTOL_ABS,
    CONVERGENCE_FTOL_REL,
    CONVERGENCE_XTOL_ABS,
    CONVERGENCE_XTOL_REL,
    STOPPING_MAXFUN,
    STOPPING_MAXFUN_GLOBAL,
)
from optimagic.parameters.nonlinear_constraints import (
    equality_as_inequality_constraints,
)

if IS_NLOPT_INSTALLED:
    import nlopt


@mark_minimizer(
    name="nlopt_bobyqa",
    primary_criterion_entry="value",
    needs_scaling=False,
    is_available=IS_NLOPT_INSTALLED,
)
def nlopt_bobyqa(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_xtol_rel=CONVERGENCE_XTOL_REL,
    convergence_xtol_abs=CONVERGENCE_XTOL_ABS,
    convergence_ftol_rel=CONVERGENCE_FTOL_REL,
    convergence_ftol_abs=CONVERGENCE_FTOL_ABS,
    stopping_maxfun=STOPPING_MAXFUN,
):
    """Minimize a scalar function using the BOBYQA algorithm.

    For details see
    :ref: `list_of_nlopt_algorithms`.

    """
    out = _minimize_nlopt(
        criterion,
        x,
        lower_bounds,
        upper_bounds,
        derivative=None,
        algorithm=nlopt.LN_BOBYQA,
        convergence_xtol_rel=convergence_xtol_rel,
        convergence_xtol_abs=convergence_xtol_abs,
        convergence_ftol_rel=convergence_ftol_rel,
        convergence_ftol_abs=convergence_ftol_abs,
        stopping_max_eval=stopping_maxfun,
    )

    return out


@mark_minimizer(
    name="nlopt_neldermead",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=IS_NLOPT_INSTALLED,
)
def nlopt_neldermead(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_xtol_rel=CONVERGENCE_XTOL_REL,
    convergence_xtol_abs=CONVERGENCE_XTOL_ABS,
    convergence_ftol_rel=0,
    convergence_ftol_abs=CONVERGENCE_FTOL_ABS,
    stopping_maxfun=STOPPING_MAXFUN,
):
    """Minimize a scalar function using the Nelder-Mead simplex algorithm.

    For details see
    :ref: `list_of_nlopt_algorithms`.

    """

    out = _minimize_nlopt(
        criterion,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LN_NELDERMEAD,
        derivative=None,
        convergence_xtol_rel=convergence_xtol_rel,
        convergence_xtol_abs=convergence_xtol_abs,
        convergence_ftol_rel=convergence_ftol_rel,
        convergence_ftol_abs=convergence_ftol_abs,
        stopping_max_eval=stopping_maxfun,
    )

    return out


@mark_minimizer(
    name="nlopt_praxis",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=IS_NLOPT_INSTALLED,
)
def nlopt_praxis(
    criterion,
    x,
    *,
    convergence_xtol_rel=CONVERGENCE_XTOL_REL,
    convergence_xtol_abs=CONVERGENCE_XTOL_ABS,
    convergence_ftol_rel=CONVERGENCE_FTOL_REL,
    convergence_ftol_abs=CONVERGENCE_FTOL_ABS,
    stopping_maxfun=STOPPING_MAXFUN,
):
    """Minimize a scalar function using principal-axis method.

    For details see
    :ref: `list_of_nlopt_algorithms`.

    """
    out = _minimize_nlopt(
        criterion,
        x,
        lower_bounds=None,
        upper_bounds=None,
        algorithm=nlopt.LN_PRAXIS,
        derivative=None,
        convergence_xtol_rel=convergence_xtol_rel,
        convergence_xtol_abs=convergence_xtol_abs,
        convergence_ftol_rel=convergence_ftol_rel,
        convergence_ftol_abs=convergence_ftol_abs,
        stopping_max_eval=stopping_maxfun,
    )

    return out


@mark_minimizer(
    name="nlopt_cobyla",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=IS_NLOPT_INSTALLED,
)
def nlopt_cobyla(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    nonlinear_constraints=(),
    convergence_xtol_rel=CONVERGENCE_XTOL_REL,
    convergence_xtol_abs=CONVERGENCE_XTOL_ABS,
    convergence_ftol_rel=CONVERGENCE_FTOL_REL,
    convergence_ftol_abs=CONVERGENCE_FTOL_ABS,
    stopping_maxfun=STOPPING_MAXFUN,
):
    """Minimize a scalar function using the cobyla method.

    For details see
    :ref: `list_of_nlopt_algorithms`.

    """

    out = _minimize_nlopt(
        criterion,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LN_COBYLA,
        derivative=None,
        nonlinear_constraints=nonlinear_constraints,
        convergence_xtol_rel=convergence_xtol_rel,
        convergence_xtol_abs=convergence_xtol_abs,
        convergence_ftol_rel=convergence_ftol_rel,
        convergence_ftol_abs=convergence_ftol_abs,
        stopping_max_eval=stopping_maxfun,
    )

    return out


@mark_minimizer(
    name="nlopt_sbplx",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=IS_NLOPT_INSTALLED,
)
def nlopt_sbplx(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_xtol_rel=CONVERGENCE_XTOL_REL,
    convergence_xtol_abs=CONVERGENCE_XTOL_ABS,
    convergence_ftol_rel=CONVERGENCE_FTOL_REL,
    convergence_ftol_abs=CONVERGENCE_FTOL_ABS,
    stopping_maxfun=STOPPING_MAXFUN,
):
    """Minimize a scalar function using the "Subplex" algorithm.

    For details see
    :ref: `list_of_nlopt_algorithms`.

    """

    out = _minimize_nlopt(
        criterion,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LN_SBPLX,
        convergence_xtol_rel=convergence_xtol_rel,
        convergence_xtol_abs=convergence_xtol_abs,
        convergence_ftol_rel=convergence_ftol_rel,
        convergence_ftol_abs=convergence_ftol_abs,
        stopping_max_eval=stopping_maxfun,
    )

    return out


@mark_minimizer(
    name="nlopt_newuoa",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=IS_NLOPT_INSTALLED,
)
def nlopt_newuoa(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_xtol_rel=CONVERGENCE_XTOL_REL,
    convergence_xtol_abs=CONVERGENCE_XTOL_ABS,
    convergence_ftol_rel=CONVERGENCE_FTOL_REL,
    convergence_ftol_abs=CONVERGENCE_FTOL_ABS,
    stopping_maxfun=STOPPING_MAXFUN,
):
    """Minimize a scalar function using the NEWUOA algorithm.

    For details see
    :ref: `list_of_nlopt_algorithms`.

    """
    if np.any(np.isfinite(lower_bounds)) or np.any(np.isfinite(upper_bounds)):
        algo = nlopt.LN_NEWUOA_BOUND
    else:
        algo = nlopt.LN_NEWUOA

    out = _minimize_nlopt(
        criterion,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=algo,
        convergence_xtol_rel=convergence_xtol_rel,
        convergence_xtol_abs=convergence_xtol_abs,
        convergence_ftol_rel=convergence_ftol_rel,
        convergence_ftol_abs=convergence_ftol_abs,
        stopping_max_eval=stopping_maxfun,
    )

    return out


@mark_minimizer(
    name="nlopt_tnewton",
    primary_criterion_entry="value",
    needs_scaling=False,
    is_available=IS_NLOPT_INSTALLED,
)
def nlopt_tnewton(
    criterion,
    derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_xtol_rel=CONVERGENCE_XTOL_REL,
    convergence_xtol_abs=CONVERGENCE_XTOL_ABS,
    convergence_ftol_rel=CONVERGENCE_FTOL_REL,
    convergence_ftol_abs=CONVERGENCE_FTOL_ABS,
    stopping_maxfun=STOPPING_MAXFUN,
):
    """Minimize a scalar function using the "TNEWTON" algorithm.

    For details see
    :ref: `list_of_nlopt_algorithms`.

    """

    out = _minimize_nlopt(
        criterion,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LD_TNEWTON,
        derivative=derivative,
        convergence_xtol_rel=convergence_xtol_rel,
        convergence_xtol_abs=convergence_xtol_abs,
        convergence_ftol_rel=convergence_ftol_rel,
        convergence_ftol_abs=convergence_ftol_abs,
        stopping_max_eval=stopping_maxfun,
    )

    return out


@mark_minimizer(
    name="nlopt_lbfgsb",
    primary_criterion_entry="value",
    needs_scaling=False,
    is_available=IS_NLOPT_INSTALLED,
)
def nlopt_lbfgs(
    criterion,
    derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_xtol_rel=CONVERGENCE_XTOL_REL,
    convergence_xtol_abs=CONVERGENCE_XTOL_ABS,
    convergence_ftol_rel=CONVERGENCE_FTOL_REL,
    convergence_ftol_abs=CONVERGENCE_FTOL_ABS,
    stopping_maxfun=STOPPING_MAXFUN,
):
    """Minimize a scalar function using the "LBFGS" algorithm.

    For details see
    :ref: `list_of_nlopt_algorithms`.

    """

    out = _minimize_nlopt(
        criterion,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LD_TNEWTON,
        derivative=derivative,
        convergence_xtol_rel=convergence_xtol_rel,
        convergence_xtol_abs=convergence_xtol_abs,
        convergence_ftol_rel=convergence_ftol_rel,
        convergence_ftol_abs=convergence_ftol_abs,
        stopping_max_eval=stopping_maxfun,
    )

    return out


@mark_minimizer(
    name="nlopt_ccsaq",
    primary_criterion_entry="value",
    needs_scaling=False,
    is_available=IS_NLOPT_INSTALLED,
)
def nlopt_ccsaq(
    criterion,
    derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_xtol_rel=CONVERGENCE_XTOL_REL,
    convergence_xtol_abs=CONVERGENCE_XTOL_ABS,
    convergence_ftol_rel=CONVERGENCE_FTOL_REL,
    convergence_ftol_abs=CONVERGENCE_FTOL_ABS,
    stopping_maxfun=STOPPING_MAXFUN,
):
    """Minimize a scalar function using CCSAQ algorithm.

    For details see
    :ref: `list_of_nlopt_algorithms`.

    """
    out = _minimize_nlopt(
        criterion,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LD_CCSAQ,
        derivative=derivative,
        convergence_xtol_rel=convergence_xtol_rel,
        convergence_xtol_abs=convergence_xtol_abs,
        convergence_ftol_rel=convergence_ftol_rel,
        convergence_ftol_abs=convergence_ftol_abs,
        stopping_max_eval=stopping_maxfun,
    )

    return out


@mark_minimizer(
    name="nlopt_mma",
    primary_criterion_entry="value",
    needs_scaling=False,
    is_available=IS_NLOPT_INSTALLED,
)
def nlopt_mma(
    criterion,
    derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    nonlinear_constraints=(),
    convergence_xtol_rel=CONVERGENCE_XTOL_REL,
    convergence_xtol_abs=CONVERGENCE_XTOL_ABS,
    convergence_ftol_rel=CONVERGENCE_FTOL_REL,
    convergence_ftol_abs=CONVERGENCE_FTOL_ABS,
    stopping_maxfun=STOPPING_MAXFUN,
):
    """Minimize a scalar function using the method of moving asymptotes (MMA).

    For details see
    :ref: `list_of_nlopt_algorithms`.

    """
    # cannot handle equality constraints
    nonlinear_constraints = equality_as_inequality_constraints(nonlinear_constraints)

    out = _minimize_nlopt(
        criterion,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LD_MMA,
        derivative=derivative,
        nonlinear_constraints=nonlinear_constraints,
        convergence_xtol_rel=convergence_xtol_rel,
        convergence_xtol_abs=convergence_xtol_abs,
        convergence_ftol_rel=convergence_ftol_rel,
        convergence_ftol_abs=convergence_ftol_abs,
        stopping_max_eval=stopping_maxfun,
    )

    return out


@mark_minimizer(
    name="nlopt_var",
    primary_criterion_entry="value",
    needs_scaling=False,
    is_available=IS_NLOPT_INSTALLED,
)
def nlopt_var(
    criterion,
    derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_xtol_rel=CONVERGENCE_XTOL_REL,
    convergence_xtol_abs=CONVERGENCE_XTOL_ABS,
    convergence_ftol_rel=CONVERGENCE_FTOL_REL,
    convergence_ftol_abs=CONVERGENCE_FTOL_ABS,
    stopping_maxfun=STOPPING_MAXFUN,
    rank_1_update=True,
):
    """Minimize a scalar function limited memory switching variable-metric method.

    For details see
    :ref: `list_of_nlopt_algorithms`.

    """
    if rank_1_update:
        algo = nlopt.LD_VAR1
    else:
        algo = nlopt.LD_VAR2
    out = _minimize_nlopt(
        criterion,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=algo,
        derivative=derivative,
        convergence_xtol_rel=convergence_xtol_rel,
        convergence_xtol_abs=convergence_xtol_abs,
        convergence_ftol_rel=convergence_ftol_rel,
        convergence_ftol_abs=convergence_ftol_abs,
        stopping_max_eval=stopping_maxfun,
    )

    return out


@mark_minimizer(
    name="nlopt_slsqp",
    primary_criterion_entry="value",
    needs_scaling=False,
    is_available=IS_NLOPT_INSTALLED,
)
def nlopt_slsqp(
    criterion,
    derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    nonlinear_constraints=(),
    convergence_xtol_rel=CONVERGENCE_XTOL_REL,
    convergence_xtol_abs=CONVERGENCE_XTOL_ABS,
    convergence_ftol_rel=CONVERGENCE_FTOL_REL,
    convergence_ftol_abs=CONVERGENCE_FTOL_ABS,
    stopping_maxfun=STOPPING_MAXFUN,
):
    """Optimize a scalar function based on SLSQP method.

    For details see
    :ref: `list_of_nlopt_algorithms`.

    """
    out = _minimize_nlopt(
        criterion,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.LD_SLSQP,
        derivative=derivative,
        nonlinear_constraints=nonlinear_constraints,
        convergence_xtol_rel=convergence_xtol_rel,
        convergence_xtol_abs=convergence_xtol_abs,
        convergence_ftol_rel=convergence_ftol_rel,
        convergence_ftol_abs=convergence_ftol_abs,
        stopping_max_eval=stopping_maxfun,
    )
    return out


@mark_minimizer(
    name="nlopt_direct",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=IS_NLOPT_INSTALLED,
    is_global=True,
)
def nlopt_direct(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_xtol_rel=CONVERGENCE_XTOL_REL,
    convergence_xtol_abs=CONVERGENCE_XTOL_ABS,
    convergence_ftol_rel=CONVERGENCE_FTOL_REL,
    convergence_ftol_abs=CONVERGENCE_FTOL_ABS,
    stopping_maxfun=STOPPING_MAXFUN_GLOBAL,
    locally_biased=False,
    random_search=False,
    unscaled_bounds=False,
):
    """Optimize a scalar function based on DIRECT method.

    For details see
    :ref: `list_of_nlopt_algorithms`.

    """
    if not locally_biased and not random_search and not unscaled_bounds:
        algo = nlopt.GN_DIRECT
    elif locally_biased and not random_search and not unscaled_bounds:
        algo = nlopt.GN_DIRECT_L
    elif locally_biased and not random_search and unscaled_bounds:
        algo = nlopt.GN_DIRECT_L_NOSCAL
    elif locally_biased and random_search and not unscaled_bounds:
        algo = nlopt.GN_DIRECT_L_RAND
    elif locally_biased and random_search and unscaled_bounds:
        algo = nlopt.GN_DIRECT_L_RAND_NOSCAL
    elif not locally_biased and not random_search and unscaled_bounds:
        algo = nlopt.GN_DIRECT_NOSCAL
    out = _minimize_nlopt(
        criterion,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=algo,
        convergence_xtol_rel=convergence_xtol_rel,
        convergence_xtol_abs=convergence_xtol_abs,
        convergence_ftol_rel=convergence_ftol_rel,
        convergence_ftol_abs=convergence_ftol_abs,
        stopping_max_eval=stopping_maxfun,
    )

    # this is a global optimizer
    out["success"] = None
    return out


@mark_minimizer(
    name="nlopt_esch",
    primary_criterion_entry="value",
    is_global=True,
    needs_scaling=True,
    is_available=IS_NLOPT_INSTALLED,
)
def nlopt_esch(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_xtol_rel=CONVERGENCE_XTOL_REL,
    convergence_xtol_abs=CONVERGENCE_XTOL_ABS,
    convergence_ftol_rel=CONVERGENCE_FTOL_REL,
    convergence_ftol_abs=CONVERGENCE_FTOL_ABS,
    stopping_maxfun=STOPPING_MAXFUN_GLOBAL,
):
    """Optimize a scalar function using the ESCH algorithm.

    For details see
    :ref: `list_of_nlopt_algorithms`.

    """
    out = _minimize_nlopt(
        criterion,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.GN_ESCH,
        convergence_xtol_rel=convergence_xtol_rel,
        convergence_xtol_abs=convergence_xtol_abs,
        convergence_ftol_rel=convergence_ftol_rel,
        convergence_ftol_abs=convergence_ftol_abs,
        stopping_max_eval=stopping_maxfun,
    )

    # this is a global optimizer
    out["success"] = None
    return out


@mark_minimizer(
    name="nlopt_isres",
    primary_criterion_entry="value",
    is_global=True,
    needs_scaling=True,
    is_available=IS_NLOPT_INSTALLED,
)
def nlopt_isres(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    nonlinear_constraints=(),
    convergence_xtol_rel=CONVERGENCE_XTOL_REL,
    convergence_xtol_abs=CONVERGENCE_XTOL_ABS,
    convergence_ftol_rel=CONVERGENCE_FTOL_REL,
    convergence_ftol_abs=CONVERGENCE_FTOL_ABS,
    stopping_maxfun=STOPPING_MAXFUN_GLOBAL,
):
    """Optimize a scalar function using the ISRES algorithm.

    For details see
    :ref: `list_of_nlopt_algorithms`.

    """
    out = _minimize_nlopt(
        criterion,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.GN_ISRES,
        nonlinear_constraints=nonlinear_constraints,
        convergence_xtol_rel=convergence_xtol_rel,
        convergence_xtol_abs=convergence_xtol_abs,
        convergence_ftol_rel=convergence_ftol_rel,
        convergence_ftol_abs=convergence_ftol_abs,
        stopping_max_eval=stopping_maxfun,
    )

    # this is a global optimizer
    out["success"] = None
    return out


@mark_minimizer(
    name="nlopt_crs2_lm",
    primary_criterion_entry="value",
    is_global=True,
    needs_scaling=True,
    is_available=IS_NLOPT_INSTALLED,
)
def nlopt_crs2_lm(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    convergence_xtol_rel=CONVERGENCE_XTOL_REL,
    convergence_xtol_abs=CONVERGENCE_XTOL_ABS,
    convergence_ftol_rel=CONVERGENCE_FTOL_REL,
    convergence_ftol_abs=CONVERGENCE_FTOL_ABS,
    stopping_maxfun=STOPPING_MAXFUN_GLOBAL,
    population_size=None,
):
    """Optimize a scalar function using the CRS2_LM algorithm.

    For details see
    :ref: `list_of_nlopt_algorithms`.

    """
    if population_size is None:
        population_size = 10 * (len(x) + 1)
    out = _minimize_nlopt(
        criterion,
        x,
        lower_bounds,
        upper_bounds,
        algorithm=nlopt.GN_CRS2_LM,
        convergence_xtol_rel=convergence_xtol_rel,
        convergence_xtol_abs=convergence_xtol_abs,
        convergence_ftol_rel=convergence_ftol_rel,
        convergence_ftol_abs=convergence_ftol_abs,
        stopping_max_eval=stopping_maxfun,
        population_size=population_size,
    )

    # this is a global optimizer
    out["success"] = None
    return out


def _minimize_nlopt(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    algorithm,
    *,
    derivative=None,
    nonlinear_constraints=(),
    convergence_xtol_rel=None,
    convergence_xtol_abs=None,
    convergence_ftol_rel=None,
    convergence_ftol_abs=None,
    stopping_max_eval=None,
    population_size=None,
):
    """Run actual nlopt optimization argument, set relevant attributes."""

    def func(x, grad):
        if grad.size > 0:
            criterion_value = criterion(x)
            grad[:] = derivative(x)
        else:
            criterion_value = criterion(x)
        return criterion_value

    opt = nlopt.opt(algorithm, x.shape[0])
    if convergence_ftol_rel is not None:
        opt.set_ftol_rel(convergence_ftol_rel)
    if convergence_ftol_abs is not None:
        opt.set_ftol_abs(convergence_ftol_abs)
    if convergence_xtol_rel is not None:
        opt.set_xtol_rel(convergence_xtol_rel)
    if convergence_xtol_abs is not None:
        opt.set_xtol_abs(convergence_xtol_abs)
    if lower_bounds is not None:
        opt.set_lower_bounds(lower_bounds)
    if upper_bounds is not None:
        opt.set_upper_bounds(upper_bounds)
    if stopping_max_eval is not None:
        opt.set_maxeval(stopping_max_eval)
    if population_size is not None:
        opt.set_population(population_size)
    for constr in _get_nlopt_constraints(nonlinear_constraints, filter_type="eq"):
        opt.add_equality_mconstraint(constr["fun"], constr["tol"])
    for constr in _get_nlopt_constraints(nonlinear_constraints, filter_type="ineq"):
        opt.add_inequality_mconstraint(constr["fun"], constr["tol"])
    opt.set_min_objective(func)
    solution_x = opt.optimize(x)
    return _process_nlopt_results(opt, solution_x)


def _get_nlopt_constraints(constraints, filter_type):
    """Transform internal nonlinear constraints to NLOPT readable format."""
    filtered = [c for c in constraints if c["type"] == filter_type]
    nlopt_constraints = [_internal_to_nlopt_constaint(c) for c in filtered]
    return nlopt_constraints


def _internal_to_nlopt_constaint(c):
    """Sign flip description:

    In optimagic, inequality constraints are internally defined as g(x) >= 0. NLOPT uses
    h(x) <= 0, which is why we need to flip the sign.

    """
    tol = c["tol"]
    if np.isscalar(tol):
        tol = np.tile(tol, c["n_constr"])

    def _constraint(result, x, grad):
        result[:] = -c["fun"](x)  # see docstring for sign flip
        if grad.size > 0:
            grad[:] = -c["jac"](x)  # see docstring for sign flip

    new_constr = {
        "fun": _constraint,
        "tol": tol,
    }
    return new_constr


def _process_nlopt_results(nlopt_obj, solution_x):
    messages = {
        1: "Convergence achieved ",
        2: (
            "Optimizer stopped because maximum value of criterion function was reached"
        ),
        3: (
            "Optimizer stopped because convergence_ftol_rel or "
            "convergence_ftol_abs was reached"
        ),
        4: (
            "Optimizer stopped because convergence_xtol_rel or "
            "convergence_xtol_abs was reached"
        ),
        5: "Optimizer stopped because max_criterion_evaluations was reached",
        6: "Optimizer stopped because max running time was reached",
        -1: "Optimizer failed",
        -2: "Invalid arguments were passed",
        -3: "Memory error",
        -4: "Halted because roundoff errors limited progress",
        -5: "Halted because of user specified forced stop",
    }
    processed = {
        "solution_x": solution_x,
        "solution_criterion": nlopt_obj.last_optimum_value(),
        "solution_derivative": None,
        "solution_hessian": None,
        "n_fun_evals": nlopt_obj.get_numevals(),
        "n_jac_evals": None,
        "n_iterations": None,
        "success": nlopt_obj.last_optimize_result() in [1, 2, 3, 4],
        "message": messages[nlopt_obj.last_optimize_result()],
        "reached_convergence_criterion": None,
    }
    return processed
