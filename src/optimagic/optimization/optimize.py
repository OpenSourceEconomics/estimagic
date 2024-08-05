"""Public functions for optimization.

This module defines the public functions `maximize` and `minimize` that will be called
by users.

Internally, `maximize` and `minimize` just call `create_optimization_problem` with
all arguments and add the `direction`. In `create_optimization_problem`, the user input
is consolidated and converted to stricter types.  The resulting `OptimizationProblem`
is then passed to `_optimize` which handles the optimization logic.

`_optimize` processes the optimization problem and performs the actual optimization.

"""

import functools
import warnings
from pathlib import Path

from optimagic.exceptions import (
    InvalidFunctionError,
    InvalidKwargsError,
)
from optimagic.logging.create_tables import (
    make_optimization_iteration_table,
    make_optimization_problem_table,
    make_steps_table,
)
from optimagic.logging.load_database import load_database
from optimagic.logging.write_to_database import append_row
from optimagic.optimization.create_optimization_problem import (
    OptimizationProblem,
    create_optimization_problem,
)
from optimagic.optimization.error_penalty import get_error_penalty_function
from optimagic.optimization.get_algorithm import (
    get_final_algorithm,
)
from optimagic.optimization.internal_criterion_template import (
    internal_criterion_and_derivative_template,
)
from optimagic.optimization.multistart import (
    run_multistart_optimization,
)
from optimagic.optimization.multistart_options import (
    get_internal_multistart_options_from_public,
)
from optimagic.optimization.optimization_logging import log_scheduled_steps_and_get_ids
from optimagic.optimization.optimize_result import OptimizeResult
from optimagic.optimization.process_results import process_internal_optimizer_result
from optimagic.parameters.bounds import Bounds
from optimagic.parameters.conversion import (
    aggregate_func_output_to_value,
    get_converter,
)
from optimagic.parameters.nonlinear_constraints import process_nonlinear_constraints


def maximize(
    fun=None,
    params=None,
    algorithm=None,
    *,
    bounds=None,
    constraints=None,
    fun_kwargs=None,
    algo_options=None,
    jac=None,
    jac_kwargs=None,
    fun_and_jac=None,
    fun_and_jac_kwargs=None,
    numdiff_options=None,
    logging=False,
    log_options=None,
    error_handling="raise",
    error_penalty=None,
    scaling=False,
    multistart=False,
    collect_history=True,
    skip_checks=False,
    # scipy aliases
    x0=None,
    method=None,
    args=None,
    # scipy arguments that are not yet supported
    hess=None,
    hessp=None,
    callback=None,
    # scipy arguments that will never be supported
    options=None,
    tol=None,
    # deprecated arguments
    criterion=None,
    criterion_kwargs=None,
    derivative=None,
    derivative_kwargs=None,
    criterion_and_derivative=None,
    criterion_and_derivative_kwargs=None,
    lower_bounds=None,
    upper_bounds=None,
    soft_lower_bounds=None,
    soft_upper_bounds=None,
    scaling_options=None,
    multistart_options=None,
):
    """Maximize fun using algorithm subject to constraints.

    TODO: Write docstring after enhancement proposals are implemented.

    Args:
        bounds: Lower and upper bounds on the parameters. The most general and preferred
            way to specify bounds is an `optimagic.Bounds` object that collects lower,
            upper, soft_lower and soft_upper bounds. The soft bounds are used for
            sampling based optimizers but are not enforced during optimization. Each
            bound type mirrors the structure of params. Check our how-to guide on bounds
            for examples. If params is a flat numpy array, you can also provide bounds
            via any format that is supported by scipy.optimize.minimize.

    """
    problem = create_optimization_problem(
        direction="maximize",
        fun=fun,
        params=params,
        bounds=bounds,
        algorithm=algorithm,
        fun_kwargs=fun_kwargs,
        constraints=constraints,
        algo_options=algo_options,
        jac=jac,
        jac_kwargs=jac_kwargs,
        fun_and_jac=fun_and_jac,
        fun_and_jac_kwargs=fun_and_jac_kwargs,
        numdiff_options=numdiff_options,
        logging=logging,
        log_options=log_options,
        error_handling=error_handling,
        error_penalty=error_penalty,
        scaling=scaling,
        multistart=multistart,
        collect_history=collect_history,
        skip_checks=skip_checks,
        # scipy aliases
        x0=x0,
        method=method,
        args=args,
        # scipy arguments that are not yet supported
        hess=hess,
        hessp=hessp,
        callback=callback,
        # scipy arguments that will never be supported
        options=options,
        tol=tol,
        # deprecated arguments
        criterion=criterion,
        criterion_kwargs=criterion_kwargs,
        derivative=derivative,
        derivative_kwargs=derivative_kwargs,
        criterion_and_derivative=criterion_and_derivative,
        criterion_and_derivative_kwargs=criterion_and_derivative_kwargs,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        soft_lower_bounds=soft_lower_bounds,
        soft_upper_bounds=soft_upper_bounds,
        scaling_options=scaling_options,
        multistart_options=multistart_options,
    )
    return _optimize(problem)


def minimize(
    fun=None,
    params=None,
    algorithm=None,
    *,
    bounds=None,
    constraints=None,
    fun_kwargs=None,
    algo_options=None,
    jac=None,
    jac_kwargs=None,
    fun_and_jac=None,
    fun_and_jac_kwargs=None,
    numdiff_options=None,
    logging=False,
    log_options=None,
    error_handling="raise",
    error_penalty=None,
    scaling=False,
    multistart=False,
    collect_history=True,
    skip_checks=False,
    # scipy aliases
    x0=None,
    method=None,
    args=None,
    # scipy arguments that are not yet supported
    hess=None,
    hessp=None,
    callback=None,
    # scipy arguments that will never be supported
    options=None,
    tol=None,
    # deprecated arguments
    criterion=None,
    criterion_kwargs=None,
    derivative=None,
    derivative_kwargs=None,
    criterion_and_derivative=None,
    criterion_and_derivative_kwargs=None,
    lower_bounds=None,
    upper_bounds=None,
    soft_lower_bounds=None,
    soft_upper_bounds=None,
    scaling_options=None,
    multistart_options=None,
):
    """Minimize criterion using algorithm subject to constraints.

    TODO: Write docstring after enhancement proposals are implemented.

    Args:
        bounds: Lower and upper bounds on the parameters. The most general and preferred
            way to specify bounds is an `optimagic.Bounds` object that collects lower,
            upper, soft_lower and soft_upper bounds. The soft bounds are used for
            sampling based optimizers but are not enforced during optimization. Each
            bound type mirrors the structure of params. Check our how-to guide on bounds
            for examples. If params is a flat numpy array, you can also provide bounds
            via any format that is supported by scipy.optimize.minimize.

    """

    problem = create_optimization_problem(
        direction="minimize",
        fun=fun,
        params=params,
        algorithm=algorithm,
        bounds=bounds,
        fun_kwargs=fun_kwargs,
        constraints=constraints,
        algo_options=algo_options,
        jac=jac,
        jac_kwargs=jac_kwargs,
        fun_and_jac=fun_and_jac,
        fun_and_jac_kwargs=fun_and_jac_kwargs,
        numdiff_options=numdiff_options,
        logging=logging,
        log_options=log_options,
        error_handling=error_handling,
        error_penalty=error_penalty,
        scaling=scaling,
        multistart=multistart,
        collect_history=collect_history,
        skip_checks=skip_checks,
        # scipy aliases
        x0=x0,
        method=method,
        args=args,
        # scipy arguments that are not yet supported
        hess=hess,
        hessp=hessp,
        callback=callback,
        # scipy arguments that will never be supported
        options=options,
        tol=tol,
        # deprecated arguments
        criterion=criterion,
        criterion_kwargs=criterion_kwargs,
        derivative=derivative,
        derivative_kwargs=derivative_kwargs,
        criterion_and_derivative=criterion_and_derivative,
        criterion_and_derivative_kwargs=criterion_and_derivative_kwargs,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        soft_lower_bounds=soft_lower_bounds,
        soft_upper_bounds=soft_upper_bounds,
        scaling_options=scaling_options,
        multistart_options=multistart_options,
    )
    return _optimize(problem)


def _optimize(problem: OptimizationProblem) -> OptimizeResult:
    """Solve an optimization problem."""
    # ==================================================================================
    # Split constraints into nonlinear and reparametrization parts
    # ==================================================================================
    constraints = problem.constraints
    if isinstance(constraints, dict):
        constraints = [constraints]

    nonlinear_constraints = [c for c in constraints if c["type"] == "nonlinear"]

    algo_kwargs = set(problem.algo_info.arguments)
    if nonlinear_constraints and "nonlinear_constraints" not in algo_kwargs:
        raise ValueError(
            f"Algorithm {problem.algo_info.name} does not support nonlinear "
            "constraints."
        )

    # the following constraints will be handled via reparametrization
    constraints = [c for c in constraints if c["type"] != "nonlinear"]

    # ==================================================================================
    # Do first evaluation of user provided functions
    # ==================================================================================
    try:
        first_crit_eval = problem.fun(problem.params)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        msg = "Error while evaluating criterion at start params."
        raise InvalidFunctionError(msg) from e

    # do first derivative evaluation (if given)
    if problem.jac is not None:
        try:
            first_deriv_eval = problem.jac(problem.params)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            msg = "Error while evaluating derivative at start params."
            raise InvalidFunctionError(msg) from e

    if problem.fun_and_jac is not None:
        try:
            first_crit_and_deriv_eval = problem.fun_and_jac(problem.params)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            msg = "Error while evaluating criterion_and_derivative at start params."
            raise InvalidFunctionError(msg) from e

    if problem.jac is not None:
        used_deriv = first_deriv_eval
    elif problem.fun_and_jac is not None:
        used_deriv = first_crit_and_deriv_eval[1]
    else:
        used_deriv = None

    # ==================================================================================
    # Get the converter (for tree flattening, constraints and scaling)
    # ==================================================================================
    converter, internal_params = get_converter(
        params=problem.params,
        constraints=constraints,
        bounds=problem.bounds,
        func_eval=first_crit_eval,
        primary_key=problem.algo_info.primary_criterion_entry,
        scaling=problem.scaling,
        derivative_eval=used_deriv,
        add_soft_bounds=problem.multistart is not None,
    )

    # ==================================================================================
    # initialize the log database
    # ==================================================================================
    if problem.logging:
        # TODO: We want to remove the optimization_problem table completely but we
        # probably do need to store the start parameters in the database because it is
        # used by the log reader.
        problem_data = {
            "direction": problem.direction,
            "params": problem.params,
        }
        database = _create_and_initialize_database(
            logging=problem.logging,
            log_options=problem.log_options,
            problem_data=problem_data,
        )
    else:
        database = None

    # ==================================================================================
    # Do some things that require internal parameters or bounds
    # ==================================================================================

    if converter.has_transforming_constraints and problem.multistart is not None:
        raise NotImplementedError(
            "multistart optimizations are not yet compatible with transforming "
            "constraints."
        )

    numdiff_options = _fill_numdiff_options_with_defaults(
        numdiff_options=problem.numdiff_options,
        lower_bounds=internal_params.lower_bounds,
        upper_bounds=internal_params.upper_bounds,
    )

    # get error penalty function
    error_penalty_func = get_error_penalty_function(
        error_handling=problem.error_handling,
        start_x=internal_params.values,
        start_criterion=converter.func_to_internal(first_crit_eval),
        error_penalty=problem.error_penalty,
        primary_key=problem.algo_info.primary_criterion_entry,
        direction=problem.direction,
    )

    # process nonlinear constraints:
    internal_constraints = process_nonlinear_constraints(
        nonlinear_constraints=nonlinear_constraints,
        params=problem.params,
        converter=converter,
        numdiff_options=numdiff_options,
        skip_checks=problem.skip_checks,
    )

    x = internal_params.values
    # ==================================================================================
    # get the internal algorithm
    # ==================================================================================
    internal_algorithm = get_final_algorithm(
        raw_algorithm=problem.algorithm,
        algo_info=problem.algo_info,
        valid_kwargs=algo_kwargs,
        lower_bounds=internal_params.lower_bounds,
        upper_bounds=internal_params.upper_bounds,
        nonlinear_constraints=internal_constraints,
        algo_options=problem.algo_options,
        logging=problem.logging,
        database=database,
        collect_history=problem.collect_history,
    )
    # ==================================================================================
    # partial arguments into the internal_criterion_and_derivative_template
    # ==================================================================================
    to_partial = {
        "direction": problem.direction,
        "criterion": problem.fun,
        "converter": converter,
        "derivative": problem.jac,
        "criterion_and_derivative": problem.fun_and_jac,
        "numdiff_options": numdiff_options,
        "logging": problem.logging,
        "database": database,
        "algo_info": problem.algo_info,
        "error_handling": problem.error_handling,
        "error_penalty_func": error_penalty_func,
    }

    internal_criterion_and_derivative = functools.partial(
        internal_criterion_and_derivative_template,
        **to_partial,
    )

    problem_functions = {}
    for task in ["criterion", "derivative", "criterion_and_derivative"]:
        if task in algo_kwargs:
            problem_functions[task] = functools.partial(
                internal_criterion_and_derivative,
                task=task,
            )

    # ==================================================================================
    # Do actual optimization
    # ==================================================================================
    if problem.multistart is None:
        steps = [{"type": "optimization", "name": "optimization"}]

        step_ids = log_scheduled_steps_and_get_ids(
            steps=steps,
            logging=problem.logging,
            database=database,
        )

        raw_res = internal_algorithm(**problem_functions, x=x, step_id=step_ids[0])
    else:
        multistart_options = get_internal_multistart_options_from_public(
            options=problem.multistart,
            params=problem.params,
            params_to_internal=converter.params_to_internal,
        )

        raw_res = run_multistart_optimization(
            local_algorithm=internal_algorithm,
            primary_key=problem.algo_info.primary_criterion_entry,
            problem_functions=problem_functions,
            x=x,
            lower_sampling_bounds=internal_params.soft_lower_bounds,
            upper_sampling_bounds=internal_params.soft_upper_bounds,
            options=multistart_options,
            logging=problem.logging,
            database=database,
            error_handling=problem.error_handling,
        )

    # ==================================================================================
    # Process the result
    # ==================================================================================

    _scalar_start_criterion = aggregate_func_output_to_value(
        converter.func_to_internal(first_crit_eval),
        problem.algo_info.primary_criterion_entry,
    )

    fixed_result_kwargs = {
        "start_fun": _scalar_start_criterion,
        "start_params": problem.params,
        "algorithm": problem.algo_info.name,
        "direction": problem.direction,
        "n_free": internal_params.free_mask.sum(),
    }

    res = process_internal_optimizer_result(
        raw_res,
        converter=converter,
        primary_key=problem.algo_info.primary_criterion_entry,
        fixed_kwargs=fixed_result_kwargs,
        skip_checks=problem.skip_checks,
    )

    return res


def _create_and_initialize_database(logging, log_options, problem_data):
    """Create and initialize to sqlite database for logging."""
    path = Path(logging)
    fast_logging = log_options.get("fast_logging", False)
    if_table_exists = log_options.get("if_table_exists", "extend")
    if_database_exists = log_options.get("if_database_exists", "extend")

    if "if_exists" in log_options and "if_table_exists" not in log_options:
        warnings.warn("The log_option 'if_exists' was renamed to 'if_table_exists'.")

    if logging.exists():
        if if_database_exists == "raise":
            raise FileExistsError(
                f"The database {logging} already exists and the log_option "
                "'if_database_exists' is set to 'raise'"
            )
        elif if_database_exists == "replace":
            logging.unlink()

    database = load_database(path_or_database=path, fast_logging=fast_logging)

    # create the optimization_iterations table
    make_optimization_iteration_table(
        database=database,
        if_exists=if_table_exists,
    )

    # create and initialize the steps table; This is alway extended if it exists.
    make_steps_table(database, if_exists=if_table_exists)

    # create_and_initialize the optimization_problem table
    make_optimization_problem_table(database, if_exists=if_table_exists)

    not_saved = [
        "criterion",
        "criterion_kwargs",
        "constraints",
        "derivative",
        "derivative_kwargs",
        "criterion_and_derivative",
        "criterion_and_derivative_kwargs",
    ]
    problem_data = {
        key: val for key, val in problem_data.items() if key not in not_saved
    }

    append_row(problem_data, "optimization_problem", database=database)

    return database


def _fill_numdiff_options_with_defaults(numdiff_options, lower_bounds, upper_bounds):
    """Fill options for numerical derivatives during optimization with defaults."""
    method = numdiff_options.get("method", "forward")
    default_error_handling = "raise" if method == "central" else "raise_strict"

    relevant = {
        "method",
        "n_steps",
        "base_steps",
        "scaling_factor",
        "lower_bounds",
        "upper_bounds",
        "step_ratio",
        "min_steps",
        "n_cores",
        "error_handling",
        "batch_evaluator",
    }

    ignored = [option for option in numdiff_options if option not in relevant]

    if ignored:
        raise InvalidKwargsError(
            f"The following numdiff_options are not allowed:\n\n{ignored}"
        )

    numdiff_options = {
        key: val for key, val in numdiff_options.items() if key in relevant
    }

    # only define the ones that deviate from the normal defaults
    default_numdiff_options = {
        "method": "forward",
        "bounds": Bounds(lower=lower_bounds, upper=upper_bounds),
        "error_handling": default_error_handling,
        "return_info": False,
    }

    numdiff_options = {**default_numdiff_options, **numdiff_options}
    return numdiff_options
