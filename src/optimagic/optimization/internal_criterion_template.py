import time
import warnings
from dataclasses import asdict

from optimagic.differentiation.derivatives import first_derivative
from optimagic.exceptions import UserFunctionRuntimeError, get_traceback
from optimagic.logging.write_to_database import append_row
from optimagic.typing import AggregationLevel


def internal_criterion_and_derivative_template(
    x,
    *,
    task,
    direction,
    criterion,
    converter,
    algo_info,
    bounds,
    derivative,
    criterion_and_derivative,
    numdiff_options,
    logging,
    database,
    error_handling,
    error_penalty_func,
    fixed_log_data,
    history_container=None,
    return_history_entry=False,
):
    """Template for the internal criterion and derivative function.

    This function forms the basis of all functions that define the optimization problem
    and are passed to the internal optimizers in optimagic. I.e. the criterion,
    derivative and criterion_and_derivative functions.

    Most of the arguments of this function will be partialled in before the functions
    are passed to internal optimizers.

    That is the reason why this function is called a template.

    Args:
        x (np.ndarray): 1d numpy array with internal parameters.
        task (str): One of "criterion", "derivative" and "criterion_and_derivative".
        direction (str): One of "maximize" or "minimize"
        criterion (callable): (partialed) user provided criterion function that takes a
            parameter dataframe as only argument and returns a scalar, an array like
            object or a dictionary. See :ref:`criterion`.
        params (pd.DataFrame): see :ref:`params`
        converter (Converter): NamedTuple with methods to convert between internal
            and external derivatives, parameters and criterion outputs.
        algo_info (AlgoInfo): NamedTuple with attributes
            - primary_criterion_entry
            - name
            - parallelizes
            - needs_scaling
            - is_available
        bounds (Bounds): Bounds on the internal parameters for the optimization problem.
        derivative (callable, optional): (partialed) user provided function that
            calculates the first derivative of criterion. For most algorithm, this is
            the gradient of the scalar output (or "value" entry of the dict). However
            some algorithms (e.g. bhhh) require the jacobian of the "contributions"
            entry of the dict. You will get an error if you provide the wrong type of
            derivative.
        criterion_and_derivative (callable): Function that returns criterion
            and derivative as a tuple. This can be used to exploit synergies in the
            evaluation of both functions. The first element of the tuple has to be
            exactly the same as the output of criterion. The second has to be exactly
            the same as the output of derivative.
        numdiff_options (dict): Keyword arguments for the calculation of numerical
            derivatives. See :ref:`first_derivative` for details. Note that the default
            method is changed to "forward" for speed reasons.
        logging (bool): Whether logging is used.
        database (DataBase): Database to which the logs are written.
        error_handling (str): Either "raise" or "continue". Note that "continue" does
            not absolutely guarantee that no error is raised but we try to handle as
            many errors as possible in that case without aborting the optimization.
        error_penalty_func (callable): Function that takes ``x`` and ``task`` and
            returns a penalized criterion function, its derivative or both (depending)
            on task.
        fixed_log_data (dict): Dictionary with fixed data to be saved in the database.
            Has the entries "stage" (str) and "substage" (int).
        history_container (list or None): List to which parameter, criterion and
            derivative histories are appended. Should be set to None if an algorithm
            parallelizes over criterion or derivative evaluations.
        return_history_entry (bool): Whether the history container should be returned.

    Returns:
        float, np.ndarray or tuple: If task=="criterion" it returns the output of
            criterion which can be a float or 1d numpy array. If task=="derivative" it
            returns the first derivative of criterion, which is a numpy array.
            If task=="criterion_and_derivative" it returns both as a tuple.

    """
    now = time.perf_counter()
    to_dos = _determine_to_dos(task, derivative, criterion_and_derivative)

    caught_exceptions = []
    new_fun, new_external_fun = None, None
    new_jac, new_external_jac = None, None
    current_params, external_x = converter.params_from_internal(
        x,
        return_type="tree_and_flat",
    )
    if to_dos == []:
        pass
    elif "numerical_fun_and_jac" in to_dos:

        def func(x):
            p = converter.params_from_internal(x, "tree")
            return criterion(p)

        try:
            numerical_derivative = first_derivative(
                func,
                x,
                bounds=bounds,
                **asdict(numdiff_options),
                unpacker=lambda x: x.internal_value(algo_info.solver_type),
                error_handling="raise_strict",
            )

            new_jac = numerical_derivative.derivative
            new_external_fun = numerical_derivative.func_value
            new_fun = new_external_fun.internal_value(algo_info.solver_type)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            tb = get_traceback()
            caught_exceptions.append(tb)
            if error_handling == "raise":
                msg = (
                    "An error occurred when evaluating criterion to calculate a "
                    "numerical derivative during optimization."
                )
                raise UserFunctionRuntimeError(msg) from e
            else:
                msg = (
                    "The following exception was caught when evaluating criterion to "
                    f"calculate a numerical derivative during optimization:\n\n{tb}"
                )
                warnings.warn(msg)

    elif "fun_and_jac" in to_dos:
        try:
            new_external_fun, new_external_jac = criterion_and_derivative(
                current_params
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            tb = get_traceback()
            caught_exceptions.append(tb)
            if error_handling == "raise":
                msg = (
                    "An error ocurred when evaluating criterion_and_derivative "
                    "during optimization."
                )
                raise UserFunctionRuntimeError(msg) from e
            else:
                msg = (
                    "The following exception was caught when evaluating "
                    f"criterion_and_derivative during optimization:\n\n{tb}"
                )
                warnings.warn(msg)

    else:
        if "fun" in to_dos:
            try:
                new_external_fun = criterion(current_params)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                tb = get_traceback()
                caught_exceptions.append(tb)
                if error_handling == "raise":
                    msg = (
                        "An error ocurred when evaluating criterion during "
                        "optimization."
                    )
                    raise UserFunctionRuntimeError(msg) from e
                else:
                    msg = (
                        "The following exception was caught when evaluating "
                        f"criterion during optimization:\n\n{tb}"
                    )
                    warnings.warn(msg)

        if "jac" in to_dos:
            try:
                new_external_jac = derivative(current_params)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                tb = get_traceback()
                caught_exceptions.append(tb)
                if error_handling == "raise":
                    msg = (
                        "An error ocurred when evaluating derivative during "
                        "optimization"
                    )
                    raise UserFunctionRuntimeError(msg) from e
                else:
                    msg = (
                        "The following exception was caught when evaluating "
                        f"derivative during optimization:\n\n{tb}"
                    )
                    warnings.warn(msg)

    if new_external_fun is not None and new_fun is None:
        new_fun = new_external_fun.internal_value(algo_info.solver_type)

    if new_external_jac is not None and new_jac is None:
        new_jac = converter.derivative_to_internal(new_external_jac, x)

    if caught_exceptions:
        new_external_fun, new_jac = error_penalty_func(
            x, task="criterion_and_derivative"
        )
        new_fun = new_external_fun.internal_value(algo_info.solver_type)

    if new_fun is not None:
        scalar_critval = new_external_fun.internal_value(AggregationLevel.SCALAR)
    else:
        scalar_critval = None

    if (new_fun is not None or new_jac is not None) and logging:
        _log_new_evaluations(
            new_criterion=new_external_fun,
            new_derivative=new_jac,
            external_x=external_x,
            caught_exceptions=caught_exceptions,
            database=database,
            fixed_log_data=fixed_log_data,
            scalar_value=scalar_critval,
            now=now,
        )

    res = _get_output_for_optimizer(
        new_criterion=new_fun,
        new_derivative=new_jac,
        task=task,
        direction=direction,
    )

    if new_fun is not None:
        hist_entry = {
            "params": current_params,
            "criterion": scalar_critval,
            "runtime": now,
        }
        if history_container is not None:
            if history_container:
                _batch = history_container[-1]["batches"] + 1
            else:
                _batch = 0

            hist_entry["batches"] = _batch
    else:
        hist_entry = None

    if history_container is not None and new_fun is not None:
        history_container.append(hist_entry)

    if return_history_entry:
        res = (res, hist_entry)

    return res


def _determine_to_dos(task, derivative, criterion_and_derivative):
    """Determine which functions have to be evaluated at the new parameters.

    Args:
        task (str): One of "criterion", "derivative", "criterion_and_derivative"
        cache_entry (dict): Possibly empty dict.
        derivative (Callable or None): Only used to determine if a closed form
            derivative is available.
        criterion_and_derivative (callable or None): Only used to determine if this
            function is available.

    Returns:
        list: List of functions that have to be evaluated. Possible values are:
            - [] if nothing has to be done
            - ["criterion_and_derivative"]
            - ["numerical_criterion_and_derivative"]
            - ["criterion", "derivative"]
            - ["criterion"]
            - ["derivative"]

    """
    fun_needed = "criterion" in task
    jac_needed = "derivative" in task

    to_dos = []
    if criterion_and_derivative is not None and fun_needed and jac_needed:
        to_dos.append("fun_and_jac")
    elif derivative is None and criterion_and_derivative is not None and jac_needed:
        to_dos.append("fun_and_jac")
    elif derivative is None and jac_needed:
        to_dos.append("numerical_fun_and_jac")
    else:
        if jac_needed:
            to_dos.append("jac")
        if fun_needed:
            to_dos.append("fun")
    return to_dos


def _log_new_evaluations(
    new_criterion,
    new_derivative,
    external_x,
    caught_exceptions,
    database,
    fixed_log_data,
    scalar_value,
    now,
):
    """Write the new evaluations and additional information into the database.

    Note: There are some seemingly unnecessary type conversions because sqlalchemy
    can fail silently when called with numpy dtypes instead of the equivalent python
    types.

    """
    data = {
        "params": external_x,
        "timestamp": now,
        "valid": True,
        "criterion_eval": new_criterion,
        "value": scalar_value,
        **fixed_log_data,
    }

    if new_derivative is not None:
        data["internal_derivative"] = new_derivative

    if caught_exceptions:
        separator = "\n" + "=" * 80 + "\n"
        data["exceptions"] = separator.join(caught_exceptions)
        data["valid"] = False

    name = "optimization_iterations"

    append_row(data, name, database=database)


def _get_output_for_optimizer(
    new_criterion,
    new_derivative,
    task,
    direction,
):
    if "criterion" in task and direction == "maximize":
        new_criterion = -new_criterion

    if "derivative" in task and direction == "maximize":
        new_derivative = -new_derivative

    if task == "criterion":
        out = new_criterion
    elif task == "derivative":
        out = new_derivative
    elif task == "criterion_and_derivative":
        out = (new_criterion, new_derivative)
    else:
        raise ValueError(f"Invalid task: {task}")
    return out
