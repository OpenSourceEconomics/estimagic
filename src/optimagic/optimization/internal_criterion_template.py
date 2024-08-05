import time
import warnings

from optimagic.differentiation.derivatives import first_derivative
from optimagic.exceptions import UserFunctionRuntimeError, get_traceback
from optimagic.logging.write_to_database import append_row
from optimagic.parameters.conversion import aggregate_func_output_to_value


def internal_criterion_and_derivative_template(
    x,
    *,
    task,
    direction,
    criterion,
    converter,
    algo_info,
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
    new_criterion, new_external_criterion = None, None
    new_derivative, new_external_derivative = None, None
    current_params, external_x = converter.params_from_internal(
        x,
        return_type="tree_and_flat",
    )
    if to_dos == []:
        pass
    elif "numerical_criterion_and_derivative" in to_dos:

        def func(x):
            p = converter.params_from_internal(x, "tree")
            crit_full = criterion(p)
            crit_relevant = converter.func_to_internal(crit_full)
            out = {"full": crit_full, "relevant": crit_relevant}
            return out

        options = numdiff_options.copy()
        options["key"] = "relevant"
        options["return_func_value"] = True

        try:
            derivative_dict = first_derivative(func, x, **options)
            new_derivative = derivative_dict["derivative"]
            new_criterion = derivative_dict["func_value"]["relevant"]
            new_external_criterion = derivative_dict["func_value"]["full"]
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

    elif "criterion_and_derivative" in to_dos:
        try:
            new_external_criterion, new_external_derivative = criterion_and_derivative(
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
        if "criterion" in to_dos:
            try:
                new_external_criterion = criterion(current_params)
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

        if "derivative" in to_dos:
            try:
                new_external_derivative = derivative(current_params)
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

    if new_external_criterion is not None and new_criterion is None:
        new_criterion = converter.func_to_internal(new_external_criterion)

    if new_external_derivative is not None and new_derivative is None:
        new_derivative = converter.derivative_to_internal(new_external_derivative, x)

    if caught_exceptions:
        new_criterion, new_derivative = error_penalty_func(
            x, task="criterion_and_derivative"
        )

    if new_criterion is not None:
        scalar_critval = aggregate_func_output_to_value(
            f_eval=new_criterion,
            primary_key=algo_info.primary_criterion_entry,
        )
    else:
        scalar_critval = None

    if (new_criterion is not None or new_derivative is not None) and logging:
        _log_new_evaluations(
            new_criterion=new_external_criterion,
            new_derivative=new_derivative,
            external_x=external_x,
            caught_exceptions=caught_exceptions,
            database=database,
            fixed_log_data=fixed_log_data,
            scalar_value=scalar_critval,
            now=now,
        )

    res = _get_output_for_optimizer(
        new_criterion=new_criterion,
        new_derivative=new_derivative,
        task=task,
        direction=direction,
    )

    if new_criterion is not None:
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

    if history_container is not None and new_criterion is not None:
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
    criterion_needed = "criterion" in task
    derivative_needed = "derivative" in task

    to_dos = []
    if criterion_and_derivative is not None and criterion_needed and derivative_needed:
        to_dos.append("criterion_and_derivative")
    elif (
        derivative is None
        and criterion_and_derivative is not None
        and derivative_needed
    ):
        to_dos.append("criterion_and_derivative")
    elif derivative is None and derivative_needed:
        to_dos.append("numerical_criterion_and_derivative")
    else:
        if derivative_needed:
            to_dos.append("derivative")
        if criterion_needed:
            to_dos.append("criterion")
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
