import functools
import warnings
from functools import partial

import numpy as np

from optimagic.algorithms import ALL_ALGORITHMS
from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.logging.read_from_database import (
    list_of_dicts_to_dict_of_lists,
)
from optimagic.logging.write_to_database import update_row
from optimagic.utilities import propose_alternatives


def process_user_algorithm(algorithm):
    """Process the user specfied algorithm.

    If the algorithm is a callable, this function just reads out the algorithm_info
    and available options. If algorithm is a string it loads the algorithm function
    from the available algorithms.

    Args:
        algorithm (str or callable): The user specified algorithm.

    Returns:
        callable: the raw internal algorithm
        AlgoInfo: Attributes of the algorithm

    """
    if isinstance(algorithm, str):
        try:
            # Use ALL_ALGORITHMS and not just AVAILABLE_ALGORITHMS such that the
            # algorithm specific error message with installation instruction will be
            # reached if an optional dependency is not installed.
            algorithm = ALL_ALGORITHMS[algorithm]
        except KeyError:
            proposed = propose_alternatives(algorithm, list(ALL_ALGORITHMS))
            raise ValueError(
                f"Invalid algorithm: {algorithm}. Did you mean {proposed}?"
            ) from None

    algo_info = algorithm._algorithm_info

    return algorithm, algo_info


def get_final_algorithm(
    raw_algorithm,
    algo_info,
    valid_kwargs,
    lower_bounds,
    upper_bounds,
    nonlinear_constraints,
    algo_options,
    logging,
    database,
    collect_history,
):
    """Get algorithm-function with partialled options.

    The resulting function only depends on ``x``,  the relevant criterion functions,
    derivatives and ``step_id``. The remaining options are partialled in.

    Moreover, we add the following capabilities over the internal algorithms:
    - log the algorithm progress in a database (if logging is True)
    - collect the history of parameters and criterion values as well as runtime and
      batch information.

    Args:
        algorithm (str or callable): String with the name of an algorithm or internal
            algorithm function.
        lower_bounds (np.ndarray): 1d numpy array with lower bounds.
        upper_bounds (np.ndarray): 1d numpy array with upper bounds.
        nonlinear_constraints (list[dict]): List of dictionaries, each containing the
            specification of a nonlinear constraint.
        algo_options (dict): Dictionary with additional keyword arguments for the
            algorithm. Entries that are not used by the algorithm are ignored with a
            warning.
        logging (bool): Whether the algorithm should do logging.
        database (DataBase): Database to which the logging should be written.

    Returns:
        callable: The algorithm.

    """
    algo_name = algo_info.name

    internal_options = _adjust_options_to_algorithm(
        algo_options=algo_options,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        nonlinear_constraints=nonlinear_constraints,
        algo_name=algo_name,
        valid_kwargs=valid_kwargs,
    )

    algorithm = partial(raw_algorithm, **internal_options)

    algorithm = _add_logging(
        algorithm,
        logging=logging,
        database=database,
    )
    is_parallel = algo_info.parallelizes
    batch_size = internal_options.get("batch_size", internal_options.get("n_cores", 1))

    if collect_history and not algo_info.disable_history:
        algorithm = _add_history_collection(algorithm, is_parallel, batch_size)

    return algorithm


def _add_logging(algorithm=None, *, logging=None, database=None):
    """Add logging of status to the algorithm."""

    def decorator_add_logging_to_algorithm(algorithm):
        @functools.wraps(algorithm)
        def wrapper_add_logging_algorithm(**kwargs):
            step_id = int(kwargs["step_id"])

            _kwargs = {k: v for k, v in kwargs.items() if k != "step_id"}

            if logging:
                update_row(
                    data={"status": "running"},
                    rowid=step_id,
                    table_name="steps",
                    database=database,
                )

            for task in ["criterion", "derivative", "criterion_and_derivative"]:
                if task in _kwargs:
                    _kwargs[task] = partial(
                        _kwargs[task], fixed_log_data={"step": step_id}
                    )

            res = algorithm(**_kwargs)

            if logging:
                update_row(
                    data={"status": "complete"},
                    rowid=step_id,
                    table_name="steps",
                    database=database,
                )

            return res

        return wrapper_add_logging_algorithm

    if callable(algorithm):
        return decorator_add_logging_to_algorithm(algorithm)
    else:
        return decorator_add_logging_to_algorithm


def _add_history_collection(algorithm, is_parallel, batch_size):
    """Add history collection to the algorithm.

    The history collection is done jointly be the internal criterion function and the
    batch evaluator. Using the batch evaluator for history collection is necessary
    for two reasons:
    1. The batch information is only known inside the batch evaluator
    2. The normal approach of appending information to a list that is partialled into
       the internal criterion function breaks down when the batch evaluator pickles
       the internal criterion function.

    We make sure that optimizers that do some function evaluations via the batch
    evaluator and others directly are handled correctly.

    The interplay between the two methods for history collections is as follows:
    - If the function is called directly, all relevant information is appended to a list
      that is partialled into the internal criterion function.
    - If the function is called via the batch evaluator, we signal this to the internal
      criterion via the two arguments ``history_container`` (which is set to None) and
      ``return_history_entry`` (which is set to True). The returned history entries
      are then collected inside the batch evaluator and appended to the history
      container after all evaluations are done.

    Args:
        algorithm (callable): The algorithm.
        is_parallel (bool): Whether the algorithm can parallelize.

    """

    @functools.wraps(algorithm)
    def algorithm_with_history_collection(**kwargs):
        # initialize the shared history container
        container = []

        # add history collection via the internal criterion functions
        func_names = {"criterion", "derivative", "criterion_and_derivative"}
        _kwargs = kwargs.copy()
        for name in func_names:
            if name in kwargs:
                _kwargs[name] = partial(kwargs[name], history_container=container)

        # add history collection via the batch evaluator
        if is_parallel:
            raw_be = kwargs.get("batch_evaluator", "joblib")
            batch_evaluator = process_batch_evaluator(raw_be)

            _kwargs["batch_evaluator"] = _get_history_collecting_batch_evaluator(
                batch_evaluator=batch_evaluator,
                container=container,
                batch_size=batch_size,
            )

        # call the algorithm
        out = algorithm(**_kwargs)

        # add the history container to the algorithm output
        if "history" not in out:
            out["history"] = container
        else:
            out["history"] = out["history"] + container

        # process the history
        out["history"] = _process_collected_history(out["history"])

        return out

    return algorithm_with_history_collection


def _get_history_collecting_batch_evaluator(batch_evaluator, container, batch_size):
    @functools.wraps(batch_evaluator)
    def history_collecting_batch_evaluator(*args, **kwargs):
        if args:
            func = args[0]
        else:
            func = kwargs["func"]

        # find out if func is our internal criterion function. This is
        # necessary because an algorithm might use the batch evaluatior for
        # other functions as well, but for those functions we do not want to
        # (and cannot) collect a history.
        if isinstance(func, partial) and "history_container" in func.keywords:
            # partial in None as history container to disable history collection
            # via criterion function, which would not work with parallelization
            _func = partial(func, history_container=None, return_history_entry=True)

            if args:
                _args = (_func, *args[1:])
                _kwargs = kwargs
            else:
                _args = args
                _kwargs = kwargs.copy()
                _kwargs["func"] = _func

            raw_out = batch_evaluator(*_args, **_kwargs)
            out = [tup[0] for tup in raw_out]
            _hist = [tup[1] for tup in raw_out if tup[1] is not None]
            _start_batch = container[-1]["batches"] + 1 if container else 0
            _offsets = np.arange(len(_hist)).repeat(batch_size)[: len(_hist)]
            _batch_info = _offsets + _start_batch
            for batch, hist_entry in zip(_batch_info, _hist, strict=False):
                hist_entry["batches"] = batch

            container.extend(_hist)

        else:
            out = batch_evaluator(*args, **kwargs)

        return out

    return history_collecting_batch_evaluator


def _process_collected_history(raw):
    history = list_of_dicts_to_dict_of_lists(raw)
    runtimes = np.array(history["runtime"])
    runtimes -= runtimes[0]
    history["runtime"] = runtimes.tolist()
    return history


def _adjust_options_to_algorithm(
    algo_options,
    lower_bounds,
    upper_bounds,
    nonlinear_constraints,
    algo_name,
    valid_kwargs,
):
    """Reduce the algo_options and check if bounds are compatible with algorithm."""

    # convert algo option keys to valid Python arguments
    algo_options = {key.replace(".", "_"): val for key, val in algo_options.items()}

    reduced = {key: val for key, val in algo_options.items() if key in valid_kwargs}

    ignored = {key: val for key, val in algo_options.items() if key not in valid_kwargs}

    if ignored:
        warnings.warn(
            "The following algo_options were ignored because they are not compatible "
            f"with {algo_name}:\n\n {ignored}"
        )

    if "lower_bounds" not in valid_kwargs and not (lower_bounds == -np.inf).all():
        raise ValueError(
            f"{algo_name} does not support lower bounds but your optimization "
            "problem has lower bounds (either because you specified them explicitly "
            "or because they were implied by other constraints)."
        )

    if "upper_bounds" not in valid_kwargs and not (upper_bounds == np.inf).all():
        raise ValueError(
            f"{algo_name} does not support upper bounds but your optimization "
            "problem has upper bounds (either because you specified them explicitly "
            "or because they were implied by other constraints)."
        )

    if "lower_bounds" in valid_kwargs:
        reduced["lower_bounds"] = lower_bounds

    if "upper_bounds" in valid_kwargs:
        reduced["upper_bounds"] = upper_bounds

    if "nonlinear_constraints" in valid_kwargs:
        reduced["nonlinear_constraints"] = nonlinear_constraints

    return reduced
