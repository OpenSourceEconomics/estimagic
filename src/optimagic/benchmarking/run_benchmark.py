"""Functions to create, run and visualize optimization benchmarks.

TO-DO:
- Add other benchmark sets:
    - finish medium scale problems from https://arxiv.org/pdf/1710.11005.pdf, Page 34.
    - add scalar problems from https://github.com/AxelThevenot
- Add option for deterministic noise or wiggle.

"""

import numpy as np
from pybaum import tree_just_flatten

from optimagic import batch_evaluators
from optimagic.algorithms import AVAILABLE_ALGORITHMS
from optimagic.optimization.optimize import minimize
from optimagic.parameters.tree_registry import get_registry


def run_benchmark(
    problems,
    optimize_options,
    *,
    batch_evaluator="joblib",
    n_cores=1,
    error_handling="continue",
    max_criterion_evaluations=1_000,
    disable_convergence=True,
):
    """Run problems with different optimize options.

    Args:
        problems (dict): Nested dictionary with benchmark problems of the structure:
            {"name": {"inputs": {...}, "solution": {...}, "info": {...}}}
            where "inputs" are keyword arguments for ``minimize`` such as the criterion
            function and start parameters. "solution" contains the entries "params" and
            "value" and "info" might  contain information about the test problem.
        optimize_options (list or dict): Either a list of algorithms or a Nested
            dictionary that maps a name for optimizer settings
            (e.g. ``"lbfgsb_strict_criterion"``) to a dictionary of keyword arguments
            for arguments for ``minimize`` (e.g. ``{"algorithm": "scipy_lbfgsb",
            "algo_options": {"convergence.ftol_rel": 1e-12}}``).
            Alternatively, the values can just be an algorithm which is then benchmarked
            at default settings.
        batch_evaluator (str or callable): See :ref:`batch_evaluators`.
        n_cores (int): Number of optimizations that is run in parallel. Note that in
            addition to that an optimizer might parallelize.
        error_handling (str): One of "raise", "continue".
        max_criterion_evaluations (int): Shortcut to set the maximum number of
            criterion evaluations instead of passing them in via algo options. In case
            an optimizer does not support this stopping criterion, we also use this as
            max iterations.
        disable_convergence (bool): If True, we set extremely strict convergence
            convergence criteria by default, such that most optimizers will exploit
            their full computation budget set by max_criterion_evaluations.

    Returns:
        dict: Nested Dictionary with information on the benchmark run. The outer keys
            are tuples where the first entry is the name of the problem and the second
            the name of the optimize options. The values are dicts with the entries:
            "params_history", "criterion_history", "time_history" and "solution".

    """
    if isinstance(batch_evaluator, str):
        batch_evaluator = getattr(
            batch_evaluators, f"{batch_evaluator}_batch_evaluator"
        )
    opt_options = _process_optimize_options(
        optimize_options,
        max_evals=max_criterion_evaluations,
        disable_convergence=disable_convergence,
    )

    minimize_arguments, keys = _get_optimization_arguments_and_keys(
        problems, opt_options
    )

    raw_results = batch_evaluator(
        func=minimize,
        arguments=minimize_arguments,
        n_cores=n_cores,
        error_handling=error_handling,
        unpack_symbol="**",
    )

    processing_arguments = []
    for name, raw_result in zip(keys, raw_results, strict=False):
        processing_arguments.append(
            {"optimize_result": raw_result, "problem": problems[name[0]]}
        )

    results = batch_evaluator(
        func=_process_one_result,
        arguments=processing_arguments,
        n_cores=n_cores,
        error_handling="raise",
        unpack_symbol="**",
    )

    results = dict(zip(keys, results, strict=False))

    return results


def _process_optimize_options(raw_options, max_evals, disable_convergence):
    if not isinstance(raw_options, dict):
        dict_options = {}
        for option in raw_options:
            if isinstance(option, str):
                dict_options[option] = option
            else:
                dict_options[option.__name__] = option
    else:
        dict_options = raw_options

    default_algo_options = {}
    if max_evals is not None:
        default_algo_options["stopping.maxfun"] = max_evals
        default_algo_options["stopping.maxiter"] = max_evals
    if disable_convergence:
        default_algo_options["convergence.ftol_rel"] = 1e-14
        default_algo_options["convergence.xtol_rel"] = 1e-14
        default_algo_options["convergence.gtol_rel"] = 1e-14

    out_options = {}
    for name, _option in dict_options.items():
        if not isinstance(_option, dict):
            option = {"algorithm": _option}
        else:
            option = _option.copy()

        algo_options = {**default_algo_options, **option.get("algo_options", {})}
        algo_options = {k.replace(".", "_"): v for k, v in algo_options.items()}
        option["algo_options"] = algo_options
        if isinstance(option.get("algo_options"), dict):
            option["algo_options"] = {**default_algo_options, **option["algo_options"]}
        else:
            option["algo_options"] = default_algo_options

        out_options[name] = option

    return out_options


def _get_optimization_arguments_and_keys(problems, opt_options):
    kwargs_list = []
    names = []

    for prob_name, problem in problems.items():
        for option_name, options in opt_options.items():
            algo = options["algorithm"]
            if isinstance(algo, str):
                if algo not in AVAILABLE_ALGORITHMS:
                    raise ValueError(f"Invalid algorithm: {algo}")
                else:
                    valid_options = AVAILABLE_ALGORITHMS[algo]._algorithm_info.arguments

            else:
                valid_options = algo._algorithm_info.arguments

            algo_options = options["algo_options"]
            algo_options = {k: v for k, v in algo_options.items() if k in valid_options}

            kwargs = {**options, **problem["inputs"]}
            kwargs["algo_options"] = algo_options
            kwargs_list.append(kwargs)
            names.append((prob_name, option_name))

    return kwargs_list, names


def _process_one_result(optimize_result, problem):
    """Process the result of one optimization run.

    Args:
        optimize_result (OptimizeResult): Result of one optimization run.
        problem (dict): Problem specification.

    Returns:
        dict: Processed result.

    """
    _registry = get_registry(extended=True)
    _criterion = problem["noise_free_fun"]
    _start_x = problem["inputs"]["params"]
    _start_crit_value = _criterion(_start_x)
    if isinstance(_start_crit_value, np.ndarray):
        _start_crit_value = (_start_crit_value**2).sum()
    _is_noisy = problem["noisy"]
    _solution_crit = problem["solution"]["value"]

    # This will happen if the optimization raised an error
    if isinstance(optimize_result, str):
        params_history_flat = [tree_just_flatten(_start_x, registry=_registry)]
        criterion_history = [_start_crit_value]
        time_history = [np.inf]
        batches_history = [0]
    else:
        history = optimize_result.history
        params_history = history["params"]
        params_history_flat = [
            tree_just_flatten(p, registry=_registry) for p in params_history
        ]
        if _is_noisy:
            criterion_history = np.array([_criterion(p) for p in params_history])
            if criterion_history.ndim == 2:
                criterion_history = (criterion_history**2).sum(axis=1)
        else:
            criterion_history = history["criterion"]
        criterion_history = np.clip(criterion_history, _solution_crit, np.inf)
        batches_history = history["batches"]
        time_history = history["runtime"]

    return {
        "params_history": params_history_flat,
        "criterion_history": criterion_history,
        "time_history": time_history,
        "batches_history": batches_history,
        "solution": optimize_result,
    }
