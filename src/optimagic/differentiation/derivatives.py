import functools
import itertools
import re
from itertools import product
from typing import NamedTuple

import numpy as np
import pandas as pd
from pybaum import tree_flatten, tree_unflatten
from pybaum import tree_just_flatten as tree_leaves

from optimagic import batch_evaluators
from optimagic.config import DEFAULT_N_CORES
from optimagic.deprecations import replace_and_warn_about_deprecated_bounds
from optimagic.differentiation import finite_differences
from optimagic.differentiation.generate_steps import generate_steps
from optimagic.differentiation.richardson_extrapolation import richardson_extrapolation
from optimagic.parameters.block_trees import hessian_to_block_tree, matrix_to_block_tree
from optimagic.parameters.bounds import Bounds, get_internal_bounds, pre_process_bounds
from optimagic.parameters.tree_registry import get_registry


class Evals(NamedTuple):
    pos: np.ndarray
    neg: np.ndarray


def first_derivative(
    func,
    params,
    *,
    bounds=None,
    func_kwargs=None,
    method="central",
    n_steps=1,
    base_steps=None,
    scaling_factor=1,
    step_ratio=2,
    min_steps=None,
    f0=None,
    n_cores=DEFAULT_N_CORES,
    error_handling="continue",
    batch_evaluator="joblib",
    return_func_value=False,
    return_info=False,
    key=None,
    # deprecated
    lower_bounds=None,
    upper_bounds=None,
):
    """Evaluate first derivative of func at params according to method and step options.

    Internally, the function is converted such that it maps from a 1d array to a 1d
    array. Then the Jacobian of that function is calculated.

    The parameters and the function output can be optimagic-pytrees; for more details on
    estimagi-pytrees see :ref:`eppytrees`. By default the resulting Jacobian will be
    returned as a block-pytree.

    For a detailed description of all options that influence the step size as well as an
    explanation of how steps are adjusted to bounds in case of a conflict, see
    :func:`~optimagic.differentiation.generate_steps.generate_steps`.

    Args:
        func (callable): Function of which the derivative is calculated.
        params (pytree): A pytree. See :ref:`params`.
        bounds: Lower and upper bounds on the parameters. The most general and preferred
            way to specify bounds is an `optimagic.Bounds` object that collects lower,
            upper, soft_lower and soft_upper bounds. The soft bounds are not used during
            numerical differentiation. Each bound type mirrors the structure of params.
            Check our how-to guide on bounds for examples. If params is a flat numpy
            array, you can also provide bounds via any format that is supported by
            scipy.optimize.minimize.
        func_kwargs (dict): Additional keyword arguments for func, optional.
        method (str): One of ["central", "forward", "backward"], default "central".
        n_steps (int): Number of steps needed. For central methods, this is
            the number of steps per direction. It is 1 if no Richardson extrapolation
            is used.
        base_steps (numpy.ndarray, optional): 1d array of the same length as params.
            base_steps * scaling_factor is the absolute value of the first (and possibly
            only) step used in the finite differences approximation of the derivative.
            If base_steps * scaling_factor conflicts with bounds, the actual steps will
            be adjusted. If base_steps is not provided, it will be determined according
            to a rule of thumb as long as this does not conflict with min_steps.
        scaling_factor (numpy.ndarray or float): Scaling factor which is applied to
            base_steps. If it is an numpy.ndarray, it needs to be as long as params.
            scaling_factor is useful if you want to increase or decrease the base_step
            relative to the rule-of-thumb or user provided base_step, for example to
            benchmark the effect of the step size. Default 1.
        step_ratio (float, numpy.array): Ratio between two consecutive Richardson
            extrapolation steps in the same direction. default 2.0. Has to be larger
            than one. The step ratio is only used if n_steps > 1.
        min_steps (numpy.ndarray): Minimal possible step sizes that can be chosen to
            accommodate bounds. Must have same length as params. By default min_steps is
            equal to base_steps, i.e step size is not decreased beyond what is optimal
            according to the rule of thumb.
        f0 (numpy.ndarray): 1d numpy array with func(x), optional.
        n_cores (int): Number of processes used to parallelize the function
            evaluations. Default 1.
        error_handling (str): One of "continue" (catch errors and continue to calculate
            derivative estimates. In this case, some derivative estimates can be
            missing but no errors are raised), "raise" (catch errors and continue
            to calculate derivative estimates at first but raise an error if all
            evaluations for one parameter failed) and "raise_strict" (raise an error
            as soon as a function evaluation fails).
        batch_evaluator (str or callable): Name of a pre-implemented batch evaluator
            (currently 'joblib' and 'pathos_mp') or Callable with the same interface
            as the optimagic batch_evaluators.
        return_func_value (bool): If True, return function value at params, stored in
            output dict under "func_value". Default False. This is useful when using
            first_derivative during optimization.
        return_info (bool): If True, return additional information on function
            evaluations and internal derivative candidates, stored in output dict under
            "func_evals" and "derivative_candidates". Derivative candidates are only
            returned if n_steps > 1. Default False.
        key (str): If func returns a dictionary, take the derivative of
            func(params)[key].

    Returns:
        result (dict): Result dictionary with keys:
            - "derivative" (numpy.ndarray, pandas.Series or pandas.DataFrame): The
                estimated first derivative of func at params. The shape of the output
                depends on the dimension of params and func(params):

                - f: R -> R leads to shape (1,), usually called derivative
                - f: R^m -> R leads to shape (m, ), usually called Gradient
                - f: R -> R^n leads to shape (n, 1), usually called Jacobian
                - f: R^m -> R^n leads to shape (n, m), usually called Jacobian

            - "func_value" (numpy.ndarray, pandas.Series or pandas.DataFrame): Function
                value at params, returned if return_func_value is True.

            - "func_evals" (pandas.DataFrame): Function evaluations produced by internal
                derivative method, returned if return_info is True.

            - "derivative_candidates" (pandas.DataFrame): Derivative candidates from
                Richardson extrapolation, returned if return_info is True and n_steps >
                1.

    """
    # ==================================================================================
    # handle deprecations
    # ==================================================================================
    bounds = replace_and_warn_about_deprecated_bounds(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        bounds=bounds,
    )

    # ==================================================================================

    bounds = pre_process_bounds(bounds)

    _is_fast_params = isinstance(params, np.ndarray) and params.ndim == 1
    registry = get_registry(extended=True)

    internal_lb, internal_ub = get_internal_bounds(params, bounds=bounds)

    # handle keyword arguments
    func_kwargs = {} if func_kwargs is None else func_kwargs
    partialed_func = functools.partial(func, **func_kwargs)

    # convert params to numpy
    if not _is_fast_params:
        x, params_treedef = tree_flatten(params, registry=registry)
        x = np.array(x, dtype=np.float64)
    else:
        x = params.astype(float)

    if np.isnan(x).any():
        raise ValueError("The parameter vector must not contain NaNs.")

    # generate the step array
    steps = generate_steps(
        x=x,
        method=method,
        n_steps=n_steps,
        target="first_derivative",
        base_steps=base_steps,
        scaling_factor=scaling_factor,
        bounds=Bounds(lower=internal_lb, upper=internal_ub),
        step_ratio=step_ratio,
        min_steps=min_steps,
    )

    # generate parameter vectors at which func has to be evaluated as numpy arrays
    evaluation_points = []
    for step_arr in steps:
        for i, j in product(range(n_steps), range(len(x))):
            if np.isnan(step_arr[i, j]):
                evaluation_points.append(np.nan)
            else:
                point = x.copy()
                point[j] += step_arr[i, j]
                evaluation_points.append(point)

    # convert the numpy arrays to whatever is needed by func
    if not _is_fast_params:
        evaluation_points = [
            # entries are either a numpy.ndarray or np.nan
            _unflatten_if_not_nan(p, params_treedef, registry)
            for p in evaluation_points
        ]

    # we always evaluate f0, so we can fall back to one-sided derivatives if
    # two-sided derivatives fail. The extra cost is negligible in most cases.
    if f0 is None:
        evaluation_points.append(params)

    # do the function evaluations, including error handling
    batch_error_handling = "raise" if error_handling == "raise_strict" else "continue"
    raw_evals = _nan_skipping_batch_evaluator(
        func=partialed_func,
        arguments=evaluation_points,
        n_cores=n_cores,
        error_handling=batch_error_handling,
        batch_evaluator=batch_evaluator,
    )

    # extract information on exceptions that occurred during function evaluations
    exc_info = "\n\n".join([val for val in raw_evals if isinstance(val, str)])
    raw_evals = [val if not isinstance(val, str) else np.nan for val in raw_evals]

    # store full function value at params as func_value and a processed version of it
    # that we need to calculate derivatives as f0
    if f0 is None:
        f0 = raw_evals[-1]
        raw_evals = raw_evals[:-1]
    func_value = f0

    use_key = key is not None and isinstance(f0, dict)
    f0_tree = f0[key] if use_key else f0
    scalar_out = np.isscalar(f0_tree)
    vector_out = isinstance(f0_tree, np.ndarray) and f0_tree.ndim == 1

    if scalar_out:
        f0 = np.array([f0_tree], dtype=float)
    elif vector_out:
        f0 = f0_tree.astype(float)
    else:
        f0 = tree_leaves(f0_tree, registry=registry)
        f0 = np.array(f0, dtype=np.float64)

    # convert the raw evaluations to numpy arrays
    raw_evals = _convert_evals_to_numpy(
        raw_evals=raw_evals,
        key=key,
        registry=registry,
        is_scalar_out=scalar_out,
        is_vector_out=vector_out,
    )

    # apply finite difference formulae
    evals = np.array(raw_evals).reshape(2, n_steps, len(x), -1)
    evals = np.transpose(evals, axes=(0, 1, 3, 2))
    evals = Evals(pos=evals[0], neg=evals[1])

    jac_candidates = {}
    for m in ["forward", "backward", "central"]:
        jac_candidates[m] = finite_differences.jacobian(evals, steps, f0, m)

    # get the best derivative estimate out of all derivative estimates that could be
    # calculated, given the function evaluations.
    orders = {
        "central": ["central", "forward", "backward"],
        "forward": ["forward", "backward"],
        "backward": ["backward", "forward"],
    }

    if n_steps == 1:
        jac = _consolidate_one_step_derivatives(jac_candidates, orders[method])
        updated_candidates = None
    else:
        richardson_candidates = _compute_richardson_candidates(
            jac_candidates, steps, n_steps
        )
        jac, updated_candidates = _consolidate_extrapolated(richardson_candidates)

    # raise error if necessary
    if error_handling in ("raise", "raise_strict") and np.isnan(jac).any():
        raise Exception(exc_info)

    # results processing
    if _is_fast_params and vector_out:
        derivative = jac
    elif _is_fast_params and scalar_out:
        derivative = jac.flatten()
    else:
        derivative = matrix_to_block_tree(jac, f0_tree, params)

    result = {"derivative": derivative}
    if return_func_value:
        result["func_value"] = func_value
    if return_info:
        info = _collect_additional_info(
            steps, evals, updated_candidates, target="first_derivative"
        )
        result = {**result, **info}
    return result


def second_derivative(
    func,
    params,
    *,
    bounds=None,
    func_kwargs=None,
    method="central_cross",
    n_steps=1,
    base_steps=None,
    scaling_factor=1,
    step_ratio=2,
    min_steps=None,
    f0=None,
    n_cores=DEFAULT_N_CORES,
    error_handling="continue",
    batch_evaluator="joblib",
    return_func_value=False,
    return_info=False,
    key=None,
    # deprecated
    lower_bounds=None,
    upper_bounds=None,
):
    """Evaluate second derivative of func at params according to method and step
    options.

    Internally, the function is converted such that it maps from a 1d array to a 1d
    array. Then the Hessians of that function are calculated. The resulting derivative
    estimate is always a :class:`numpy.ndarray`.

    The parameters and the function output can be pandas objects (Series or DataFrames
    with value column). In that case the output of second_derivative is also a pandas
    object and with appropriate index and columns.

    Detailed description of all options that influence the step size as well as an
    explanation of how steps are adjusted to bounds in case of a conflict,
    see :func:`~optimagic.differentiation.generate_steps.generate_steps`.

    Args:
        func (callable): Function of which the derivative is calculated.
        params (numpy.ndarray, pandas.Series or pandas.DataFrame): 1d numpy array or
            :class:`pandas.DataFrame` with parameters at which the derivative is
            calculated. If it is a DataFrame, it can contain the columns "lower_bound"
            and "upper_bound" for bounds. See :ref:`params`.
        bounds: Lower and upper bounds on the parameters. The most general and preferred
            way to specify bounds is an `optimagic.Bounds` object that collects lower,
            upper, soft_lower and soft_upper bounds. The soft bounds are not used during
            numerical differentiation. Each bound type mirrors the structure of params.
            Check our how-to guide on bounds for examples. If params is a flat numpy
            array, you can also provide bounds via any format that is supported by
            scipy.optimize.minimize.
        func_kwargs (dict): Additional keyword arguments for func, optional.
        method (str): One of {"forward", "backward", "central_average", "central_cross"}
            These correspond to the finite difference approximations defined in
            equations [7, x, 8, 9] in Rideout [2009], where ("backward", x) is not found
            in Rideout [2009] but is the natural extension of equation 7 to the backward
            case. Default "central_cross".
        n_steps (int): Number of steps needed. For central methods, this is
            the number of steps per direction. It is 1 if no Richardson extrapolation
            is used.
        base_steps (numpy.ndarray, optional): 1d array of the same length as params.
            base_steps * scaling_factor is the absolute value of the first (and possibly
            only) step used in the finite differences approximation of the derivative.
            If base_steps * scaling_factor conflicts with bounds, the actual steps will
            be adjusted. If base_steps is not provided, it will be determined according
            to a rule of thumb as long as this does not conflict with min_steps.
        scaling_factor (numpy.ndarray or float): Scaling factor which is applied to
            base_steps. If it is an numpy.ndarray, it needs to be as long as params.
            scaling_factor is useful if you want to increase or decrease the base_step
            relative to the rule-of-thumb or user provided base_step, for example to
            benchmark the effect of the step size. Default 1.
        step_ratio (float, numpy.array): Ratio between two consecutive Richardson
            extrapolation steps in the same direction. default 2.0. Has to be larger
            than one. The step ratio is only used if n_steps > 1.
        min_steps (numpy.ndarray): Minimal possible step sizes that can be chosen to
            accommodate bounds. Must have same length as params. By default min_steps is
            equal to base_steps, i.e step size is not decreased beyond what is optimal
            according to the rule of thumb.
        f0 (numpy.ndarray): 1d numpy array with func(x), optional.
        n_cores (int): Number of processes used to parallelize the function
            evaluations. Default 1.
        error_handling (str): One of "continue" (catch errors and continue to calculate
            derivative estimates. In this case, some derivative estimates can be
            missing but no errors are raised), "raise" (catch errors and continue
            to calculate derivative estimates at first but raise an error if all
            evaluations for one parameter failed) and "raise_strict" (raise an error
            as soon as a function evaluation fails).
        batch_evaluator (str or callable): Name of a pre-implemented batch evaluator
            (currently 'joblib' and 'pathos_mp') or Callable with the same interface
            as the optimagic batch_evaluators.
        return_func_value (bool): If True, return function value at params, stored in
            output dict under "func_value". Default False. This is useful when using
            first_derivative during optimization.
        return_info (bool): If True, return additional information on function
            evaluations and internal derivative candidates, stored in output dict under
            "func_evals" and "derivative_candidates". Derivative candidates are only
            returned if n_steps > 1. Default False.
        key (str): If func returns a dictionary, take the derivative of
            func(params)[key].


    Returns:
        result (dict): Result dictionary with keys:
            - "derivative" (numpy.ndarray, pandas.Series or pandas.DataFrame): The
                estimated second derivative of func at params. The shape of the output
                depends on the dimension of params and func(params):

                - f: R -> R leads to shape (1,), usually called second derivative
                - f: R^m -> R leads to shape (m, m), usually called Hessian
                - f: R -> R^n leads to shape (n,), usually called Hessian
                - f: R^m -> R^n leads to shape (n, m, m), usually called Hessian tensor

            - "func_value" (numpy.ndarray, pandas.Series or pandas.DataFrame): Function
                value at params, returned if return_func_value is True.

            - "func_evals_one_step" (pandas.DataFrame): Function evaluations produced by
                internal derivative method when altering the params vector at one
                dimension, returned if return_info is True.

            - "func_evals_two_step" (pandas.DataFrame): This features is not implemented
                yet and therefore set to None. Once implemented it will contain
                function evaluations produced by internal derivative method when
                altering the params vector at two dimensions, returned if return_info is
                True.

            - "func_evals_cross_step" (pandas.DataFrame): This features is not
                implemented yet and therefore set to None. Once implemented it will
                contain function evaluations produced by internal derivative method when
                altering the params vector at two dimensions in different directions,
                returned if return_info is True.

    """

    # ==================================================================================
    # handle deprecations
    # ==================================================================================
    bounds = replace_and_warn_about_deprecated_bounds(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        bounds=bounds,
    )
    # ==================================================================================

    bounds = pre_process_bounds(bounds)

    internal_lb, internal_ub = get_internal_bounds(params, bounds=bounds)

    # handle keyword arguments
    func_kwargs = {} if func_kwargs is None else func_kwargs
    partialed_func = functools.partial(func, **func_kwargs)

    # convert params to numpy
    registry = get_registry(extended=True)
    x, params_treedef = tree_flatten(params, registry=registry)
    x = np.atleast_1d(x).astype(np.float64)

    if np.isnan(x).any():
        raise ValueError("The parameter vector must not contain NaNs.")

    implemented_methods = {"forward", "backward", "central_average", "central_cross"}
    if method not in implemented_methods:
        raise ValueError(f"Method has to be in {implemented_methods}.")

    # generate the step array
    steps = generate_steps(
        x=x,
        method=("central" if "central" in method else method),
        n_steps=n_steps,
        target="second_derivative",
        base_steps=base_steps,
        scaling_factor=scaling_factor,
        bounds=Bounds(lower=internal_lb, upper=internal_ub),
        step_ratio=step_ratio,
        min_steps=min_steps,
    )

    # generate parameter vectors at which func has to be evaluated as numpy arrays
    evaluation_points = {"one_step": [], "two_step": [], "cross_step": []}
    for step_arr in steps:
        # single direction steps
        for i, j in product(range(n_steps), range(len(x))):
            if np.isnan(step_arr[i, j]):
                evaluation_points["one_step"].append(np.nan)
            else:
                point = x.copy()
                point[j] += step_arr[i, j]
                evaluation_points["one_step"].append(point)
        # two and cross direction steps
        for i, j, k in product(range(n_steps), range(len(x)), range(len(x))):
            if j > k or np.isnan(step_arr[i, j]) or np.isnan(step_arr[i, k]):
                evaluation_points["two_step"].append(np.nan)
                evaluation_points["cross_step"].append(np.nan)
            else:
                point = x.copy()
                point[j] += step_arr[i, j]
                point[k] += step_arr[i, k]
                evaluation_points["two_step"].append(point)
                if j == k:
                    evaluation_points["cross_step"].append(np.nan)
                else:
                    point = x.copy()
                    point[j] += step_arr[i, j]
                    point[k] -= step_arr[i, k]
                    evaluation_points["cross_step"].append(point)

    # convert the numpy arrays to whatever is needed by func
    evaluation_points = {
        # entries are either a numpy.ndarray or np.nan, we unflatten only
        step_type: [_unflatten_if_not_nan(p, params_treedef, registry) for p in points]
        for step_type, points in evaluation_points.items()
    }

    # we always evaluate f0, so we can fall back to one-sided derivatives if
    # two-sided derivatives fail. The extra cost is negligible in most cases.
    if f0 is None:
        evaluation_points["one_step"].append(params)

    # do the function evaluations for one and two step, including error handling
    batch_error_handling = "raise" if error_handling == "raise_strict" else "continue"
    raw_evals = _nan_skipping_batch_evaluator(
        func=partialed_func,
        arguments=list(itertools.chain.from_iterable(evaluation_points.values())),
        n_cores=n_cores,
        error_handling=batch_error_handling,
        batch_evaluator=batch_evaluator,
    )

    # extract information on exceptions that occurred during function evaluations
    exc_info = "\n\n".join([val for val in raw_evals if isinstance(val, str)])
    raw_evals = [val if not isinstance(val, str) else np.nan for val in raw_evals]

    n_one_step, n_two_step, n_cross_step = map(len, evaluation_points.values())
    raw_evals = {
        "one_step": raw_evals[:n_one_step],
        "two_step": raw_evals[n_one_step : n_two_step + n_one_step],
        "cross_step": raw_evals[n_two_step + n_one_step :],
    }

    # store full function value at params as func_value and a processed version of it
    # that we need to calculate derivatives as f0
    if f0 is None:
        f0 = raw_evals["one_step"][-1]
        raw_evals["one_step"] = raw_evals["one_step"][:-1]
    func_value = f0

    f0_tree = f0[key] if key is not None and isinstance(f0, dict) else f0
    f0 = tree_leaves(f0_tree, registry=registry)
    f0 = np.array(f0, dtype=np.float64)

    # convert the raw evaluations to numpy arrays
    raw_evals = {
        step_type: _convert_evals_to_numpy(evals, key, registry)
        for step_type, evals in raw_evals.items()
    }

    # reshape arrays into dimension (n_steps, dim_f, dim_x) or (n_steps, dim_f, dim_x,
    # dim_x) for finite differences
    evals = {}
    evals["one_step"] = _reshape_one_step_evals(raw_evals["one_step"], n_steps, len(x))
    evals["two_step"] = _reshape_two_step_evals(raw_evals["two_step"], n_steps, len(x))
    evals["cross_step"] = _reshape_cross_step_evals(
        raw_evals["cross_step"], n_steps, len(x), f0
    )

    # apply finite difference formulae
    hess_candidates = {}
    for m in ["forward", "backward", "central_average", "central_cross"]:
        hess_candidates[m] = finite_differences.hessian(evals, steps, f0, m)

    # get the best derivative estimate out of all derivative estimates that could be
    # calculated, given the function evaluations.
    orders = {
        "central_cross": ["central_cross", "central_average", "forward", "backward"],
        "central_average": ["central_average", "central_cross", "forward", "backward"],
        "forward": ["forward", "backward", "central_average", "central_cross"],
        "backward": ["backward", "forward", "central_average", "central_cross"],
    }

    if n_steps == 1:
        hess = _consolidate_one_step_derivatives(hess_candidates, orders[method])
        updated_candidates = None
    else:
        raise ValueError(
            "Richardson extrapolation is not implemented for the second derivative yet."
        )

    # raise error if necessary
    if error_handling in ("raise", "raise_strict") and np.isnan(hess).any():
        raise Exception(exc_info)

    # results processing
    derivative = hessian_to_block_tree(hess, f0_tree, params)

    result = {"derivative": derivative}
    if return_func_value:
        result["func_value"] = func_value
    if return_info:
        info = _collect_additional_info(
            steps, evals, updated_candidates, target="second_derivative"
        )
        result = {**result, **info}
    return result


def _reshape_one_step_evals(raw_evals_one_step, n_steps, dim_x):
    """Reshape raw_evals for evaluation points with one step.

    Returned object is a namedtuple with entries 'pos' and 'neg' corresponding to
    positive and negative steps. Each entry will be a numpy array with dimension
    (n_steps, dim_f, dim_x).

    Mathematical:

            evals.pos = (f(x0 + delta_jl e_j))
            evals.neg = (f(x0 - delta_jl e_j))

        for j=1,...,dim_x and l=1,...,n_steps

    """
    evals = np.array(raw_evals_one_step).reshape(2, n_steps, dim_x, -1)
    evals = evals.swapaxes(2, 3)
    evals = Evals(pos=evals[0], neg=evals[1])
    return evals


def _reshape_two_step_evals(raw_evals_two_step, n_steps, dim_x):
    """Reshape raw_evals for evaluation points with two steps.

    Returned object is a namedtuple with entries 'pos' and 'neg' corresponding to
    positive and negative steps. Each entry will be a numpy array with dimension
    (n_steps, dim_f, dim_x, dim_x). Since the array is, by definition, symmetric over
    the last two dimensions, the function is not evaluated on both sides to save
    computation time and the information is simply copied here.

    Mathematical:

            evals.pos = (f(x0 + delta_jl e_j + delta_kl e_k))
            evals.neg = (f(x0 - delta_jl e_j - delta_kl e_k))

        for j,k=1,...,dim_x and l=1,...,n_steps

    """
    tril_idx = np.tril_indices(dim_x, -1)
    evals = np.array(raw_evals_two_step).reshape(2, n_steps, dim_x, dim_x, -1)
    evals = evals.transpose(0, 1, 4, 2, 3)
    evals[..., tril_idx[0], tril_idx[1]] = evals[..., tril_idx[1], tril_idx[0]]
    evals = Evals(pos=evals[0], neg=evals[1])
    return evals


def _reshape_cross_step_evals(raw_evals_cross_step, n_steps, dim_x, f0):
    """Reshape raw_evals for evaluation points with cross steps.

    Returned object is a namedtuple with entries 'pos' and 'neg' corresponding to
    positive and negative steps. Each entry will be a numpy array with dimension
    (n_steps, dim_f, dim_x, dim_x). Since the array is, by definition, symmetric over
    the last two dimensions, the function is not evaluated on both sides to save
    computation time and the information is simply copied here. In comparison to the
    two_step case, however, this symmetry holds only over the dimension 'pos' and 'neg'.
    That is, the lower triangular of the last two dimensions of 'pos' must equal the
    upper triangular of the last two dimensions of 'neg'. Further, the diagonal of the
    last two dimensions must be equal to f0.

    Mathematical:

            evals.pos = (f(x0 + delta_jl e_j - delta_kl e_k))
            evals.neg = (f(x0 - delta_jl e_j + delta_kl e_k))

        for j,k=1,...,dim_x and l=1,...,n_steps

    """
    tril_idx = np.tril_indices(dim_x, -1)
    diag_idx = np.diag_indices(dim_x)
    evals = np.array(raw_evals_cross_step).reshape(2, n_steps, dim_x, dim_x, -1)
    evals = evals.transpose(0, 1, 4, 2, 3)
    evals[0][..., tril_idx[0], tril_idx[1]] = evals[1][..., tril_idx[1], tril_idx[0]]
    evals[0][..., diag_idx[0], diag_idx[1]] = np.atleast_2d(f0).T[np.newaxis, ...]
    evals = Evals(pos=evals[0], neg=evals[0].swapaxes(2, 3))
    return evals


def _convert_evaluation_data_to_frame(steps, evals):
    """Convert evaluation data to (tidy) data frame.

    Args:
        steps (namedtuple): Namedtuple with field names pos and neg. Is generated by
            :func:`~optimagic.differentiation.generate_steps.generate_steps`.
        evals (namedtuple): Namedtuple with field names pos and neg. Contains function
            evaluation corresponding to steps.

    Returns:
        df (pandas.DataFrame): Tidy data frame with index (sign, step_number, dim_x
            dim_f), where sign corresponds to pos or neg in steps and evals, step_number
            indexes the step, dim_x is the dimension of the input vector and dim_f is
            the dimension of the function output. The data is given by the two columns
            step and eval. The data frame has 2 * n_steps * dim_x * dim_f rows.

    """
    n_steps, dim_f, dim_x = evals.pos.shape

    dfs = []
    for direction, step_arr, eval_arr in zip((1, -1), steps, evals, strict=False):
        df_steps = pd.DataFrame(step_arr, columns=range(dim_x))
        df_steps = df_steps.reset_index()
        df_steps = df_steps.rename(columns={"index": "step_number"})
        df_steps = df_steps.melt(
            id_vars="step_number", var_name="dim_x", value_name="step"
        )
        df_steps = df_steps.sort_values("step_number")
        df_steps = df_steps.reset_index(drop=True)
        df_steps = df_steps.apply(lambda col: col.abs() if col.name == "step" else col)

        reshaped_eval_arr = np.transpose(eval_arr, (0, 2, 1)).reshape(-1, dim_f)
        df_evals = pd.concat((df_steps, pd.DataFrame(reshaped_eval_arr)), axis=1)
        df_evals = df_evals.melt(
            id_vars=["step_number", "dim_x", "step"],
            var_name="dim_f",
            value_name="eval",
        )
        df_evals = df_evals.assign(sign=direction)
        df_evals = df_evals.set_index(["sign", "step_number", "dim_x", "dim_f"])
        df_evals = df_evals.sort_index()

        dfs.append(df_evals)

    df = pd.concat(dfs).astype({"step": float, "eval": float})
    return df


def _convert_richardson_candidates_to_frame(jac, err):
    """Convert (richardson) jacobian candidates and errors to pandas data frame.

    Args:
        jac (dict): Dict with richardson jacobian candidates.
        err (dict): Dict with errors corresponding to richardson jacobian candidates.

    Returns:
        df (pandas.DataFrame): Frame with column "der" and "err" and index ["method",
            "num_term", "dim_x", "dim_f"] with respective meaning: type of method used,
            e.g. central or foward; kind of value, e.g. derivative or error.

    """
    dim_f, dim_x = jac["forward1"].shape
    dfs = []
    for key, value in jac.items():
        method, num_term = _split_into_str_and_int(key)
        df = pd.DataFrame(value.T, columns=range(dim_f))
        df = df.assign(dim_x=range(dim_x))
        df = df.melt(id_vars="dim_x", var_name="dim_f", value_name="der")
        df = df.assign(method=method, num_term=num_term, err=err[key].T.flatten())
        dfs.append(df)

    df = pd.concat(dfs)
    df = df.set_index(["method", "num_term", "dim_x", "dim_f"])
    return df


def _convert_evals_to_numpy(
    raw_evals, key, registry, is_scalar_out=False, is_vector_out=False
):
    """Harmonize the output of the function evaluations.

    The raw_evals might contain dictionaries of which we only need one entry, scalar
    np.nan where we need arrays filled with np.nan or pandas objects. The processed
    evals only contain numpy arrays.

    """
    # get rid of dictionaries
    evals = [
        val[key] if isinstance(val, dict) and key is not None else val
        for val in raw_evals
    ]

    # convert pytrees to arrays
    if is_scalar_out:
        evals = [
            np.array([val], dtype=float) if not _is_scalar_nan(val) else val
            for val in evals
        ]

    elif is_vector_out:
        evals = [val.astype(float) if not _is_scalar_nan(val) else val for val in evals]
    else:
        evals = [
            (
                np.array(tree_leaves(val, registry=registry), dtype=np.float64)
                if not _is_scalar_nan(val)
                else val
            )
            for val in evals
        ]

    # find out the correct output shape
    try:
        array = next(x for x in evals if hasattr(x, "shape") or isinstance(x, dict))
        out_shape = array.shape
    except StopIteration:
        out_shape = "scalar"

    # convert to correct output shape
    if out_shape == "scalar":
        evals = [np.atleast_1d(val) for val in evals]
    else:
        for i in range(len(evals)):
            if isinstance(evals[i], float) and np.isnan(evals[i]):
                evals[i] = np.full(out_shape, np.nan)

    return evals


def _consolidate_one_step_derivatives(candidates, preference_order):
    """Replace missing derivative estimates of preferred method with others.

    Args:
        candidates (dict): Dictionary with derivative estimates from different methods.
        preference_order (list): Order on (a subset of) the keys in candidates. Earlier
        entries are preferred.

    Returns:
        consolidated (np.ndarray): Array of same shape as input derivative estimates.

    """
    preferred, others = preference_order[0], preference_order[1:]
    consolidated = candidates[preferred].copy()
    for other in others:
        consolidated = np.where(np.isnan(consolidated), candidates[other], consolidated)

    return consolidated.reshape(consolidated.shape[1:])


def _consolidate_extrapolated(candidates):
    """Get the best possible derivative estimate, given an error estimate.

    Going through ``candidates`` select the best derivative estimate element-wise using
    the estimated candidates, where best is defined as minimizing the error estimate
    from the Richardson extrapolation.

    See https://tinyurl.com/ubn3nv5 for corresponding code in numdifftools and
    https://tinyurl.com/snle7mb for an explanation of how errors of Richardson
    extrapolated derivative estimates can be estimated.

    Args:
        candidates (dict): Dictionary containing different derivative estimates and
            their error estimates.

    Returns:
        consolidated (np.ndarray): Array of same shape as input derivative estimates.
        candidate_der_dict (dict): Best derivative estimate given method.
        candidate_err_dict (dict): Errors corresponding to best derivatives given method

    """
    # first find minimum over steps for each method
    candidate_der_dict = {}
    candidate_err_dict = {}

    for key in candidates:
        _der = candidates[key]["derivative"]
        _err = candidates[key]["error"]
        derivative, error = _select_minimizer_along_axis(_der, _err)
        candidate_der_dict[key] = derivative
        candidate_err_dict[key] = error

    # second find minimum over methods
    candidate_der = np.stack(list(candidate_der_dict.values()))
    candidate_err = np.stack(list(candidate_err_dict.values()))
    consolidated, _ = _select_minimizer_along_axis(candidate_der, candidate_err)

    updated_candidates = (candidate_der_dict, candidate_err_dict)
    return consolidated, updated_candidates


def _compute_richardson_candidates(jac_candidates, steps, n_steps):
    """Compute derivative candidates using Richardson extrapolation.

    Args:
        jac_candidates (dict): Dictionary with (traditional) derivative estimates from
            different methods.
        steps (namedtuple): Namedtuple with the field names pos and neg. Each field
            contains a numpy array of shape (n_steps, len(x)) with the steps in
            the corresponding direction. The steps are always symmetric, in the sense
            that steps.neg[i, j] = - steps.pos[i, j] unless one of them is NaN.
        n_steps (int): Number of steps needed. For central methods, this is
            the number of steps per direction. It is 1 if no Richardson extrapolation
            is used.

    Returns:
        richardson_candidates (dict): Dictionary with derivative estimates and error
            estimates from different methods.
            - Keys correspond to the method used, i.e. forward, backward or central
            differences and the number of terms used in the Richardson extrapolation.
            - Values represent the corresponding derivative estimate and error
            estimate, stored as np.ndarrays in a sub-dictionary under "derivative" and
            "error" respectively, with the first dimensions coinciding with that of an
            element of ``jac_candidates`` and depending on num_terms, possibly one
            further dimension.

    """
    richardson_candidates = {}
    for method in ["forward", "backward", "central"]:
        for num_terms in range(1, n_steps):
            derivative, error = richardson_extrapolation(
                jac_candidates[method], steps, method, num_terms
            )
            richardson_candidates[method + str(num_terms)] = {
                "derivative": derivative,
                "error": error,
            }

    return richardson_candidates


def _select_minimizer_along_axis(derivative, errors):
    """Select best derivative estimates element wise.

    Select elements from ``derivative`` which correspond to minimum in ``errors`` along
    first axis.

    Args:
        derivative (np.ndarray): Derivative estimates from Richardson approximation.
            First axis (axis 0) denotes the potentially multiple estimates. Following
            dimensions represent the dimension of the derivative, i.e. for a classical
            gradient ``derivative`` has 2 dimensions, while for a classical jacobian
            ``derivative`` has 3 dimensions.
        errors (np.ndarray): Error estimates of ``derivative`` estimates. Has the same
            shape as ``derivative``.

    Returns:
        derivative_minimal (np.ndarray): Best derivate estimates chosen with respect
            to minimizing ``errors``. Note that the best values are selected
            element-wise. Has shape ``(derivative.shape[1], derivative.shape[2])``.

        error_minimal (np.ndarray): Minimal errors selected element-wise along axis
            0 of ``errors``.

    """
    if derivative.shape[0] == 1:
        jac_minimal = np.squeeze(derivative, axis=0)
        error_minimal = np.squeeze(errors, axis=0)
    else:
        minimizer = np.nanargmin(errors, axis=0)
        jac_minimal = np.take_along_axis(derivative, minimizer[np.newaxis, :], axis=0)
        jac_minimal = np.squeeze(jac_minimal, axis=0)
        error_minimal = np.nanmin(errors, axis=0)

    return jac_minimal, error_minimal


def _nan_skipping_batch_evaluator(
    func, arguments, n_cores, error_handling, batch_evaluator
):
    """Evaluate func at each entry in arguments, skipping np.nan entries.

    The function is only evaluated at inputs that are not a scalar np.nan.
    The outputs corresponding to skipped inputs as well as for inputs on which func
    returns np.nan are np.nan.

    Args:
        func (function): Python function that returns a numpy array. The shape
            of the output of func has to be the same for all elements in arguments.
        arguments (list): List with inputs for func.
        n_cores (int): Number of processes.

    Returns
        evaluations (list): The function evaluations, same length as arguments.

    """
    # extract information
    nan_indices = {
        i for i, arg in enumerate(arguments) if isinstance(arg, float) and np.isnan(arg)
    }
    real_args = [arg for i, arg in enumerate(arguments) if i not in nan_indices]

    # get the batch evaluator if it was provided as string
    if not callable(batch_evaluator):
        batch_evaluator = getattr(
            batch_evaluators, f"{batch_evaluator}_batch_evaluator"
        )

    # evaluate functions
    evaluations = batch_evaluator(
        func=func, arguments=real_args, n_cores=n_cores, error_handling=error_handling
    )

    # combine results
    evaluations = iter(evaluations)
    results = []
    for i in range(len(arguments)):
        if i in nan_indices:
            results.append(np.nan)
        else:
            results.append(next(evaluations))

    return results


def _split_into_str_and_int(s):
    """Splits string in str and int parts.

    Args:
        s (str): The string.

    Returns:
        str_part (str): The str part.
        int_part (int): The int part.

    Example:
    >>> s = "forward1"
    >>> _split_into_str_and_int(s)
    ('forward', 1)

    """
    str_part, int_part = re.findall(r"(\w+?)(\d+)", s)[0]
    return str_part, int(int_part)


def _collect_additional_info(steps, evals, updated_candidates, target):
    """Combine additional information in dict if return_info is True."""
    info = {}
    # save function evaluations to accessible data frame
    if target == "first_derivative":
        func_evals = _convert_evaluation_data_to_frame(steps, evals)
        info["func_evals"] = func_evals
    else:
        one_step = _convert_evaluation_data_to_frame(steps, evals["one_step"])
        info["func_evals_one_step"] = one_step
        info["func_evals_two_step"] = None
        info["func_evals_cross_step"] = None

    if updated_candidates is not None:
        # combine derivative candidates in accessible data frame
        derivative_candidates = _convert_richardson_candidates_to_frame(
            *updated_candidates
        )
        info["derivative_candidates"] = derivative_candidates

    return info


def _is_scalar_nan(value):
    return isinstance(value, float) and np.isnan(value)


def _unflatten_if_not_nan(leaves, treedef, registry):
    if isinstance(leaves, np.ndarray):
        out = tree_unflatten(treedef, leaves, registry=registry)
    else:
        out = leaves
    return out
