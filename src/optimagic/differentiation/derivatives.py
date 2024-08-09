import functools
import itertools
import re
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Literal, NamedTuple, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pybaum import tree_flatten, tree_unflatten
from pybaum import tree_just_flatten as tree_leaves

from optimagic import batch_evaluators, deprecations
from optimagic.config import DEFAULT_N_CORES
from optimagic.deprecations import (
    replace_and_warn_about_deprecated_base_steps,
    replace_and_warn_about_deprecated_bounds,
)
from optimagic.differentiation import finite_differences
from optimagic.differentiation.generate_steps import generate_steps
from optimagic.differentiation.richardson_extrapolation import richardson_extrapolation
from optimagic.parameters.block_trees import hessian_to_block_tree, matrix_to_block_tree
from optimagic.parameters.bounds import Bounds, get_internal_bounds, pre_process_bounds
from optimagic.parameters.tree_registry import get_registry
from optimagic.typing import PyTree


@dataclass(frozen=True)
class NumdiffResult:
    """Result of a numerical differentiation.

    The following relationship holds for vector-valued functions with vector-valued
    parameters:

    First Derivative:
    -----------------

    - f: R -> R leads to shape (1,), usually called derivative
    - f: R^m -> R leads to shape (m, ), usually called Gradient
    - f: R -> R^n leads to shape (n, 1), usually called Jacobian
    - f: R^m -> R^n leads to shape (n, m), usually called Jacobian

    Second Derivative:
    ------------------

    - f: R -> R leads to shape (1,), usually called second derivative
    - f: R^m -> R leads to shape (m, m), usually called Hessian
    - f: R -> R^n leads to shape (n,), usually called Hessian
    - f: R^m -> R^n leads to shape (n, m, m), usually called Hessian tensor

    Attributes:
        derivative: The estimated derivative at the parameters. The structure of the
            derivative depends on the input parameters and the output of the function.
        func_value: The value of the function at the parameters.

    """

    derivative: PyTree
    func_value: PyTree | None = None
    # deprecated
    _func_evals: pd.DataFrame | dict[str, pd.DataFrame | None] | None = None
    _derivative_candidates: pd.DataFrame | None = None

    @property
    def func_evals(self) -> pd.DataFrame | dict[str, pd.DataFrame | None] | None:
        deprecations.throw_numdiff_result_func_evals_future_warning()
        return self._func_evals

    @property
    def derivative_candidates(self) -> pd.DataFrame | None:
        deprecations.throw_numdiff_result_derivative_candidates_future_warning()
        return self._derivative_candidates

    def __getitem__(self, key: str) -> Any:
        deprecations.throw_dict_access_future_warning(key, obj_name=type(self).__name__)
        return getattr(self, key)


class Evals(NamedTuple):
    pos: NDArray[np.float64]
    neg: NDArray[np.float64]


def first_derivative(
    func: Callable[[PyTree], PyTree],
    params: PyTree,
    *,
    bounds: Bounds | None = None,
    func_kwargs: dict[str, Any] | None = None,
    method: Literal["central", "forward", "backward"] = "central",
    step_size: float | PyTree | None = None,
    scaling_factor: float | PyTree = 1,
    min_steps: float | PyTree | None = None,
    f0: PyTree | None = None,
    n_cores: int = DEFAULT_N_CORES,
    error_handling: Literal["continue", "raise", "raise_strict"] = "continue",
    batch_evaluator: Literal["joblib", "pathos"] | Callable = "joblib",
    unpacker: Callable[[Any], PyTree] | None = None,
    # deprecated
    lower_bounds: PyTree | None = None,
    upper_bounds: PyTree | None = None,
    base_steps: PyTree | None = None,
    key: str | None = None,
    step_ratio: float | None = None,
    n_steps: int | None = None,
    return_info: bool | None = None,
    return_func_value: bool | None = None,
) -> NumdiffResult:
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
        func: Function of which the derivative is calculated.
        params: A pytree. See :ref:`params`.
        bounds: Lower and upper bounds on the parameters. The most general and preferred
            way to specify bounds is an `optimagic.Bounds` object that collects lower,
            upper, soft_lower and soft_upper bounds. The soft bounds are not used during
            numerical differentiation. Each bound type mirrors the structure of params.
            Check our how-to guide on bounds for examples. If params is a flat numpy
            array, you can also provide bounds via any format that is supported by
            scipy.optimize.minimize.
        func_kwargs: Additional keyword arguments for func, optional.
        method: One of ["central", "forward", "backward"], default "central".
        step_size: 1d array of the same length as params.
            step_size * scaling_factor is the absolute value of the first (and possibly
            only) step used in the finite differences approximation of the derivative.
            If step_size * scaling_factor conflicts with bounds, the actual steps will
            be adjusted. If step_size is not provided, it will be determined according
            to a rule of thumb as long as this does not conflict with min_steps.
        scaling_factor: Scaling factor which is applied to step_size. If it is an
            numpy.ndarray, it needs to be as long as params. scaling_factor is useful if
            you want to increase or decrease the base_step relative to the rule-of-thumb
            or user provided base_step, for example to benchmark the effect of the step
            size. Default 1.
        min_steps: Minimal possible step sizes that can be chosen to accommodate bounds.
            Must have same length as params. By default min_steps is equal to step_size,
            i.e step size is not decreased beyond what is optimal according to the rule
            of thumb.
        f0: 1d numpy array with func(x), optional.
        n_cores: Number of processes used to parallelize the function evaluations.
            Default 1.
        error_handling: One of "continue" (catch errors and continue to calculate
            derivative estimates. In this case, some derivative estimates can be
            missing but no errors are raised), "raise" (catch errors and continue
            to calculate derivative estimates at first but raise an error if all
            evaluations for one parameter failed) and "raise_strict" (raise an error
            as soon as a function evaluation fails).
        batch_evaluator (str or callable): Name of a pre-implemented batch evaluator
            (currently 'joblib' and 'pathos_mp') or Callable with the same interface
            as the optimagic batch_evaluators.
        unpacker: A callable that takes the output of func and returns the part of the
            output that is needed for the derivative calculation. If None, the output of
            func is used as is. Default None.

    Returns:
        NumdiffResult: A numerical differentiation result.

    """
    # ==================================================================================
    # handle deprecations
    # ==================================================================================
    bounds = replace_and_warn_about_deprecated_bounds(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        bounds=bounds,
    )

    step_size = replace_and_warn_about_deprecated_base_steps(
        step_size=step_size,
        base_steps=base_steps,
    )

    if key is not None:
        deprecations.throw_key_warning_in_derivatives()
        if unpacker is None:
            unpacker = lambda x: x[key]

    if step_ratio is not None:
        deprecations.throw_derivatives_step_ratio_future_warning()
    else:
        step_ratio = 2

    if n_steps is not None:
        deprecations.throw_derivatives_n_steps_future_warning()
    else:
        n_steps = 1

    if return_info is not None:
        deprecations.throw_derivatives_return_info_future_warning()
    else:
        return_info = False

    if return_func_value is not None:
        deprecations.throw_derivatives_return_func_value_future_warning()
    else:
        return_func_value = True

    # ==================================================================================

    bounds = pre_process_bounds(bounds)
    unpacker = _process_unpacker(unpacker)

    # ==================================================================================
    # Convert arguments to numpy
    # ==================================================================================
    registry = get_registry(extended=True)

    fast_path = _is_1d_array(params)

    if not fast_path:
        x, params_treedef = tree_flatten(params, registry=registry)
        x = np.asarray(x, dtype=np.float64)

        if not isinstance(scaling_factor, int | float):
            scaling_factor, _ = tree_flatten(scaling_factor, registry=registry)

        if min_steps is not None:
            min_steps, _ = tree_flatten(min_steps, registry=registry)

        if step_size is not None:
            step_size, _ = tree_flatten(step_size, registry=registry)

    else:
        x = params.astype(np.float64)

        if min_steps is not None:
            min_steps = np.atleast_1d(min_steps)

        if step_size is not None:
            step_size = np.atleast_1d(step_size)

    if isinstance(scaling_factor, int | float):
        pass
    elif len(scaling_factor) == len(x):
        scaling_factor = np.asarray(scaling_factor, dtype=np.float64)
    else:
        raise ValueError(
            "scaling_factor must be an int, a float or have the same structure as "
            "params."
        )

    if min_steps is None:
        pass
    elif len(min_steps) == 1:
        min_steps = np.full_like(x, min_steps[0])
    elif len(min_steps) == len(x):
        min_steps = np.asarray(min_steps, dtype=np.float64)
    else:
        raise ValueError(
            "min_steps must be a float or have the same structure as params."
        )

    if step_size is None:
        pass
    elif len(step_size) == 1:
        step_size = np.full_like(x, step_size[0])
    elif len(step_size) == len(x):
        step_size = np.asarray(step_size, dtype=np.float64)
    else:
        raise ValueError(
            "step_size must be a float or have the same structure as params."
        )

    if np.isnan(x).any():
        raise ValueError("The parameter vector must not contain NaNs.")

    internal_lb, internal_ub = get_internal_bounds(params, bounds=bounds)

    # handle kwargs
    func_kwargs = {} if func_kwargs is None else func_kwargs
    partialed_func = functools.partial(func, **func_kwargs)

    implemented_methods = {"forward", "backward", "central"}
    if method not in implemented_methods:
        raise ValueError(f"Method has to be in {implemented_methods}.")

    # generate the step array
    step_size = generate_steps(
        x=x,
        method=method,
        n_steps=n_steps,
        target="first_derivative",
        base_steps=step_size,
        scaling_factor=scaling_factor,
        bounds=Bounds(lower=internal_lb, upper=internal_ub),
        step_ratio=step_ratio,
        min_steps=min_steps,
    )
    step_size = cast(NDArray[np.float64], step_size)

    # generate parameter vectors at which func has to be evaluated as numpy arrays
    evaluation_points = []
    for step_arr in step_size:
        for i, j in product(range(n_steps), range(len(x))):
            if np.isnan(step_arr[i, j]):
                evaluation_points.append(np.nan)
            else:
                point = x.copy()
                point[j] += step_arr[i, j]
                evaluation_points.append(point)

    # convert the numpy arrays to whatever is needed by func
    if not fast_path:
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

    f0_tree = unpacker(f0)
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
        unpacker=unpacker,
        registry=registry,
        is_scalar_out=scalar_out,
        is_vector_out=vector_out,
    )

    # apply finite difference formulae
    evals_data = np.array(raw_evals).reshape(2, n_steps, len(x), -1)
    evals_data = np.transpose(evals_data, axes=(0, 1, 3, 2))
    evals = Evals(pos=evals_data[0], neg=evals_data[1])

    jac_candidates = {}
    for m in ["forward", "backward", "central"]:
        jac_candidates[m] = finite_differences.jacobian(evals, step_size, f0, m)

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
            jac_candidates, step_size, n_steps
        )
        jac, updated_candidates = _consolidate_extrapolated(richardson_candidates)

    # raise error if necessary
    if error_handling in ("raise", "raise_strict") and np.isnan(jac).any():
        raise Exception(exc_info)

    # results processing
    if fast_path and vector_out:
        derivative = jac
    elif fast_path and scalar_out:
        derivative = jac.flatten()
    else:
        derivative = matrix_to_block_tree(jac, f0_tree, params)

    result = {"derivative": derivative}
    if return_func_value:
        result["func_value"] = func_value
    if return_info:
        info = _collect_additional_info(
            step_size, evals, updated_candidates, target="first_derivative"
        )
        result = {**result, **info}
    return NumdiffResult(**result)


def second_derivative(
    func: Callable[[PyTree], PyTree],
    params: PyTree,
    *,
    bounds: Bounds | None = None,
    func_kwargs: dict[str, Any] | None = None,
    method: Literal[
        "forward", "backward", "central_average", "central_cross"
    ] = "central_cross",
    step_size: float | PyTree | None = None,
    scaling_factor: float | PyTree = 1,
    min_steps: float | PyTree | None = None,
    f0: PyTree | None = None,
    n_cores: int = DEFAULT_N_CORES,
    error_handling: Literal["continue", "raise", "raise_strict"] = "continue",
    batch_evaluator: Literal["joblib", "pathos"] | Callable = "joblib",
    unpacker: Callable[[Any], PyTree] | None = None,
    # deprecated
    lower_bounds: PyTree | None = None,
    upper_bounds: PyTree | None = None,
    base_steps: PyTree | None = None,
    step_ratio: float | None = None,
    n_steps: int | None = None,
    return_info: bool | None = None,
    return_func_value: bool | None = None,
    key: str | None = None,
) -> NumdiffResult:
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
        func: Function of which the derivative is calculated.
        params: 1d numpy array or
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
        func_kwargs: Additional keyword arguments for func, optional.
        method: One of {"forward", "backward", "central_average", "central_cross"}
            These correspond to the finite difference approximations defined in
            equations [7, x, 8, 9] in Rideout [2009], where ("backward", x) is not found
            in Rideout [2009] but is the natural extension of equation 7 to the backward
            case. Default "central_cross".
        step_size: 1d array of the same length as params.
            step_size * scaling_factor is the absolute value of the first (and possibly
            only) step used in the finite differences approximation of the derivative.
            If step_size * scaling_factor conflicts with bounds, the actual steps will
            be adjusted. If step_size is not provided, it will be determined according
            to a rule of thumb as long as this does not conflict with min_steps.
        scaling_factor: Scaling factor which is applied to
            step_size. If it is an numpy.ndarray, it needs to be as long as params.
            scaling_factor is useful if you want to increase or decrease the base_step
            relative to the rule-of-thumb or user provided base_step, for example to
            benchmark the effect of the step size. Default 1.
        min_steps: Minimal possible step sizes that can be chosen to
            accommodate bounds. Must have same length as params. By default min_steps is
            equal to step_size, i.e step size is not decreased beyond what is optimal
            according to the rule of thumb.
        f0: 1d numpy array with func(x), optional.
        n_cores: Number of processes used to parallelize the function
            evaluations. Default 1.
        error_handling: One of "continue" (catch errors and continue to calculate
            derivative estimates. In this case, some derivative estimates can be
            missing but no errors are raised), "raise" (catch errors and continue
            to calculate derivative estimates at first but raise an error if all
            evaluations for one parameter failed) and "raise_strict" (raise an error
            as soon as a function evaluation fails).
        batch_evaluator: Name of a pre-implemented batch evaluator
            (currently 'joblib' and 'pathos_mp') or Callable with the same interface
            as the optimagic batch_evaluators.
        unpacker: A callable that takes the output of func and returns the part of the
            output that is needed for the derivative calculation. If None, the output of
            func is used as is. Default None.


    Returns:
        NumdiffResult: A numerical differentiation result.

    """

    # ==================================================================================
    # handle deprecations
    # ==================================================================================
    bounds = replace_and_warn_about_deprecated_bounds(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        bounds=bounds,
    )

    step_size = replace_and_warn_about_deprecated_base_steps(
        step_size=step_size,
        base_steps=base_steps,
    )

    if step_ratio is not None:
        deprecations.throw_derivatives_step_ratio_future_warning()
    else:
        step_ratio = 2

    if n_steps is not None:
        deprecations.throw_derivatives_n_steps_future_warning()
    else:
        n_steps = 1

    if return_info is not None:
        deprecations.throw_derivatives_return_info_future_warning()
    else:
        return_info = False

    if return_func_value is not None:
        deprecations.throw_derivatives_return_func_value_future_warning()
    else:
        return_func_value = True

    if key is not None:
        deprecations.throw_key_warning_in_derivatives()
        if unpacker is None:
            unpacker = lambda x: x[key]

    # ==================================================================================
    bounds = pre_process_bounds(bounds)
    unpacker = _process_unpacker(unpacker)

    # ==================================================================================
    # Convert arguments to numpy
    # ==================================================================================
    registry = get_registry(extended=True)

    fast_path = _is_1d_array(params)

    if not fast_path:
        x, params_treedef = tree_flatten(params, registry=registry)
        x = np.asarray(x, dtype=np.float64)

        if not isinstance(scaling_factor, int | float):
            scaling_factor, _ = tree_flatten(scaling_factor, registry=registry)

        if min_steps is not None:
            min_steps, _ = tree_flatten(min_steps, registry=registry)

        if step_size is not None:
            step_size, _ = tree_flatten(step_size, registry=registry)

    else:
        x = params.astype(np.float64)

        if min_steps is not None:
            min_steps = np.atleast_1d(min_steps)

        if step_size is not None:
            step_size = np.atleast_1d(step_size)

    if isinstance(scaling_factor, int | float):
        pass
    elif len(scaling_factor) == len(x):
        scaling_factor = np.asarray(scaling_factor, dtype=np.float64)
    else:
        raise ValueError(
            "scaling_factor must be an int, a float or have the same structure as "
            "params."
        )

    if min_steps is None:
        pass
    elif len(min_steps) == 1:
        min_steps = np.full_like(x, min_steps[0])
    elif len(min_steps) == len(x):
        min_steps = np.asarray(min_steps, dtype=np.float64)
    else:
        raise ValueError(
            "min_steps must be a float or have the same structure as params."
        )

    if step_size is None:
        pass
    elif len(step_size) == 1:
        step_size = np.full_like(x, step_size[0])
    elif len(step_size) == len(x):
        step_size = np.asarray(step_size, dtype=np.float64)
    else:
        raise ValueError(
            "step_size must be a float or have the same structure as params."
        )

    unpacker = _process_unpacker(unpacker)

    internal_lb, internal_ub = get_internal_bounds(params, bounds=bounds)

    # handle kwargs
    func_kwargs = {} if func_kwargs is None else func_kwargs
    partialed_func = functools.partial(func, **func_kwargs)

    implemented_methods = {"forward", "backward", "central_average", "central_cross"}
    if method not in implemented_methods:
        raise ValueError(f"Method has to be in {implemented_methods}.")

    # generate the step array
    step_size = generate_steps(
        x=x,
        method=("central" if "central" in method else method),
        n_steps=n_steps,
        target="second_derivative",
        base_steps=step_size,
        scaling_factor=scaling_factor,
        bounds=Bounds(lower=internal_lb, upper=internal_ub),
        step_ratio=step_ratio,
        min_steps=min_steps,
    )
    step_size = cast(NDArray[np.float64], step_size)

    # generate parameter vectors at which func has to be evaluated as numpy arrays
    evaluation_points = {  # type: ignore
        "one_step": [],
        "two_step": [],
        "cross_step": [],
    }
    for step_arr in step_size:
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
    if not fast_path:
        evaluation_points = {
            # entries are either a numpy.ndarray or np.nan, we unflatten only
            step_type: [
                _unflatten_if_not_nan(p, params_treedef, registry) for p in points
            ]
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

    f0_tree = unpacker(f0)
    f0 = tree_leaves(f0_tree, registry=registry)
    f0 = np.array(f0, dtype=np.float64)

    # convert the raw evaluations to numpy arrays
    raw_evals = {
        step_type: _convert_evals_to_numpy(
            raw_evals=evals, unpacker=unpacker, registry=registry
        )
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
        hess_candidates[m] = finite_differences.hessian(evals, step_size, f0, m)

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
            step_size, evals, updated_candidates, target="second_derivative"
        )
        result = {**result, **info}
    return NumdiffResult(**result)


def _is_1d_array(candidate: Any) -> bool:
    return isinstance(candidate, np.ndarray) and candidate.ndim == 1


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


def _process_unpacker(
    unpacker: None | Callable[[Any], PyTree],
) -> Callable[[Any], PyTree]:
    """Process the user provided unpacker function.

    If the unpacker was None, we set it to the identity.

    """
    if unpacker is None:
        unpacker = lambda x: x
    else:
        raw_unpacker = unpacker

        def unpacker(x):
            if isinstance(x, float) and np.isnan(x):
                return x
            return raw_unpacker(x)

    return unpacker


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
    raw_evals, unpacker, registry, is_scalar_out=False, is_vector_out=False
):
    """Harmonize the output of the function evaluations.

    The raw_evals might contain dictionaries of which we only need one entry, scalar
    np.nan where we need arrays filled with np.nan or pandas objects. The processed
    evals only contain numpy arrays.

    """
    # get rid of additional output
    evals = [unpacker(val) for val in raw_evals]

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
        info["_func_evals"] = func_evals
    else:
        one_step = _convert_evaluation_data_to_frame(steps, evals["one_step"])
        info["_func_evals"] = {
            "one_step": one_step,
            "two_step": None,
            "cross_step": None,
        }

    if updated_candidates is not None:
        # combine derivative candidates in accessible data frame
        derivative_candidates = _convert_richardson_candidates_to_frame(
            *updated_candidates
        )
        info["_derivative_candidates"] = derivative_candidates

    return info


def _is_scalar_nan(value):
    return isinstance(value, float) and np.isnan(value)


def _unflatten_if_not_nan(leaves, treedef, registry):
    if isinstance(leaves, np.ndarray):
        out = tree_unflatten(treedef, leaves, registry=registry)
    else:
        out = leaves
    return out
