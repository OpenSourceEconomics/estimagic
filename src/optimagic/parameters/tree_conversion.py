from typing import Callable, NamedTuple

import numpy as np
from pybaum import leaf_names, tree_flatten, tree_just_flatten, tree_unflatten

from optimagic.exceptions import InvalidFunctionError
from optimagic.parameters.block_trees import block_tree_to_matrix
from optimagic.parameters.bounds import get_internal_bounds
from optimagic.parameters.tree_registry import get_registry
from optimagic.utilities import isscalar


def get_tree_converter(
    params,
    bounds,
    func_eval,
    primary_key,
    derivative_eval=None,
    add_soft_bounds=False,
):
    """Get flatten and unflatten functions for criterion and its derivative.

    The function creates a converter with methods to convert parameters, derivatives
    and the output of the criterion function between the user provided pytree structure
    and flat representations.

    The main motivation for bundling all of this together (as opposed to handling
    parameters, derivatives and function outputs separately) is that the derivative
    conversion needs to know about the structure of params and the criterion output.

    Args:
        params (pytree): The user provided parameters.
        lower_bounds (pytree): The user provided lower_bounds
        upper_bounds (pytree): The user provided upper bounds
        func_eval (float, dict or pytree): An evaluation of ``func`` at ``params``.
            Used to deterimine how the function output has to be transformed for the
            optimizer.
        primary_key (str): One of "value", "contributions" and "root_contributions".
            Used to determine how the function and derivative output has to be
            transformed for the optimzer.
        derivative_eval (dict, pytree or None): Evaluation of the derivative of
            func at params. Used for consistency checks.
        soft_lower_bounds (pytree): As lower_bounds
        soft_upper_bounds (pytree): As upper_bounds
        add_soft_bounds (bool): Whether soft bounds should be added to the flat_params

    Returns:
        TreeConverter: NamedTuple with flatten and unflatten methods.
        FlatParams: NamedTuple of 1d arrays with flattened bounds and param names.

    """
    _registry = get_registry(extended=True)
    _params_vec, _params_treedef = tree_flatten(params, registry=_registry)
    _params_vec = np.array(_params_vec).astype(float)
    _lower, _upper = get_internal_bounds(
        params=params,
        bounds=bounds,
        registry=_registry,
    )

    if add_soft_bounds:
        _soft_lower, _soft_upper = get_internal_bounds(
            params=params,
            bounds=bounds,
            registry=_registry,
            add_soft_bounds=add_soft_bounds,
        )
    else:
        _soft_lower, _soft_upper = None, None

    _param_names = leaf_names(params, registry=_registry)

    flat_params = FlatParams(
        values=_params_vec,
        lower_bounds=_lower,
        upper_bounds=_upper,
        names=_param_names,
        soft_lower_bounds=_soft_lower,
        soft_upper_bounds=_soft_upper,
    )

    _params_flatten = _get_params_flatten(registry=_registry)
    _params_unflatten = _get_params_unflatten(
        registry=_registry, treedef=_params_treedef
    )
    _func_flatten = _get_func_flatten(
        registry=_registry,
        func_eval=func_eval,
        primary_key=primary_key,
    )
    _derivative_flatten = _get_derivative_flatten(
        registry=_registry,
        primary_key=primary_key,
        params=params,
        func_eval=func_eval,
        derivative_eval=derivative_eval,
    )

    converter = TreeConverter(
        params_flatten=_params_flatten,
        params_unflatten=_params_unflatten,
        func_flatten=_func_flatten,
        derivative_flatten=_derivative_flatten,
    )

    return converter, flat_params


def _get_params_flatten(registry):
    def params_flatten(params):
        return np.array(tree_just_flatten(params, registry=registry)).astype(float)

    return params_flatten


def _get_params_unflatten(registry, treedef):
    def params_unflatten(x):
        return tree_unflatten(treedef=treedef, leaves=list(x), registry=registry)

    return params_unflatten


def _get_func_flatten(registry, func_eval, primary_key):
    if isscalar(func_eval):
        if primary_key == "value":
            func_flatten = lambda func_eval: float(func_eval)
        else:
            msg = (
                "criterion returns a scalar value but the requested optimizer "
                "requires a vector or pytree output. criterion can either return this "
                f"output alone or inside a dictionary with the key {primary_key}."
            )
            raise InvalidFunctionError(msg)
    elif not isinstance(func_eval, dict):
        raise ValueError()  # xxxx
    else:
        key, aggregate = _get_best_key_and_aggregator(primary_key, func_eval)

        def func_flatten(func_eval):
            # the if condition is necessary, such that we can also accept func_evals
            # where the primary entry has already been extracted. This is for example
            # necessary if the criterion_and_derivative returns only the relevant
            # entry of criterion, whereas criterion returns a dict.
            if isinstance(func_eval, dict) and key in func_eval:
                func_eval = func_eval[key]
            return aggregate(tree_just_flatten(func_eval, registry=registry))

    return func_flatten


def _get_best_key_and_aggregator(needed_key, available_keys):
    if needed_key in available_keys:
        key = needed_key
        if needed_key == "value":
            aggregate = lambda x: float(x[0])
        else:
            aggregate = lambda x: np.array(x).astype(float)
    elif needed_key == "contributions" and "root_contributions" in available_keys:
        key = "root_contributions"
        aggregate = lambda x: np.array(x).astype(float) ** 2
    elif needed_key == "value" and "contributions" in available_keys:
        key = "contributions"
        aggregate = lambda x: float(np.sum(x))
    elif needed_key == "value" and "root_contributions" in available_keys:
        key = "root_contributions"
        aggregate = lambda x: float((np.array(x) ** 2).sum())
    else:
        msg = (
            "The optimizer you requested requires a criterion function that returns "
            f"a dictionary with the entry '{needed_key}'. Your function returns a "
            f"dictionary that only contains the entries {available_keys}."
        )
        raise InvalidFunctionError(msg)

    return key, aggregate


def _get_derivative_flatten(registry, primary_key, params, func_eval, derivative_eval):
    # gradient case
    if primary_key == "value":

        def derivative_flatten(derivative_eval):
            flat = np.array(
                tree_just_flatten(derivative_eval, registry=registry)
            ).astype(float)
            return flat

    # jacobian case
    else:
        key, _ = _get_best_key_and_aggregator(primary_key, func_eval)

        def derivative_flatten(derivative_eval):
            flat = block_tree_to_matrix(
                derivative_eval,
                outer_tree=func_eval[key],
                inner_tree=params,
            )
            return flat

    if derivative_eval is not None:
        try:
            derivative_flatten(derivative_eval)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            msg = "The output of derivative and criterion cannot be aligned."
            raise InvalidFunctionError(msg) from e

    return derivative_flatten


class TreeConverter(NamedTuple):
    params_flatten: Callable
    params_unflatten: Callable
    func_flatten: Callable
    derivative_flatten: Callable


class FlatParams(NamedTuple):
    values: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    soft_lower_bounds: np.ndarray | None = None
    soft_upper_bounds: np.ndarray | None = None
    names: list | None = None
    free_mask: np.ndarray | None = None
