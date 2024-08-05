"""Aggregate the multiple parameter and function output conversions into on."""

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from optimagic.parameters.process_selectors import process_selectors
from optimagic.parameters.scale_conversion import get_scale_converter
from optimagic.parameters.space_conversion import InternalParams, get_space_converter
from optimagic.parameters.tree_conversion import get_tree_converter


def get_converter(
    params,
    constraints,
    bounds,
    func_eval,
    primary_key,
    scaling=None,
    derivative_eval=None,
    add_soft_bounds=False,
):
    """Get a converter between external and internal params and internal params.

    This combines the following conversions:
    - Flattening parameters provided as pytrees (tree_conversion)
    - Enforcing constraints via reparametrizations (space_conversion)
    - Scaling of the parameter space (scale_conversion)

    The resulting converter can transform parameters, function outputs and derivatives.

    If possible, fast paths for some or all transformations are chosen.

    Args:
        params (pytree): The user provided parameters.
        constraints (list): The user provided constraints.
        lower_bounds (pytree): The user provided lower_bounds
        upper_bounds (pytree): The user provided upper bounds
        func_eval (float, dict or pytree): An evaluation of ``func`` at ``params``.
            Used to deterimine how the function output has to be transformed for the
            optimizer.
        primary_key (str): One of "value", "contributions" and "root_contributions".
            Used to determine how the function and derivative output has to be
            transformed for the optimzer.
        scaling (ScalingOptions | None): Scaling options. If None, no scaling is
            performed.
        derivative_eval (dict, pytree or None): Evaluation of the derivative of
            func at params. Used for consistency checks.
        soft_lower_bounds (pytree): As lower_bounds
        soft_upper_bounds (pytree): As upper_bounds
        add_soft_bounds (bool): Whether soft bounds should be added to the
            internal_params

    Returns:
        Converter: NamedTuple with methods to convert between internal and external
            parameters, derivatives and function outputs.
        InternalParams: NamedTuple with internal parameter values, lower_bounds and
            upper_bounds.

    """
    fast_path = _is_fast_path(
        params=params,
        constraints=constraints,
        func_eval=func_eval,
        primary_key=primary_key,
        scaling=scaling,
        derivative_eval=derivative_eval,
        add_soft_bounds=add_soft_bounds,
    )
    if fast_path:
        return _get_fast_path_converter(
            params=params,
            bounds=bounds,
            primary_key=primary_key,
        )

    tree_converter, internal_params = get_tree_converter(
        params=params,
        bounds=bounds,
        func_eval=func_eval,
        derivative_eval=derivative_eval,
        primary_key=primary_key,
        add_soft_bounds=add_soft_bounds,
    )

    flat_constraints = process_selectors(
        constraints=constraints,
        params=params,
        tree_converter=tree_converter,
        param_names=internal_params.names,
    )

    space_converter, internal_params = get_space_converter(
        internal_params=internal_params, internal_constraints=flat_constraints
    )

    scale_converter, scaled_params = get_scale_converter(
        internal_params=internal_params,
        scaling=scaling,
    )

    def _params_to_internal(params):
        x_flat = tree_converter.params_flatten(params)
        x_internal = space_converter.params_to_internal(x_flat)
        x_scaled = scale_converter.params_to_internal(x_internal)
        return x_scaled

    def _params_from_internal(x, return_type="tree"):
        x_unscaled = scale_converter.params_from_internal(x)
        x_external = space_converter.params_from_internal(x_unscaled)

        x_tree = tree_converter.params_unflatten(x_external)
        if return_type == "tree":
            out = x_tree
        elif return_type == "tree_and_flat":
            out = x_tree, x_external
        elif return_type == "flat":
            out = x_external
        else:
            msg = (
                f"Invalid return type: {return_type}. Must be one of 'tree', 'flat', "
                "'tree_and_flat'"
            )
            raise ValueError(msg)
        return out

    def _derivative_to_internal(derivative_eval, x, jac_is_flat=False):
        if jac_is_flat:
            jacobian = derivative_eval
        else:
            jacobian = tree_converter.derivative_flatten(derivative_eval)
        x_unscaled = scale_converter.params_from_internal(x)
        jac_with_space_conversion = space_converter.derivative_to_internal(
            jacobian, x_unscaled
        )
        jac_with_unscaling = scale_converter.derivative_to_internal(
            jac_with_space_conversion
        )
        return jac_with_unscaling

    def _func_to_internal(func_eval):
        return tree_converter.func_flatten(func_eval)

    internal_params = replace(scaled_params, free_mask=internal_params.free_mask)

    converter = Converter(
        params_to_internal=_params_to_internal,
        params_from_internal=_params_from_internal,
        derivative_to_internal=_derivative_to_internal,
        func_to_internal=_func_to_internal,
        has_transforming_constraints=space_converter.has_transforming_constraints,
    )

    return converter, internal_params


@dataclass(frozen=True)
class Converter:
    params_to_internal: Callable
    params_from_internal: Callable
    derivative_to_internal: Callable
    func_to_internal: Callable
    has_transforming_constraints: bool


def aggregate_func_output_to_value(f_eval, primary_key):
    if primary_key == "value":
        return f_eval
    elif primary_key == "contributions":
        return f_eval.sum()
    elif primary_key == "root_contributions":
        return f_eval @ f_eval


def _unpack_value_if_needed(func_eval):
    if isinstance(func_eval, dict):
        return float(func_eval["value"])
    else:
        return func_eval


def _unpack_contributions_if_needed(func_eval):
    if isinstance(func_eval, dict):
        return func_eval["contributions"].astype(float)
    else:
        return func_eval.astype(float)


def _unpack_root_contributions_if_needed(func_eval):
    if isinstance(func_eval, dict):
        return func_eval["root_contributions"].astype(float)
    else:
        return func_eval.astype(float)


UNPACK_FUNCTIONS = {
    "value": _unpack_value_if_needed,
    "contributions": _unpack_contributions_if_needed,
    "root_contributions": _unpack_root_contributions_if_needed,
}


def _fast_params_from_internal(x, return_type="tree"):
    x = x.astype(float)
    if return_type == "tree_and_flat":
        return x, x
    else:
        return x


def _get_fast_path_converter(params, bounds, primary_key):
    def _fast_derivative_to_internal(
        derivative_eval,
        x,  # noqa: ARG001
        jac_is_flat=True,  # noqa: ARG001
    ):
        # make signature compatible with non-fast path
        return derivative_eval

    converter = Converter(
        params_to_internal=lambda params: params.astype(float),
        params_from_internal=_fast_params_from_internal,
        derivative_to_internal=_fast_derivative_to_internal,
        func_to_internal=UNPACK_FUNCTIONS[primary_key],
        has_transforming_constraints=False,
    )

    if bounds is None or bounds.lower is None:
        lower_bounds = np.full(len(params), -np.inf)
    else:
        lower_bounds = bounds.lower.astype(float)

    if bounds is None or bounds.upper is None:
        upper_bounds = np.full(len(params), np.inf)
    else:
        upper_bounds = bounds.upper.astype(float)

    internal_params = InternalParams(
        values=params.astype(float),
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        free_mask=np.full(len(params), True),
        names=[str(i) for i in range(len(params))],
    )
    return converter, internal_params


def _is_fast_path(
    params,
    constraints,
    func_eval,
    primary_key,
    scaling,
    derivative_eval,
    add_soft_bounds,
):
    if not _is_1d_arr(params):
        return False
    if constraints:
        return False
    if not _is_fast_func_eval(func_eval, primary_key):
        return False

    if scaling is not None:
        return False

    if not _is_fast_deriv_eval(derivative_eval, primary_key):
        return False

    if add_soft_bounds:
        return False

    return True


def _is_fast_func_eval(f, key):
    if key == "value":
        if not (np.isscalar(f) or (_is_dict_with(f, key) and np.isscalar(f[key]))):
            return False
    else:
        if not (_is_1d_arr(f) or (_is_dict_with(f, key)) and _is_1d_arr(f[key])):
            return False

    return True


def _is_fast_deriv_eval(d, key):
    # this is the case if no or closed form derivatives are used
    if d is None:
        return True

    if key == "value":
        if not (_is_1d_arr(d) or (_is_dict_with(d, key) and _is_1d_arr(d[key]))):
            return False
    else:
        if not (_is_2d_arr(d) or (_is_dict_with(d, key)) and _is_2d_arr(d[key])):
            return False

    return True


def _is_1d_arr(candidate):
    return isinstance(candidate, np.ndarray) and candidate.ndim == 1


def _is_2d_arr(candidate):
    return isinstance(candidate, np.ndarray) and candidate.ndim == 2


def _is_dict_with(candidate, key):
    return isinstance(candidate, dict) and key in candidate
