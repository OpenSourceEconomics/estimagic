from functools import partial

import numpy as np

from optimagic.config import CRITERION_PENALTY_CONSTANT, CRITERION_PENALTY_SLOPE
from optimagic.parameters.conversion import aggregate_func_output_to_value


def _penalty_value(x, constant, slope, x0, dim_out=None):  # noqa: ARG001
    return constant + slope * np.linalg.norm(x - x0)


def _penalty_contributions(x, constant, slope, x0, dim_out):
    contrib = (constant + slope * np.linalg.norm(x - x0)) / dim_out
    return np.ones(dim_out) * contrib


def _penalty_root_contributions(x, constant, slope, x0, dim_out):
    contrib = np.sqrt((constant + slope * np.linalg.norm(x - x0)) / dim_out)
    return np.ones(dim_out) * contrib


def _penalty_value_derivative(x, constant, slope, x0, dim_out=None):  # noqa: ARG001
    return slope * (x - x0) / np.linalg.norm(x - x0)


def _penalty_contributions_derivative(x, constant, slope, x0, dim_out):  # noqa: ARG001
    row = slope * (x - x0) / (dim_out * np.linalg.norm(x - x0))
    return np.full((dim_out, len(x)), row)


def _penalty_root_contributions_derivative(x, constant, slope, x0, dim_out):
    inner_deriv = slope * (x - x0) / np.linalg.norm(x - x0)
    outer_deriv = 0.5 / np.sqrt(_penalty_value(x, constant, slope, x0) * dim_out)
    row = outer_deriv * inner_deriv
    return np.full((dim_out, len(x)), row)


def get_error_penalty_function(
    error_handling,
    start_x,
    start_criterion,
    error_penalty,
    primary_key,
    direction,
):
    if error_handling == "raise":
        return None
    elif error_handling != "continue":
        raise ValueError("Error handling must be 'raise' or 'continue'")

    error_penalty = {} if error_penalty is None else error_penalty

    first_value = aggregate_func_output_to_value(
        f_eval=start_criterion, primary_key=primary_key
    )

    constant, slope = _process_error_penalty(
        error_penalty=error_penalty,
        first_value=first_value,
        direction=direction,
    )

    dim_out = 1 if primary_key == "value" else len(start_criterion)

    kwargs = {"constant": constant, "slope": slope, "x0": start_x, "dim_out": dim_out}

    if primary_key == "value":
        _penalty = partial(_penalty_value, **kwargs)
        _derivative = partial(_penalty_value_derivative, **kwargs)
    elif primary_key == "contributions":
        _penalty = partial(_penalty_contributions, **kwargs)
        _derivative = partial(
            _penalty_contributions_derivative,
            **kwargs,
        )
    elif primary_key == "root_contributions":
        _penalty = partial(
            _penalty_root_contributions,
            **kwargs,
        )
        _derivative = partial(_penalty_root_contributions_derivative, **kwargs)

    else:
        raise ValueError()

    def penalty(x, task):
        if task == "criterion":
            out = _penalty(x)
        elif task == "derivative":
            out = _derivative(x)
        elif task == "criterion_and_derivative":
            out = (_penalty(x), _derivative(x))

        return out

    return penalty


def _process_error_penalty(error_penalty, first_value, direction):
    """Add default options to error_penalty options."""
    error_penalty = error_penalty.copy()

    if direction == "minimize":
        default_constant = (
            first_value + np.abs(first_value) + CRITERION_PENALTY_CONSTANT
        )
        default_slope = CRITERION_PENALTY_SLOPE
    else:
        default_constant = (
            first_value - np.abs(first_value) - CRITERION_PENALTY_CONSTANT
        )
        default_slope = -CRITERION_PENALTY_SLOPE

    constant = error_penalty.get("constant", default_constant)
    slope = error_penalty.get("slope", default_slope)

    return constant, slope
