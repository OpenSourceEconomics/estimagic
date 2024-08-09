import warnings
from typing import NamedTuple

import numpy as np


class Steps(NamedTuple):
    pos: np.ndarray
    neg: np.ndarray


def generate_steps(
    x,
    method,
    n_steps,
    target,
    base_steps,
    scaling_factor,
    bounds,
    step_ratio,
    min_steps,
):
    """Generate steps for finite differences with or without Richardson Extrapolation.

    steps can be used to construct x-vectors at which the function has to be evaluated
    for finite difference formulae. How the vectors are constructed from the steps
    differs between first and second derivative. Note that both positive and negative
    steps are returned, even for one-sided methods, because bounds might make it
    necessary to flip the direction of the method.

    The rule of thumb for the generation of base_steps is:
    - first_derivative: `np.finfo(float).eps ** (1 / 2) * np.maximum(np.abs(x), 0.1)`
    - second_derivative: `np.finfo(float).eps ** (1 / 3) * np.maximum(np.abs(x), 0.1)`
    Where `np.finfo(float).eps` is machine accuracy. This rule of thumb
    is also used in statsmodels and scipy.

    The step generation is bound aware and will try to find a good solution if
    any step would violate a bound. For this, we use the following rules until
    no bounds are violated:

    1. If a one sided method is used, flip to the direction with more distance
        to the bound.
    2. Decrease the base_steps, unless this would mean to go below min_steps. By default
        min_steps is equal to base_steps, so no squeezing happens unless explicitly
        requested by setting a smaller min_step.
    3. Set the conflicting steps to NaN, which means that this step won't be
        usable in the calculation of derivatives. All derivative functions can
        handle NaNs and will produce the best possible derivative estimate given
        the remaining steps. If all steps of one parameter are set to NaN, no
        derivative estimate will be produced for that parameter.

    Args:
        x (numpy.ndarray): 1d array at which the derivative is calculated.
        method (str): One of ["central", "forward", "backward"]
        n_steps (int): Number of steps needed. For central methods, this is
            the number of steps per direction. It is 1 if no Richardson extrapolation
            is used.
        target (str): One of ["first_derivative", "second_derivative"]. This is used to
            choose the appropriate rule of thumb for the base_steps.
        base_steps (numpy.ndarray, optional): 1d array of the same length as x.
            base_steps * scaling_factor is the absolute value of the first (and possibly
            only) step used in the finite differences approximation of the derivative.
            If base_steps * scaling_factor conflicts with bounds, the actual steps will
            be adjusted. If base_steps is not provided, it will be determined according
            to a rule of thumb as long as this does not conflict with min_steps.
        scaling_factor (numpy.ndarray or float): Scaling factor which is applied to
            base_steps. If it is an numpy.ndarray, it needs to have the same shape as x.
            scaling_factor is useful if you want to increase or decrease the base_step
            relative to the rule-of-thumb or user provided base_step, for example to
            benchmark the effect of the step size.
        lower_bounds (numpy.ndarray): 1d array with lower bounds for each parameter.
        upper_bounds (numpy.ndarray): 1d array with upper bounds for each parameter.
        step_ratio (float or array): Ratio between two consecutive Richardson
            extrapolation steps in the same direction. default 2.0. Has to be larger
            than one. step ratio is only used if n_steps > 1.
        min_steps (numpy.ndarray): Minimal possible step sizes that can be chosen to
            accommodate bounds. Needs to have same length as x. By default min_steps is
            equal to base_steps, i.e step size is not decreased beyond what is optimal
            according to the rule of thumb.

    Returns:
        steps (namedtuple): Namedtuple with the field names pos and neg. Each field
            contains a numpy array of shape (n_steps, len(x)) with the steps in
            the corresponding direction. The steps are always symmetric, in the sense
            that steps.neg[i, j] = - steps.pos[i, j] unless one of them is NaN.

    """
    base_steps = _calculate_or_validate_base_steps(
        base_steps, x, target, min_steps, scaling_factor
    )
    min_steps = base_steps if min_steps is None else min_steps

    assert (
        bounds.upper - bounds.lower >= 2 * min_steps
    ).all(), "min_steps is too large to fit into bounds."

    upper_step_bounds = bounds.upper - x
    lower_step_bounds = bounds.lower - x

    pos = step_ratio ** np.arange(n_steps) * base_steps.reshape(-1, 1)
    neg = -pos.copy()

    if method in ["forward", "backward"]:
        pos, neg = _set_unused_side_to_nan(
            x, pos, neg, method, lower_step_bounds, upper_step_bounds
        )

    if np.isfinite(bounds.lower).any() or np.isfinite(bounds.upper).any():
        pos, neg = _rescale_to_accomodate_bounds(
            base_steps, pos, neg, lower_step_bounds, upper_step_bounds, min_steps
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pos[pos > upper_step_bounds.reshape(-1, 1)] = np.nan
        neg[neg < lower_step_bounds.reshape(-1, 1)] = np.nan

    steps = Steps(pos=pos.T, neg=neg.T)

    return steps


def _calculate_or_validate_base_steps(base_steps, x, target, min_steps, scaling_factor):
    """Validate user provided base_steps or generate them with rule of thumb.

    Args:
        base_steps (numpy.ndarray, optional): 1d array of the same length as x.
            base_steps * scaling_factor is the absolute value of the first (and possibly
            only) step used in the finite differences approximation of the derivative.
        x (numpy.ndarray): 1d array at which the derivative is evaluated
        target (str): One of ["first_derivative", "second_derivative"]. This is used to
            choose the appropriate rule of thumb for the base_steps.
        min_steps (numpy.ndarray or None): Minimal possible step sizes that can be
            chosen to accommodate bounds. Needs to have same length as x.
        scaling_factor (numpy.ndarray or float): Scaling factor which is applied to
            base_steps. If it is an :class:`numpy.ndarray`, it needs to have the same
            shape as x.

    Returns:
        base_steps (numpy.ndarray): 1d array of the same length as x with the
            absolute value of the first step.

    """
    if np.any(scaling_factor <= 0):
        raise ValueError("Scaling factor must be strictly positive.")

    if base_steps is not None:
        if np.isscalar(base_steps):
            base_steps = np.full(len(x), base_steps)

        if base_steps.shape != x.shape:
            raise ValueError("base_steps has to have the same shape as x.")

        base_steps = base_steps * scaling_factor

        if np.isscalar(min_steps):
            min_steps = np.full(len(x), min_steps)

        if min_steps is not None and (base_steps <= min_steps).any():
            raise ValueError(
                "scaling_factor * base_steps must be larger than min_steps."
            )
    else:
        eps = np.finfo(float).eps
        if target == "first_derivative":
            base_steps = eps ** (1 / 2) * np.maximum(np.abs(x), 0.1) * scaling_factor
        elif target == "second_derivative":
            base_steps = eps ** (1 / 3) * np.maximum(np.abs(x), 0.1) * scaling_factor
        else:
            raise ValueError(f"Invalid target: {target}.")
        if min_steps is not None:
            base_steps = np.clip(base_steps, a_min=min_steps, a_max=None)
    return base_steps


def _set_unused_side_to_nan(
    x,  # noqa: ARG001
    pos,
    neg,
    method,
    lower_step_bounds,
    upper_step_bounds,
):
    """Set unused side (i.e. pos or neg) to np.nan.

    A side is not used if:
    - It was not requested due to one sided derivatives.
    - It was requested but a side switch was better due to bounds.

    This function does not yet guarantee that all bounds are fulfilled. It only switches
    to the side that has more space to the bound if there is a bound violation.

    Args:
        x (numpy.ndarray): 1d array with parameters.
        pos (numpy.ndarray): Array with positive steps of shape (n_steps, len(x))
        neg (numpy.ndarray): Array with negative steps of shape (n_steps, len(x))
        method (str): One of ["forward", "backward"]
        lower_step_bounds (numpy.ndarray): Lower bounds for steps.
        upper_step_bounds (numpy.ndarray): Upper bounds for steps.

    Returns:
        pos (numpy.ndarray): Copy of pos with additional NaNs
        neg (numpy.ndarray): Copy of neg with additional NaNs

    """
    pos = pos.copy()
    neg = neg.copy()
    better_side = np.where(upper_step_bounds >= -lower_step_bounds, 1, -1)
    max_abs_step = pos[:, -1]
    if method == "forward":
        used_side = np.where(upper_step_bounds >= max_abs_step, 1, better_side)
    elif method == "backward":
        used_side = np.where(-lower_step_bounds >= max_abs_step, -1, better_side)
    else:
        raise ValueError("This function only works for forward or backward method.")

    pos[used_side == -1] = np.nan
    neg[used_side == 1] = np.nan
    return pos, neg


def _rescale_to_accomodate_bounds(
    base_steps, pos, neg, lower_step_bounds, upper_step_bounds, min_steps
):
    """Rescale steps to make them compatible with bounds unless this violates min_steps.

    Args:
        base_steps (np.ndarray, optional): 1d array of the same length as x.
            base_steps * scaling_factor is the absolute value of the first (and possibly
            only) step used in the finite differences approximation of the derivative.
        pos (np.ndarray): Array with positive steps of shape (n_steps, len(x))
        neg (np.ndarray): Array with negative steps of shape (n_steps, len(x))
        lower_step_bounds (np.ndarray): Lower bounds for steps.
        upper_step_bounds (np.ndarray): Upper bounds for steps.
        min_steps (np.ndarray): Minimal possible step sizes that can be chosen
            to accomodate bounds. Needs to have same length as x.

    Returns:
        pos (np.ndarray): Copy of pos with rescaled steps.
        neg (np.ndarray): Copy of neg with rescaled steps.

    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pos_needed_scaling = _fillna(
            upper_step_bounds / np.nanmax(pos, axis=1), 1
        ).clip(0, 1)
        neg_needed_scaling = _fillna(
            lower_step_bounds / np.nanmin(neg, axis=1), 1
        ).clip(0, 1)
    needed_scaling = np.minimum(pos_needed_scaling, neg_needed_scaling)

    min_possible_scaling = min_steps / base_steps

    scaling = np.maximum(needed_scaling, min_possible_scaling).reshape(-1, 1)

    pos = pos * scaling
    neg = neg * scaling
    return pos, neg


def _fillna(x, val):
    return np.where(np.isnan(x), val, x)
