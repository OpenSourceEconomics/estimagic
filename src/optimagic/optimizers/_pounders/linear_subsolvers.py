"""Collection of linear trust-region subsolvers."""

from typing import NamedTuple

import numpy as np


class LinearModel(NamedTuple):
    intercept: float | None = None
    linear_terms: np.ndarray | None = None  # shape (n_params, n_params)


def minimize_trsbox_linear(
    linear_model, lower_bounds, upper_bounds, trustregion_radius, *, zero_treshold=1e-14
):
    """Minimize a linear trust-region subproblem using the trsbox algorithm.

    Solve the linear subproblem:

      min_x   g.T @ x
        s.t.   lower_bound <= x <= upper_bound
              ||x||**2 <= trustregion_radius**2

    using an active-set approach.

    This algorithm is an implementation of the TRSBOX routine from
    M. J. D. Powell (2009) "The BOBYQA algorithm for bound constrained
    optimization without derivatives." (cite:`Powell2009`).

    Args:
        linear_model (NamedTuple): Named tuple containing the parameters of the
            linear model, i.e.:
            - ``intercept`` (float): Intercept of the linear model.
            - ``linear_terms`` (np.ndarray): 1d array of shape (n,) with the linear
            terms of the mdoel.
        lower_bounds (np.ndarray): 1d array of shape (n,) with lower bounds
            for the parameter vector x.
        upper_bounds (np.ndarray): 1d array of shape (n,) with upper bounds
            for the parameter vector x.
        trustregion_radius (float): Radius of the trust-region.
        zero_treshold (float): Treshold for treating numerical values as zero.
            Numbers smaller than this are considered zero up to machine precision.

    Returns:
        (np.ndarray): Solution vector for the linear trust-region subproblem.
            Array of shape (n,).

    """
    lower_bounds_internal = np.minimum(lower_bounds, -zero_treshold)
    upper_bounds_internal = np.maximum(upper_bounds, zero_treshold)

    model_gradient = linear_model.linear_terms
    n = len(model_gradient)
    x_candidate = np.zeros(n)

    direction = -model_gradient

    indices_inactive_directions = np.where(np.abs(direction) < zero_treshold)[0]
    direction[indices_inactive_directions] = 0

    active_directions = np.setdiff1d(np.arange(n), indices_inactive_directions)
    set_active_directions = iter(active_directions)

    for _ in range(n):
        if np.linalg.norm(direction) < zero_treshold:
            break

        x_candidate_unconstr = _take_unconstrained_step_up_to_boundary(
            x_candidate, direction, trustregion_radius, zero_treshold=zero_treshold
        )

        active_bound, index_active_bound = _find_next_active_bound(
            x_candidate_unconstr,
            lower_bounds_internal,
            upper_bounds_internal,
            set_active_directions,
        )

        if active_bound is None:
            x_candidate = x_candidate_unconstr
            break

        else:
            x_candidate, direction = _take_constrained_step_up_to_boundary(
                x_candidate,
                direction,
                active_bound,
                index_active_bound,
            )

    return x_candidate


def improve_geomtery_trsbox_linear(
    x_center,
    linear_model,
    lower_bounds,
    upper_bounds,
    trustregion_radius,
    *,
    zero_treshold=1e-14,
):
    """Maximize a Lagrange polynomial of degree one to improve geometry of the model.

    Let a Lagrange polynomial of degree one be defined by:
        L(x) = c + g.T @ (x - x_center),

    where c and g denote the constant term and the linear terms (gradient)
    of the linear model, respectively.

    In order to maximize L(x), we maximize the absolute value of L(x) in a
    trust-region setting. I.e. we solve:

        max_x  abs(c + g.T @ (x - x_center))
            s.t. lower_bound <= x <= upper_bound
                 ||x - x_center|| <= trustregion_radius

    In order to find the solution x*, we first minimize and then maximize
    g.T @ (x - center).
    The resulting candidate vectors are then plugged into the objective function L(x)
    to check which one yields the largest absolute value of the Lagrange polynomial.

    Args:
        x_center (np.ndarray): 1d array of shape (n,) containing the center of the
            parameter vector.
        linear_model (NamedTuple): Named tuple containing the parameters of the
            linear model that form the Lagrange polynomial, including:
            - ``intercept`` (float): Intercept of the linear model.
            - ``linear_terms`` (np.ndarray): 1d array of shape (n,) with the linear
            terms of the mdoel.
        lower_bounds (np.ndarray): 1d array of shape (n,) with lower bounds
            for the parameter vector x.
        upper_bounds (np.ndarray): 1d array of shape (n,) with upper bounds
            for the parameter vector x.
        trustregion_radius (float): Radius of the trust-region.
        zero_treshold (float): Treshold for treating numerical values as zero.
            Numbers smaller than this are considered zero up to machine precision.

    Returns:
        np.ndarray: Solution vector of shape (n,) that maximizes the Lagrange
            polynomial.

    """
    if np.any(lower_bounds > x_center + zero_treshold):
        raise ValueError("x_center violates lower bound.")
    if np.any(x_center - zero_treshold > upper_bounds):
        raise ValueError("x_center violates upper bound.")

    # Minimize and maximize g.T @ (x - x_center), respectively
    linear_model_to_minimize = linear_model
    linear_model_to_maximize = linear_model._replace(
        linear_terms=-linear_model.linear_terms
    )

    x_candidate_min = minimize_trsbox_linear(
        linear_model_to_minimize,
        lower_bounds - x_center,
        upper_bounds - x_center,
        trustregion_radius,
        zero_treshold=zero_treshold,
    )
    x_candidate_max = minimize_trsbox_linear(
        linear_model_to_maximize,
        lower_bounds - x_center,
        upper_bounds - x_center,
        trustregion_radius,
        zero_treshold=zero_treshold,
    )

    lagrange_polynomial = lambda x: abs(
        linear_model.intercept + linear_model.linear_terms.T @ x
    )

    if lagrange_polynomial(x_candidate_min) >= lagrange_polynomial(x_candidate_max):
        x_lagrange = x_candidate_min + x_center
    else:
        x_lagrange = x_candidate_max + x_center

    return x_lagrange


def _find_next_active_bound(
    x_candidate_unconstr,
    lower_bounds,
    upper_bounds,
    set_active_directions,
):
    """Find the next active bound and return its index.

    A (lower or upper) bound is considered active if
        x_candidate <= lower_bounds
        x_candidate >= upper_bounds

    Args:
        x_candidate_unconstr (np.ndarray): Unconstrained candidate vector of shape (n,),
            which ignores bound constraints.
        lower_bounds (np.ndarray): 1d array of shape (n,) with lower bounds
            for the parameter vector x.
        upper_bounds (np.ndarray): 1d array of shape (n,) with upper bounds
            for the parameter vector x.
        set_active_directions (iterator): Iterator over the indices of active search
            directions, i.e. directions that are not zero.

    Returns:
        Tuple:
            - active_bound (float or None): The next active bound. It can be a lower
                or active bound. If None, there are no more active bounds left in the
                set of active search directions.
            - index_bound_active (int or None): Index where an active lower or
                upper bound has been found. None, if no active bound has been detected.

    """
    index_active = next(set_active_directions)

    while True:
        if x_candidate_unconstr[index_active] >= upper_bounds[index_active]:
            active_bound = upper_bounds[index_active]
            break

        elif x_candidate_unconstr[index_active] <= lower_bounds[index_active]:
            active_bound = lower_bounds[index_active]
            break

        else:
            try:
                index_active = next(set_active_directions)
            except StopIteration:
                active_bound = None
                break

    return active_bound, index_active


def _take_constrained_step_up_to_boundary(
    x_candidate, direction, active_bound, index_bound_active
):
    """Take largest constrained step possible until trust-region boundary is hit.

    Args:
        x_candidate (np.ndarray): Current candidate vector of shape (n,).
        direction (np.ndarray): Direction vector of shape (n,).
        active_bound (float): The active (lower or upper) bound.
        index_bound_active (int): Index where an active lower or upper bound
            has been found.

    Returns:
        Tuple:
        - x_candidate (np.ndarray): New candidate vector of shape (n,).
        - direction (np.ndarray): New direction vector of shape (n,), where the
            search direction of the active_bound has been set to zero.

    """
    step_size_constr = (active_bound - x_candidate[index_bound_active]) / direction[
        index_bound_active
    ]

    x_candidate = x_candidate + step_size_constr * direction
    x_candidate[index_bound_active] = active_bound

    # Do not search in this direction anymore
    direction[index_bound_active] = 0

    return x_candidate, direction


def _take_unconstrained_step_up_to_boundary(
    x_candidate, direction, trustregion_radius, zero_treshold
):
    """Take largest unconstrained step possible until trust-region boundary is hit.

    Args:
        x_candidate (np.ndarray): Current candidate vector of shape (n,).
        direction (np.ndarray): Direction vector of shape (n,).
        trustregion_radius (float): Radius of the trust-region.
        zero_treshold (float): Treshold for treating numerical values as zero.
            Numbers smaller than this are considered zero up to machine precision.

    Returns:
        np.ndarray: Updated, unconstrained candidate vector of shape (n,).

    """
    step_size_unconstr = _get_distance_to_trustregion_boundary(
        x_candidate, direction, trustregion_radius, zero_treshold
    )
    x_candidate_unconstr = x_candidate + step_size_unconstr * direction

    return x_candidate_unconstr


def _get_distance_to_trustregion_boundary(
    x, direction, trustregion_radius, zero_treshold
):
    """Compute the candidate vector's distance to the trustregion boundary.

    Given the candidate vector, find the largest step alpha in direction g
    that satisfies ||x|| <= trustregion_radius,

    where g denotes the direction vector.

    To find alpha, i.e. the candidate's distance to the trust-region boundary, solve
      ||x + alpha * g||**2 = trustregion_radius**2
         s.t. alpha >= 0

    Using this method, the solution exists whenever ||x|| <= trustregion_radius**2.

    Choose alpha = 0, if the direction vector is zero everywhere.

    Args:
        x (np.ndarray): Candidate vector of shape (n,).
        direction (np.ndarray): Direction vector of shape (n,).
        trustregion_radius (float): Radius of the trust-region.
        zero_treshold (float): Treshold for treating numerical values as zero.
            Numbers smaller than this are considered zero up to machine precision.

    Returns:
        float: Distance between the candidate vector and the trust-region boundary.

    """
    g_dot_x = direction.T @ x
    g_sumsq = direction @ direction
    x_sumsq = x @ x

    l2_norm = np.sqrt(g_sumsq)

    if l2_norm < zero_treshold:
        distance_to_boundary = 0
    else:
        distance_to_boundary = (
            np.sqrt(
                np.maximum(0, g_dot_x**2 + g_sumsq * (trustregion_radius**2 - x_sumsq))
            )
            - g_dot_x
        ) / g_sumsq

    return distance_to_boundary
