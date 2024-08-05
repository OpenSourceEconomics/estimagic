"""Implementation of the quadratic trustregion solver TRSBOX."""

import numpy as np


def minimize_trust_trsbox(
    model_gradient,
    model_hessian,
    trustregion_radius,
    *,
    lower_bounds,
    upper_bounds,
):
    """Minimize a qaudratic trust-region subproblem using the trsbox algorithm.

    Solve the quadratic trust-region subproblem:
      min_x   g.T @ x + 0.5 * x.T @ hess @ x
        s.t.   ||x|| <= trustregion_radius
               lower_bounds <= x <= upper_bounds

    approximately, using an active-set approach, where g denotes the gradient
    and hess the hessian of the quadratic model (i.e. the linear terms and
    square_terms), respectively.

    The subproblem is assumed to be centered, i.e. ``x_center`` is the zero vector.
    The trsbox algorithm applies a conjugate gradient step in its main loop.

    This implementation of the quadratic trsbox algorithm is based on
    M. J. D. Powell (2009) "The BOBYQA algorithm for bound constrained
    optimization without derivatives." (cite:`Powell2009`).

    Some modifications to the termination conditions are taken from the
    DFBOLS method by Zhang et al. (:cite:`Zhang2010`).

    Args:
        model_gradient (np.ndarray): 1d array of shape (n,) containing the
            gradient (i.e. linear terms) of the quadratic model.
        model_hessian (np.ndarray): 2d array of shape (n, n) containing the
            hessian (i.e .square terms) of the quadratic model.
        lower_bounds (np.ndarray): 1d array of shape (n,) with lower bounds
            for the parameter vector x.
        upper_bounds (np.ndarray): 1d array of shape (n,) with upper bounds
            for the parameter vector x.
        trustregion_radius (float): Radius of the trust-region.
    Returns:
        np.ndarray: Solution vector for the quadratic trust-region subproblem
            of shape (n,).

    """
    n = len(model_gradient)
    x_center = np.zeros(n)

    n_iter = 0
    n_fixed_variables = 0

    x_bounded = np.zeros(n)
    x_bounded[(x_center <= lower_bounds) & (model_gradient >= 0.0)] = -1
    x_bounded[(x_center >= upper_bounds) & (model_gradient <= 0.0)] = 1

    x_candidate = np.zeros(n)
    gradient_projected = np.zeros(n)
    gradient_candidate = model_gradient

    total_reduction = 0
    delta_sq = trustregion_radius**2
    curve_min = -1.0
    beta = 0

    need_alt_trust_step = False
    max_iter = 100 * n**2

    # Main Conjugate Gradient loop
    for _ in range(max_iter):
        gradient_projected[x_bounded != 0] = 0
        if beta == 0:
            gradient_projected[x_bounded == 0] = -gradient_candidate[x_bounded == 0]
        else:
            gradient_projected[x_bounded == 0] = (
                beta * gradient_projected[x_bounded == 0]
                - gradient_candidate[x_bounded == 0]
            )
        gradient_projected_sumsq = gradient_projected @ gradient_projected

        if gradient_projected_sumsq == 0:
            need_alt_trust_step = False
            break

        if beta == 0:
            gradient_sumsq = gradient_projected_sumsq
            max_iter = n_iter + n - n_fixed_variables

        if n_iter == 0:
            gradient_sumsq_initial = gradient_sumsq

        if gradient_sumsq <= min(
            1.0e-6 * gradient_sumsq_initial, 1.0e-18
        ) or gradient_sumsq * delta_sq <= min(1.0e-6 * total_reduction**2, 1.0e-18):
            need_alt_trust_step = False
            break

        hess_g = model_hessian @ gradient_projected
        g_x = gradient_projected[x_bounded == 0] @ x_candidate[x_bounded == 0]
        g_hess_g = gradient_projected[x_bounded == 0] @ hess_g[x_bounded == 0]
        raw_distance = (
            delta_sq - x_candidate[x_bounded == 0] @ x_candidate[x_bounded == 0]
        )

        if raw_distance <= 0:
            need_alt_trust_step = True
            break

        step_len, distance_to_boundary = _take_unconstrained_step_up_to_boundary(
            raw_distance, gradient_sumsq, gradient_projected_sumsq, g_x, g_hess_g
        )

        if step_len <= 1.0e-30:
            need_alt_trust_step = False
            break

        step_len, index_bound_active = _take_constrained_step_up_to_boundary(
            x_candidate, gradient_projected, step_len, lower_bounds, upper_bounds
        )

        current_reduction = 0
        if step_len > 0:
            n_iter += 1
            (
                x_candidate,
                gradient_candidate,
                current_reduction,
                total_reduction,
                curve_min,
                gradient_sumsq,
                gradient_sumsq_old,
            ) = _update_candidate_vectors_and_reduction(
                x_candidate,
                x_bounded,
                gradient_candidate,
                gradient_projected,
                step_len,
                total_reduction,
                curve_min,
                index_bound_active,
                gradient_projected_sumsq,
                gradient_sumsq,
                g_hess_g,
                hess_g,
            )

        if index_bound_active is not None:
            n_fixed_variables += 1
            if gradient_projected[index_bound_active] >= 0:
                x_bounded[index_bound_active] = 1
            else:
                x_bounded[index_bound_active] = -1

            delta_sq = delta_sq - x_candidate[index_bound_active] ** 2
            if delta_sq <= 0:
                need_alt_trust_step = True
                break

            beta = 0
            continue

        if step_len >= distance_to_boundary:
            need_alt_trust_step = True
            break

        if n_iter == max_iter or current_reduction <= 1.0e-6 * total_reduction:
            need_alt_trust_step = False
            break

        beta = gradient_sumsq / gradient_sumsq_old
        continue

    if need_alt_trust_step:
        curve_min = 0
        x_candidate = _perform_alternative_trustregion_step(
            x_candidate=x_candidate,
            x_bounded=x_bounded,
            gradient_candidate=gradient_candidate,
            model_hessian=model_hessian,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            n_fixed_variables=n_fixed_variables,
            total_reduction=total_reduction,
        )
    else:
        x_candidate = _apply_bounds_to_candidate_vector(
            x_candidate, x_bounded, lower_bounds, upper_bounds
        )

    return x_candidate


def _perform_alternative_trustregion_step(
    x_candidate,
    x_bounded,
    gradient_candidate,
    model_hessian,
    lower_bounds,
    upper_bounds,
    n_fixed_variables,
    total_reduction,
):
    """Perform the alternative trust-region step."""
    n = len(x_candidate)
    max_iter = 100 * n**2

    for _ in range(max_iter):
        if n_fixed_variables >= n - 1:
            x_candidate = _apply_bounds_to_candidate_vector(
                x_candidate, x_bounded, lower_bounds, upper_bounds
            )
            break

        search_direction = np.zeros(n)
        search_direction[x_bounded == 0] = x_candidate[x_bounded == 0]

        x_reduced = x_candidate[x_bounded == 0] @ x_candidate[x_bounded == 0]
        x_grad = x_candidate[x_bounded == 0] @ gradient_candidate[x_bounded == 0]
        gradient_reduced = (
            gradient_candidate[x_bounded == 0] @ gradient_candidate[x_bounded == 0]
        )
        hess_s = model_hessian @ search_direction
        hessian_reduced = hess_s

        restart_alt_loop = False

        for _ in range(max_iter):
            raw_reduction = gradient_reduced * x_reduced - x_grad**2
            if raw_reduction <= 1.0e-4 * total_reduction**2:
                restart_alt_loop = False
                break

            search_direction, s_norm = _compute_new_search_direction_and_norm(
                x_candidate,
                x_bounded,
                x_reduced,
                gradient_candidate,
                x_grad,
                raw_reduction,
            )

            (
                x_bounded,
                index_active_bound,
                n_fixed_variables,
                active_bound,
                bound_on_tangent,
                free_variable_reached_bound,
            ) = _calc_upper_bound_on_tangent(
                x_candidate,
                search_direction,
                x_bounded,
                lower_bounds,
                upper_bounds,
                n_fixed_variables,
            )

            if free_variable_reached_bound:
                restart_alt_loop = True
                break

            hess_s = model_hessian @ search_direction

            s_hess_s = np.sum(search_direction[x_bounded == 0] * hess_s[x_bounded == 0])
            x_hess_s = np.sum(x_candidate[x_bounded == 0] * hess_s[x_bounded == 0])
            x_hess_x = np.sum(
                x_candidate[x_bounded == 0] * hessian_reduced[x_bounded == 0]
            )

            (
                previous_reduction,
                next_reduction,
                max_reduction,
                tangent,
                index_angle_greatest_reduction,
                n_angles,
            ) = _calc_greatest_criterion_reduction(
                bound_on_tangent, s_hess_s, x_hess_s, x_hess_x, x_grad, s_norm
            )

            if index_angle_greatest_reduction == -1:
                restart_alt_loop = False
                break

            if index_angle_greatest_reduction < n_angles - 1:
                tangent = _update_tangent(
                    index_angle_greatest_reduction,
                    bound_on_tangent,
                    n_angles,
                    next_reduction,
                    previous_reduction,
                    max_reduction,
                )

            cosine = (1.0 - tangent**2) / (1.0 + tangent**2)
            sine = 2.0 * tangent / (1.0 + tangent**2)
            current_reduction = _calc_new_reduction(
                tangent, sine, s_hess_s, x_hess_x, x_hess_s, x_grad, s_norm
            )

            if current_reduction <= 0.0:
                restart_alt_loop = False
                break

            (
                x_candidate,
                gradient_candidate,
                x_grad,
                gradient_reduced,
                hessian_reduced,
            ) = _update_candidate_vectors_and_reduction_alt_step(
                x_candidate,
                search_direction,
                x_bounded,
                gradient_candidate,
                cosine,
                sine,
                hess_s,
                hessian_reduced,
            )

            total_reduction = total_reduction + current_reduction
            if (
                index_active_bound is not None
                and index_angle_greatest_reduction == n_angles - 1
            ):
                n_fixed_variables += 1
                x_bounded[index_active_bound] = active_bound
                restart_alt_loop = True
                break

            if current_reduction <= 0.01 * total_reduction:
                restart_alt_loop = False
                break

            continue

        if restart_alt_loop:
            continue
        else:
            break

    x_candidate = _apply_bounds_to_candidate_vector(
        x_candidate, x_bounded, lower_bounds, upper_bounds
    )

    return x_candidate


def _apply_bounds_to_candidate_vector(
    x_candidate,
    x_bounded,
    lower_bounds,
    upper_bounds,
):
    """Force candidate vector to lie within bounds."""
    x_candidate_new = np.clip(lower_bounds, x_candidate, upper_bounds)
    x_candidate_new[x_bounded == -1] = lower_bounds[x_bounded == -1]
    x_candidate_new[x_bounded == 1] = upper_bounds[x_bounded == 1]

    return x_candidate_new


def _take_unconstrained_step_up_to_boundary(
    raw_distance, gradient_sumsq, gradient_projected_sumsq, g_x, g_hess_g
):
    """Take unconstrained step, ignoring bounds, up to boundary."""
    temp = np.sqrt(gradient_projected_sumsq * raw_distance + g_x**2)

    if g_x >= 0:
        distance_to_boundary = raw_distance / (temp + g_x)
    else:
        distance_to_boundary = (temp - g_x) / gradient_projected_sumsq

    if g_hess_g <= 0:
        step_len = distance_to_boundary
    else:
        step_len = min(distance_to_boundary, gradient_sumsq / g_hess_g)

    return step_len, distance_to_boundary


def _update_candidate_vectors_and_reduction(
    x_candidate,
    x_bounded,
    gradient_candidate,
    gradient_projected,
    step_len,
    total_reduction,
    curve_min,
    index_bound_active,
    gradient_projected_sumsq,
    gradient_sumsq,
    g_hess_g,
    hess_g,
):
    """Update candidate vectors and the associated criterion reduction."""
    current_min = g_hess_g / gradient_projected_sumsq

    if index_bound_active is None and current_min > 0:
        if curve_min != -1.0:
            curve_min = min(curve_min, current_min)
        else:
            curve_min = current_min

    gradient_sumsq_old = gradient_sumsq

    gradient_candidate = gradient_candidate + step_len * hess_g
    x_candidate = x_candidate + step_len * gradient_projected

    gradient_sumsq = (
        gradient_candidate[x_bounded == 0] @ gradient_candidate[x_bounded == 0]
    )

    current_reduction = max(
        step_len * (gradient_sumsq_old - 0.5 * step_len * g_hess_g), 0
    )
    total_reduction = total_reduction + current_reduction

    return (
        x_candidate,
        gradient_candidate,
        current_reduction,
        total_reduction,
        curve_min,
        gradient_sumsq,
        gradient_sumsq_old,
    )


def _take_constrained_step_up_to_boundary(
    x_candidate, gradient_projected, step_len, lower_bounds, upper_bounds
):
    """Reduce step length, where boundary is hit, to preserve simple bounds."""
    index_bound_active = None

    for i in range(len(x_candidate)):
        if gradient_projected[i] != 0:
            if gradient_projected[i] > 0:
                step_len_constr = (
                    upper_bounds[i] - x_candidate[i]
                ) / gradient_projected[i]
            else:
                step_len_constr = (
                    lower_bounds[i] - x_candidate[i]
                ) / gradient_projected[i]

            if step_len_constr < step_len:
                step_len = step_len_constr
                index_bound_active = i

    return step_len, index_bound_active


def _calc_upper_bound_on_tangent(
    x_candidate,
    search_direction,
    x_bounded,
    lower_bounds,
    upper_bounds,
    n_fixed_variables,
):
    """Calculate upper bound on tangent of half the angle to the boundary."""
    bound_on_tangent = 1
    free_variable_reached_bound = False
    index_active_bound = None
    active_bound = None

    for i in range(len(x_candidate)):
        if x_bounded[i] == 0:
            lower_bound_centered = x_candidate[i] - lower_bounds[i]
            upper_bound_centered = upper_bounds[i] - x_candidate[i]

            if lower_bound_centered <= 0.0:
                n_fixed_variables += 1
                x_bounded[i] = -1
                free_variable_reached_bound = True
                break

            elif upper_bound_centered <= 0.0:
                n_fixed_variables += 1
                x_bounded[i] = 1
                free_variable_reached_bound = True
                break

            ssq = x_candidate[i] ** 2 + search_direction[i] ** 2

            ssq_lower = ssq - lower_bounds[i] ** 2
            if ssq_lower > 0.0:
                ssq_lower = np.sqrt(ssq_lower) - search_direction[i]
                if bound_on_tangent * ssq_lower > lower_bound_centered:
                    bound_on_tangent = lower_bound_centered / ssq_lower
                    index_active_bound = i
                    active_bound = -1

            ssq_upper = ssq - upper_bounds[i] ** 2
            if ssq_upper > 0.0:
                ssq_upper = np.sqrt(ssq_upper) + search_direction[i]
                if bound_on_tangent * ssq_upper > upper_bound_centered:
                    bound_on_tangent = upper_bound_centered / ssq_upper
                    index_active_bound = i
                    active_bound = 1

    return (
        x_bounded,
        index_active_bound,
        n_fixed_variables,
        active_bound,
        bound_on_tangent,
        free_variable_reached_bound,
    )


def _calc_greatest_criterion_reduction(
    bound_on_tangent, s_hess_s, x_hess_s, x_hess_x, x_grad, s_norm
):
    """Calculate the greatest feasible reduction in the criterion function.

    The largest reduction is found by looking at a range of equally spaced values of
    ``tangent`` in the interval [0, ``bound_on_tangent``], where ``tangent`` is the
    tangent of half the angle to the trust-region boundary.

    """
    previous_reduction = None
    next_reduction = None

    max_reduction = 0
    index_angle_greatest_reduction = -1
    old_reduction = 0
    n_angles = int(17 * bound_on_tangent + 3.1)

    for i in range(n_angles):
        tangent = bound_on_tangent * (i + 1) / n_angles
        sine = 2.0 * tangent / (1.0 + tangent**2)

        new_reduction = _calc_new_reduction(
            tangent, sine, s_hess_s, x_hess_x, x_hess_s, x_grad, s_norm
        )

        if new_reduction > max_reduction:
            max_reduction = new_reduction
            index_angle_greatest_reduction = i
            previous_reduction = old_reduction
        elif i == index_angle_greatest_reduction + 1:
            next_reduction = new_reduction
        old_reduction = new_reduction

    return (
        previous_reduction,
        next_reduction,
        max_reduction,
        tangent,
        index_angle_greatest_reduction,
        n_angles,
    )


def _update_candidate_vectors_and_reduction_alt_step(
    x_candidate,
    search_direction,
    x_bounded,
    gradient_candidate,
    cosine,
    sine,
    hess_s,
    hessian_reduced,
):
    """Update candidate vectors and the associated criterion reduction.

    If the angle of the alternative iteration is restricted by a bound on a free
    variable, that variable is fixed at the bound.

    """
    gradient_candidate_new = (
        gradient_candidate + (cosine - 1.0) * hessian_reduced + sine * hess_s
    )
    x_candidate_new = np.copy(x_candidate)
    x_candidate_new[x_bounded == 0] = (
        cosine * x_candidate[x_bounded == 0] + sine * search_direction[x_bounded == 0]
    )
    x_grad = x_candidate_new[x_bounded == 0] @ gradient_candidate_new[x_bounded == 0]
    gradient_reduced = (
        gradient_candidate_new[x_bounded == 0] @ gradient_candidate_new[x_bounded == 0]
    )
    hessian_reduced = cosine * hessian_reduced + sine * hess_s

    return (
        x_candidate_new,
        gradient_candidate_new,
        x_grad,
        gradient_reduced,
        hessian_reduced,
    )


def _compute_new_search_direction_and_norm(
    x_candidate, x_bounded, x_reduced, gradient_candidate, x_grad, raw_reduction
):
    """Compute the new search direction and its norm."""
    raw_reduction = np.sqrt(raw_reduction)
    search_direction = np.zeros_like(x_candidate)

    search_direction[x_bounded == 0] = (
        x_grad * x_candidate[x_bounded == 0]
        - x_reduced * gradient_candidate[x_bounded == 0]
    ) / raw_reduction
    s_norm = -raw_reduction

    return search_direction, s_norm


def _calc_new_reduction(tangent, sine, s_hess_s, x_hess_x, x_hess_s, x_grad, s_norm):
    """Calculate the new reduction in the criterion function."""
    raw_reduction = s_hess_s + tangent * (tangent * x_hess_x - 2.0 * x_hess_s)
    current_reduction = sine * (tangent * x_grad - s_norm - 0.5 * sine * raw_reduction)

    return current_reduction


def _update_tangent(
    index_angle_greatest_reduction,
    bound_on_tangent,
    n_angles,
    next_reduction,
    previous_reduction,
    max_reduction,
):
    """Update the tangent of half the angle to the trust-region boundary."""
    raw_reduction = (next_reduction - previous_reduction) / (
        2.0 * max_reduction - previous_reduction - next_reduction
    )
    tangent = (
        bound_on_tangent
        * ((index_angle_greatest_reduction + 1) + 0.5 * raw_reduction)
        / n_angles
    )
    return tangent
