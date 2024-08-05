"""Implementation of the Conjugate Gradient algorithm."""

import numpy as np


def minimize_trust_cg(
    model_gradient, model_hessian, trustregion_radius, *, gtol_abs=1e-8, gtol_rel=1e-6
):
    """Minimize the quadratic subproblem via (standard) conjugate gradient.

    Solve the trust-region quadratic subproblem:
      min_x   g.T @ x + 0.5 * x.T @ H @ x
        s.t.   ||x|| <= trustregion_radius

    approximately, where g denotes the gradient and H the hessian of the quadratic
    model (i.e. the linear terms and square_terms), respectively.

    Args:
        model_gradient (np.ndarray): 1d array of shape (n,) containing the
            gradient (i.e. linear terms) of the quadratic model.
        model_hessian (np.ndarray): 2d array of shape (n, n) containing the
            hessian (i.e .square terms) of the quadratic model.
        trustregion_radius (float): Radius of the trust-region.
        gtol_abs (float): Convergence tolerance for the absolute gradient norm.
        gtol_rel (float): Convergence tolerance for the relative gradient norm.

    Returns:
        np.ndarray: Solution vector of shape (n,).

    """
    n = len(model_gradient)
    max_iter = n * 2
    x_candidate = np.zeros(n)

    residual = model_gradient
    direction = -model_gradient

    gradient_norm = np.linalg.norm(residual)
    stop_tol = max(gtol_abs, gtol_rel * gradient_norm)

    for _ in range(max_iter):
        if gradient_norm <= stop_tol:
            break

        square_terms = direction.T @ model_hessian @ direction

        distance_to_boundary = _get_distance_to_trustregion_boundary(
            x_candidate, direction, trustregion_radius
        )

        # avoid divide by zero warning
        if square_terms > 0:
            step_size = (residual @ residual) / square_terms
        else:
            step_size = np.inf

        if square_terms <= 0 or step_size > distance_to_boundary:
            x_candidate = x_candidate + distance_to_boundary * direction
            break

        x_candidate, residual, direction = _update_vectors_for_next_iteration(
            x_candidate, residual, direction, model_hessian, step_size
        )
        gradient_norm = np.linalg.norm(residual)

    return x_candidate


def _update_vectors_for_next_iteration(
    x_candidate, residual, direction, hessian, alpha
):
    """Update candidate, residual, and direction vectors for the next iteration.

    Args:
        x_candidate (np.ndarray): Candidate vector of shape (n,).
        residual (np.ndarray): Array of residuals of shape (n,). The residual vector
            is defined as `r = Ax - b`, where `A` denotes the hessian matrix and `b` the
            gradient vector of the quadratic trust-region subproblem.
            `r` is equivalent to the first derivative of the quadratic subproblem.
        direction (np.ndarray): Direction vector of shape (n,).

    Returns:
        (tuple) Tuple containing:
            - x_candidate (np.ndarray): Updated candidate vector of shape (n,).
            - residual (np.ndarray): Updated array of residuals of shape (n,).
            - direction (np.darray): Updated direction vector of shape (n,).

    """
    residual_old = residual

    x_candidate = x_candidate + alpha * direction
    residual = residual_old + alpha * (hessian @ direction)

    beta = (residual @ residual) / (residual_old @ residual_old)
    direction = -residual + beta * direction

    return x_candidate, residual, direction


def _get_distance_to_trustregion_boundary(candidate, direction, radius):
    """Compute the distance of the candidate vector to trustregion boundary.

    The positive distance sigma is defined in Eculidean norm, as follows:

        || x + sigma * d || = radius

    where x denotes the candidate vector, and d the direction vector.

    Args:
        candidate(np.ndarray): Candidate vector of shape (n,).
        direction (np.ndarray): Direction vector of shape (n,).
        radius (floar): Radius of the trust-region

    Returns:
        float: The candidate vector's distance to the trustregion
            boundary.

    """
    cc = candidate @ candidate
    cd = candidate @ direction
    dd = direction @ direction

    sigma = -cd + np.sqrt(cd * cd + dd * (radius**2 - cc))
    sigma /= dd

    return sigma
