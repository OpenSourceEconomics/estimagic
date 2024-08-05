"""Implementation of the Steihaug-Toint Conjugate Gradient algorithm."""

import numpy as np


def minimize_trust_stcg(model_gradient, model_hessian, trustregion_radius):
    """Minimize the quadratic subproblem via Steihaug-Toint conjugate gradient.

    Solve the quadratic trust-region subproblem:

      min_x   g.T @ x + 0.5 * x.T @ hess @ x
        s.t.   ||x|| <= trustregion_radius

    approximately, where g denotes the gradient and hess the hessian of the quadratic
    model (i.e. the linear terms and square_terms), respectively.

    The Steihaug-Toint conjugate gradient method is based on Steihaug
    (:cite:`Steihaug1983`) and Toint (:cite:`Toint1981`).

    Args:
        model_gradient (np.ndarray): 1d array of shape (n,) containing the
            gradient (i.e. linear terms) of the quadratic model.
        model_hessian (np.ndarray): 2d array of shape (n, n) containing the
            hessian (i.e .square terms) of the quadratic model.
        trustregion_radius (float): Radius of the trust-region.

    Returns:
        np.ndarray: Solution vector of shape (n,).

    """
    abstol = 1e-50
    rtol = 1e-5
    divtol = 10_000

    n = len(model_gradient)
    radius_sq = trustregion_radius**2

    residual = -model_gradient
    rr = residual.T @ residual

    x_candidate = np.zeros(n)

    max_iter = min(n, 10_000)

    z = np.linalg.pinv(model_hessian) @ residual
    rz = residual @ residual

    n_iter = 0
    diverged = False
    converged = False

    norm_r = np.sqrt(rr)
    norm_r0 = norm_r
    ttol = max(rtol * norm_r0, abstol)

    converged, diverged = _check_convergence(
        norm_r, norm_r0, abstol, ttol, divtol, converged, diverged
    )

    p = model_hessian @ z
    z = model_hessian @ p
    n_iter += 1

    kappa = p @ z

    dp = 0
    norm_d = 0
    norm_p = p @ p

    if kappa <= 0:
        converged = True

        x_candidate, z, n_iter = _update_candidate_vector_and_iteration_number(
            x_candidate,
            residual,
            p,
            z,
            model_gradient,
            model_hessian,
            rr,
            trustregion_radius,
            norm_p,
            n_iter,
        )

    for _ in range(max_iter):
        alpha = rz / kappa
        norm_dp1 = norm_d + alpha * (2 * dp + alpha * norm_p)

        if trustregion_radius != 0 and norm_dp1 >= radius_sq:
            converged = True

            if norm_p > 0:
                x_candidate = _take_step_to_trustregion_boundary(
                    x_candidate, p, dp, radius_sq, norm_d, norm_p
                )

            break

        x_candidate = x_candidate + alpha * p
        residual = residual - alpha * (model_hessian @ p)

        norm_d = x_candidate @ x_candidate

        rzm1 = rz
        rz = residual @ residual

        norm_r = np.linalg.norm(residual)

        converged, diverged = _check_convergence(
            norm_r, norm_r0, abstol, ttol, divtol, converged, diverged
        )

        if converged or diverged:
            break

        beta = rz / rzm1

        if abs(beta) <= 0:
            diverged = True
            break

        if n_iter >= max_iter:
            diverged = True
            break

        p = residual + beta * p

        dp = x_candidate @ p
        norm_p = p @ p

        z = model_hessian @ p
        kappa = p @ z
        n_iter += 1

        if kappa <= 0:
            converged = True

            if trustregion_radius != 0 and norm_p > 0:
                x_candidate = _take_step_to_trustregion_boundary(
                    x_candidate, p, dp, radius_sq, norm_d, norm_p
                )

            break

    return x_candidate


def _update_candidate_vector_and_iteration_number(
    x_candidate,
    residual,
    p,
    z,
    model_gradient,
    model_hessian,
    rr,
    radius,
    norm_p,
    n_iter,
):
    """Update candidate, z vector, and iteration number."""
    radius_sq = radius**2

    if radius != 0 and norm_p > 0:
        # Take step to boundary
        step = np.sqrt(radius_sq / norm_p)
        x_candidate = x_candidate + step * p

    elif radius != 0:
        if radius_sq >= rr:
            alpha = 1.0
        else:
            alpha = np.sqrt(radius_sq / rr)

        x_candidate = x_candidate + alpha * residual
        z = model_gradient - 0.5 * (model_hessian @ x_candidate)

        n_iter += 1

    return x_candidate, z, n_iter


def _take_step_to_trustregion_boundary(x_candidate, p, dp, radius_sq, norm_d, norm_p):
    """Take step to trust-region boundary."""
    step = (np.sqrt(dp * dp + norm_p * (radius_sq - norm_d)) - dp) / norm_p
    x_candidate = x_candidate + step * p

    return x_candidate


def _check_convergence(
    rnorm,
    rnorm0,
    abstol,  # noqa: ARG001
    ttol,
    divtol,
    converged,
    diverged,
):
    """Check for convergence."""
    if rnorm <= ttol:
        converged = True
    elif rnorm >= divtol * rnorm0:
        diverged = True

    return converged, diverged
