"""Auxiliary functions for the pounders algorithm."""

from typing import NamedTuple

import numpy as np
from scipy.linalg import qr_multiply

from optimagic.optimizers._pounders.bntr import (
    bntr,
)
from optimagic.optimizers._pounders.gqtpar import (
    gqtpar,
)


class ResidualModel(NamedTuple):
    intercepts: np.ndarray | None = None  # shape (n_residuals,)
    linear_terms: np.ndarray | None = None  # shape (n_residuals, n_params)
    square_terms: np.ndarray | None = None  # shape (n_residuals, n_params, n_params)


class MainModel(NamedTuple):
    linear_terms: np.ndarray | None = None  # shape (n_params,)
    square_terms: np.ndarray | None = None  # shape (n_params, n_params)


def create_initial_residual_model(history, accepted_index, delta):
    """Update linear and square terms of the initial residual model.

    Args:
        history (LeastSquaresHistory): Class storing history of xs, residuals, and
            critvals.
        accepted_index (int): Index in history pointing to the currently
            accepted candidate vector.
        delta (float): Trust-region radius.

    Returns:
        ResidualModel: Residual model containing the initial parameters for
             ``linear_terms`` and ``square_terms``.

    """
    center_info = {
        "x": history.get_best_x(),
        "residuals": history.get_best_residuals(),
        "radius": delta,
    }
    n_params = len(center_info["x"])
    n_residuals = center_info["residuals"].shape[0]

    indices_not_min = [i for i in range(n_params + 1) if i != accepted_index]

    x_candidate, residuals_candidate, _ = history.get_centered_entries(
        center_info=center_info,
        index=indices_not_min,
    )

    linear_terms = np.linalg.solve(x_candidate, residuals_candidate)
    square_terms = np.zeros((n_residuals, n_params, n_params))

    residual_model = ResidualModel(
        intercepts=history.get_best_residuals(),
        linear_terms=linear_terms,
        square_terms=square_terms,
    )

    return residual_model


def update_residual_model(residual_model, coefficients_to_add, delta, delta_old):
    """Update linear and square terms of the residual model.

    Args:
        residual_model (ResidualModel): Residual model with the following parameters:
            ``intercepts``, ``linear_terms``, and ``square terms``.
        coefficients_to_add (dict): Coefficients used for updating the
            parameters of the residual model.
        delta (float): Trust region radius of the current iteration.
        delta_old (float): Trust region radius of the previous iteration.

    Returns:
        ResidualModel: Residual model containing the updated parameters
            ``linear_terms`` and ``square_terms``.

    """
    linear_terms_new = (
        coefficients_to_add["linear_terms"]
        + (delta / delta_old) * residual_model.linear_terms
    )

    square_terms_new = (
        coefficients_to_add["square_terms"]
        + (delta / delta_old) ** 2 * residual_model.square_terms
    )

    residual_model_updated = residual_model._replace(
        linear_terms=linear_terms_new, square_terms=square_terms_new
    )

    return residual_model_updated


def create_main_from_residual_model(
    residual_model, multiply_square_terms_with_intercepts=True
):
    """Update linear and square terms of the main model via the residual model.

    Args:
        residual_model (ResidualModel): Residual model with the following parameters:
            ``intercepts``, ``linear_terms``, and ``square terms``.
        multiply_square_terms_with_residuals (bool): Indicator whether we
            multiply the main model's ``square terms`` with the
            intercepts of the residual model.

    Returns:
        MainModel: Main model containing the updated parameters
            ``linear_terms`` and ``square terms``.

    """
    linear_terms_main_model = residual_model.linear_terms @ residual_model.intercepts
    square_terms_main_model = (
        residual_model.linear_terms @ residual_model.linear_terms.T
    )

    if multiply_square_terms_with_intercepts is True:
        square_terms_main_model = (
            square_terms_main_model
            + residual_model.square_terms.T @ residual_model.intercepts
        )

    main_model = MainModel(
        linear_terms=linear_terms_main_model, square_terms=square_terms_main_model
    )

    return main_model


def update_main_model_with_new_accepted_x(main_model, x_candidate):
    """Use accepted candidate to update the linear terms of the residual model.

    Args:
         main_model (MainModel): Main model with the following parameters:
             ``linear_terms`` and ``square terms``.
        x_candidate (np.ndarray): Vector of centered x candidates of shape (n_params,).

    Returns:
        MainModel: Main model containing the updated ``linear_terms``.

    """
    linear_terms_new = main_model.linear_terms + main_model.square_terms @ x_candidate
    main_model_updated = main_model._replace(linear_terms=linear_terms_new)

    return main_model_updated


def update_residual_model_with_new_accepted_x(residual_model, x_candidate):
    """Use accepted candidate to update residual model.

    Args:
        residual_model (ResidualModel): Residual model containing the parameters of
            the residual model, i.e. ``intercepts``, ``linear_terms``, and
            ``square terms``.
        x_candidate (np.ndarray): Vector of centered x candidates of shape (n_params,).

    Returns:
        ResidualModel: Residual model containing the updated parameters
            `intercepts`` and ``linear_terms``.

    """
    intercepts_new = (
        residual_model.intercepts
        + x_candidate @ residual_model.linear_terms
        + 0.5 * (x_candidate.T @ residual_model.square_terms @ x_candidate)
    )

    linear_terms_new = (
        residual_model.linear_terms + (residual_model.square_terms @ x_candidate).T
    )

    residual_model_updated = residual_model._replace(
        intercepts=intercepts_new, linear_terms=linear_terms_new
    )

    return residual_model_updated


def solve_subproblem(
    x_accepted,
    main_model,
    lower_bounds,
    upper_bounds,
    delta,
    solver,
    *,
    conjugate_gradient_method,
    maxiter,
    maxiter_gradient_descent,
    gtol_abs,
    gtol_rel,
    gtol_scaled,
    gtol_abs_conjugate_gradient,
    gtol_rel_conjugate_gradient,
    k_easy,
    k_hard,
):
    """Solve the quadratic subproblem.

    Args:
        x_accepted (np.ndarray): Currently accepted candidate vector of shape
            (n_params,).
        delta (float): Current trust region radius.
        main_model (MainModel): Main model with the following parameters:
             ``linear_terms`` and ``square terms``.
        lower_bounds (np.ndarray): 1d array of shape (n_params,) with lower bounds
            for the parameter vector x.
        upper_bounds (np.ndarray): 1d array of shape (n_params,) with upper bounds
            for the parameter vector x.
        delta (float) Current trust-region radius
        solver (str): Trust-region subsolver to use. Currently, two internal solvers
            are supported:
            - "bntr" (default, supports bound constraints)
            - "gqtpar" (does not support bound constraints)
        conjugate_gradient_method (str): Method for computing the conjugate gradient
            step. Available conjugate gradient methods are:
                - "cg"
                - "steihaug_toint"
                - "trsbox" (default)
        maxiter (int): Maximum number of iterations to perform when solving the
            trust-region subproblem.
        maxiter_gradient_descent (int): Maximum number of gradient descent iterations
            to perform when the trust-region subsolver "bntr" is used.
        gtol_abs (float): Convergence tolerance for the absolute gradient norm
            in the trust-region subproblem ("bntr").
        gtol_rel (float): Convergence tolerance for the relative gradient norm
            in the trust-region subproblem ("bntr").
        gtol_scaled (float): Convergence tolerance for the scaled gradient norm
            in the trust-region subproblem ("bntr").
        gtol_abs_conjugate_gradient (float): Convergence tolerance for the absolute
            gradient norm in the conjugate gradient step of the trust-region
            subproblem ("bntr").
        gtol_rel_conjugate_gradient (float): Convergence tolerance for the relative
            gradient norm in the conjugate gradient step of the trust-region
            subproblem ("bntr").
        k_easy (float): topping criterion for the "easy" case in the trust-region
            subproblem ("gqtpar").
        k_hard (float): Stopping criterion for the "hard" case in the trust-region
            subproblem ("gqtpar").

    Returns:
        (dict): Result dictionary containing the followng keys:
            - "x" (np.ndarray): The solution vector of shape (n_params,)
            - "criterion" (float): The value of the criterion functions associated
                with the solution
            - "n_iterations" (int): Number of iterations performed before termination.
            - "success" (bool): Boolean indicating whether a solution has been found
                before reaching maxiter.

    """
    x0 = np.zeros_like(x_accepted)

    # Normalize bounds. If none provided, use unit cube [-1, 1]
    if lower_bounds is not None:
        lower_bounds = (lower_bounds - x_accepted) / delta
        lower_bounds[lower_bounds < -1] = -1
    else:
        lower_bounds = -np.ones_like(x_accepted)

    if upper_bounds is not None:
        upper_bounds = (upper_bounds - x_accepted) / delta
        upper_bounds[upper_bounds > 1] = 1
    else:
        upper_bounds = np.ones_like(x_accepted)

    # Check if bounds are valid
    if np.max(lower_bounds - upper_bounds) > 1e-10:
        raise ValueError("Upper bounds < lower bounds in subproblem.")
    if np.max(lower_bounds - x0) > 1e-10:
        raise ValueError("Initial guess < lower bounds in subproblem.")
    if np.max(x0 - upper_bounds) > 1e-10:
        raise ValueError("Initial guess > upper bounds in subproblem.")

    if solver == "bntr":
        options = {
            "conjugate_gradient_method": conjugate_gradient_method,
            "maxiter": maxiter,
            "maxiter_gradient_descent": maxiter_gradient_descent,
            "gtol_abs": gtol_abs,
            "gtol_rel": gtol_rel,
            "gtol_scaled": gtol_scaled,
            "gtol_abs_conjugate_gradient": gtol_abs_conjugate_gradient,
            "gtol_rel_conjugate_gradient": gtol_rel_conjugate_gradient,
        }
        result = bntr(main_model, lower_bounds, upper_bounds, x_candidate=x0, **options)
    elif solver == "gqtpar":
        result = gqtpar(
            main_model,
            x_candidate=x0,
            k_easy=k_easy,
            k_hard=k_hard,
            maxiter=maxiter,
        )
    else:
        raise ValueError(
            "Invalid subproblem solver: {solver}. Must be one of bntr, gqtpar."
        )

    # Test bounds post-solution
    if np.max(lower_bounds - result["x"]) > 1e-5:
        raise ValueError("Subproblem solution < lower bounds.")
    if np.max(result["x"] - upper_bounds) > 1e-5:
        raise ValueError("Subproblem solution > upper bounds.")

    return result


def find_affine_points(
    history,
    x_accepted,
    model_improving_points,
    project_x_onto_null,
    delta,
    theta1,
    c,
    model_indices,
    n_modelpoints,
):
    """Find affine points.

    Args:
        history (LeastSquaresHistory): Class storing history of xs, residuals,
            and critvals.
        x_accepted (np.ndarray): Accepted solution vector of the subproblem.
            Shape (n_params,).
        model_improving_points (np.ndarray): Array of shape (n_params, n_params)
            including points to improve the main model, i.e. make the main model
            fully linear, i.e. just-identified.
            If *project_x_onto_null* is False, it is an array filled with zeros.
        project_x_onto_null (int): Indicator whether to calculate the QR
            decomposition of *model_improving_points* and multiply it
             with vector *x_projected*.
        delta (float): Delta, current trust-region radius.
        theta1 (float): Threshold for adding the current x candidate to the model.
        c (float): Threshold for acceptance of the norm of our current x candidate.
        model_indices (np.ndarray): Indices related to the candidates of x
            that are currently in the main model. Shape (2 * n_params + 1,).
        n_modelpoints (int): Current number of model points.

    Returns:
        Tuple:
        - model_improving_points (np.ndarray):  Array of shape (n_params, n_params)
            including points to improve the main model, i.e. make the main model
            fully linear, i.e. just-identified.
        - model_indices (np.ndarray): Indices related to the candidates of x
            that are currently in the main model. Shape (2 *n_params* + 1,).
        - n_modelpoints (int): Current number of model points.
        - project_x_onto_null (int): Indicator whether to calculate the QR
            decomposition of *model_improving_points* and multiply it
            with vector *x_projected*.
            Relevant for next call of *find_affine_points()*.

    """
    n_params = len(x_accepted)

    for i in range(history.get_n_fun() - 1, -1, -1):
        center_info = {"x": x_accepted, "radius": delta}
        x_candidate = history.get_centered_xs(center_info, index=i)
        candidate_norm = np.linalg.norm(x_candidate)

        x_projected = x_candidate

        if candidate_norm <= c:
            if project_x_onto_null is True:
                x_projected, _ = qr_multiply(model_improving_points, x_projected)

            proj = np.linalg.norm(x_projected[n_modelpoints:])

            # Add this index to the model
            if proj >= theta1:
                model_indices[n_modelpoints] = i
                model_improving_points[:, n_modelpoints] = x_candidate
                project_x_onto_null = True
                n_modelpoints += 1

            if n_modelpoints == n_params:
                break

    return model_improving_points, model_indices, n_modelpoints, project_x_onto_null


def add_geomtery_points_to_make_main_model_fully_linear(
    history,
    main_model,
    model_improving_points,
    model_indices,
    x_accepted,
    n_modelpoints,
    delta,
    criterion,
    lower_bounds,
    upper_bounds,
    batch_evaluator,
    n_cores,
):
    """Add points until main model is fully linear.

    Args:
        history (LeastSquaresHistory): Class storing history of xs, residuals, and
            critvals.
        main_model (MainModel): Main model with the following parameters:
             ``linear_terms`` and ``square terms``.
        model_improving_points (np.ndarray): Array of shape (n_params, n_params)
            including points to improve the main model.
        model_indices (np.ndarray): Indices of the candidates of x that are
            currently in the main model. Shape (2 * n_params + 1,).
        x_accepted (np.ndarray): Accepted solution vector of the subproblem.
            Shape (n_params,).
        n_modelpoints (int): Current number of model points.
        delta (float): Delta, current trust-region radius.
        criterion (callable): Criterion function.
        lower_bounds (np.ndarray): Lower bounds.
            Must have same length as the initial guess of the
            parameter vector. Equal to -1 if not provided by the user.
        upper_bounds (np.ndarray): Upper bounds.
            Must have same length as the initial guess of the
            parameter vector. Equal to 1 if not provided by the user.
        batch_evaluator (str or callable): Name of a pre-implemented batch evaluator
            (currently 'joblib' and 'pathos_mp') or Callable with the same interface
            as the optimagic batch_evaluators.
        n_cores (int): Number of processes used to parallelize the function
            evaluations.

    Returns:
        Tuple:
        - history (class): Class storing history of xs, residuals, and critvals.
        - model_indices (np.ndarray): Indices of the candidates of x that are
            currently in the main model. Shape (2 * n_params + 1,).

    """
    n_params = len(x_accepted)

    current_history = history.get_n_fun()

    x_candidate = np.zeros_like(x_accepted)
    x_candidates_list = []
    criterion_candidates_list = []

    model_improving_points, _ = qr_multiply(model_improving_points, np.eye(n_params))

    for i in range(n_modelpoints, n_params):
        change_direction = model_improving_points[:, i] @ main_model.linear_terms

        if change_direction > 0:
            model_improving_points[:, i] *= -1

        x_candidate = delta * model_improving_points[:, i] + x_accepted

        # Project into feasible region
        if lower_bounds is not None and upper_bounds is not None:
            x_candidate = np.median(
                np.stack([lower_bounds, x_candidate, upper_bounds]), axis=0
            )
        x_candidates_list.append(x_candidate)
        model_indices[i] = current_history + i - n_modelpoints

    criterion_candidates_list = batch_evaluator(
        criterion, arguments=x_candidates_list, n_cores=n_cores
    )

    history.add_entries(x_candidates_list, criterion_candidates_list)

    return history, model_indices


def evaluate_residual_model(
    centered_xs,
    centered_residuals,
    residual_model,
):
    """Compute the difference between observed and predicted model evaluations.

    We use a quadratic model of the form:

        f(x) = a + x.T @ b + 0.5 x.T @ C @ x ,

    where C is lower triangular. Note the connection of b and C to the gradient:
    f'(x) = b + (C + C.T) @ x, and the Hessian: f''(x) = C + C.T.

    Args:
        residual_model (ResidualModel): The residual model. Has entries:
            - ``intercept``: corresponds to 'a' in the above equation
            - ``linear_terms``: corresponds to 'b' in the above equation
            - ``square_terms``: corresponds to 'C' in the above equation
        centered_xs (np.ndarray): Centered x sample. Shape (n_modelpoints, n_params).
        centered_residuals (np.ndarray): Centered residuals, i.e. the observed model
            evaluations. Shape (n_maxinterp, n_residuals).

    Returns:
        np.ndarray: Observed minus predicted model evaluations,
            has shape (n_modelpoints, n_residuals).

    """
    n_residuals = centered_residuals.shape[1]
    n_modelpoints = centered_xs.shape[0]
    y_residuals = np.empty((n_modelpoints, n_residuals), dtype=np.float64)

    for j in range(n_residuals):
        x_dot_square_terms = centered_xs @ residual_model.square_terms[j, :, :]

        for i in range(n_modelpoints):
            y_residuals[i, j] = (
                centered_residuals[i, j]
                - residual_model.linear_terms[:, j] @ centered_xs[i, :]
                - 0.5 * (x_dot_square_terms[i, :] @ centered_xs[i, :])
            )

    return y_residuals


def get_feature_matrices_residual_model(
    history, x_accepted, model_indices, delta, c2, theta2, n_maxinterp
):
    """Obtain the feature matrices for fitting the residual model.

    Pounders uses underdetermined sample sets, with at most n_maxinterp
    points in the model. Hence, the fitting method is interpolation,
    where the solution represents the quadratic whose Hessian matrix is of
    minimum Frobenius norm.

    For a mathematical exposition see :cite:`Wild2008`, p. 3-5.

    Args:
        history (LeastSquaresHistory): Class storing history of xs, residuals, and
            critvals.
        x_accepted (np.ndarray): Accepted solution vector of the subproblem.
            Shape (n_params,).
        model_indices (np.ndarray): Indices of the candidates of x that are
            currently in the model. Shape (2 * n_params + 1,).
        delta (float): Delta, current trust-region radius.
        c2 (int): Threshold for acceptance of the norm of our current x candidate.
            Equal to 10 by default.
        theta2 (float): Threshold for adding the current x candidate to the model.
        n_maxinterp (int): Maximum number of interpolation points. By default,
            2 * n_params + 1 points.

    Returns:
        Tuple:
        - m_mat (np.ndarray): Polynomial feature matrix of the linear terms.
            Shape(n_params + 1, n_params + 1).
        - n_mat (np.ndarray): Polynomial feature matrix of the square terms.
            Shape(n_modelpoints, n_poly_features).
        - z_mat (np.ndarray): Basis for the null space of m_mat.
            Shape(n_modelpoints, n_modelpoints - n_params - 1).
        - n_z_mat (np.ndarray): Lower triangular matrix of xs that form
            the monomial basis. Shape(n_poly_features, n_modelpoints - n_params - 1).
        - n_modelpoints (int): Current number of model points.

    """
    n_params = len(x_accepted)
    n_poly_features = n_params * (n_params + 1) // 2

    m_mat = np.zeros((n_maxinterp, n_params + 1))
    m_mat[:, 0] = 1
    m_mat_pad = np.zeros((n_maxinterp, n_maxinterp))
    m_mat_pad[:n_maxinterp, : n_params + 1] = m_mat

    n_mat = np.zeros((n_maxinterp, n_poly_features))

    center_info = {"x": x_accepted, "radius": delta}
    for i in range(n_params + 1):
        m_mat[i, 1:] = history.get_centered_xs(center_info, index=model_indices[i])
        n_mat[i, :] = _get_monomial_basis(m_mat[i, 1:])

    point = history.get_n_fun() - 1
    n_modelpoints = n_params + 1

    while (n_modelpoints < n_maxinterp) and (point >= 0):
        reject = False

        # Reject any points already in the model
        for i in range(n_params + 1):
            if point == model_indices[i]:
                reject = True
                break

        if reject is False:
            candidate_x = history.get_centered_xs(center_info, index=point)
            candidate_norm = np.linalg.norm(candidate_x)

            if candidate_norm > c2:
                reject = True

        if reject is True:
            point -= 1
            continue

        m_mat[n_modelpoints, 1:] = history.get_centered_xs(center_info, index=point)
        n_mat[n_modelpoints, :] = _get_monomial_basis(m_mat[n_modelpoints, 1:])

        m_mat_pad = np.zeros((n_maxinterp, n_maxinterp))
        m_mat_pad[:n_maxinterp, : n_params + 1] = m_mat

        _n_z_mat, _ = qr_multiply(
            m_mat_pad[: n_modelpoints + 1, :],
            n_mat.T[:n_poly_features, : n_modelpoints + 1],
        )
        beta = np.linalg.svd(_n_z_mat.T[n_params + 1 :], compute_uv=False)

        if beta[min(n_modelpoints - n_params, n_poly_features) - 1] > theta2:
            # Accept point
            model_indices[n_modelpoints] = point
            n_z_mat = _n_z_mat

            n_modelpoints += 1

        point -= 1

    z_mat, _ = qr_multiply(
        m_mat_pad[:n_modelpoints, :],
        np.eye(n_maxinterp)[:, :n_modelpoints],
    )

    # Just-identified case
    if n_modelpoints == (n_params + 1):
        n_z_mat = np.zeros((n_maxinterp, n_poly_features))
        n_z_mat[:n_params, :n_params] = np.eye(n_params)

    return (
        m_mat[: n_params + 1, : n_params + 1],
        n_mat[:n_modelpoints],
        z_mat[:n_modelpoints, n_params + 1 : n_modelpoints],
        n_z_mat[:, n_params + 1 : n_modelpoints],
        n_modelpoints,
    )


def fit_residual_model(
    m_mat,
    n_mat,
    z_mat,
    n_z_mat,
    y_residuals,
    n_modelpoints,
):
    """Fit a linear model using the pounders fitting method.

    Pounders uses underdetermined sample sets, with at most 2 * n_params + 1
    points in the model. Hence, the fitting method is interpolation, where
    the solution represents the quadratic whose Hessian matrix is of
    minimum Frobenius norm.

    For a mathematical exposition, see :cite:`Wild2008`, p. 3-5.

    Args:
        m_mat (np.ndarray): Polynomial feature matrix of the linear terms.
            Shape(n_params + 1, n_params + 1).
        n_mat (np.ndarray): Polynomial feature matrix of the square terms.
            Shape(n_modelpoints, n_poly_features).
        z_mat (np.ndarray): Basis for the null space of m_mat.
            Shape(n_modelpoints, n_modelpoints - n_params - 1).
        n_z_mat (np.ndarray): Lower triangular matrix of xs that form
            the monomial basis. Shape(n_poly_features, n_modelpoints - n_params - 1).
        n_modelpoints (int): Current number of model points.
        y_residuals (np.ndarray): The dependent variable. Observed minus predicted
            evaluations of the residual model. Shape (n_modelpoints, n_residuals).
        n_maxinterp (int): Maximum number of interpolation points. By default,
            2 * n_params + 1 points.

    Returns:
        dict: The coefficients of the residual model.

    """
    n_params = m_mat.shape[1] - 1
    n_residuals = y_residuals.shape[1]
    n_poly_terms = n_params * (n_params + 1) // 2
    _is_just_identified = n_modelpoints == (n_params + 1)

    coeffs_linear = np.empty((n_residuals, n_params))
    coeffs_square = np.empty((n_residuals, n_params, n_params))

    if _is_just_identified:
        coeffs_first_stage = np.zeros(n_params)
        beta = np.zeros(n_poly_terms)
    else:
        n_z_mat_square = n_z_mat.T @ n_z_mat

    for k in range(n_residuals):
        if not _is_just_identified:
            z_y_vec = np.dot(z_mat.T, y_residuals[:, k])
            coeffs_first_stage = np.linalg.solve(
                np.atleast_2d(n_z_mat_square),
                np.atleast_1d(z_y_vec),
            )

            beta = np.atleast_2d(n_z_mat) @ coeffs_first_stage

        rhs = y_residuals[:, k] - n_mat @ beta

        alpha = np.linalg.solve(m_mat, rhs[: n_params + 1])
        coeffs_linear[k, :] = alpha[1 : (n_params + 1)]

        num = 0
        for i in range(n_params):
            coeffs_square[k, i, i] = beta[num]
            num += 1
            for j in range(i + 1, n_params):
                coeffs_square[k, j, i] = beta[num] / np.sqrt(2)
                coeffs_square[k, i, j] = beta[num] / np.sqrt(2)
                num += 1

    coef = {
        "linear_terms": coeffs_linear.T,
        "square_terms": coeffs_square,
    }

    return coef


def update_trustregion_radius(
    result_subproblem,
    rho,
    model_is_valid,
    delta,
    delta_min,
    delta_max,
    eta1,
    gamma0,
    gamma1,
):
    """Update the trust-region radius."""
    norm_x_sub = np.sqrt(np.sum(result_subproblem["x"] ** 2))

    if rho >= eta1 and norm_x_sub > 0.5 * delta:
        delta = min(delta * gamma1, delta_max)
    elif model_is_valid is True:
        delta = max(delta * gamma0, delta_min)

    return delta


def get_last_model_indices_and_check_for_repeated_model(
    model_indices, last_model_indices, n_modelpoints, n_last_modelpoints
):
    """Get the last model_indices and check if we have reused the same model."""
    if n_modelpoints == n_last_modelpoints:
        same_model_used = True
    else:
        same_model_used = False

    for i in range(n_modelpoints):
        if same_model_used:
            if model_indices[i] == last_model_indices[i]:
                same_model_used = True
            else:
                same_model_used = False
        last_model_indices[i] = model_indices[i]

    n_last_modelpoints = n_modelpoints

    return last_model_indices, n_last_modelpoints, same_model_used


def add_accepted_point_to_residual_model(model_indices, accepted_index, n_modelpoints):
    """Add accepted point to the residual model."""
    model_indices[1 : n_modelpoints + 1] = model_indices[:n_modelpoints]
    model_indices[0] = accepted_index

    return model_indices


def _get_monomial_basis(x):
    """Get the monomial basis (basis for quadratic functions) of x.

    Monomial basis = .5*[x(1)^2  sqrt(2)*x(1)*x(2) ... sqrt(2)*x(1)*x(n_params) ...
        ... x(2)^2 sqrt(2)*x(2)*x(3) .. x(n_params)^2]

    Args:
        x (np.ndarray): Parameter vector of shape (n_params,).

    Returns:
        np.ndarray: Monomial basis of x of shape (n_params * (n_params + 1) / 2,).

    """
    n_params = len(x)
    monomial_basis = np.zeros(int(n_params * (n_params + 1) / 2))

    j = 0
    for i in range(n_params):
        monomial_basis[j] = 0.5 * x[i] ** 2
        j += 1

        for k in range(i + 1, n_params):
            monomial_basis[j] = x[i] * x[k] / np.sqrt(2)
            j += 1

    return monomial_basis
