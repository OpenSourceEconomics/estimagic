"""Auxiliary functions for the quadratic BNTR trust-region subsolver."""

from functools import reduce
from typing import NamedTuple

import numpy as np

from optimagic.optimizers._pounders._conjugate_gradient import (
    minimize_trust_cg,
)
from optimagic.optimizers._pounders._steihaug_toint import (
    minimize_trust_stcg,
)
from optimagic.optimizers._pounders._trsbox import minimize_trust_trsbox

EPSILON = np.finfo(float).eps ** (2 / 3)


class ActiveBounds(NamedTuple):
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    fixed: np.ndarray | None = None
    active: np.ndarray | None = None
    inactive: np.ndarray | None = None


def bntr(
    model,
    lower_bounds,
    upper_bounds,
    x_candidate,
    *,
    conjugate_gradient_method,
    maxiter,
    maxiter_gradient_descent,
    gtol_abs,
    gtol_rel,
    gtol_scaled,
    gtol_abs_conjugate_gradient,
    gtol_rel_conjugate_gradient,
):
    """Minimize a bounded trust-region subproblem via Newton Conjugate Gradient method.

    The BNTR (Bounded Newton Trust Rregion) algorithm uses an active-set approach
    to solve the symmetric system of equations:

        hessian @ x = - gradient

    only for the inactive parameters of x that lie within the bounds. The active-set
    estimation employed here is based on Bertsekas (:cite:`Bertsekas1982`).

    In the main loop, BNTR globalizes the Newton step using a trust-region method
    based on the predicted versus actual reduction in the criterion function.
    The trust-region radius is increased only if the accepted step is at the
    trust-region boundary.


    Args:
        model (NamedTuple): NamedTuple containing the parameters of the
            main model, i.e.:
            - ``linear_terms`` (np.ndarray): 1d array of shape (n,)
            - ``square_terms`` (np.ndarray): 2d array of shape (n,n).
        lower_bounds (np.ndarray): 1d array of shape (n,) with lower bounds
            for the parameter vector x.
        upper_bounds (np.ndarray): 1d array of shape (n,) with upper bounds
            for the parameter vector x.
        x_candidate (np.ndarray): Initial guess for the solution of the subproblem.
        conjugate_gradient_method (str): Method for computing the conjugate gradient
            step. Available conjugate gradient methods are:
                - "cg"
                - "steihaug_toint"
                - "trsbox" (default)
        maxiter (int): Maximum number of iterations. If reached, terminate.
        maxiter_gradient_descent (int): Maximum number of steepest descent iterations
            to perform when the trust-region subsolver BNTR is used.
        gtol_abs (float): Convergence tolerance for the absolute gradient norm.
        gtol_rel (float): Convergence tolerance for the relative gradient norm.
        gtol_scaled (float): Convergence tolerance for the scaled gradient norm.
        gtol_abs_conjugate_gradient (float): Convergence tolerance for the absolute
            gradient norm in the conjugate gradient step of the trust-region
            subproblem ("BNTR").
        gtol_rel_conjugate_gradient (float): Convergence tolerance for the relative
            gradient norm in the conjugate gradient step of the trust-region
            subproblem ("BNTR").

    Returns:
        (dict): Result dictionary containing the following keys:
            - ``x`` (np.ndarray): Solution vector of the subproblem of shape (n,)
            - ``criterion`` (float): Minimum function value associated with the
                solution.
            - ``n_iterations`` (int): Number of iterations the algorithm ran before
                termination.
            - ``success`` (bool): Boolean indicating whether a solution has been found
                before reaching maxiter.

    """
    options_update_radius = {
        "eta1": 1.0e-4,
        "eta2": 0.25,
        "eta3": 0.50,
        "eta4": 0.90,
        "alpha1": 0.25,
        "alpha2": 0.50,
        "alpha3": 1.00,
        "alpha4": 2.00,
        "alpha5": 4.00,
        "min_radius": 1e-10,
        "max_radius": 1e10,
        "default_radius": 100.00,
    }

    (
        x_candidate,
        f_candidate,
        gradient_unprojected,
        hessian_bounds_inactive,
        trustregion_radius,
        active_bounds_info,
        converged,
        convergence_reason,
    ) = _take_preliminary_gradient_descent_step_and_check_for_solution(
        x_candidate,
        model,
        lower_bounds,
        upper_bounds,
        maxiter_gradient_descent,
        gtol_abs,
        gtol_rel,
        gtol_scaled,
    )

    for niter in range(maxiter + 1):
        if converged:
            break

        x_old = x_candidate
        f_old = f_candidate
        accept_step = False

        while not accept_step and not converged:
            gradient_bounds_inactive = gradient_unprojected[active_bounds_info.inactive]
            hessian_bounds_inactive = _find_hessian_submatrix_where_bounds_inactive(
                model, active_bounds_info
            )
            (
                conjugate_gradient_step,
                conjugate_gradient_step_inactive_bounds,
                cg_step_norm,
            ) = _compute_conjugate_gradient_step(
                x_candidate,
                gradient_bounds_inactive,
                hessian_bounds_inactive,
                lower_bounds,
                upper_bounds,
                active_bounds_info,
                trustregion_radius,
                conjugate_gradient_method=conjugate_gradient_method,
                gtol_abs_conjugate_gradient=gtol_abs_conjugate_gradient,
                gtol_rel_conjugate_gradient=gtol_rel_conjugate_gradient,
                options_update_radius=options_update_radius,
            )

            x_unbounded = x_candidate + conjugate_gradient_step
            x_candidate = _apply_bounds_to_x_candidate(
                x_unbounded, lower_bounds, upper_bounds
            )

            predicted_reduction = (
                _compute_predicted_reduction_from_conjugate_gradient_step(
                    conjugate_gradient_step,
                    conjugate_gradient_step_inactive_bounds,
                    gradient_unprojected,
                    gradient_bounds_inactive,
                    hessian_bounds_inactive,
                    active_bounds_info,
                )
            )

            f_candidate = _evaluate_model_criterion(
                x_candidate, model.linear_terms, model.square_terms
            )
            actual_reduction = f_old - f_candidate

            trustregion_radius_old = trustregion_radius
            (
                trustregion_radius,
                accept_step,
            ) = _update_trustregion_radius_conjugate_gradient(
                f_candidate,
                predicted_reduction,
                actual_reduction,
                cg_step_norm,
                trustregion_radius,
                options_update_radius,
            )

            if accept_step:
                gradient_unprojected = (
                    model.linear_terms + model.square_terms @ x_candidate
                )

                active_bounds_info = _get_information_on_active_bounds(
                    x_candidate,
                    gradient_unprojected,
                    lower_bounds,
                    upper_bounds,
                )
            else:
                x_candidate = x_old
                f_candidate = f_old

                if trustregion_radius == trustregion_radius_old:
                    converged = True
                    break

            converged, convergence_reason = _check_for_convergence(
                x_candidate,
                f_candidate,
                gradient_unprojected,
                model,
                lower_bounds,
                upper_bounds,
                converged,
                convergence_reason,
                niter,
                maxiter=maxiter,
                gtol_abs=gtol_abs,
                gtol_rel=gtol_rel,
                gtol_scaled=gtol_scaled,
            )

    result = {
        "x": x_candidate,
        "criterion": f_candidate,
        "n_iterations": niter,
        "success": converged,
        "message": convergence_reason,
    }

    return result


def _take_preliminary_gradient_descent_step_and_check_for_solution(
    x_candidate,
    model,
    lower_bounds,
    upper_bounds,
    maxiter_gradient_descent,
    gtol_abs,
    gtol_rel,
    gtol_scaled,
):
    """Take a preliminary gradient descent step and check if we found a solution."""
    options_update_radius = {
        "mu1": 0.35,
        "mu2": 0.50,
        "gamma1": 0.0625,
        "gamma2": 0.5,
        "gamma3": 2.0,
        "gamma4": 5.0,
        "theta": 0.25,
        "min_radius": 1e-10,
        "max_radius": 1e10,
        "default_radius": 100.0,
    }

    converged = False
    convergence_reason = "Continue iterating."

    criterion_candidate = _evaluate_model_criterion(
        x_candidate, model.linear_terms, model.square_terms
    )

    active_bounds_info = _get_information_on_active_bounds(
        x_candidate,
        model.linear_terms,
        lower_bounds,
        upper_bounds,
    )

    gradient_unprojected = model.linear_terms + model.square_terms @ x_candidate
    gradient_projected = _project_gradient_onto_feasible_set(
        gradient_unprojected, active_bounds_info
    )

    converged, convergence_reason = _check_for_convergence(
        x_candidate,
        criterion_candidate,
        gradient_unprojected,
        model,
        lower_bounds,
        upper_bounds,
        converged,
        convergence_reason,
        niter=None,
        maxiter=None,
        gtol_abs=gtol_abs,
        gtol_rel=gtol_rel,
        gtol_scaled=gtol_scaled,
    )

    if converged:
        hessian_inactive = model.square_terms
        trustregion_radius = options_update_radius["default_radius"]
    else:
        hessian_inactive = _find_hessian_submatrix_where_bounds_inactive(
            model, active_bounds_info
        )

        (
            x_candidate_gradient_descent,
            f_min_gradient_descent,
            step_size_gradient_descent,
            trustregion_radius,
            radius_lower_bound,
        ) = _perform_gradient_descent_step(
            x_candidate,
            criterion_candidate,
            gradient_projected,
            hessian_inactive,
            model,
            lower_bounds,
            upper_bounds,
            active_bounds_info,
            maxiter_gradient_descent,
            options_update_radius,
        )

        if f_min_gradient_descent < criterion_candidate:
            criterion_candidate = f_min_gradient_descent

            x_unbounded = (
                x_candidate_gradient_descent
                - step_size_gradient_descent * gradient_projected
            )
            x_candidate = _apply_bounds_to_x_candidate(
                x_unbounded, lower_bounds, upper_bounds
            )

            gradient_unprojected = model.linear_terms + model.square_terms @ x_candidate
            active_bounds_info = _get_information_on_active_bounds(
                x_candidate,
                gradient_unprojected,
                lower_bounds,
                upper_bounds,
            )

            gradient_projected = _project_gradient_onto_feasible_set(
                gradient_unprojected, active_bounds_info
            )
            hessian_inactive = _find_hessian_submatrix_where_bounds_inactive(
                model, active_bounds_info
            )

            converged, convergence_reason = _check_for_convergence(
                x_candidate,
                criterion_candidate,
                gradient_projected,
                model,
                lower_bounds,
                upper_bounds,
                converged,
                convergence_reason,
                niter=None,
                maxiter=None,
                gtol_abs=gtol_abs,
                gtol_rel=gtol_rel,
                gtol_scaled=gtol_scaled,
            )

        if not converged:
            trustregion_radius = np.clip(
                max(trustregion_radius, radius_lower_bound),
                options_update_radius["min_radius"],
                options_update_radius["max_radius"],
            )

    return (
        x_candidate,
        criterion_candidate,
        gradient_unprojected,
        hessian_inactive,
        trustregion_radius,
        active_bounds_info,
        converged,
        convergence_reason,
    )


def _compute_conjugate_gradient_step(
    x_candidate,
    gradient_inactive,
    hessian_inactive,
    lower_bounds,
    upper_bounds,
    active_bounds_info,
    trustregion_radius,
    *,
    conjugate_gradient_method,
    gtol_abs_conjugate_gradient,
    gtol_rel_conjugate_gradient,
    options_update_radius,
):
    """Compute the bounded Conjugate Gradient trust-region step."""
    conjugate_gradient_step = np.zeros_like(x_candidate)

    if active_bounds_info.inactive.size == 0:
        # Save some computation and return an adjusted zero step
        step_inactive = _apply_bounds_to_x_candidate(
            x_candidate, lower_bounds, upper_bounds
        )
        step_norm = np.linalg.norm(step_inactive)

        conjugate_gradient_step = _apply_bounds_to_conjugate_gradient_step(
            step_inactive,
            x_candidate,
            lower_bounds,
            upper_bounds,
            active_bounds_info,
        )

    else:
        if conjugate_gradient_method == "cg":
            step_inactive = minimize_trust_cg(
                gradient_inactive,
                hessian_inactive,
                trustregion_radius,
                gtol_abs=gtol_abs_conjugate_gradient,
                gtol_rel=gtol_rel_conjugate_gradient,
            )
            step_norm = np.linalg.norm(step_inactive)
        elif conjugate_gradient_method == "steihaug_toint":
            step_inactive = minimize_trust_stcg(
                gradient_inactive,
                hessian_inactive,
                trustregion_radius,
            )
            step_norm = np.linalg.norm(step_inactive)
        elif conjugate_gradient_method == "trsbox":
            step_inactive = minimize_trust_trsbox(
                gradient_inactive,
                hessian_inactive,
                trustregion_radius,
                lower_bounds=lower_bounds[active_bounds_info.inactive],
                upper_bounds=upper_bounds[active_bounds_info.inactive],
            )
            step_norm = np.linalg.norm(step_inactive)
        else:
            raise ValueError(
                "Invalid method: {conjugate_gradient_method}. "
                "Must be one of cg, steihaug_toint, trsbox."
            )

        if trustregion_radius == 0:
            if step_norm > 0:
                # Accept
                trustregion_radius = np.clip(
                    step_norm,
                    options_update_radius["min_radius"],
                    options_update_radius["max_radius"],
                )

            else:
                # Re-solve
                trustregion_radius = np.clip(
                    options_update_radius["default_radius"],
                    options_update_radius["min_radius"],
                    options_update_radius["max_radius"],
                )

                if conjugate_gradient_method == "cg":
                    step_inactive = minimize_trust_cg(
                        gradient_inactive,
                        hessian_inactive,
                        trustregion_radius,
                        gtol_abs=gtol_abs_conjugate_gradient,
                        gtol_rel=gtol_rel_conjugate_gradient,
                    )
                    step_norm = np.linalg.norm(step_inactive)
                elif conjugate_gradient_method == "steihaug_toint":
                    step_inactive = minimize_trust_stcg(
                        gradient_inactive,
                        hessian_inactive,
                        trustregion_radius,
                    )
                    step_norm = np.linalg.norm(step_inactive)
                elif conjugate_gradient_method == "trsbox":
                    step_inactive = minimize_trust_trsbox(
                        gradient_inactive,
                        hessian_inactive,
                        trustregion_radius,
                        lower_bounds=lower_bounds[active_bounds_info.inactive],
                        upper_bounds=upper_bounds[active_bounds_info.inactive],
                    )
                    step_norm = np.linalg.norm(step_inactive)

                if step_norm == 0:
                    raise ValueError("Initial direction is zero.")

        conjugate_gradient_step = _apply_bounds_to_conjugate_gradient_step(
            step_inactive,
            x_candidate,
            lower_bounds,
            upper_bounds,
            active_bounds_info,
        )

    return (
        conjugate_gradient_step,
        step_inactive,
        step_norm,
    )


def _compute_predicted_reduction_from_conjugate_gradient_step(
    conjugate_gradient_step,
    conjugate_gradient_step_inactive,
    gradient_unprojected,
    gradient_inactive,
    hessian_inactive,
    active_bounds_info,
):
    """Compute predicted reduction induced by the Conjugate Gradient step."""
    if active_bounds_info.active.size > 0:
        # Projection changed the step, so we have to recompute the step
        # and the predicted reduction. Leave the rust radius unchanged.
        cg_step_recomp = conjugate_gradient_step[active_bounds_info.inactive]
        gradient_inactive_recomp = gradient_unprojected[active_bounds_info.inactive]

        predicted_reduction = _evaluate_model_criterion(
            cg_step_recomp, gradient_inactive_recomp, hessian_inactive
        )
    else:
        # Step did not change, so we can just recover the
        # pre-computed prediction
        predicted_reduction = _evaluate_model_criterion(
            conjugate_gradient_step_inactive,
            gradient_inactive,
            hessian_inactive,
        )

    return -predicted_reduction


def _perform_gradient_descent_step(
    x_candidate,
    f_candidate_initial,
    gradient_projected,
    hessian_inactive,
    model,
    lower_bounds,
    upper_bounds,
    active_bounds_info,
    maxiter_steepest_descent,
    options_update_radius,
):
    """Perform gradient descent step and update trust-region radius."""
    f_min = f_candidate_initial
    gradient_norm = np.linalg.norm(gradient_projected)

    trustregion_radius = options_update_radius["default_radius"]
    radius_lower_bound = 0
    step_size_accepted = 0

    for _ in range(maxiter_steepest_descent):
        x_old = x_candidate

        step_size_candidate = trustregion_radius / gradient_norm
        x_candidate = x_old - step_size_candidate * gradient_projected

        x_candidate = _apply_bounds_to_x_candidate(
            x_candidate, lower_bounds, upper_bounds
        )
        f_candidate = _evaluate_model_criterion(
            x_candidate, model.linear_terms, model.square_terms
        )

        x_diff = x_candidate - x_old

        if f_candidate < f_min:
            f_min = f_candidate
            step_size_accepted = step_size_candidate

        x_inactive = x_diff[active_bounds_info.inactive]
        square_terms = x_inactive.T @ hessian_inactive @ x_inactive

        predicted_reduction = trustregion_radius * (
            gradient_norm - 0.5 * trustregion_radius * square_terms / (gradient_norm**2)
        )
        actual_reduction = f_candidate_initial - f_candidate

        (
            trustregion_radius,
            radius_lower_bound,
        ) = _update_trustregion_radius_and_gradient_descent(
            trustregion_radius,
            radius_lower_bound,
            predicted_reduction,
            actual_reduction,
            gradient_norm,
            options_update_radius,
        )

    return (
        x_candidate,
        f_min,
        step_size_accepted,
        trustregion_radius,
        radius_lower_bound,
    )


def _update_trustregion_radius_conjugate_gradient(
    f_candidate,
    predicted_reduction,
    actual_reduction,
    x_norm_cg,
    trustregion_radius,
    options,
):
    """Update the trust-region radius based on predicted and actual reduction."""
    accept_step = False

    if predicted_reduction < 0 or ~np.isfinite(predicted_reduction):
        # Reject and start over
        trustregion_radius = options["alpha1"] * min(trustregion_radius, x_norm_cg)

    else:
        if ~np.isfinite(actual_reduction):
            trustregion_radius = options["alpha1"] * min(trustregion_radius, x_norm_cg)
        else:
            if abs(actual_reduction) <= max(1, abs(f_candidate) * EPSILON) and abs(
                predicted_reduction
            ) <= max(1, abs(f_candidate) * EPSILON):
                kappa = 1
            else:
                kappa = actual_reduction / predicted_reduction

            if kappa < options["eta1"]:
                # Reject the step
                trustregion_radius = options["alpha1"] * min(
                    trustregion_radius, x_norm_cg
                )
            else:
                accept_step = True

                # Update the trust-region radius only if the computed step is at the
                # trust-radius boundary
                if x_norm_cg == trustregion_radius:
                    if kappa < options["eta2"]:
                        # Marginal bad step
                        trustregion_radius = options["alpha2"] * trustregion_radius
                    elif kappa < options["eta3"]:
                        # Reasonable step
                        trustregion_radius = options["alpha3"] * trustregion_radius
                    elif kappa < options["eta4"]:
                        trustregion_radius = options["alpha4"] * trustregion_radius
                    else:
                        # Very good step
                        trustregion_radius = options["alpha5"] * trustregion_radius

    trustregion_radius = np.clip(
        trustregion_radius, options["min_radius"], options["max_radius"]
    )

    return trustregion_radius, accept_step


def _get_information_on_active_bounds(
    x,
    gradient_unprojected,
    lower_bounds,
    upper_bounds,
):
    """Return the index set of active bounds."""
    active_lower = np.where((x <= lower_bounds) & (gradient_unprojected > 0))[0]
    active_upper = np.where((x >= upper_bounds) & (gradient_unprojected < 0))[0]
    active_fixed = np.where(lower_bounds == upper_bounds)[0]
    active_all = reduce(np.union1d, (active_fixed, active_lower, active_upper))
    inactive = np.setdiff1d(np.arange(len(x)), active_all)

    active_bounds_info = ActiveBounds(
        lower=active_lower,
        upper=active_upper,
        fixed=active_fixed,
        active=active_all,
        inactive=inactive,
    )

    return active_bounds_info


def _find_hessian_submatrix_where_bounds_inactive(model, active_bounds_info):
    """Find the submatrix of the initial hessian where bounds are inactive."""
    hessian_inactive = model.square_terms[
        active_bounds_info.inactive[:, np.newaxis], active_bounds_info.inactive
    ]

    return hessian_inactive


def _check_for_convergence(
    x_candidate,
    f_candidate,
    gradient_candidate,
    model,
    lower_bounds,
    upper_bounds,
    converged,
    reason,
    niter,
    *,
    maxiter,
    gtol_abs,
    gtol_rel,
    gtol_scaled,
):
    """Check if we have found a solution."""
    direction_fischer_burmeister = _get_fischer_burmeister_direction_vector(
        x_candidate, gradient_candidate, lower_bounds, upper_bounds
    )
    gradient_norm = np.linalg.norm(direction_fischer_burmeister)
    gradient_norm_initial = np.linalg.norm(model.linear_terms)

    if gradient_norm < gtol_abs:
        converged = True
        reason = "Norm of the gradient is less than absolute_gradient_tolerance."
    elif f_candidate != 0 and abs(gradient_norm / f_candidate) < gtol_rel:
        converged = True
        reason = (
            "Norm of the gradient relative to the criterion value is less than "
            "relative_gradient_tolerance."
        )
    elif (
        gradient_norm_initial != 0
        and gradient_norm / gradient_norm_initial < gtol_scaled
    ):
        converged = True
        reason = (
            "Norm of the gradient divided by norm of the gradient at the "
            "initial parameters is less than scaled_gradient_tolerance."
        )
    elif gradient_norm_initial != 0 and gradient_norm == 0 and gtol_scaled == 0:
        converged = True
        reason = (
            "Norm of the gradient divided by norm of the gradient at the "
            "initial parameters is less than scaled_gradient_tolerance."
        )
    elif f_candidate <= -np.inf:
        converged = True
        reason = "Criterion value is negative infinity."
    elif niter is not None and niter == maxiter:
        reason = "Maximum number of iterations reached."

    return converged, reason


def _apply_bounds_to_x_candidate(x, lower_bounds, upper_bounds, bound_tol=0):
    """Apply upper and lower bounds to the candidate vector."""
    x = np.where(x <= lower_bounds + bound_tol, lower_bounds, x)
    x = np.where(x >= upper_bounds - bound_tol, upper_bounds, x)

    return x


def _project_gradient_onto_feasible_set(gradient_unprojected, active_bounds_info):
    """Project gradient onto feasible set, where search directions unconstrained."""
    gradient_projected = np.zeros_like(gradient_unprojected)
    gradient_projected[active_bounds_info.inactive] = gradient_unprojected[
        active_bounds_info.inactive
    ]

    return gradient_projected


def _apply_bounds_to_conjugate_gradient_step(
    step_inactive,
    x_candidate,
    lower_bounds,
    upper_bounds,
    active_bounds_info,
):
    """Apply lower and upper bounds to the Conjugate Gradient step."""
    cg_step = np.zeros_like(x_candidate)
    cg_step[active_bounds_info.inactive] = step_inactive

    if active_bounds_info.lower.size > 0:
        x_active_lower = x_candidate[active_bounds_info.lower]
        lower_bound_active = lower_bounds[active_bounds_info.lower]

        cg_step[active_bounds_info.lower] = lower_bound_active - x_active_lower

    if active_bounds_info.upper.size > 0:
        x_active_upper = x_candidate[active_bounds_info.upper]
        upper_bound_active = upper_bounds[active_bounds_info.upper]

        cg_step[active_bounds_info.upper] = upper_bound_active - x_active_upper

    if active_bounds_info.fixed.size > 0:
        cg_step[active_bounds_info.fixed] = 0

    return cg_step


def _update_trustregion_radius_and_gradient_descent(
    trustregion_radius,
    radius_lower_bound,
    predicted_reduction,
    actual_reduction,
    gradient_norm,
    options,
):
    """Update the trust-region radius and its upper bound."""
    if abs(actual_reduction) <= EPSILON and abs(predicted_reduction) <= EPSILON:
        kappa = 1
    else:
        kappa = actual_reduction / predicted_reduction

    tau_1 = (
        options["theta"]
        * gradient_norm
        * trustregion_radius
        / (
            options["theta"] * gradient_norm * trustregion_radius
            + (1 - options["theta"]) * predicted_reduction
            - actual_reduction
        )
    )
    tau_2 = (
        options["theta"]
        * gradient_norm
        * trustregion_radius
        / (
            options["theta"] * gradient_norm * trustregion_radius
            - (1 + options["theta"]) * predicted_reduction
            + actual_reduction
        )
    )

    tau_min = min(tau_1, tau_2)
    tau_max = max(tau_1, tau_2)

    if abs(kappa - 1) <= options["mu1"]:
        # Great agreement
        radius_lower_bound = max(radius_lower_bound, trustregion_radius)

        if tau_max < 1:
            tau = options["gamma3"]
        elif tau_max > options["gamma4"]:
            tau = options["gamma4"]
        else:
            tau = tau_max

    elif abs(kappa - 1) <= options["mu2"]:
        # Good agreement
        radius_lower_bound = max(radius_lower_bound, trustregion_radius)

        if tau_max < options["gamma2"]:
            tau = options["gamma2"]
        elif tau_max > options["gamma3"]:
            tau = options["gamma3"]
        else:
            tau = tau_max

    else:
        # Not good agreement
        if tau_min > 1:
            tau = options["gamma2"]
        elif tau_max < options["gamma1"]:
            tau = options["gamma1"]
        elif (tau_min < options["gamma1"]) and (tau_max >= 1):
            tau = options["gamma1"]
        elif (
            (tau_1 >= options["gamma1"])
            and (tau_1 < 1.0)
            and ((tau_2 < options["gamma1"]) or (tau_2 >= 1.0))
        ):
            tau = tau_1
        elif (
            (tau_2 >= options["gamma1"])
            and (tau_2 < 1.0)
            and ((tau_1 < options["gamma1"]) or (tau_2 >= 1.0))
        ):
            tau = tau_2
        else:
            tau = tau_max

    trustregion_radius = trustregion_radius * tau

    return trustregion_radius, radius_lower_bound


def _get_fischer_burmeister_direction_vector(x, gradient, lower_bounds, upper_bounds):
    """Compute the constrained direction vector via the Fischer-Burmeister function."""
    fischer_vec = np.vectorize(_get_fischer_burmeister_scalar)

    fischer_burmeister = reduce(
        fischer_vec, (upper_bounds - x, -gradient, x - lower_bounds)
    )
    direction = np.where(
        lower_bounds == upper_bounds, lower_bounds - x, fischer_burmeister
    )

    return direction


def _get_fischer_burmeister_scalar(a, b):
    """Get the value of the Fischer-Burmeister function for two scalar inputs.

    This method was suggested by Bob Vanderbei. Since the Fischer-Burmeister
    is symmetric, the order of the scalar inputs does not matter.

    Args:
        a (float): First input.
        b (float): Second input.

    Returns:
        float: Value of the Fischer-Burmeister function for inputs a and b.

    """
    if a + b <= 0:
        fischer_burmeister = np.sqrt(a**2 + b**2) - (a + b)
    else:
        fischer_burmeister = -2 * a * b / (np.sqrt(a**2 + b**2) + (a + b))

    return fischer_burmeister


def _evaluate_model_criterion(
    x,
    gradient,
    hessian,
):
    """Evaluate the criterion function value of the main model.

    Args:
        x (np.ndarray): Parameter vector of shape (n,).
        gradient (np.ndarray): Gradient of shape (n,) for which the main model
            shall be evaluated.
        hessian (np.ndarray): Hessian of shape (n, n) for which the main model
            shall be evaulated.

    Returns:
        float: Criterion value of the main model.

    """
    return gradient.T @ x + 0.5 * x.T @ hessian @ x
