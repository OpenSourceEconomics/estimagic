"""Finite difference formulae for jacobians and hessians.

All functions in this module should not only work for the simple case of one positive
and/or one negative step, but also for the Richardson Extrapolation case with several
positive and/or several negative steps.

Since steps and evals contain NaNs, we have to make sure that the functions do not raise
warnings or errors for that case.

"""

from typing import NamedTuple

import numpy as np


class Evals(NamedTuple):
    pos: np.ndarray
    neg: np.ndarray


def jacobian(evals, steps, f0, method):
    """Calculate a Jacobian estimate with finite differences according to method.

    Notation: f:R^dim_x -> R^dim_f. We compute the derivative at x0, with f0 = f(x0).

    Args:
        evals (namedtuple): It has the fields called pos and neg for evaluations with
            positive and negative steps, respectively. Each field is a numpy array
            of shape (n_steps, dim_f, dim_x). It contains np.nan for evaluations that
            failed or were not attempted because a one-sided derivative rule was chosen.
        steps (namedtuple): Namedtuple with the fields pos and neg. Each field
            contains a numpy array of shape (n_steps, dim_x) with the steps in
            the corresponding direction. The steps are always symmetric, in the sense
            that steps.neg[i, j] = - steps.pos[i, j] unless one of them is NaN.
        f0 (numpy.ndarray): Numpy array of length dim_f with the output of the function
            at the user supplied parameters.
        method (str): One of ["forward", "backward", "central"]

    Returns:
        jac (numpy.ndarray): Numpy array of shape (n_steps, dim_f, dim_x) with estimated
            Jacobians. I.e. there are n_step jacobian estimates.

    """
    n_steps, dim_f, dim_x = evals.pos.shape
    if method == "forward":
        diffs = evals.pos - f0.reshape(1, dim_f, 1)
        jac = diffs / steps.pos.reshape(n_steps, 1, dim_x)
    elif method == "backward":
        diffs = evals.neg - f0.reshape(1, dim_f, 1)
        jac = diffs / steps.neg.reshape(n_steps, 1, dim_x)
    elif method == "central":
        diffs = evals.pos - evals.neg
        deltas = steps.pos - steps.neg
        jac = diffs / deltas.reshape(n_steps, 1, dim_x)
    else:
        raise ValueError("Method has to be 'forward', 'backward' or 'central'.")
    return jac


def hessian(evals, steps, f0, method):
    """Calculate a Hessian estimate with finite differences according to method.

    Notation: f:R^dim_x -> R^dim_f. We compute the derivative at x0, with f0 = f(x0).

    The formulae in Rideout [2009] which are implemented here use three types of
    function evaluations:

    1. f(theta + delta_j e_j)
    2. f(theta + delta_j e_j + delta_k e_k)
    3. f(theta + delta_j e_j - delta_k e_k)

    Which are called here: 1. ``evals_one``, 2. ``evals_two`` and 3. ``evals_cross``,
    corresponding to the idea that we are moving in one direction, in two directions and
    in two cross directions (opposite signs). Note that theta denotes x0, delta_j the
    step size for the j-th variable and e_j the j-th standard basis vector.

    Note also that the brackets in the finite difference formulae are not arbitrary but
    improve the numerical accuracy, see Rideout [2009].

    Args:
        evals (dict[namedtuple]): Dictionary with keys "one_step" for function evals in
            a single step direction, "two_step" for evals in two steps in the same
            direction, and "cross_step" for evals in two steps in the opposite
            direction. Each dict item has the fields called pos and neg for evaluations
            with positive and negative steps, respectively. Each field is a numpy array
            of shape (n_steps, dim_f, dim_x). It contains np.nan for evaluations that
            failed or were not attempted because a one-sided derivative rule was chosen.
        steps (namedtuple): Namedtuple with the fields pos and neg. Each field
            contains a numpy array of shape (n_steps, dim_x) with the steps in
            the corresponding direction. The steps are always symmetric, in the sense
            that steps.neg[i, j] = - steps.pos[i, j] unless one of them is NaN.
        f0 (numpy.ndarray): Numpy array of length dim_f with the output of the function
            at the user supplied parameters.
        method (str): One of {"forward", "backward", "central_average", "central_cross"}
            These correspond to the finite difference approximations defined in
            equations [7, x, 8, 9] in Rideout [2009], where ("backward", x) is not found
            in Rideout [2009] but is the natural extension of equation 7 to the backward
            case.

    Returns:
        hess (numpy.ndarray): Numpy array of shape (n_steps, dim_f, dim_x, dim_x) with
            estimated Hessians. I.e. there are n_step hessian estimates.

    """
    n_steps, dim_f, dim_x = evals["one_step"].pos.shape
    f0 = f0.reshape(1, dim_f, 1, 1)

    # rename variables to increase readability in formulas
    evals_one = Evals(
        pos=np.expand_dims(evals["one_step"].pos, axis=3),
        neg=np.expand_dims(evals["one_step"].neg, axis=3),
    )
    evals_two = evals["two_step"]
    evals_cross = evals["cross_step"]

    if method == "forward":
        outer_product_steps = _calculate_outer_product_steps(steps.pos, n_steps, dim_x)
        diffs = (evals_two.pos - evals_one.pos.swapaxes(2, 3)) - (evals_one.pos - f0)
        hess = diffs / outer_product_steps
    elif method == "backward":
        outer_product_steps = _calculate_outer_product_steps(steps.neg, n_steps, dim_x)
        diffs = (evals_two.neg - evals_one.neg.swapaxes(2, 3)) - (evals_one.neg - f0)
        hess = diffs / outer_product_steps
    elif method == "central_average":
        outer_product_steps = _calculate_outer_product_steps(steps.pos, n_steps, dim_x)
        forward = (evals_two.pos - evals_one.pos.swapaxes(2, 3)) - (evals_one.pos - f0)
        backward = (evals_two.neg - evals_one.neg.swapaxes(2, 3)) - (evals_one.neg - f0)
        hess = (forward + backward) / (2 * outer_product_steps)
    elif method == "central_cross":
        outer_product_steps = _calculate_outer_product_steps(steps.pos, n_steps, dim_x)
        diffs = (evals_two.pos - evals_cross.pos) - (evals_cross.neg - evals_two.neg)
        hess = diffs / (4 * outer_product_steps)
    else:
        raise ValueError(
            "Method has to be 'forward', 'backward', 'central_average' or ",
            "'central_cross'.",
        )
    return hess


def _calculate_outer_product_steps(signed_steps, n_steps, dim_x):
    """Calculate array of outer product of steps.

    Args:
        signed_steps (np.ndarray): Square array with either pos or neg steps returned
            by :func:`~optimagic.differentiation.generate_steps.generate_steps` function
        n_steps (int): Number of steps needed. For central methods, this is
            the number of steps per direction. It is 1 if no Richardson extrapolation
            is used.
        dim_x (int): Dimension of input vector x.

    Returns:
        outer_product_steps (np.ndarray): Array with outer product of steps. Has
            dimension (n_steps, 1, dim_x, dim_x).

    """
    outer_product_steps = np.array(
        [np.outer(signed_steps[j], signed_steps[j]) for j in range(n_steps)]
    ).reshape(n_steps, 1, dim_x, dim_x)
    return outer_product_steps
