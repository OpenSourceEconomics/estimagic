"""Implementation of parallelosation of Nelder-Mead algorithm."""

import numpy as np

from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.decorators import mark_minimizer
from optimagic.optimization.algo_options import (
    CONVERGENCE_SECOND_BEST_FTOL_ABS,
    CONVERGENCE_SECOND_BEST_XTOL_ABS,
    STOPPING_MAXITER,
)


@mark_minimizer(
    name="neldermead_parallel",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=True,
    disable_history=True,
)
def neldermead_parallel(
    criterion,
    x,
    *,
    init_simplex_method="gao_han",
    n_cores=1,
    adaptive=True,
    stopping_maxiter=STOPPING_MAXITER,
    convergence_ftol_abs=CONVERGENCE_SECOND_BEST_FTOL_ABS,  # noqa: E501
    convergence_xtol_abs=CONVERGENCE_SECOND_BEST_XTOL_ABS,  # noqa: E501
    batch_evaluator="joblib",
):
    """Parallel Nelder-Mead algorithm following Lee D., Wiswall M., A parallel
    implementation of the simplex function minimization routine, Computational
    Economics, 2007.

    Parameters
    ----------
    criterion (callable): A function that takes a Numpy array_like as
        an argument and return scalar floating point.

    x (array_like): 1-D array of initial value of parameters

    init_simplex_method (string or callable): Name of the method to create initial
        simplex or callable which takes as an argument initial value of parameters
        and returns initial simplex as j+1 x j array, where j is length of x.
        The default is "gao_han".

    n_cores (int): Degrees of parallization. The default is 1 (no parallelization).

    adaptive (bool): Adjust parameters of Nelder-Mead algorithm to accounf
        for simplex size.
        The default is True.

    stopping_maxiter (int): Maximum number of algorithm iterations.
        The default is STOPPING_MAX_ITERATIONS.

    convergence_ftol_abs (float): maximal difference between
        function value evaluated on simplex points.

    convergence_xtol_abs (float): maximal distance between points in
        the simplex.

    batch_evaluator (string or callable): See :ref:`batch_evaluators` for
        details. Default "joblib".

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if x.ndim >= 1:
        x = x.ravel()  # check if the vector of initial values is one-dimensional

    j = len(x)  # size of the parameter vector

    if n_cores <= 1:
        p = 1  # if number of cores is nonpositive, set it to 1
    else:
        if n_cores >= j:  # number of parallelisation cannot be bigger than
            # the number of parameters minus 1
            p = int(j - 1)
        else:
            p = int(n_cores)

    # set parameters of Nelder-Mead algorithm
    # for a discussion about Nlder-Mead parameters see Gao F., Han L., Implementing the
    # Nelder-Mead siplex algorithm with adaptive parameters, Computational Optimization
    # and Applications, 2012
    alpha, gamma, beta, tau = _init_algo_params(adaptive, j)

    # construct initial simplex using one of feasible methods
    # see Wssing, Simon, Proper initialization is crucial for
    # the Nelder-Mead simplex search, Optimization Letters, 2019
    # for a discussion about the choice of initialization

    if not callable(init_simplex_method):
        s = globals()["_" + init_simplex_method](x)
    else:
        s = init_simplex_method(x)

    batch_evaluator = process_batch_evaluator(batch_evaluator)

    # calculate criterion values for the initial simplex
    f_s = np.array(batch_evaluator(func=criterion, arguments=s, n_cores=n_cores))[
        :, None
    ]

    # parallelized function
    def func_parallel(args):
        criterion, s_j, s_j_r, f_s_0, f_s_j, f_s_j_1, m = args  # read arguments

        f_s_j_r = criterion(
            s_j_r
        )  # calculate value of the criterion at the reflection point

        if f_s_j_r < f_s_0:  # if the reflection point is better than the best point
            s_j_e = m + gamma * (s_j_r - m)  # calculate expansion point
            f_s_j_e = criterion(
                s_j_e
            )  # calculate value of the criterion at the expansion point

            if f_s_j_e < f_s_0:  # if the expansion point is better than the best point
                return np.hstack(
                    [s_j_e, f_s_j_e, 0]
                )  # return the expansion point as a new point

            else:  # if the expansion point is worse than the best point
                return np.hstack(
                    [s_j_r, f_s_j_r, 0]
                )  # return the reflection point as a new point

        elif (
            f_s_j_r < f_s_j_1
        ):  # if reflection point is better than the next worst point
            return np.hstack(
                [s_j_r, f_s_j_r, 0]
            )  # return reflection point as a new point

        else:  # if the reflection point is worse than the next worst point
            if (
                f_s_j_r < f_s_j
            ):  # if value of the criterion at reflection point is better than
                # value of the criterion at initial point
                s_j_c = m + beta * (s_j_r - m)  # calculate outside contraction point
            else:
                s_j_c = m - beta * (s_j_r - m)  # calculate inside contraction point

            f_s_j_c = criterion(
                s_j_c
            )  # calculate a value of the criterion at contraction point

            if f_s_j_c < np.minimum(
                f_s_j, f_s_j_r
            ):  # if ta value of the criterion at contraction point is better
                # than original and refrelction point

                return np.hstack(
                    [s_j_c, f_s_j_c, 0]
                )  # return contraction point as as new point

            else:
                if f_s_j_r < f_s_j:
                    return np.hstack(
                        [s_j_r, f_s_j_r, 1]
                    )  # return reflection point as a new point

                else:  # if value of the criterion at contraction point is worse
                    # than the value uf the criterion at the reflection
                    # and the initial points
                    return np.hstack(
                        [s_j, f_s_j, 1]
                    )  # return the old point as a new point

    optimal = False  # optmisation condition, if True stop the algorithem
    iterations = 0  # number of criterion evaluations

    while not optimal:
        iterations += 1  # new iteration

        # sort points and arguments increasing
        row = np.argsort(f_s.ravel())
        s = np.take(s, row, axis=0)
        f_s = np.take(f_s, row, axis=0)

        # calculate centroid
        m = (s[:-p, :].sum(axis=0)) / (j - p + 1)

        # calculate reflaction points
        s_j_r = m + alpha * (m - s[-p:, :])

        # calculate new points of simplex
        s[-p:, :], f_s[-p:, :], shrink_count = np.split(
            np.vstack(
                batch_evaluator(
                    func=func_parallel,
                    arguments=tuple(
                        (
                            criterion,
                            s[j + 1 - p + i, :],
                            s_j_r[i, :],
                            f_s[0, :],
                            f_s[j + 1 - p + i, :],
                            f_s[j - p + i, :],
                            m,
                        )
                        for i in range(p)
                    ),
                    n_cores=p,
                )
            ),
            [-2, -1],
            axis=1,
        )

        # shrink simplex if there is no improvement in every process
        if shrink_count.sum() == p:
            s = (
                tau * s[0:1, :] + (1 - tau) * s
            )  # new simplex is a linear combination of the best point
            # and remaining points
            # evaluate function at new simplex
            f_s = np.array(
                batch_evaluator(
                    func=criterion,
                    arguments=s,
                    n_cores=n_cores,
                )
            )[:, None]

        # termination criteria
        if (
            np.max(np.abs(f_s[0, :] - f_s[1:, :])) <= convergence_ftol_abs
            and np.max(np.abs(s[0, :] - s[1:,])) <= convergence_xtol_abs
        ):
            optimal = True
            converge = True
            reason_to_stop = "Termination codition satisfied"
        elif (
            iterations >= stopping_maxiter
        ):  # if maximum amount of iteration is exceeded
            optimal = True
            converge = False
            reason_to_stop = "Maximum number of interation exceeded"

    # save results
    result = {
        "solution_x": s[np.nonzero(f_s == f_s.min())[0][0], :],
        "solution_criterion": f_s.min(),
        "n_iterations": iterations,
        "success": converge,
        "reached_convergence_criterion": reason_to_stop,
    }
    return result


# set parameters of Nelder-Mead algorithm
# for a discussion about Nlder-Mead parameters see Gao F., Han L., Implementing the
# Nelder-Mead siplex algorithm with adaptive parameters, Computational Optimization
# and Applications, 2012
def _init_algo_params(adaptive, j):
    if adaptive:
        # algorithem parameters alla Gao-Han (adaptive)
        return (
            1,
            1 + 2 / j,
            0.75 - 1 / (2 * j),
            1 - 1 / j,
        )
    else:
        # standard setting of Nelder-Mead
        return (
            1,
            2,
            0.5,
            0.5,
        )


# initial structure of the simplex
def _init_simplex(x):
    s = np.vstack(
        [
            x,
        ]
        * (len(x) + 1)
    ).astype(np.float64)

    return s


# initilize due to L. Pfeffer at Standford (Matlab fminsearch and SciPy default option)
def _pfeffer(x):
    s = _init_simplex(x)

    # method parameters
    c_p = 1.05

    # initial simplex
    np.fill_diagonal(s[1:, :], x * c_p * (x != 0) + 0.00025 * (x == 0))

    return s


# adopted from Nash (R default option)
# see Nash, J.C.: Compact numerical methods for computers: linear algebra and
# function minimisation, 2nd edn. Adam Hilger Ltd., Bristol (1990) for details
def _nash(x):
    s = _init_simplex(x)

    # method parameters
    c_n = 0.1

    # initial simplex
    np.fill_diagonal(s[1:, :], (x != 0) * (np.max(x) * c_n + x) + c_n * (x == 0))
    return s


# adopted from Gao F., Han L., Implementing the
# Nelder-Mead siplex algorithm with adaptive parameters, Computational Optimizatio
def _gao_han(x):
    s = _init_simplex(x)

    # method parameters
    c_h = np.minimum(np.maximum(np.max(x), 1), 10)
    j = len(x)

    # initial simplex
    s = (
        s
        + np.vstack(
            [
                np.array([[(1 - (j + 1) ** 0.5) / j]]) * np.ones([1, j]),
                np.eye(j),
            ]
        )
        * c_h
    )

    return s


# adopted by Varadhan and Borchers for the R package dfoptim
# see Varadhan, R., Borchers, H.W.: Dfoptim: derivative-free optimization (2016).
# https://CRAN.R-project. org/package=dfoptim. R package version 2016.7-1 for details
def _varadhan_borchers(x):
    s = _init_simplex(x)

    # method parameters
    j = len(x)
    c_s = np.maximum(1, ((x**2).sum()) ** 0.5)
    beta1 = c_s / (j * 2**0.5) * ((j + 1) ** 0.5 + j - 1)
    beta2 = c_s / (j * 2**0.5) * ((j + 1) ** 0.5 - 1)

    # initial simplex
    s[1:, :] = s[1:, :] + np.full([j, j], beta2) + np.eye(j) * (beta1 - beta2)

    return s
