import numpy as np
from scipy import stats
from scipy.linalg import pinv
from scipy.ndimage import convolve1d


def richardson_extrapolation(sequence, steps, method="central", num_terms=None):
    """Apply Richardson extrapolation to sequence.

    Suppose you have a series expansion

        L = g(h) + a0*(h**p_0) + a1*(h**p_1) + a2*(h**p_2) + ... ,

    where p_i = order + exponentiation_step * i  and g(h) -> L as h -> 0, but g(0) != L.

    For ``method``='central', that is, for a sequence resulting from a central
    differences derivative approximation, we get ``order`` = 2 and
    ``exponentiation_step`` = 2, which would result in

        L = g(h) + a0*(h**2) + a1*(h**4) + a2*(h**6) + ...,

    where g(h) := [f(x + h) - f(x - h)] / 2h and f the function of interest. See
    function ``_get_order_and_exponentiation_step`` for more details.

    If we evaluate the right hand side for different stepsizes h we can fit a polynomial
    to that sequence of approximations and use the estimated intercept as a better
    approximation for L. Further, we can compute estimation errors of our approximation.


    Args:
        sequence (np.ndarray): The sequence of which we want to approximate the limit.
            Has dimension (k x n x m), where k denotes the number of sequence elements
            and an element ``sequence[l, :, :]`` denotes the (n x m) dimensional element

        steps (namedtuple): Namedtuple with the field names pos and neg. Each field
            contains a numpy array of shape (n_steps, len(x)) with the steps in
            the corresponding direction. The steps are always symmetric, in the sense
            that steps.neg[i, j] = - steps.pos[i, j] unless one of them is NaN.

        method (str): One of ["central", "forward", "backward"], default "central".

        num_terms (int): Number of terms needed to construct one estimate. Default is
            ``steps.shape[0] - 1``.

    Returns:
        limit (np.ndarray): The refined limit.
        error (np.ndarray): The error approximation of ``limit``.

    """
    seq_len = sequence.shape[0]
    steps = steps.pos
    n_steps = steps.shape[0]
    num_terms = n_steps if num_terms is None else num_terms

    assert (
        seq_len == n_steps
    ), "Length of ``steps`` must coincide with length of ``sequence``."
    assert num_terms > 0, "``num_terms`` must be greater than zero."
    assert (
        seq_len - 1 >= num_terms
    ), "``num_terms`` cannot be greater than ``seq_len`` - 1."

    step_ratio = _compute_step_ratio(steps)
    order, exponentiation_step = _get_order_and_exponentiation_step(method)

    richardson_coef = _richardson_coefficients(
        num_terms,
        step_ratio,
        exponentiation_step,
        order,
    )

    new_sequence = convolve1d(
        input=sequence, weights=richardson_coef[::-1], axis=0, origin=num_terms // 2
    )

    m = seq_len - num_terms
    mm = m + 1 if num_terms >= 2 else seq_len

    abserr = _estimate_error(new_sequence[:mm], sequence, richardson_coef)

    limit = new_sequence[:m]
    error = abserr[:m]

    return limit, error


def _richardson_coefficients(num_terms, step_ratio, exponentiation_step, order):
    """Compute Richardson coefficients.

    Let e := ``exponentiation_step``, r := ``step_ratio``, o := ``order`` and
    n := ``num_terms``. We build a matrix of the form

            [[1      1                  ...         1                ],
             [1    1/(s)**(2*o)         ...  1/(s)**(2*(o+n))        ],
        R =  [1    1/(s**2)**(2*o)      ...  1/(s**2)**(2*(o+n))     ],
             [...                       ...        ...               ],
             [1    1/(s**(n+1))**(2*o)  ...  1/(s**(n+1))**(2*(o+n)) ]]

    which is the weighting matrix in equation 24 in https://tinyurl.com/ybtfj4pm.
    We then return the first row of R^{-1} as the coefficients, as can be seen in
    equation 25 in https://tinyurl.com/ybtfj4pm.


    Args:
        num_terms (int): Number of terms needed to construct one estimate. Default is
            ``steps.shape[0] - 1``.

        step_ratio (float): Ratio between two consecutive steps. Order is chosen such
            that ``step_ratio`` >= 1.

        exponentiation_step (int): Step representing the growth of the exponent in
            the series expansions of the limit.
            For central difference derivative approximation ``exponentiation_step`` = 2.

        order (int): Initial order of the approximation error of sequence elements.
            For central difference derivative approximation ``order`` = 2.

    Returns:
        coef (np.ndarray): Richardson coefficients, array has length num_terms + 1.

    Example:
    >>> import numpy as np
    >>> num_terms = 2
    >>> step_ratio = 2.
    >>> exponentiation_step = 2
    >>> order = 2
    >>> _richardson_coefficients(num_terms, step_ratio, exponentiation_step, order)
    array([ 0.02222222, -0.44444444,  1.42222222])

    """
    rows, cols = np.ogrid[: num_terms + 1, :num_terms]

    coef_mat = np.ones((num_terms + 1, num_terms + 1))
    coef_mat[:, 1:] = (1.0 / step_ratio) ** (
        rows @ (exponentiation_step * cols + order)
    )

    coef = pinv(coef_mat)[0]
    return coef


def _estimate_error(new_seq, old_seq, richardson_coef):
    """Estimate error of multiple Richardson limit approximation.

    Args:
        new_seq (np.ndarray): Multiple estimates of the limit of ``old_seq``. The last
            two dimensions coincide with those of ``old_seq``. The first dimensions
            denotes the number of different estimates.

        old_seq (np.ndarray): The sequence of which we want to approximate the limit.
            Has dimension (k x n x m), where k denotes the number of sequence elements
            and an element ``sequence[l, :, :]`` denotes the (n x m) dimensional element

        richardson_coef (np.ndarray): Richardson coefficients. See function
            ``_richardson_coefficient`` for details.

    Returns:
        abserr (np.ndarray): The error estimate for each limit approximation in
            ``new_seq``.

    """
    eps = np.finfo(float).eps
    t_quantile = stats.t(df=1).ppf(0.975)  # 12.7062047361747 in numdifftools
    new_seq_len = new_seq.shape[0]

    unnormalized_covariance = np.sum(richardson_coef**2)
    fact = np.maximum(t_quantile * np.sqrt(unnormalized_covariance), eps * 10.0)

    if new_seq_len <= 1:
        delta = np.diff(old_seq, axis=0)
        tol = np.maximum(np.abs(old_seq[:-1]), np.abs(old_seq[1:])) * fact
        err = np.abs(delta)
        converged = err <= tol
        abserr = err[-new_seq_len:] + np.where(
            converged[-new_seq_len:],
            tol[-new_seq_len:] * 10,
            abs(new_seq - old_seq[-new_seq_len:]) * fact,
        )
    else:
        err = np.abs(np.diff(new_seq, axis=0)) * fact
        tol = np.maximum(np.abs(new_seq[1:]), np.abs(new_seq[:-1])) * eps * fact
        converged = err <= tol
        abserr = err + np.where(
            converged,
            tol * 10,
            abs(new_seq[:-1] - old_seq[-new_seq_len + 1 :]) * fact,
        )

    return abserr


def _get_order_and_exponentiation_step(method):
    """Return order and exponentiation step given ``method``.

    Given ``method`` we return the initial order of the approximation error of the
    sequence under consideration (order) as well as the step size representing the
    growth of the exponent in the series expansion of the limit (exponentiation_step).
    See function ``richardson_extrapolation`` for more details.

    For different methods, different values of order and exponentiation step apply.
    Consider the following examples, where we continue the notation from function
    ``richardson_extrapolation`` and use O() to denote the Big O Laundau symbol.

    Central Differences.
        Derivative approximation via central difference is given by
            g(h) := [f(x + h) - f(x - h)] / 2h = f'(x) + r(x, h),
        where r(x, h) denotes the remainder term.

        If we expand the remainder term r(x, h) we get
            r(x, h) = a0*(h**2) + a1*(h**4) + a2*(h**6) + ...
        with a0 = f''(x) / 2!, a1 = f'''(x) / 3! etc.

        Rearanging terms we can write L := f'(x) = g(h) - r(x, h) = g(h) + O(h**2) and
        we notice that order = 2 and exponentiation_step = 2.

    Forward Differences.
        Derivative approximation via forward difference is given by
            g(h) := [f(x + h) - f(x)] / h = f'(x) + r(x, h),
            where again r(x, h) denotes the remainder term.

        If we expand the remainder term r(x, h) we get
            r(x, h) = a0*(h**1) + a1*(h**2) + a2*(h**3) + ...
        with a0 = f''(x) / 2!, a1 = f'''(x) / 3! etc.

        Rearanging terms we can write L := f'(x) = g(h) - r(x, h) = g(h) + O(h) and
        we notice that order = 1 and exponentiation_step = 1.

    Backward Differences.
        Analogous to forward differences.


    Args:
        method (str): One of ["central", "forward", "backward"], default "central".

    Returns:
        order (int): Initial order of the approximation error of sequence elements.

        exponentiation_step (int): Step representing the growth of the exponent in the
            series expansions of the limit.

    Example:
    >>> _get_order_and_exponentiation_step('central')
    (2, 2)

    """
    lookup = {
        "central": (2, 2),
        "forward": (1, 1),
        "backward": (1, 1),
    }

    order, exponentiation_step = lookup[method]
    return order, exponentiation_step


def _compute_step_ratio(steps):
    """Compute the step ratio used in producing ``steps``.

    Args:
        steps (np.ndarray): Array of shape (n_steps, len(x)) with the steps in the
            corresponding direction.

    Returns:
        step_ratio (float): The step ratio used in producing ``steps``.

    Example:
    >>> import numpy as np
    >>> steps = np.array([[2., np.nan, 2], [4, 4, 4], [8, 8, np.nan]])
    >>> _compute_step_ratio(steps)
    2.0

    """
    ratios = steps[1:, :] / steps[:-1, :]
    ratios = ratios[np.isfinite(ratios)]

    step_ratio = ratios.flat[0]
    return step_ratio
