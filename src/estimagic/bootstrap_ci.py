import numpy as np
from scipy.stats import norm

from estimagic.bootstrap_helpers import check_inputs


def calculate_ci(
    base_outcome,
    estimates,
    ci_method="percentile",
    ci_level=0.95,
):
    """Compute confidence interval of bootstrap estimates.

    Parts of the code of the subfunctions of this function are taken from
    Daniel Saxton's resample library, as found on
    https://github.com/dsaxton/resample/


    Args:
        base_outcome (list): List of flat base outcomes, i.e. the outcome
            statistic(s) evaluated on the original data set.
        estimates (np.ndarray): Array of estimates computed on the bootstrapped
            samples.
        ci_method (str): Method of choice for computing confidence intervals.
            The default is "percentile".
        ci_level (float): Confidence level for the calculation of confidence
            intervals. The default is 0.95.

    Returns:
        np.ndarray: 1d array of the lower confidence interval, where the k'th entry
            contains the lower confidence interval for the k'th parameter.
        np.ndarray: 1d array of the upper confidence interval, where the k'th entry
            contains the upper confidence interval for the k'th parameter.

    """
    check_inputs(ci_method=ci_method, ci_level=ci_level, skipdata=True)

    alpha = 1 - ci_level

    if ci_method == "percentile":
        cis = _ci_percentile(estimates, alpha)
    elif ci_method == "bc":
        cis = _ci_bc(estimates, base_outcome, alpha)
    elif ci_method == "t":
        cis = _ci_t(estimates, base_outcome, alpha)
    elif ci_method == "basic":
        cis = _ci_basic(estimates, base_outcome, alpha)
    elif ci_method == "normal":
        cis = _ci_normal(estimates, base_outcome, alpha)

    return cis[:, 0], cis[:, 1]


def _ci_percentile(estimates, alpha):
    """Compute percentile type confidence interval of bootstrap estimates.

    Args:
        estimates (np.ndarray): Array of estimates computed on the bootstrapped
            samples.
        alpha (float): Statistical significance level of choice.

    Returns:
        cis (np.ndarray): 2d array where k'th row contains the upper and lower CI
            for k'th parameter.

    """
    num_params = estimates.shape[1]
    cis = np.zeros((num_params, 2))

    for k in range(num_params):
        q = _eqf(estimates[:, k])
        cis[k, :] = q(alpha / 2), q(1 - alpha / 2)

    return cis


def _ci_bc(estimates, base_outcome, alpha):
    """Compute bc type confidence interval of bootstrap estimates.

    Args:
        estimates (np.ndarray): Array of estimates computed on the bootstrapped
            samples.
        base_outcome (list): List of flat base outcomes, i.e. the outcome
            statistics evaluated on the original data set.
        alpha (float): Statistical significance level of choice.

    Returns:
        cis (np.ndarray): 2d array where k'th row contains the upper and lower CI
            for k'th parameter.

    """
    num_params = estimates.shape[1]
    cis = np.zeros((num_params, 2))

    for k in range(num_params):
        q = _eqf(estimates[:, k])
        params = estimates[:, k]

        # Bias correction
        z_naught = norm.ppf(np.mean(params <= base_outcome[k]))
        z_low = norm.ppf(alpha)
        z_high = norm.ppf(1 - alpha)

        p1 = norm.cdf(z_naught + (z_naught + z_low))
        p2 = norm.cdf(z_naught + (z_naught + z_high))

        cis[k, :] = q(p1), q(p2)

    return cis


def _ci_t(estimates, base_outcome, alpha):
    """Compute studentized confidence interval of bootstrap estimates.

    Args:
        estimates (np.ndarray): Array of estimates computed on the bootstrapped
            samples.
        base_outcome (list): List of flat base outcomes, i.e. the outcome
            statistics evaluated on the original data set.
        alpha (float): Statistical significance level of choice.

    Returns:
        cis (np.ndarray): 2d array where k'th row contains the upper and lower CI
            for k'th parameter.

    """
    num_params = estimates.shape[1]
    cis = np.zeros((num_params, 2))

    for k in range(num_params):
        params = estimates[:, k]

        theta_std = np.std(params)

        tq = _eqf((params - base_outcome[k]) / theta_std)
        t1 = tq(1 - alpha / 2)
        t2 = tq(alpha / 2)

        cis[k, :] = base_outcome[k] - theta_std * t1, base_outcome[k] - theta_std * t2

    return cis


def _ci_normal(estimates, base_outcome, alpha):
    """Compute approximate normal confidence interval of bootstrap estimates.

    Args:
        estimates (np.ndarray): Array of estimates computed on the bootstrapped
            samples.
        base_outcome (list): List of flat base outcomes, i.e. the outcome
            statistics evaluated on the original data set.
        alpha (float): Statistical significance level of choice.

    Returns:
        cis (np.ndarray): 2d array where k'th row contains the upper and lower CI
            for k'th parameter.

    """
    num_params = estimates.shape[1]
    cis = np.zeros((num_params, 2))

    for k in range(num_params):
        params = estimates[:, k]
        theta_std = np.std(params)
        t = norm.ppf(alpha / 2)

        cis[k, :] = base_outcome[k] + theta_std * t, base_outcome[k] - theta_std * t

    return cis


def _ci_basic(estimates, base_outcome, alpha):
    """Compute basic bootstrap confidence interval of bootstrap estimates.

    Args:
        estimates (np.ndarray): Array of estimates computed on the bootstrapped
            samples.
        base_outcome (list): List of flat base outcomes, i.e. the outcome
            statistics evaluated on the original data set.
        alpha (float): Statistical significance level of choice.

    Returns:
        cis (np.ndarray): 2d array where k'th row contains the upper and lower CI
            for k'th parameter.

    """
    num_params = estimates.shape[1]
    cis = np.zeros((num_params, 2))

    for k in range(num_params):
        q = _eqf(estimates[:, k])

        cis[k, :] = (
            2 * base_outcome[k] - q(1 - alpha / 2),
            2 * base_outcome[k] - q(alpha / 2),
        )

    return cis


def _eqf(sample):
    """Return empirical quantile function of the given sample.

    Args:
        sample (np.ndarray): Sample to base quantile function on.

    Returns:
        f (callable): Quantile function for given sample.

    """

    def f(x):
        return np.quantile(sample, x)

    return f
