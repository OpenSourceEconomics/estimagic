import pandas as pd
from optimagic.exceptions import INVALID_INFERENCE_MSG
from optimagic.utilities import robust_inverse

from estimagic.shared_covs import process_pandas_arguments


def cov_robust(jac, weights, moments_cov):
    """Calculate the cov of msm estimates with asymptotically non-efficient weights.

    Note that asymptotically non-efficient weights are typically preferrable because
    they lead to less finite sample bias.

    Args:
        jac (np.ndarray or pandas.DataFrame): Numpy array or DataFrame with the jacobian
            of simulate_moments with respect to params. The derivative needs to be taken
            at the estimated parameters. Has shape n_moments, n_params.
        weights (np.ndarray): The weighting matrix for msm estimation.
        moments_cov (np.ndarray): The covariance matrix of the empirical moments.

    Returns:
        numpy.ndarray: numpy array with covariance matrix.

    """
    _jac, _weights, _moments_cov, names = process_pandas_arguments(
        jac=jac, weights=weights, moments_cov=moments_cov
    )

    bread = robust_inverse(
        _jac.T @ _weights @ _jac,
        msg=INVALID_INFERENCE_MSG,
    )

    butter = _jac.T @ _weights @ _moments_cov @ _weights @ _jac

    cov = bread @ butter @ bread

    if names:
        cov = pd.DataFrame(cov, columns=names.get("params"), index=names.get("params"))

    return cov


def cov_optimal(jac, weights):
    """Calculate the cov of msm estimates with asymptotically efficient weights.

    Note that asymptotically efficient weights have substantial finite sample
    bias and are typically not a good choice.

    Args:
        jac (np.ndarray or pandas.DataFrame): Numpy array or DataFrame with the jacobian
            of simulate_moments with respect to params. The derivative needs to be taken
            at the estimated parameters. Has shape n_moments, n_params.
        weights (np.ndarray): The weighting matrix for msm estimation.
        moments_cov (np.ndarray): The covariance matrix of the empirical moments.

    Returns:
        numpy.ndarray: numpy array with covariance matrix.

    """
    _jac, _weights, names = process_pandas_arguments(jac=jac, weights=weights)

    cov = robust_inverse(_jac.T @ _weights @ _jac, msg=INVALID_INFERENCE_MSG)

    if names:
        cov = pd.DataFrame(cov, columns=names.get("params"), index=names.get("params"))

    return cov
