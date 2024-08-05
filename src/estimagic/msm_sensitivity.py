"""Implement local sensitivity measures for method of moments.

measures:
m1: Andrews, Gentzkow & Shapiro (2017, Quarterly Journal of Economics)

epsilon 2-6: Honore, Jorgensen & de Paula
(https://onlinelibrary.wiley.com/doi/full/10.1002/jae.2779)

"""

import numpy as np
import pandas as pd
from optimagic.exceptions import INVALID_SENSITIVITY_MSG
from optimagic.utilities import robust_inverse

from estimagic.msm_covs import cov_robust
from estimagic.shared_covs import process_pandas_arguments


def calculate_sensitivity_to_bias(jac, weights):
    """Calculate the sensitivity to bias.

    The sensitivity measure is calculated for each parameter wrt each moment.

    It answers the following question: How strongly would the parameter estimates be
        biased if the kth moment was misspecified, i.e not zero in expectation?

    Args:
        jac (np.ndarray or pandas.DataFrame): The jacobian of simulate_moments with
            respect to params, evaluated at the  point estimates.
        weights (np.ndarray or pandas.DataFrame): The weighting matrix used for
            msm estimation.

    Returns:
        np.ndarray or pd.DataFrame: Sensitivity measure with shape (n_params, n_moments)

    """
    _jac, _weights, names = process_pandas_arguments(jac=jac, weights=weights)
    gwg = _sandwich(_jac, _weights)
    gwg_inverse = robust_inverse(gwg, INVALID_SENSITIVITY_MSG)
    m1 = -gwg_inverse @ _jac.T @ _weights

    if names:
        m1 = pd.DataFrame(m1, index=names.get("params"), columns=names.get("moments"))

    return m1


def calculate_fundamental_sensitivity_to_noise(
    jac, weights, moments_cov, params_cov_opt
):
    """Calculate the fundamental sensitivity to noise.

    The sensitivity measure is calculated for each parameter wrt each moment.

    It answers the following question: How much precision would be lost if the kth
        moment was subject to a little additional noise if the optimal weighting matrix
        is used?

    Args:
        jac (np.ndarray or pandas.DataFrame): The jacobian of simulate_moments with
            respect to params, evaluated at the  point estimates.
        weights (np.ndarray or pandas.DataFrame): The weighting matrix used for
            msm estimation.
        moments_cov (numpy.ndarray or pandas.DataFrame): The covariance matrix of the
            empirical moments.
        params_cov_opt (numpy.ndarray or pandas.DataFrame): The covariance matrix of the
            parameter estimates. Note that this needs to be the parameter covariance
            matrix using the formula for asymptotically optimal MSM.

    Returns:
        np.ndarray or pd.DataFrame: Sensitivity measure with shape (n_params, n_moments)

    """
    _jac, _weights, _moments_cov, _params_cov_opt, names = process_pandas_arguments(
        jac=jac, weights=weights, moments_cov=moments_cov, params_cov_opt=params_cov_opt
    )

    m2 = []

    for k in range(len(_weights)):
        mask_matrix_o = np.zeros(shape=_weights.shape)
        mask_matrix_o[k, k] = 1

        meat = _sandwich_plus(_jac, _weights, mask_matrix_o)

        m2k = _params_cov_opt @ meat @ _params_cov_opt
        m2k = np.diagonal(m2k)

        m2.append(m2k)

    m2 = np.array(m2).T

    moments_variances = np.diagonal(_moments_cov)
    params_variances = np.diagonal(_params_cov_opt)

    e2 = m2 / params_variances.reshape(-1, 1)
    e2 = e2 * moments_variances

    if names:
        e2 = pd.DataFrame(e2, index=names.get("params"), columns=names.get("moments"))

    return e2


def calculate_actual_sensitivity_to_noise(
    sensitivity_to_bias, weights, moments_cov, params_cov
):
    """Calculate the actual sensitivity to noise.

    The sensitivity measure is calculated for each parameter wrt each moment.

    It answers the following question: How much precision would be lost if the kth
        moment was subjet to a little additional noise if "weights" is used as
        weighting matrix?

    Args:
        sensitivity_to_bias (np.ndarray or pandas.DataFrame): See
            ``calculate_sensitivity_to_bias`` for details.
        weights (np.ndarray or pandas.DataFrame): The weighting matrix used for
            msm estimation.
        moments_cov (numpy.ndarray or pandas.DataFrame): The covariance matrix of the
            empirical moments.
        params_cov (numpy.ndarray or pandas.DataFrame): The covariance matrix of the
            parameter estimates.

    Returns:
        np.ndarray or pd.DataFrame: Sensitivity measure with shape (n_params, n_moments)

    """
    if isinstance(sensitivity_to_bias, pd.DataFrame):
        sensitivity_to_bias = sensitivity_to_bias.to_numpy()

    _weights, _moments_cov, _params_cov, names = process_pandas_arguments(
        weights=weights, moments_cov=moments_cov, params_cov=params_cov
    )

    m3 = []

    for k in range(len(_weights)):
        mask_matrix_o = np.zeros(shape=_weights.shape)
        mask_matrix_o[k, k] = 1

        m3k = _sandwich(sensitivity_to_bias.T, mask_matrix_o)
        m3k = np.diagonal(m3k)

        m3.append(m3k)

    m3 = np.array(m3).T

    moments_variances = np.diagonal(_moments_cov)
    params_variances = np.diagonal(_params_cov)

    e3 = m3 / params_variances.reshape(-1, 1)
    e3 = e3 * moments_variances

    if names:
        e3 = pd.DataFrame(e3, index=names.get("params"), columns=names.get("moments"))

    return e3


def calculate_actual_sensitivity_to_removal(jac, weights, moments_cov, params_cov):
    """Calculate the actual sensitivity to removal.

    The sensitivity measure is calculated for each parameter wrt each moment.

    It answers the following question: How much precision would be lost if the kth
        moment was excluded from the estimation if "weights" is used as weighting
        matrix?

    Args:
        sensitivity_to_bias (np.ndarray or pandas.DataFrame): See
            ``calculate_sensitivity_to_bias`` for details.
        weights (np.ndarray or pandas.DataFrame): The weighting matrix used for
            msm estimation.
        moments_cov (numpy.ndarray or pandas.DataFrame): The covariance matrix of the
            empirical moments.
        params_cov (numpy.ndarray or pandas.DataFrame): The covariance matrix of the
            parameter estimates.

    Returns:
        np.ndarray or pd.DataFrame: Sensitivity measure with shape (n_params, n_moments)

    """
    m4 = []

    _jac, _weights, _moments_cov, _params_cov, names = process_pandas_arguments(
        jac=jac, weights=weights, moments_cov=moments_cov, params_cov=params_cov
    )

    for k in range(len(_weights)):
        weight_tilde_k = np.copy(_weights)
        weight_tilde_k[k, :] = 0
        weight_tilde_k[:, k] = 0

        sigma_tilde_k = cov_robust(_jac, weight_tilde_k, _moments_cov)

        m4k = sigma_tilde_k - _params_cov
        m4k = m4k.diagonal()

        m4.append(m4k)

    m4 = np.array(m4).T

    params_variances = np.diagonal(_params_cov)
    e4 = m4 / params_variances.reshape(-1, 1)

    if names:
        e4 = pd.DataFrame(e4, index=names.get("params"), columns=names.get("moments"))

    return e4


def calculate_fundamental_sensitivity_to_removal(jac, moments_cov, params_cov_opt):
    """Calculate the fundamental sensitivity to removal.

    The sensitivity measure is calculated for each parameter wrt each moment.

    It answers the following question: How much precision would be lost if the kth
        moment was excluded from the estimation with if the optimal weighting matrix is
        used?

    Args:
        jac (np.ndarray or pandas.DataFrame): The jacobian of simulate_moments with
            respect to params, evaluated at the  point estimates.
        weights (np.ndarray or pandas.DataFrame): The weighting matrix used for
            msm estimation.
        moments_cov (numpy.ndarray or pandas.DataFrame): The covariance matrix of the
            empirical moments.
        params_cov_opt (numpy.ndarray or pandas.DataFrame): The covariance matrix of the
            parameter estimates. Note that this needs to be the parameter covariance
            matrix using the formula for asymptotically optimal MSM.

    Returns:
        np.ndarray or pd.DataFrame: Sensitivity measure with shape (n_params, n_moments)

    """
    _jac, _moments_cov, _params_cov_opt, names = process_pandas_arguments(
        jac=jac,
        moments_cov=moments_cov,
        params_cov_opt=params_cov_opt,
    )
    m5 = []

    for k in range(len(_moments_cov)):
        g_k = np.copy(_jac)
        g_k = np.delete(g_k, k, axis=0)

        s_k = np.copy(_moments_cov)
        s_k = np.delete(s_k, k, axis=0)
        s_k = np.delete(s_k, k, axis=1)

        sigma_k = _sandwich(g_k, robust_inverse(s_k, INVALID_SENSITIVITY_MSG))
        sigma_k = robust_inverse(sigma_k, INVALID_SENSITIVITY_MSG)

        m5k = sigma_k - _params_cov_opt
        m5k = m5k.diagonal()

        m5.append(m5k)

    m5 = np.array(m5).T

    params_variances = np.diagonal(_params_cov_opt)
    e5 = m5 / params_variances.reshape(-1, 1)

    if names:
        e5 = pd.DataFrame(e5, index=names.get("params"), columns=names.get("moments"))

    return e5


def calculate_sensitivity_to_weighting(jac, weights, moments_cov, params_cov):
    """Calculate the sensitivity to weighting.

    The sensitivity measure is calculated for each parameter wrt each moment.

    It answers the following question: How would the precision change if the weight of
        the kth moment is increased a little?

    Args:
        sensitivity_to_bias (np.ndarray or pandas.DataFrame): See
            ``calculate_sensitivity_to_bias`` for details.
        weights (np.ndarray or pandas.DataFrame): The weighting matrix used for
            msm estimation.
        moments_cov (numpy.ndarray or pandas.DataFrame): The covariance matrix of the
            empirical moments.
        params_cov (numpy.ndarray or pandas.DataFrame): The covariance matrix of the
            parameter estimates.

    Returns:
        np.ndarray or pd.DataFrame: Sensitivity measure with shape (n_params, n_moments)

    """
    _jac, _weights, _moments_cov, _params_cov, names = process_pandas_arguments(
        jac=jac, weights=weights, moments_cov=moments_cov, params_cov=params_cov
    )
    gwg_inverse = _sandwich(_jac, _weights)
    gwg_inverse = robust_inverse(gwg_inverse, INVALID_SENSITIVITY_MSG)

    m6 = []

    for k in range(len(_weights)):
        mask_matrix_o = np.zeros(shape=_weights.shape)
        mask_matrix_o[k, k] = 1

        m6k_1 = gwg_inverse @ _sandwich(_jac, mask_matrix_o) @ _params_cov
        m6k_2 = (
            gwg_inverse
            @ _jac.T
            @ mask_matrix_o
            @ _moments_cov
            @ _weights
            @ _jac
            @ gwg_inverse
        )
        m6k_3 = (
            gwg_inverse
            @ _jac.T
            @ _weights
            @ _moments_cov
            @ mask_matrix_o
            @ _jac
            @ gwg_inverse
        )
        m6k_4 = _params_cov @ _sandwich(_jac, mask_matrix_o) @ gwg_inverse

        m6k = -m6k_1 + m6k_2 + m6k_3 - m6k_4
        m6k = m6k.diagonal()

        m6.append(m6k)

    m6 = np.array(m6).T

    weights_diagonal = np.diagonal(_weights)
    params_variances = np.diagonal(_params_cov)

    e6 = m6 / params_variances.reshape(-1, 1)
    e6 = e6 * weights_diagonal

    if names:
        e6 = pd.DataFrame(e6, index=names.get("params"), columns=names.get("moments"))

    return e6


def _sandwich(a, b):
    """Calculate the sandwich product of two matrices: a.T * b * a."""
    sandwich = a.T @ b @ a
    return sandwich


def _sandwich_plus(a, b, c):
    """Calculate the sandwich product of three matrices: a.T * b.T * c * b * a."""
    sandwich = a.T @ b.T @ c @ b @ a
    return sandwich
