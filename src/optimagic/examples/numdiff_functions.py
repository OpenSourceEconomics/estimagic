"""Functions with known gradients, jacobians or hessians.

All functions take a numpy array with parameters as their first argument.

Example inputs for the binary choice functions are in binary_choice_inputs.pickle. They
come from the statsmodels documentation:
https://tinyurl.com/y4x67vwl
We pickled them so we don't need statsmodels as a dependency.

"""

import numpy as np
from scipy.stats import norm

FLOAT_EPS = np.finfo(float).eps

# ======================================================================================
# Logit
# ======================================================================================


def logit_loglike(params, y, x):
    return logit_loglikeobs(params, y, x).sum()


def logit_loglikeobs(params, y, x):
    q = 2 * y - 1
    return np.log(1 / (1 + np.exp(-(q * np.dot(x, params)))))


def logit_loglike_gradient(params, y, x):
    c = 1 / (1 + np.exp(-(np.dot(x, params))))
    return np.dot(y - c, x)


def logit_loglikeobs_jacobian(params, y, x):
    c = 1 / (1 + np.exp(-(np.dot(x, params))))
    return (y - c).reshape(-1, 1) * x


def logit_loglike_hessian(params, y, x):  # noqa: ARG001
    c = 1 / (1 + np.exp(-(np.dot(x, params))))
    return -np.dot(c * (1 - c) * x.T, x)


# ======================================================================================
# Probit
# ======================================================================================


def probit_loglike(params, y, x):
    return probit_loglikeobs(params, y, x).sum()


def probit_loglikeobs(params, y, x):
    q = 2 * y - 1
    return np.log(np.clip(norm.cdf(q * np.dot(x, params)), FLOAT_EPS, 1))


def probit_loglike_gradient(params, y, x):
    xb = np.dot(x, params)
    q = 2 * y - 1
    c = q * norm.pdf(q * xb) / np.clip(norm.cdf(q * xb), FLOAT_EPS, 1 - FLOAT_EPS)
    return np.dot(c, x)


def probit_loglikeobs_jacobian(params, y, x):
    xb = np.dot(x, params)
    q = 2 * y - 1
    c = q * norm.pdf(q * xb) / np.clip(norm.cdf(q * xb), FLOAT_EPS, 1 - FLOAT_EPS)
    return c.reshape(-1, 1) * x


def probit_loglike_hessian(params, y, x):
    xb = np.dot(x, params)
    q = 2 * y - 1
    c = q * norm.pdf(q * xb) / norm.cdf(q * xb)
    return np.dot(-c * (c + xb) * x.T, x)
