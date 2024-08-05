import numpy as np
import pandas as pd

# ======================================================================================
# Define criterion functions
# ======================================================================================


def trid_scalar_criterion(params):
    """Implement Trid function. Function description:
    https://www.sfu.ca/~ssurjano/trid.html.

    Args:
        params (pd.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        int: Trid function output.

    """
    x = _get_x(params)
    return ((x - 1) ** 2).sum() - (x[1:] * x[:-1]).sum()


def trid_gradient(params):
    """Calculate gradient of trid function.

    Args:
        params(pandas.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        np.ndarray: gradient of trid function.

    """
    x = _get_x(params)
    l1 = np.insert(x, 0, 0)
    l1 = np.delete(l1, [-1])
    l2 = np.append(x, 0)
    l2 = np.delete(l2, [0])
    return 2 * (x - 1) - l1 - l2


def trid_criterion_and_gradient(params):
    """Implement Trid function and calculate gradient.

    Args:
        params (pd.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        int: trid function output.
        np.ndarray: gradient of trid function.

    """
    val = trid_scalar_criterion(params)
    grad = trid_gradient(params)
    return val, grad


def trid_dict_criterion(params):
    """Implement trid function.

    Args:
        params(pandas.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        Dictionary with the following entries:
        "value" (a scalar float): trid function output.

    """
    out = {
        "value": trid_scalar_criterion(params),
    }
    return out


def rotated_hyper_ellipsoid_scalar_criterion(params):
    """Implement Rotated Hyper Ellipsoid function. Function description:
    https://www.sfu.ca/~ssurjano/rothyp.html.

    Args:
        params (pd.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        int: Rotated Hyper Ellipsoid function output.

    """
    return rotated_hyper_ellipsoid_contributions(params).sum()


def rotated_hyper_ellipsoid_gradient(params):
    """Calculate gradient of rotated_hyper_ellipsoid function.

    Args:
        params(pandas.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        np.ndarray: gradient of rotated hyper ellipsoid function.

    """
    x = _get_x(params)
    return np.arange(2 * len(x), 0, -2) * x


def rotated_hyper_ellipsoid_criterion_and_gradient(params):
    """Implement Rotated Hyper Ellipsoid function and calculate gradient.

    Args:
        params (pd.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        int: Rotated Hyper Ellipsoid function output.
        np.ndarray: gradient of rotated hyper ellipsoid function.

    """
    val = rotated_hyper_ellipsoid_scalar_criterion(params)
    grad = rotated_hyper_ellipsoid_gradient(params)
    return val, grad


def rotated_hyper_ellipsoid_contributions(params):
    """Compute contributions of Rotated Hyper Ellipsoid function.

    Args:
        params (pd.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        np.ndarray: array with contributions of function output as elements.

    """
    x = _get_x(params)
    dim = len(params)
    out = np.zeros(dim)
    for i in range(dim):
        out[i] = (x[: i + 1] ** 2).sum()
    return out


def rotated_hyper_ellipsoid_dict_criterion(params):
    """Implement Rotated Hyper Ellipsoid function and compute contributions and
    root_contributions.

    Args:
        params(pandas.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        Dictionary with the following entries:
        "value" (a scalar float): rotated hyper ellipsoid function output.
        "contributions" (np.ndarray): array with contributions of function output
        as elements.
        "root_contributions" (np.ndarray): array with root of contributions of
        function output as elements.

    """
    contribs = rotated_hyper_ellipsoid_contributions(params)
    out = _out_dict_from_contribs(contribs)
    return out


def rosenbrock_scalar_criterion(params):
    """Implement Rosenbrock function. Function description:
    https://www.sfu.ca/~ssurjano/rosen.html.

    Args:
        params (pd.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        int: Rosenbrock function output.

    """
    return rosenbrock_contributions(params).sum()


def rosenbrock_gradient(params):
    """Calculate gradient of rosenbrock function.

    Args:
        params(pandas.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        np.ndarray: gradient of rosenbrock function.

    """
    x = _get_x(params)
    l1 = np.delete(x, [-1])
    l1 = np.append(l1, 0)
    l2 = np.insert(x, 0, 0)
    l2 = np.delete(l2, [1])
    l3 = np.insert(x, 0, 0)
    l3 = np.delete(l3, [-1])
    l4 = np.delete(x, [0])
    l4 = np.append(l4, 0)
    l5 = np.full((len(x) - 1), 2)
    l5 = np.append(l5, 0)
    return 100 * (4 * (l1**3) + 2 * l2 - 2 * (l3**2) - 4 * (l4 * x)) + 2 * l1 - l5


def rosenbrock_criterion_and_gradient(params):
    """Implement rosenbrock function and calculate gradient.

    Args:
        params (pd.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        int: rosenbrock function output.
        np.ndarray: gradient of rosenbrock function.

    """
    return rosenbrock_scalar_criterion(params), rosenbrock_gradient(params)


def rosenbrock_contributions(params):
    """Compute contributions of rosenbrock function.

    Args:
        params (pd.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        np.ndarray: array with contributions of function output as elements.

    """
    x = _get_x(params)
    dim = len(params)
    out = np.zeros(dim)
    for i in range(dim - 1):
        out[i] = ((x[i + 1] - x[i] ** 2) ** 2) * 100 + ((x[i] - 1) ** 2)
    return out


def rosenbrock_dict_criterion(params):
    """Implement Rosenbrock function and compute contributions and root_contributions.

    Args:
        params(pandas.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        Dictionary with the following entries:
        "value" (a scalar float): Rosenbrock function value.
        "contributions" (np.ndarray): array with contributions of function output
        as elements.
        "root_contributions" (np.ndarray): array with root of contributions of
        function output as elements.

    """
    contribs = rosenbrock_contributions(params)
    out = _out_dict_from_contribs(contribs)
    return out


def sos_dict_criterion(params):
    """Calculate the sum of squares function and compute contributions and
    root_contributions.

    Args:
        params(pandas.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        Dictionary with the following entries:
        "value" (a scalar float): sum of squares function value.
        "contributions" (np.ndarray): array with contributions of function output
        as elements.
        "root_contributions" (np.ndarray): array with root of contributions of
        function output as elements.

    """
    root_contribs = _get_x(params)
    out = _out_dict_from_root_contribs(root_contribs)
    return out


def sos_dict_criterion_with_pd_objects(params):
    """Calculate the sum of squares function and compute contributions and
    root_contributions as pandas objects.

    Args:
        params(pandas.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        Dictionary with the following pandas object entries:
        "value" (a scalar float): sum of squares function value.
        "contributions" (np.ndarray): array with contributions of function output
        as elements.
        "root_contributions" (np.ndarray): array with root of contributions of
        function output as elements.

    """
    out = sos_dict_criterion(params)
    out["contributions"] = pd.Series(out["contributions"])
    out["root_contributions"] = pd.Series(out["root_contributions"])

    return out


def sos_scalar_criterion(params):
    """Calculate the sum of squares."""
    return (_get_x(params) ** 2).sum()


def sos_gradient(params):
    """Calculate the gradient of the sum of squares function."""
    return 2 * _get_x(params)


def sos_jacobian(params):
    """Calculate the Jacobian of the sum of squares function."""
    return np.diag(2 * _get_x(params))


def sos_ls_jacobian(params):
    return np.eye(len(params))


def sos_pandas_gradient(params):
    """Calculate the gradient of the sum of squares function."""
    return 2 * pd.Series(_get_x(params))


def sos_pandas_jacobian(params):
    """Calculate the Jacobian of the sum of squares function."""
    return pd.DataFrame(np.diag(2 * _get_x(params)))


def sos_criterion_and_gradient(params):
    """Calculate sum of squares criterion value and gradient."""
    x = _get_x(params)
    return (x**2).sum(), 2 * x


def sos_criterion_and_jacobian(params):
    """Calculate sum of squares criterion value and Jacobian."""
    x = _get_x(params)
    return {"contributions": x**2, "value": (x**2).sum()}, np.diag(2 * x)


sos_dict_derivative = {
    "value": sos_gradient,
    "contributions": sos_jacobian,
    "root_contributions": sos_ls_jacobian,
}


def _out_dict_from_root_contribs(root_contribs):
    contribs = root_contribs**2
    out = {
        "value": contribs.sum(),
        "contributions": contribs,
        "root_contributions": root_contribs,
    }
    return out


def _out_dict_from_contribs(contribs):
    out = {
        "value": contribs.sum(),
        "contributions": contribs,
        "root_contributions": np.sqrt(contribs),
    }
    return out


def _get_x(params):
    if isinstance(params, pd.DataFrame) and "value" in params:
        x = params["value"].to_numpy()
    else:
        x = params
    return x
