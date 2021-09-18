import numpy as np
import pandas as pd

# ======================================================================================
# Define criterion functions
# ======================================================================================


def trid_scalar_criterion(params):
    """Implement Trid function.
    Function description: https://www.sfu.ca/~ssurjano/trid.html

    Args:
        params (pd.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        int: Trid function output.
    """
    x = params["value"].to_numpy()
    return ((params["value"] - 1) ** 2).sum() - (params["value"][1:] * x[:-1]).sum()


def trid_gradient(params):
    """Calculate gradient of trid function.

    Args:
        params(pandas.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        np.ndarray: gradient of trid function.
    """
    x = params["value"].to_numpy()
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
    """Implement Rotated Hyper Ellipsoid function.
    Function description: https://www.sfu.ca/~ssurjano/rothyp.html.

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
    x = params["value"].to_numpy()
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
    x = params["value"].to_numpy()
    dim = len(params)
    out = np.zeros(dim)
    for i in range(dim):
        out[i] = (x[: i + 1] ** 2).sum()
    return out


def rotated_hyper_ellipsoid_dict_criterion(params):
    """Implement Rotated Hyper Ellipsoid function and compute
        contributions and root_contributions.

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
    """Implement Rosenbrock function.
    Function description: https://www.sfu.ca/~ssurjano/rosen.html

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
    x = params["value"].to_numpy()
    l1 = np.delete(x, [-1])
    l1 = np.append(l1, 0)
    l2 = np.insert(x, 0, 0)
    l2 = np.delete(l2, [1])
    l3 = np.insert(x, 0, 0)
    l3 = np.delete(l3, [-1])
    l4 = np.delete(x, [0])
    l4 = np.append(l4, 0)
    l5 = np.full((len(params["value"]) - 1), 2)
    l5 = np.append(l5, 0)
    return 100 * (4 * (l1 ** 3) + 2 * l2 - 2 * (l3 ** 2) - 4 * (l4 * x)) + 2 * l1 - l5


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
    x = params["value"].to_numpy()
    dim = len(params)
    out = np.zeros(dim)
    for i in range(dim - 1):
        out[i] = ((x[i + 1] - x[i] ** 2) ** 2) * 100 + ((x[i] - 1) ** 2)
    return out


def rosenbrock_dict_criterion(params):
    """Implement Rosenbrock function and compute
        contributions and root_contributions.

    Args:
        params(pandas.DataFrame): Must have the column "value" containing
        input values for parameters. Accepts arbitrary numbers of input values.

    Returns:
        Dictionary with the following entries:
        "value" (a scalar float): Rosenbrock function output.
        "contributions" (np.ndarray): array with contributions of function output
        as elements.
        "root_contributions" (np.ndarray): array with root of contributions of
        function output as elements.
    """
    contribs = rosenbrock_contributions(params)
    out = _out_dict_from_contribs(contribs)
    return out


def sos_dict_criterion(params):
    root_contribs = params["value"].to_numpy()
    out = _out_dict_from_root_contribs(root_contribs)
    return out


def sos_dict_criterion_with_pd_objects(params):
    out = sos_dict_criterion(params)
    out["contributions"] = pd.Series(out["contributions"])
    out["root_contributions"] = pd.Series(out["root_contributions"])

    return out


def sos_scalar_criterion(params):
    return (params["value"].to_numpy() ** 2).sum()


def sos_gradient(params):
    return 2 * params["value"].to_numpy()


def sos_jacobian(params):
    return np.diag(2 * params["value"])


def sos_ls_jacobian(params):
    return np.eye(len(params))


def sos_pandas_gradient(params):
    return 2 * params["value"]


def sos_pandas_jacobian(params):
    return pd.DataFrame(np.diag(2 * params["value"]))


def sos_criterion_and_gradient(params):
    x = params["value"].to_numpy()
    return (x ** 2).sum(), 2 * x


def sos_criterion_and_jacobian(params):
    x = params["value"].to_numpy()
    return {"contributions": x ** 2, "value": (x ** 2).sum()}, np.diag(2 * x)


def sos_dict_derivative(params):
    x = params["value"].to_numpy()

    out = {
        "value": 2 * x,
        "contributions": np.diag(2 * x),
        "root_contributions": np.eye(len(x)),
    }
    return out


def sos_dict_derivative_with_pd_objects(params):
    dict_np = sos_dict_derivative(params)
    out = {
        "value": pd.Series(dict_np["value"]),
        "contributions": pd.DataFrame(dict_np["contributions"]),
        "root_contributions": pd.DataFrame(dict_np["root_contributions"]),
    }
    return out


def sos_double_dict_criterion_and_derivative_with_pd_objects(params):
    val = sos_dict_criterion_with_pd_objects(params)
    deriv = sos_dict_derivative_with_pd_objects(params)
    return val, deriv


def _out_dict_from_root_contribs(root_contribs):
    contribs = root_contribs ** 2
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
