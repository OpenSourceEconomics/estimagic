"""
This module contains auxiliary functions for a one step differentiation.
"""


def central(func, f_x0, params_sr, var, h, *func_args, **func_kwargs):
    """
    This function calculates the central difference.
    Args:
        func (function): A function that maps params_sr into a float.
        f_x0 (float): The value of func for the observations.
        params_sr (Series): see :ref:`parmas_df`
        var (string): The central difference is calculated w.r.t. to the variable var
        h (float): The step size for the central difference.
        *func_args (list): Additional positional arguments for func.
        **func_kwargs (dict): Additional positional arguments for func.

    Returns:
        The central difference w.r.t. to variable var.
    """
    params_r = params_sr.copy()
    params_r[var] += h
    params_l = params_sr.copy()
    params_l[var] -= h
    central_diff = func(params_r, *func_args, **func_kwargs) - func(
        params_l, *func_args, **func_kwargs
    )
    return central_diff / 2.0


def forward(func, f_x0, params_sr, var, h, *func_args, **func_kwargs):
    """
    This function calculates the forward difference.
    Args:
        func (function): A function that maps params_sr into a float.
        f_x0 (float): The value of func for the observations.
        params_sr (Series): see :ref:`parmas_df`
        var (string): The central difference is calculated w.r.t. to the variable var
        h (float): The step size for the central difference.
        *func_args (list): Additional positional arguments for func.
        **func_kwargs (dict): Additional positional arguments for func

    Returns:
        The forward difference w.r.t. to variable var.
    """
    params = params_sr.copy()
    params[var] += h
    return func(params, *func_args, **func_kwargs) - f_x0


def backward(func, f_x0, params_sr, var, h, *func_args, **func_kwargs):
    """

    Args:
        func:
        f_x0:
        params_sr:
        var:
        h:
        *func_args:
        **func_kwargs:

    Returns:

    """
    params = params_sr.copy()
    params[var] -= h
    return f_x0 - func(params, *func_args, **func_kwargs)


def richardson(f, func, f_x0, params_sr, var, h, method, *func_args, **func_kwargs):
    pol = []
    for i in [1, 2, 4]:
        pol += [f(func, f_x0, params_sr, var, h * i, *func_args, **func_kwargs)]
    if method == "central":
        f_diff = (pol[2] / 4 - 10 * pol[1] + 64 * pol[0]) / 45
    else:
        f_diff = (32 * pol[0] - 12 * pol[1] + pol[0]) / 12
    return f_diff
