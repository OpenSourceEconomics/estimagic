import numdifftools as nd
import numpy as np
import pandas as pd

from estimagic.differentiation.first_order_auxiliary import backward
from estimagic.differentiation.first_order_auxiliary import central
from estimagic.differentiation.first_order_auxiliary import forward


def gradient(
    func,
    params_sr,
    method="central",
    extrapolation="richardson",
    func_args=None,
    func_kwargs=None,
):
    """
    Calculate the gradient of *func*.

    Args:
        func (function): A function that maps params_sr into a float.
        params_sr (Series): see :ref:`parmas_df`
        func_args (list): additional positional arguments for func.
        func_kwargs (dict): additional positional arguments for func.
        method (string): The method for the computation of the derivative. Default is
                         central as it gives the highest accuracy.
        extrapolation (string): This variable allows to specify the use of the
        richardson extrapolation.

    Returns:
        Series: The index is the index of params_sr.

    """
    if method not in ["central", "forward", "backward"]:
        raise ValueError("The given method is not supported.")
    # set default arguments
    func_args = [] if func_args is None else func_args
    func_kwargs = {} if func_kwargs is None else func_kwargs
    if extrapolation == "richardson":
        # For the richardson extrapolation we use, the numdifftools library.
        grad_np = nd.Gradient(func, method=method)(params_sr, *func_args, *func_kwargs)
        return pd.Series(data=grad_np, index=params_sr.index)
    else:
        grad = pd.Series(index=params_sr.index)
        f_x0 = func(params_sr, *func_args, **func_kwargs)
        if method == "forward":
            f = forward
        elif method == "backward":
            f = backward
        else:
            f = central
        for var in params_sr.index:
            h = (1 + abs(params_sr[var])) * np.sqrt(np.finfo(float).eps)
            grad[var] = f(func, f_x0, params_sr, var, h, *func_args, **func_kwargs) / h
        return grad
