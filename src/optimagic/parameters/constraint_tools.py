from optimagic.deprecations import replace_and_warn_about_deprecated_bounds
from optimagic.parameters.bounds import pre_process_bounds
from optimagic.parameters.conversion import get_converter


def count_free_params(
    params,
    constraints=None,
    bounds=None,
    # deprecated
    lower_bounds=None,
    upper_bounds=None,
):
    """Count the (free) parameters of an optimization problem.

    Args:
        params (pytree): The parameters.
        constraints (list): The constraints for the optimization problem. If constraints
            are provided, only the free parameters are counted.
        bounds: Lower and upper bounds on the parameters. The most general and preferred
            way to specify bounds is an `optimagic.Bounds` object that collects lower,
            upper, soft_lower and soft_upper bounds. The soft bounds are used for
            sampling based optimizers but are not enforced during optimization. Each
            bound type mirrors the structure of params. Check our how-to guide on bounds
            for examples. If params is a flat numpy array, you can also provide bounds
            via any format that is supported by scipy.optimize.minimize.

    Returns:
        int: Number of (free) parameters

    """
    bounds = replace_and_warn_about_deprecated_bounds(
        bounds=bounds,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    bounds = pre_process_bounds(bounds)

    _, internal_params = get_converter(
        params=params,
        constraints=constraints,
        bounds=bounds,
        func_eval=3,
        primary_key="value",
    )

    return int(internal_params.free_mask.sum())


def check_constraints(
    params,
    constraints,
    bounds=None,
    # deprecated
    lower_bounds=None,
    upper_bounds=None,
):
    """Raise an error if constraints are invalid or not satisfied in params.

    Args:
        params (pytree): The parameters.
        constraints (list): The constraints for the optimization problem.
        bounds: Lower and upper bounds on the parameters. The most general and preferred
            way to specify bounds is an `optimagic.Bounds` object that collects lower,
            upper, soft_lower and soft_upper bounds. The soft bounds are used for
            sampling based optimizers but are not enforced during optimization. Each
            bound type mirrors the structure of params. Check our how-to guide on bounds
            for examples. If params is a flat numpy array, you can also provide bounds
            via any format that is supported by scipy.optimize.minimize.

    Raises:
        InvalidParamsError: If constraints are valid but not satisfied.
        InvalidConstraintError: If constraints are invalid.

    """
    bounds = replace_and_warn_about_deprecated_bounds(
        bounds=bounds,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    bounds = pre_process_bounds(bounds)

    get_converter(
        params=params,
        constraints=constraints,
        bounds=bounds,
        func_eval=3,
        primary_key="value",
    )
