import warnings
from dataclasses import replace

from optimagic.parameters.bounds import Bounds


def throw_criterion_future_warning():
    msg = (
        "To align optimagic with scipy.optimize, the `criterion` argument has been "
        "renamed to `fun`. Please use `fun` instead of `criterion`. Using `criterion` "
        " will become an error in optimagic version 0.6.0 and later."
    )
    warnings.warn(msg, FutureWarning)


def throw_criterion_kwargs_future_warning():
    msg = (
        "To align optimagic with scipy.optimize, the `criterion_kwargs` argument has "
        "been renamed to `fun_kwargs`. Please use `fun_kwargs` instead of "
        "`criterion_kwargs`. Using `criterion_kwargs` will become an error in "
        "optimagic version 0.6.0 and later."
    )
    warnings.warn(msg, FutureWarning)


def throw_derivative_future_warning():
    msg = (
        "To align optimagic with scipy.optimize, the `derivative` argument has been "
        "renamed to `jac`. Please use `jac` instead of `derivative`. Using `derivative`"
        " will become an error in optimagic version 0.6.0 and later."
    )
    warnings.warn(msg, FutureWarning)


def throw_derivative_kwargs_future_warning():
    msg = (
        "To align optimagic with scipy.optimize, the `derivative_kwargs` argument has "
        "been renamed to `jac_kwargs`. Please use `jac_kwargs` instead of "
        "`derivative_kwargs`. Using `derivative_kwargs` will become an error in "
        "optimagic version 0.6.0 and later."
    )
    warnings.warn(msg, FutureWarning)


def throw_criterion_and_derivative_future_warning():
    msg = (
        "To align optimagic with scipy.optimize, the `criterion_and_derivative` "
        "argument has been renamed to `fun_and_jac`. Please use `fun_and_jac` "
        "instead of `criterion_and_derivative`. Using `criterion_and_derivative` "
        "will become an error in optimagic version 0.6.0 and later."
    )
    warnings.warn(msg, FutureWarning)


def throw_criterion_and_derivative_kwargs_future_warning():
    msg = (
        "To align optimagic with scipy.optimize, the `criterion_and_derivative_kwargs` "
        "argument has been renamed to `fun_and_jac_kwargs`. Please use "
        "`fun_and_jac_kwargs` instead of `criterion_and_derivative_kwargs`. Using "
        "`criterion_and_derivative_kwargs` will become an error in optimagic version "
        "0.6.0 and later."
    )
    warnings.warn(msg, FutureWarning)


def throw_scaling_options_future_warning():
    msg = (
        "Specifying scaling options via the argument `scaling_options` is deprecated "
        "and will be removed in optimagic version 0.6.0 and later. You can pass these "
        "options directly to the `scaling` argument instead."
    )
    warnings.warn(msg, FutureWarning)


def throw_multistart_options_future_warning():
    msg = (
        "Specifying multistart options via the argument `multistart_options` is "
        "deprecated and will be removed in optimagic version 0.6.0 and later. You can "
        "pass these options directly to the `multistart` argument instead."
    )
    warnings.warn(msg, FutureWarning)


def replace_and_warn_about_deprecated_algo_options(algo_options):
    if not isinstance(algo_options, dict):
        return algo_options

    algo_options = {k.replace(".", "_"): v for k, v in algo_options.items()}

    replacements = {
        "stopping_max_criterion_evaluations": "stopping_maxfun",
        "stopping_max_iterations": "stopping_maxiter",
        "convergence_absolute_criterion_tolerance": "convergence_ftol_abs",
        "convergence_relative_criterion_tolerance": "convergence_ftol_rel",
        "convergence_scaled_criterion_tolerance": "convergence_ftol_scaled",
        "convergence_absolute_params_tolerance": "convergence_xtol_abs",
        "convergence_relative_params_tolerance": "convergence_xtol_rel",
        "convergence_absolute_gradient_tolerance": "convergence_gtol_abs",
        "convergence_relative_gradient_tolerance": "convergence_gtol_rel",
        "convergence_scaled_gradient_tolerance": "convergence_gtol_scaled",
    }

    present = sorted(set(algo_options) & set(replacements))
    if present:
        msg = (
            "The following keys in `algo_options` are deprecated and will be removed "
            "in optimagic version 0.6.0 and later. Please replace them as follows:\n"
        )
        for k in present:
            msg += f"  {k} -> {replacements[k]}\n"

        warnings.warn(msg, FutureWarning)

    out = {k: v for k, v in algo_options.items() if k not in present}
    for k in present:
        out[replacements[k]] = algo_options[k]

    return out


def replace_and_warn_about_deprecated_bounds(
    lower_bounds,
    upper_bounds,
    bounds,
    soft_lower_bounds=None,
    soft_upper_bounds=None,
):
    old_bounds = {
        "lower": lower_bounds,
        "upper": upper_bounds,
        "soft_lower": soft_lower_bounds,
        "soft_upper": soft_upper_bounds,
    }

    old_present = [k for k, v in old_bounds.items() if v is not None]

    if old_present:
        substring = ", ".join(f"{b}_bound" for b in old_present)
        substring = substring.replace(", ", ", and ", -1)
        msg = (
            f"Specifying bounds via the arguments {substring} is "
            "deprecated and will be removed in optimagic version 0.6.0 and later. "
            "Please use the `bounds` argument instead."
        )
        warnings.warn(msg, FutureWarning)

    if bounds is None and old_present:
        bounds = Bounds(**old_bounds)

    return bounds


def replace_and_warn_about_deprecated_multistart_options(options):
    """Replace deprecated multistart options and warn about them.

    Args:
        options (MultistartOptions): The multistart options to replace.

    Returns:
        MultistartOptions: The replaced multistart options.

    """
    replacements = {}

    if options.share_optimization is not None:
        msg = (
            "The share_optimization option is deprecated and will be removed in "
            "version 0.6.0. Use stopping_maxopt instead to specify the number of "
            "optimizations directly."
        )
        warnings.warn(msg, FutureWarning)

    if options.convergence_relative_params_tolerance is not None:
        msg = (
            "The convergence_relative_params_tolerance option is deprecated and will "
            "be removed in version 0.6.0. Use convergence_xtol_rel instead."
        )
        warnings.warn(msg, FutureWarning)
        if options.convergence_xtol_rel is None:
            replacements["convergence_xtol_rel"] = (
                options.convergence_relative_params_tolerance
            )

    if options.optimization_error_handling is not None:
        msg = (
            "The optimization_error_handling option is deprecated and will be removed "
            "in version 0.6.0. Setting this attribute also sets the error handling "
            "for exploration. Use the new error_handling option to set the error "
            "handling for both optimization and exploration."
        )
        warnings.warn(msg, FutureWarning)
        if options.error_handling is None:
            replacements["error_handling"] = options.optimization_error_handling

    if options.exploration_error_handling is not None:
        msg = (
            "The exploration_error_handling option is deprecated and will be "
            "removed in version 0.6.0. Setting this attribute also sets the error "
            "handling for exploration. Use the new error_handling option to set the "
            "error handling for both optimization and exploration."
        )
        warnings.warn(msg, FutureWarning)
        if options.error_handling is None:
            replacements["error_handling"] = options.exploration_error_handling

    return replace(options, **replacements)
