"""Implement algorithms by the (Numerical Algorithms Group)[https://www.nag.com/].

The following arguments are not supported as ``algo_options``:

- ``scaling_within_bounds``
- ``init.run_in_parallel``
- ``do_logging``, ``print_progress`` and all their advanced options.

"""

import warnings

import numpy as np

from optimagic.config import IS_DFOLS_INSTALLED, IS_PYBOBYQA_INSTALLED
from optimagic.decorators import mark_minimizer
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import STOPPING_MAXFUN
from optimagic.utilities import calculate_trustregion_initial_radius

if IS_PYBOBYQA_INSTALLED:
    import pybobyqa

if IS_DFOLS_INSTALLED:
    import dfols


CONVERGENCE_MINIMAL_TRUSTREGION_RADIUS_TOLERANCE = 1e-8
"""float: Stop when the lower trust region radius falls below this value."""

CONVERGENCE_SLOW_PROGRESS = {
    "threshold_to_characterize_as_slow": 1e-8,
    "max_insufficient_improvements": None,
    "comparison_period": 5,
}
"""dict: Specification of when to terminate or reset the optimization because of only
    slow improvements. This is similar to an absolute criterion tolerance only that
    instead of a single improvement the average over several iterations must be small.

    Possible entries are:
        threshold_to_characterize_as_slow (float): Threshold whether an improvement
            is insufficient. Note: the improvement is divided by the
            ``comparison_period``.
            So this is the required average improvement per iteration over the
            comparison period.
        max_insufficient_improvements (int): Number of consecutive
            insufficient improvements before termination (or reset). Default is
            ``20 * len(x)``.
        comparison_period (int):
            How many iterations to go back to calculate the improvement.
            For example 5 would mean that each criterion evaluation is compared to the
            criterion value from 5 iterations before.

"""

THRESHOLD_FOR_SAFETY_STEP = 0.5
r"""float: Threshold for when to call the safety step (:math:`\gamma_s`).

    :math:`\text{proposed step} \leq \text{threshold_for_safety_step} \cdot
    \text{current_lower_trustregion_radius}`.

"""

CONVERGENCE_NOISE_CORRECTED_FTOL = 1.0
"""float: Stop when the evaluations on the set of interpolation points all fall within
    this factor of the noise level. The default is 1, i.e. when all evaluations are
    within the noise level. If you want to not use this criterion but still flag your
    criterion function as noisy, set this tolerance to 0.0.

    .. warning::
        Very small values, as in most other tolerances don't make sense here.

"""


TRUSTREGION_THRESHOLD_SUCCESSFUL = 0.1
"""float: Share of the predicted improvement that has to be achieved for a trust
    region iteration to count as successful.

"""

TRUSTREGION_THRESHOLD_VERY_SUCCESSFUL = 0.7
"""float: Share of predicted improvement that has to be achieved for a trust region
    iteration to count as very successful.``criterion_noisy``

"""

TRUSTREGION_SHRINKING_FACTOR_NOT_SUCCESSFUL = None
"""float: Ratio by which to shrink the upper trust region radius when realized
    improvement does not match the ``threshold_successful``. The default is 0.98
    if the criterion is noisy and 0.5 else.

"""

TRUSTREGION_EXPANSION_FACTOR_SUCCESSFUL = 2.0
r"""float: Ratio by which to expand the upper trust region radius :math:`\Delta_k`
    in very successful iterations (:math:`\gamma_{inc}` in the notation of the paper).

"""

TRUSTREGION_EXPANSION_FACTOR_VERY_SUCCESSFUL = 4.0
r"""float: Ratio of the proposed step ($\|s_k\|$) by which to expand the upper trust
    region radius (:math:`\Delta_k`) in very successful iterations
    (:math:`\overline{\gamma}_{inc}` in the notation of the paper).

"""

TRUSTREGION_SHRINKING_FACTOR_LOWER_RADIUS = None
r"""float: Ratio by which to shrink the lower trust region radius (:math:`\rho_k`)
    (:math:`\alpha_1` in the notation of the paper). Default is 0.9 if
    the criterion is noisy and 0.1 else.

"""

TRUSTREGION_SHRINKING_FACTOR_UPPER_RADIUS = None
r"""float: Ratio of the current lower trust region (:math:`\rho_k`) by which to shrink
    the upper trust region radius (:math:`\Delta_k`) when the lower one is shrunk
    (:math:`\alpha_2` in the notation of the paper). Default is 0.95 if the
    criterion is noisy and 0.5 else."""


INITIAL_DIRECTIONS = "coordinate"
"""string: How to draw the initial directions. Possible values are "coordinate" for
    coordinate directions (the default) or "random".

"""

RANDOM_DIRECTIONS_ORTHOGONAL = True
"""bool: Whether to make randomly drawn initial directions orthogonal."""


INTERPOLATION_ROUNDING_ERROR = 0.1
r"""float: Internally, all the NAG algorithms store interpolation points with respect
    to a base point :math:`x_b`; that is, we store :math:`\{y_t-x_b\}`,
    which reduces the risk of roundoff errors. We shift :math:`x_b` to :math:`x_k` when
    :math:`\text{proposed step} \leq \text{interpolation_rounding_error} \cdot
    \|x_k-x_b\|`.

"""

CLIP_CRITERION_IF_OVERFLOWING = True
"""bool: Whether to clip the criterion to avoid ``OverflowError``."""


TRUSTREGION_PRECONDITION_INTERPOLATION = True
"""bool: whether to scale the interpolation linear system to improve conditioning."""


RESET_OPTIONS = {
    "use_resets": None,
    "minimal_trustregion_radius_tolerance_scaling_at_reset": 1.0,
    "reset_type": "soft",
    "move_center_at_soft_reset": True,
    "reuse_criterion_value_at_hard_reset": True,
    "max_iterations_without_new_best_after_soft_reset": None,
    "auto_detect": True,
    "auto_detect_history": 30,
    "auto_detect_min_jacobian_increase": 0.015,
    "auto_detect_min_correlations": 0.1,
    "points_to_replace_at_soft_reset": 3,
    "max_consecutive_unsuccessful_resets": 10,
    # just bobyqa
    "max_unsuccessful_resets": None,
    "trust_region_scaling_at_unsuccessful_reset": None,
    # just dfols
    "max_interpolation_points": None,
    "n_extra_interpolation_points_per_soft_reset": 0,
    "n_extra_interpolation_points_per_hard_reset": 0,
    "n_additional_extra_points_to_replace_per_reset": 0,
}
r"""dict: Options for reseting the optimization.

    Possible entries are:

        use_resets (bool): Whether to do resets when the lower trust
            region radius (:math:`\rho_k`) reaches the stopping criterion
            (:math:`\rho_{end}`), or (optionally) when all interpolation points are
            within noise level. Default is ``True`` if the criterion is noisy.
        minimal_trustregion_radius_tolerance_scaling_at_reset (float): Factor with
            which the trust region stopping criterion is multiplied at each reset.

        reset_type (str): Whether to use "soft" or "hard" resets. Default is "soft".

        move_center_at_soft_reset (bool): Whether to move the trust region center
            ($x_k$) to the best new point evaluated in stead of keeping it constant.
        points_to_replace_at_soft_reset (int): Number of interpolation points to move
            at each soft reset.
        reuse_criterion_value_at_hard_reset (bool): Whether or not to recycle the
            criterion value at the best iterate found when performing a hard reset.
            This saves one criterion evaluation.
        max_iterations_without_new_best_after_soft_reset (int):
            The maximum number of successful steps in a given run where the new
            criterion value is worse than the best value found in previous runs before
            terminating. Default is ``max_criterion_evaluations``.
        auto_detect (bool): Whether or not to
            automatically determine when to reset. This is an additional condition
            and resets can still be triggered by small upper trust region radius, etc.
            There are two criteria used: upper trust region radius shrinkage
            (no increases over the history, more decreases than no changes) and
            changes in the model Jacobian (consistently increasing trend as measured
            by slope and correlation coefficient of the line of best fit).
        auto_detect_history (int):
            How many iterations of model changes and trust region radii to store.
        auto_detect_min_jacobian_increase (float):
            Minimum rate of increase of the Jacobian over past iterations to cause a
            reset.
        auto_detect_min_correlations (float):
            Minimum correlation of the Jacobian data set required to cause a reset.
        max_consecutive_unsuccessful_resets (int): maximum number of consecutive
            unsuccessful resets allowed (i.e. resets which did not outperform the
            best known value from earlier runs).

    Only used when using nag_bobyqa:

        max_unsuccessful_resets (int):
            number of total unsuccessful resets allowed.
            Default is 20 if ``seek_global_optimum`` and else unrestricted.
        trust_region_scaling_at_unsuccessful_reset (float): Factor by which to
            expand the initial lower trust region radius (:math:`\rho_{beg}`) after
            unsuccessful resets. Default is 1.1 if ``seek_global_optimum`` else 1.

    Only used when using nag_dfols:

        max_interpolation_points (int): Maximum allowed value of the number of
            interpolation points. This is useful if the number of interpolation points
            increases with each reset, e.g. when
            ``n_extra_interpolation_points_per_soft_reset > 0``. The default is
            ``n_interpolation_points``.
        n_extra_interpolation_points_per_soft_reset (int): Number of points to add to
            the interpolation set with each soft reset.
        n_extra_interpolation_points_per_hard_reset (int): Number of points to add to
            the interpolation set with each hard reset.
        n_additional_extra_points_to_replace_per_reset (int): This parameter modifies
            ``n_extra_points_to_replace_successful``. With each reset
            ``n_extra_points_to_replace_successful`` is increased by this number.

"""


TRUSTREGION_FAST_START_OPTIONS = {
    "min_inital_points": None,
    "method": "auto",
    "scale_of_trustregion_step_perturbation": None,
    "scale_of_jacobian_components_perturbation": 1e-2,
    # the following will be growing.full_rank.min_sing_val
    # but it not supported yet by DF-OLS.
    "floor_of_jacobian_singular_values": 1,
    "jacobian_max_condition_number": 1e8,
    "geometry_improving_steps": False,
    "safety_steps": True,
    "shrink_upper_radius_in_safety_steps": False,
    "full_geometry_improving_step": False,
    "reset_trustregion_radius_after_fast_start": False,
    "reset_min_trustregion_radius_after_fast_start": False,
    "shrinking_factor_not_successful": None,
    "n_extra_search_directions_per_iteration": 0,
}
r"""dict: Options to start the optimization while building the full trust region model.

    To activate this, set the number of interpolation points at which to evaluate the
    criterion before doing the first step, `min_initial_points`, to something smaller
    than the number of parameters.

    The following options can be specified:

        min_initial_points (int): Number of initial interpolation
            points in addition to the start point. This should only be changed to
            a value less than ``len(x)``, and only if the default setup cost
            of ``len(x) + 1`` evaluations of the criterion is impractical.
            If this is set to be less than the default, the input value of
            ``n_interpolation_points`` should be set to ``len(x)``.
            If the default is used, all the other parameters have no effect.
            Default is ``n_interpolation_points - 1``.
            If the default setup costs of the evaluations are very large, DF-OLS
            can start with less than ``len(x)`` interpolation points and add points
            to the trust region model with every iteration.
        method ("jacobian", "trustregion" or "auto"):
            When there are less interpolation points than ``len(x)`` the model is
            underdetermined. This can be fixed in two ways:
            If "jacobian", the interpolated Jacobian is perturbed to have full
            rank, allowing the trust region step to include components in the full
            search space. This is the default if
            ``len(x) \geq number of root contributions``.
            If "trustregion_step", the trust region step is perturbed by an
            orthogonal direction not yet searched. It is the default if
            ``len(x) < number of root contributions``.
        scale_of_trustregion_step_perturbation (float):
            When adding new search directions, the length of the step is the trust
            region radius multiplied by this value. The default is 0.1 if
            ``method == "trustregion"`` else 1.
        scale_of_jacobian_components_perturbation (float): Magnitude of extra
            components added to the Jacobian. Default is 1e-2.
        floor_of_jacobian_singular_values (float): Floor singular
            values of the Jacobian at this factor of the last non zero value.
            As of version 1.2.1 this option is not yet supported by DF-OLS!
        scale_of_jacobian_singular_value_floor (float):
            Floor singular values of the Jacobian at this factor of the last nonzero
            value.
        jacobian_max_condition_number (float): Cap on the condition number
            of Jacobian after applying floors to singular values
            (effectively another floor on the smallest singular value, since the
            largest singular value is fixed).
        geometry_improving_steps (bool): Whether to do geometry-improving steps in the
            trust region algorithm, as per the usual algorithm during the fast start.
        safety_steps (bool):
            Whether to perform safety steps.
        shrink_upper_radius_in_safety_steps (bool): During the fast start whether to
            shrink the upper trust region radius in safety steps.
        full_geometry_improving_step (bool): During the fast start whether to do a
            full geometry-improving step within safety steps (the same as the post fast
            start phase of the algorithm). Since this involves reducing the upper trust
            region radius, this can only be `True` if
            `shrink_upper_radius_in_safety_steps == False`.
        reset_trustregion_radius_after_fast_start (bool):
            Whether or not to reset the upper trust region radius to its initial value
            at the end of the fast start phase.
        reset_min_trustregion_radius_after_fast_start (bool):
            Whether or not to reset the minimum trust region radius
            (:math:`\rho_k`) to its initial value at the end of the fast start phase.
        shrinking_factor_not_successful (float):
            Ratio by which to shrink the trust region radius when realized
            improvement does not match the ``threshold_for_successful_iteration``
            during the fast start phase.  By default it is the same as
            ``reduction_when_not_successful``.
        n_extra_search_directions_per_iteration (int): Number of new search
            directions to add with each iteration where we do not have a full set
            of search directions. This approach is not recommended! Default is 0.

"""


@mark_minimizer(
    name="nag_dfols",
    primary_criterion_entry="root_contributions",
    needs_scaling=True,
    is_available=IS_DFOLS_INSTALLED,
)
def nag_dfols(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    clip_criterion_if_overflowing=CLIP_CRITERION_IF_OVERFLOWING,
    convergence_minimal_trustregion_radius_tolerance=CONVERGENCE_MINIMAL_TRUSTREGION_RADIUS_TOLERANCE,  # noqa: E501
    convergence_noise_corrected_criterion_tolerance=CONVERGENCE_NOISE_CORRECTED_FTOL,  # noqa: E501
    convergence_ftol_scaled=0.0,
    convergence_slow_progress=None,
    initial_directions=INITIAL_DIRECTIONS,
    interpolation_rounding_error=INTERPOLATION_ROUNDING_ERROR,
    noise_additive_level=None,
    noise_multiplicative_level=None,
    noise_n_evals_per_point=None,
    random_directions_orthogonal=RANDOM_DIRECTIONS_ORTHOGONAL,
    stopping_maxfun=STOPPING_MAXFUN,
    threshold_for_safety_step=THRESHOLD_FOR_SAFETY_STEP,
    trustregion_expansion_factor_successful=TRUSTREGION_EXPANSION_FACTOR_SUCCESSFUL,
    trustregion_expansion_factor_very_successful=TRUSTREGION_EXPANSION_FACTOR_VERY_SUCCESSFUL,  # noqa: E501
    trustregion_fast_start_options=None,
    trustregion_initial_radius=None,
    trustregion_method_to_replace_extra_points="geometry_improving",
    trustregion_n_extra_points_to_replace_successful=0,
    trustregion_n_interpolation_points=None,
    trustregion_precondition_interpolation=TRUSTREGION_PRECONDITION_INTERPOLATION,
    trustregion_reset_options=None,
    trustregion_shrinking_factor_not_successful=TRUSTREGION_SHRINKING_FACTOR_NOT_SUCCESSFUL,  # noqa: E501
    trustregion_shrinking_factor_lower_radius=TRUSTREGION_SHRINKING_FACTOR_LOWER_RADIUS,
    trustregion_shrinking_factor_upper_radius=TRUSTREGION_SHRINKING_FACTOR_UPPER_RADIUS,
    trustregion_threshold_successful=TRUSTREGION_THRESHOLD_SUCCESSFUL,
    trustregion_threshold_very_successful=TRUSTREGION_THRESHOLD_VERY_SUCCESSFUL,
):
    r"""Minimize a function with least squares structure using DFO-LS.

    For details see
    :ref: `list_of_nag_algorithms`.

    """
    if not IS_DFOLS_INSTALLED:
        raise NotInstalledError(
            "The 'nag_dfols' algorithm requires the DFO-LS package to be installed."
            "You can install it with 'pip install DFO-LS'. "
            "For additional installation instructions visit: ",
            r"https://numericalalgorithmsgroup.github.io/dfols/build/html/install.html",
        )
    if trustregion_method_to_replace_extra_points == "momentum":
        trustregion_use_momentum = True
    elif trustregion_method_to_replace_extra_points in ["geometry_improving", None]:
        trustregion_use_momentum = False
    else:
        raise ValueError(
            "trustregion_method_to_replace_extra_points must be "
            "'geometry_improving', 'momentum' or None."
        )

    advanced_options, trustregion_reset_options = _create_nag_advanced_options(
        x=x,
        noise_multiplicative_level=noise_multiplicative_level,
        noise_additive_level=noise_additive_level,
        noise_n_evals_per_point=noise_n_evals_per_point,
        convergence_noise_corrected_criterion_tolerance=convergence_noise_corrected_criterion_tolerance,  # noqa: E501
        trustregion_initial_radius=trustregion_initial_radius,
        trustregion_reset_options=trustregion_reset_options,
        convergence_slow_progress=convergence_slow_progress,
        interpolation_rounding_error=interpolation_rounding_error,
        threshold_for_safety_step=threshold_for_safety_step,
        clip_criterion_if_overflowing=clip_criterion_if_overflowing,
        initial_directions=initial_directions,
        random_directions_orthogonal=random_directions_orthogonal,
        trustregion_precondition_interpolation=trustregion_precondition_interpolation,
        trustregion_threshold_successful=trustregion_threshold_successful,
        trustregion_threshold_very_successful=trustregion_threshold_very_successful,
        trustregion_shrinking_factor_not_successful=trustregion_shrinking_factor_not_successful,  # noqa: E501
        trustregion_expansion_factor_successful=trustregion_expansion_factor_successful,
        trustregion_expansion_factor_very_successful=trustregion_expansion_factor_very_successful,  # noqa: E501
        trustregion_shrinking_factor_lower_radius=trustregion_shrinking_factor_lower_radius,  # noqa: E501
        trustregion_shrinking_factor_upper_radius=trustregion_shrinking_factor_upper_radius,  # noqa: E501
    )

    fast_start = _build_options_dict(
        user_input=trustregion_fast_start_options,
        default_options=TRUSTREGION_FAST_START_OPTIONS,
    )
    if fast_start["floor_of_jacobian_singular_values"] != 1:
        warnings.warn(
            "Setting the `floor_of_jacobian_singular_values` is not supported by "
            "DF-OLS as of version 1.2.1."
        )
    if (
        fast_start["shrink_upper_radius_in_safety_steps"]
        and fast_start["full_geometry_improving_step"]
    ):
        raise ValueError(
            "full_geometry_improving_step of the trustregion_fast_start_options can "
            "only be True if shrink_upper_radius_in_safety_steps is False."
        )

    (
        faststart_jac,
        faststart_step,
    ) = _get_fast_start_method(fast_start["method"])

    if (
        trustregion_reset_options["n_extra_interpolation_points_per_soft_reset"]
        < trustregion_reset_options["n_extra_interpolation_points_per_soft_reset"]
    ):
        raise ValueError(
            "In the trustregion_reset_options "
            "'n_extra_interpolation_points_per_soft_reset' must "
            "be larger or the same as 'n_extra_interpolation_points_per_hard_reset'."
        )

    dfols_options = {
        "growing.full_rank.use_full_rank_interp": faststart_jac,
        "growing.perturb_trust_region_step": faststart_step,
        "restarts.hard.use_old_rk": trustregion_reset_options[
            "reuse_criterion_value_at_hard_reset"
        ],
        "restarts.auto_detect.min_chgJ_slope": trustregion_reset_options[
            "auto_detect_min_jacobian_increase"
        ],
        "restarts.max_npt": trustregion_reset_options["max_interpolation_points"],
        "restarts.increase_npt": trustregion_reset_options[
            "n_extra_interpolation_points_per_soft_reset"
        ]
        > 0,
        "restarts.increase_npt_amt": trustregion_reset_options[
            "n_extra_interpolation_points_per_soft_reset"
        ],
        "restarts.hard.increase_ndirs_initial_amt": trustregion_reset_options[
            "n_extra_interpolation_points_per_hard_reset"
        ]
        - trustregion_reset_options["n_extra_interpolation_points_per_soft_reset"],
        "model.rel_tol": convergence_ftol_scaled,
        "regression.num_extra_steps": trustregion_n_extra_points_to_replace_successful,
        "regression.momentum_extra_steps": trustregion_use_momentum,
        "regression.increase_num_extra_steps_with_restart": trustregion_reset_options[
            "n_additional_extra_points_to_replace_per_reset"
        ],
        "growing.ndirs_initial": fast_start["min_inital_points"],
        "growing.delta_scale_new_dirns": fast_start[
            "scale_of_trustregion_step_perturbation"
        ],
        "growing.full_rank.scale_factor": fast_start[
            "scale_of_jacobian_components_perturbation"
        ],
        "growing.full_rank.svd_max_jac_cond": fast_start[
            "jacobian_max_condition_number"
        ],
        "growing.do_geom_steps": fast_start["geometry_improving_steps"],
        "growing.safety.do_safety_step": fast_start["safety_steps"],
        "growing.safety.reduce_delta": fast_start[
            "shrink_upper_radius_in_safety_steps"
        ],
        "growing.safety.full_geom_step": fast_start["full_geometry_improving_step"],
        "growing.reset_delta": fast_start["reset_trustregion_radius_after_fast_start"],
        "growing.reset_rho": fast_start[
            "reset_min_trustregion_radius_after_fast_start"
        ],
        "growing.gamma_dec": fast_start["shrinking_factor_not_successful"],
        "growing.num_new_dirns_each_iter": fast_start[
            "n_extra_search_directions_per_iteration"
        ],
        "logging.save_diagnostic_info": True,
        "logging.save_xk": True,
    }

    advanced_options.update(dfols_options)

    res = dfols.solve(
        criterion,
        x0=x,
        bounds=(lower_bounds, upper_bounds),
        maxfun=stopping_maxfun,
        rhobeg=trustregion_initial_radius,
        npt=trustregion_n_interpolation_points,
        rhoend=convergence_minimal_trustregion_radius_tolerance,
        nsamples=noise_n_evals_per_point,
        objfun_has_noise=noise_additive_level or noise_multiplicative_level,
        scaling_within_bounds=False,
        do_logging=False,
        print_progress=False,
        user_params=advanced_options,
    )

    return _process_nag_result(res, len(x))


@mark_minimizer(
    name="nag_pybobyqa",
    needs_scaling=True,
    is_available=IS_PYBOBYQA_INSTALLED,
)
def nag_pybobyqa(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    clip_criterion_if_overflowing=CLIP_CRITERION_IF_OVERFLOWING,
    convergence_criterion_value=None,
    convergence_minimal_trustregion_radius_tolerance=CONVERGENCE_MINIMAL_TRUSTREGION_RADIUS_TOLERANCE,  # noqa: E501
    convergence_noise_corrected_criterion_tolerance=CONVERGENCE_NOISE_CORRECTED_FTOL,  # noqa: E501
    convergence_slow_progress=None,
    initial_directions=INITIAL_DIRECTIONS,
    interpolation_rounding_error=INTERPOLATION_ROUNDING_ERROR,
    noise_additive_level=None,
    noise_multiplicative_level=None,
    noise_n_evals_per_point=None,
    random_directions_orthogonal=RANDOM_DIRECTIONS_ORTHOGONAL,
    seek_global_optimum=False,
    stopping_max_criterion_evaluations=STOPPING_MAXFUN,
    threshold_for_safety_step=THRESHOLD_FOR_SAFETY_STEP,
    trustregion_expansion_factor_successful=TRUSTREGION_EXPANSION_FACTOR_SUCCESSFUL,
    trustregion_expansion_factor_very_successful=TRUSTREGION_EXPANSION_FACTOR_VERY_SUCCESSFUL,  # noqa: E501
    trustregion_initial_radius=None,
    trustregion_minimum_change_hession_for_underdetermined_interpolation=True,
    trustregion_n_interpolation_points=None,
    trustregion_precondition_interpolation=TRUSTREGION_PRECONDITION_INTERPOLATION,
    trustregion_reset_options=None,
    trustregion_shrinking_factor_not_successful=TRUSTREGION_SHRINKING_FACTOR_NOT_SUCCESSFUL,  # noqa: E501
    trustregion_shrinking_factor_lower_radius=TRUSTREGION_SHRINKING_FACTOR_LOWER_RADIUS,
    trustregion_shrinking_factor_upper_radius=TRUSTREGION_SHRINKING_FACTOR_UPPER_RADIUS,
    trustregion_threshold_successful=TRUSTREGION_THRESHOLD_SUCCESSFUL,
    trustregion_threshold_very_successful=TRUSTREGION_THRESHOLD_VERY_SUCCESSFUL,
):
    r"""Minimize a function using the BOBYQA algorithm.

    For details see
    :ref: `list_of_nag_algorithms`.

    """
    if not IS_PYBOBYQA_INSTALLED:
        raise NotInstalledError(
            "The 'nag_pybobyqa' algorithm requires the Py-BOBYQA package to be "
            "installed. You can install it with 'pip install Py-BOBYQA'. "
            "For additional installation instructions visit: ",
            r"https://numericalalgorithmsgroup.github.io/pybobyqa/build/html/"
            "install.html",
        )

    if convergence_criterion_value is None:
        convergence_criterion_value = -np.inf

    advanced_options, trustregion_reset_options = _create_nag_advanced_options(
        x=x,
        noise_multiplicative_level=noise_multiplicative_level,
        noise_additive_level=noise_additive_level,
        trustregion_initial_radius=trustregion_initial_radius,
        noise_n_evals_per_point=noise_n_evals_per_point,
        convergence_noise_corrected_criterion_tolerance=convergence_noise_corrected_criterion_tolerance,  # noqa: E501
        trustregion_reset_options=trustregion_reset_options,
        convergence_slow_progress=convergence_slow_progress,
        interpolation_rounding_error=interpolation_rounding_error,
        threshold_for_safety_step=threshold_for_safety_step,
        clip_criterion_if_overflowing=clip_criterion_if_overflowing,
        initial_directions=initial_directions,
        random_directions_orthogonal=random_directions_orthogonal,
        trustregion_precondition_interpolation=trustregion_precondition_interpolation,
        trustregion_threshold_successful=trustregion_threshold_successful,
        trustregion_threshold_very_successful=trustregion_threshold_very_successful,
        trustregion_shrinking_factor_not_successful=trustregion_shrinking_factor_not_successful,  # noqa: E501
        trustregion_expansion_factor_successful=trustregion_expansion_factor_successful,
        trustregion_expansion_factor_very_successful=trustregion_expansion_factor_very_successful,  # noqa: E501
        trustregion_shrinking_factor_lower_radius=trustregion_shrinking_factor_lower_radius,  # noqa: E501
        trustregion_shrinking_factor_upper_radius=trustregion_shrinking_factor_upper_radius,  # noqa: E501
    )

    pybobyqa_options = {
        "model.abs_tol": convergence_criterion_value,
        "interpolation.minimum_change_hessian": trustregion_minimum_change_hession_for_underdetermined_interpolation,  # noqa: E501
        "restarts.max_unsuccessful_restarts_total": trustregion_reset_options[
            "max_unsuccessful_resets"
        ],
        "restarts.rhobeg_scale_after_unsuccessful_restart": trustregion_reset_options[
            "trust_region_scaling_at_unsuccessful_reset"
        ],
        "restarts.hard.use_old_fk": trustregion_reset_options[
            "reuse_criterion_value_at_hard_reset"
        ],
        "restarts.auto_detect.min_chg_model_slope": trustregion_reset_options[
            "auto_detect_min_jacobian_increase"
        ],
        "logging.save_diagnostic_info": True,
        "logging.save_xk": True,
    }

    advanced_options.update(pybobyqa_options)

    res = pybobyqa.solve(
        criterion,
        x0=x,
        bounds=(lower_bounds, upper_bounds),
        maxfun=stopping_max_criterion_evaluations,
        rhobeg=trustregion_initial_radius,
        user_params=advanced_options,
        scaling_within_bounds=False,
        do_logging=False,
        print_progress=False,
        objfun_has_noise=noise_additive_level or noise_multiplicative_level,
        nsamples=noise_n_evals_per_point,
        npt=trustregion_n_interpolation_points,
        rhoend=convergence_minimal_trustregion_radius_tolerance,
        seek_global_minimum=seek_global_optimum,
    )

    return _process_nag_result(res, len(x))


def _process_nag_result(nag_result_obj, len_x):
    """Convert the NAG result object to our result dictionary.

    Args:
        nag_result_obj: NAG result object
        len_x (int): length of the supplied parameters, i.e. the dimensionality of the
            problem.


    Returns:
        results (dict): See :ref:`internal_optimizer_output` for details.

    """
    processed = {
        "solution_criterion": nag_result_obj.f,
        "n_fun_evals": nag_result_obj.nx,
        "message": nag_result_obj.msg,
        "success": nag_result_obj.flag == nag_result_obj.EXIT_SUCCESS,
        "reached_convergence_criterion": None,
        "diagnostic_info": nag_result_obj.diagnostic_info,
    }
    try:
        processed["n_iterations"] = nag_result_obj.diagnostic_info["iters_total"].iloc[
            -1
        ]
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        processed["n_iterations"] = None

    if hasattr(nag_result_obj, "states"):
        processed.update({"states": nag_result_obj.states})
    if hasattr(nag_result_obj, "history_params"):
        processed.update({"history_params": nag_result_obj.history_params})
    if nag_result_obj.x is not None:
        processed["solution_x"] = nag_result_obj.x
    else:
        processed["solution_x"] = np.array([np.nan] * len_x)
    return processed


def _create_nag_advanced_options(
    x,
    noise_multiplicative_level,
    noise_additive_level,
    trustregion_initial_radius,
    noise_n_evals_per_point,
    convergence_noise_corrected_criterion_tolerance,
    trustregion_reset_options,
    convergence_slow_progress,
    interpolation_rounding_error,
    threshold_for_safety_step,
    clip_criterion_if_overflowing,
    initial_directions,
    random_directions_orthogonal,
    trustregion_precondition_interpolation,
    trustregion_threshold_successful,
    trustregion_threshold_very_successful,
    trustregion_shrinking_factor_not_successful,
    trustregion_expansion_factor_successful,
    trustregion_expansion_factor_very_successful,
    trustregion_shrinking_factor_lower_radius,
    trustregion_shrinking_factor_upper_radius,
):
    if noise_multiplicative_level is not None and noise_additive_level is not None:
        raise ValueError("You cannot specify both multiplicative and additive noise.")
    if trustregion_initial_radius is None:
        trustregion_initial_radius = calculate_trustregion_initial_radius(x)
    # -np.inf as a default leads to errors when building the documentation with sphinx.
    noise_n_evals_per_point = _change_evals_per_point_interface(noise_n_evals_per_point)
    trustregion_reset_options = _build_options_dict(
        user_input=trustregion_reset_options,
        default_options=RESET_OPTIONS,
    )
    if trustregion_reset_options["reset_type"] not in ["soft", "hard"]:
        raise ValueError(
            "reset_type in the trustregion_reset_options must be soft or hard."
        )
    if initial_directions not in ["coordinate", "random"]:
        raise ValueError("inital_directions must be either 'coordinate' or 'random'.")
    convergence_slow_progress = _build_options_dict(
        user_input=convergence_slow_progress,
        default_options=CONVERGENCE_SLOW_PROGRESS,
    )

    is_noisy = bool(noise_additive_level or noise_multiplicative_level)

    advanced_options = {
        "general.rounding_error_constant": interpolation_rounding_error,
        "general.safety_step_thresh": threshold_for_safety_step,
        "general.check_objfun_for_overflow": clip_criterion_if_overflowing,
        "tr_radius.eta1": trustregion_threshold_successful,
        "tr_radius.eta2": trustregion_threshold_very_successful,
        "tr_radius.gamma_dec": trustregion_shrinking_factor_not_successful,
        "tr_radius.gamma_inc": trustregion_expansion_factor_successful,
        "tr_radius.gamma_inc_overline": trustregion_expansion_factor_very_successful,
        "tr_radius.alpha1": trustregion_shrinking_factor_lower_radius,
        "tr_radius.alpha2": trustregion_shrinking_factor_upper_radius,
        "init.random_initial_directions": initial_directions == "random",
        "init.random_directions_make_orthogonal": random_directions_orthogonal,
        "slow.thresh_for_slow": convergence_slow_progress[
            "threshold_to_characterize_as_slow"
        ],
        "slow.max_slow_iters": convergence_slow_progress[
            "max_insufficient_improvements"
        ],
        "slow.history_for_slow": convergence_slow_progress["comparison_period"],
        "noise.multiplicative_noise_level": noise_multiplicative_level,
        "noise.additive_noise_level": noise_additive_level,
        "noise.quit_on_noise_level": (
            convergence_noise_corrected_criterion_tolerance > 0
        )
        and is_noisy,
        "noise.scale_factor_for_quit": convergence_noise_corrected_criterion_tolerance,
        "interpolation.precondition": trustregion_precondition_interpolation,
        "restarts.use_restarts": trustregion_reset_options["use_resets"],
        "restarts.max_unsuccessful_restarts": trustregion_reset_options[
            "max_consecutive_unsuccessful_resets"
        ],
        "restarts.rhoend_scale": trustregion_reset_options[
            "minimal_trustregion_radius_tolerance_scaling_at_reset"
        ],
        "restarts.use_soft_restarts": trustregion_reset_options["reset_type"] == "soft",
        "restarts.soft.move_xk": trustregion_reset_options["move_center_at_soft_reset"],
        "restarts.soft.max_fake_successful_steps": trustregion_reset_options[
            "max_iterations_without_new_best_after_soft_reset"
        ],
        "restarts.auto_detect": trustregion_reset_options["auto_detect"],
        "restarts.auto_detect.history": trustregion_reset_options[
            "auto_detect_history"
        ],
        "restarts.auto_detect.min_correl": trustregion_reset_options[
            "auto_detect_min_correlations"
        ],
        "restarts.soft.num_geom_steps": trustregion_reset_options[
            "points_to_replace_at_soft_reset"
        ],
    }

    return advanced_options, trustregion_reset_options


def _change_evals_per_point_interface(func):
    """Change the interface of the user supplied function to the one expected by NAG.

    Args:
        func (callable or None): function mapping from our names to
            noise_n_evals_per_point.

    Returns:
        adjusted_noise_n_evals_per_point (callable): function mapping from the
            argument names expected by pybobyqa and df-ols to noise_n_evals_per_point.

    """
    if func is not None:

        def adjusted_noise_n_evals_per_point(delta, rho, iter, nrestarts):  # noqa: A002
            return func(
                upper_trustregion_radius=delta,
                lower_trustregion_radius=rho,
                n_iterations=iter,
                n_resets=nrestarts,
            )

        return adjusted_noise_n_evals_per_point


def _build_options_dict(user_input, default_options):
    """Create the full dictionary of trust region fast start options from user input.

    Args:
        user_input (dict or None): dictionary to update the default options with.
            May only contain keys present in the default options.
        default_options (dict): the default values.

    Returns:
        full_options (dict)

    """
    full_options = default_options.copy()
    user_input = {} if user_input is None else user_input
    invalid = [x for x in user_input if x not in full_options]
    if len(invalid) > 0:
        raise ValueError(
            f"You specified illegal options {', '.join(invalid)}. Allowed are: "
            ", ".join(full_options.keys())
        )
    full_options.update(user_input)
    return full_options


def _get_fast_start_method(user_value):
    """Get fast start method arguments from user value."""
    allowed_values = ["auto", "jacobian", "trustregion"]
    if user_value not in allowed_values:
        raise ValueError(
            "`perturb_jacobian_or_trustregion_step` must be one of "
            f"{allowed_values}. You provided {user_value}."
        )
    if user_value == "auto":
        faststart_jac = None
        faststart_step = None
    else:
        faststart_jac = user_value == "jacobian"
        faststart_step = not faststart_jac

    return faststart_jac, faststart_step
