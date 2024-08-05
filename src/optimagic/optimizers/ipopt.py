"""Implement cyipopt's Interior Point Optimizer."""

import numpy as np

from optimagic.config import IS_CYIPOPT_INSTALLED
from optimagic.decorators import mark_minimizer
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import (
    CONVERGENCE_FTOL_REL,
    STOPPING_MAXITER,
)
from optimagic.optimizers.scipy_optimizers import process_scipy_result

if IS_CYIPOPT_INSTALLED:
    import cyipopt


@mark_minimizer(
    name="ipopt",
    primary_criterion_entry="value",
    needs_scaling=False,
    is_available=IS_CYIPOPT_INSTALLED,
)
def ipopt(
    criterion,
    derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    # nonlinear constraints
    nonlinear_constraints=(),
    # convergence criteria
    convergence_ftol_rel=CONVERGENCE_FTOL_REL,
    dual_inf_tol=1.0,
    constr_viol_tol=0.0001,
    compl_inf_tol=0.0001,
    s_max=100.0,
    mu_target=0.0,
    # stopping criteria
    stopping_maxiter=STOPPING_MAXITER,
    stopping_max_wall_time_seconds=1e20,
    stopping_max_cpu_time=1e20,
    # acceptable heuristic
    acceptable_iter=15,
    acceptable_tol=1e-6,
    acceptable_dual_inf_tol=1e-10,
    acceptable_constr_viol_tol=0.01,
    acceptable_compl_inf_tol=0.01,
    acceptable_obj_change_tol=1e20,
    diverging_iterates_tol=1e20,
    nlp_lower_bound_inf=-1e19,
    nlp_upper_bound_inf=1e19,
    fixed_variable_treatment="make_parameter",
    dependency_detector=None,
    dependency_detection_with_rhs=False,
    # bounds
    kappa_d=1e-5,
    bound_relax_factor=1e-8,
    honor_original_bounds=False,
    # derivatives
    check_derivatives_for_naninf=False,
    # not sure if we should support the following:
    jac_c_constant=False,
    jac_d_constant=False,
    hessian_constant=False,
    # scaling
    nlp_scaling_method="gradient-based",
    obj_scaling_factor=1,
    nlp_scaling_max_gradient=100,
    nlp_scaling_obj_target_gradient=0.0,
    nlp_scaling_constr_target_gradient=0.0,
    nlp_scaling_min_value=1e-8,
    # initialization
    bound_push=0.01,
    bound_frac=0.01,
    slack_bound_push=0.01,
    slack_bound_frac=0.01,
    constr_mult_init_max=1000,
    bound_mult_init_val=1,
    bound_mult_init_method="constant",
    least_square_init_primal=False,
    least_square_init_duals=False,
    # warm start
    warm_start_init_point=False,
    warm_start_same_structure=False,
    warm_start_bound_push=0.001,
    warm_start_bound_frac=0.001,
    warm_start_slack_bound_push=0.001,
    warm_start_slack_bound_frac=0.001,
    warm_start_mult_bound_push=0.001,
    warm_start_mult_init_max=1e6,
    warm_start_entire_iterate=False,
    warm_start_target_mu=0.0,
    # miscellaneous
    option_file_name="",
    replace_bounds=False,
    skip_finalize_solution_call=False,
    timing_statistics=False,
    # barrier parameter update
    mu_max_fact=1000,
    mu_max=100_000,
    mu_min=1e-11,
    adaptive_mu_globalization="obj-constr-filter",
    adaptive_mu_kkterror_red_iters=4,
    adaptive_mu_kkterror_red_fact=0.9999,
    filter_margin_fact=1e-5,
    filter_max_margin=1,
    adaptive_mu_restore_previous_iterate=False,
    adaptive_mu_monotone_init_factor=0.8,
    adaptive_mu_kkt_norm_type="2-norm-squared",
    mu_strategy="monotone",
    mu_oracle="quality-function",
    fixed_mu_oracle="average_compl",
    mu_init=0.1,
    barrier_tol_factor=10,
    mu_linear_decrease_factor=0.2,
    mu_superlinear_decrease_power=1.5,
    mu_allow_fast_monotone_decrease=True,
    tau_min=0.99,
    sigma_max=100,
    sigma_min=1e-6,
    quality_function_norm_type="2-norm-squared",
    quality_function_centrality=None,
    quality_function_balancing_term=None,
    quality_function_max_section_steps=8,
    quality_function_section_sigma_tol=0.01,
    quality_function_section_qf_tol=0.0,
    # line search
    line_search_method="filter",
    alpha_red_factor=0.5,
    accept_every_trial_step=False,
    accept_after_max_steps=-1,
    alpha_for_y="primal",
    alpha_for_y_tol=10,
    tiny_step_tol=2.22045 * 1e-15,
    tiny_step_y_tol=0.01,
    watchdog_shortened_iter_trigger=10,
    watchdog_trial_iter_max=3,
    theta_max_fact=10_000,
    theta_min_fact=0.0001,
    eta_phi=1e-8,
    delta=1,
    s_phi=2.3,
    s_theta=1.1,
    gamma_phi=1e-8,
    gamma_theta=1e-5,
    alpha_min_frac=0.05,
    max_soc=4,
    kappa_soc=0.99,
    obj_max_inc=5,
    max_filter_resets=5,
    filter_reset_trigger=5,
    corrector_type=None,
    skip_corr_if_neg_curv=True,
    skip_corr_in_monotone_mode=True,
    corrector_compl_avrg_red_fact=1,
    soc_method=0,
    nu_init=1e-6,
    nu_inc=0.0001,
    rho=0.1,
    kappa_sigma=1e10,
    recalc_y=False,
    recalc_y_feas_tol=1e-6,
    slack_move=1.81899 * 1e-12,
    constraint_violation_norm_type="1-norm",
    # step calculation
    mehrotra_algorithm=False,
    fast_step_computation=False,
    min_refinement_steps=1,
    max_refinement_steps=10,
    residual_ratio_max=1e-10,
    residual_ratio_singular=1e-5,
    residual_improvement_factor=1,
    neg_curv_test_tol=0,
    neg_curv_test_reg=True,
    max_hessian_perturbation=1e20,
    min_hessian_perturbation=1e-20,
    perturb_inc_fact_first=100,
    perturb_inc_fact=8,
    perturb_dec_fact=0.333333,
    first_hessian_perturbation=0.0001,
    jacobian_regularization_value=1e-8,
    jacobian_regularization_exponent=0.25,
    perturb_always_cd=False,
    # restoration phase
    expect_infeasible_problem=False,
    expect_infeasible_problem_ctol=0.001,
    expect_infeasible_problem_ytol=1e8,
    start_with_resto=False,
    soft_resto_pderror_reduction_factor=0.9999,
    max_soft_resto_iters=10,
    required_infeasibility_reduction=0.9,
    max_resto_iter=3_000_000,
    evaluate_orig_obj_at_resto_trial=True,
    resto_penalty_parameter=1000,
    resto_proximity_weight=1,
    bound_mult_reset_threshold=1000,
    constr_mult_reset_threshold=0,
    resto_failure_feasibility_threshold=None,
    # hessian approximation
    limited_memory_aug_solver="sherman-morrison",
    limited_memory_max_history=6,
    limited_memory_update_type="bfgs",
    limited_memory_initialization="scalar1",
    limited_memory_init_val=1,
    limited_memory_init_val_max=1e8,
    limited_memory_init_val_min=1e-8,
    limited_memory_max_skipping=2,
    limited_memory_special_for_resto=False,
    hessian_approximation="limited-memory",
    hessian_approximation_space="nonlinear-variables",
    # linear solver
    linear_solver="mumps",
    linear_solver_options=None,
):
    """Minimize a scalar function using the Interior Point Optimizer.

    For details see
    :ref: `ipopt_algorithm`.

    """
    if not IS_CYIPOPT_INSTALLED:
        raise NotInstalledError(
            "The 'ipopt' algorithm requires the cyipopt package to be installed. "
            "You can it with: `conda install -c conda-forge cyipopt`."
        )
    if acceptable_tol <= convergence_ftol_rel:
        raise ValueError(
            "The acceptable tolerance must be larger than the desired tolerance."
        )
    if mu_strategy not in ["monotone", "adaptive"]:
        raise ValueError(
            f"Unknown barrier strategy: {mu_strategy}. It must be 'monotone' or "
            "'adaptive'."
        )
    if nlp_upper_bound_inf < 0:
        raise ValueError("nlp_upper_bound_inf should be > 0.")
    if nlp_lower_bound_inf > 0:
        raise ValueError("nlp_lower_bound_inf should be < 0.")

    linear_solver_options = (
        {} if linear_solver_options is None else linear_solver_options
    )

    # The default value is actually 1e2*tol, where tol is the general
    # termination tolerance.
    if resto_failure_feasibility_threshold is None:
        resto_failure_feasibility_threshold = 1e2 * convergence_ftol_rel

    # convert None to str none section
    linear_solver_options_with_none = [
        "ma86_scaling",
        "ma97_scaling",
        "ma97_scaling1",
        "ma97_scaling2",
        "ma97_scaling3",
        "spral_scaling",
        "spral_scaling_1",
        "spral_scaling_2",
        "spral_scaling_3",
        "linear_system_scaling",
    ]
    for key, val in linear_solver_options.items():
        if key in linear_solver_options_with_none:
            linear_solver_options[key] = _convert_none_to_str(val)
    boolean_linear_solver_options = [
        "linear_scaling_on_demand"
        "ma27_skip_inertia_check"
        "ma27_ignore_singularity"
        "ma57_automatic_scaling"
        "ma97_solve_blas3"
        "pardiso_redo_symbolic_fact_only_if_inertia_wrong"
        "pardiso_repeated_perturbation_means_singular"
        "pardiso_skip_inertia_check"
        "pardiso_iterative"
        "pardisomkl_redo_symbolic_fact_only_if_inertia_wrong"
        "pardisomkl_repeated_perturbation_means_singular"
        "pardisomkl_skip_inertia_check"
        "spral_ignore_numa"
        "spral_use_gpu"
        "wsmp_skip_inertia_check"
        "wsmp_no_pivoting"
    ]
    for key, val in linear_solver_options.items():
        if key in boolean_linear_solver_options:
            linear_solver_options[key] = _convert_bool_to_str(val, key)

    convert_bool_to_str_options = {
        "dependency_detection_with_rhs": dependency_detection_with_rhs,
        "check_derivatives_for_naninf": check_derivatives_for_naninf,
        "jac_c_constant": jac_c_constant,
        "jac_d_constant": jac_d_constant,
        "hessian_constant": hessian_constant,
        "least_square_init_primal": least_square_init_primal,
        "least_square_init_duals": least_square_init_duals,
        "warm_start_init_point": warm_start_init_point,
        "warm_start_same_structure": warm_start_same_structure,
        "warm_start_entire_iterate": warm_start_entire_iterate,
        "replace_bounds": replace_bounds,
        "skip_finalize_solution_call": skip_finalize_solution_call,
        "timing_statistics": timing_statistics,
        "adaptive_mu_restore_previous_iterate": adaptive_mu_restore_previous_iterate,
        "mu_allow_fast_monotone_decrease": mu_allow_fast_monotone_decrease,
        "accept_every_trial_step": accept_every_trial_step,
        "skip_corr_if_neg_curv": skip_corr_if_neg_curv,
        "skip_corr_in_monotone_mode": skip_corr_in_monotone_mode,
        "recalc_y": recalc_y,
        "mehrotra_algorithm": mehrotra_algorithm,
        "fast_step_computation": fast_step_computation,
        "neg_curv_test_reg": neg_curv_test_reg,
        "perturb_always_cd": perturb_always_cd,
        "expect_infeasible_problem": expect_infeasible_problem,
        "start_with_resto": start_with_resto,
        "evaluate_orig_obj_at_resto_trial": evaluate_orig_obj_at_resto_trial,
        "limited_memory_special_for_resto": limited_memory_special_for_resto,
        "honor_original_bounds": honor_original_bounds,
    }
    converted_bool_to_str_options = {
        key: _convert_bool_to_str(val, key)
        for key, val in convert_bool_to_str_options.items()
    }

    options = {
        # disable verbosity
        "print_level": 0,
        "ma77_print_level": -1,
        "ma86_print_level": -1,
        "ma97_print_level": -1,
        "pardiso_msglvl": 0,
        # disable derivative checker
        "derivative_test": "none",
        "s_max": float(s_max),
        "max_iter": stopping_maxiter,
        "max_wall_time": float(stopping_max_wall_time_seconds),
        "max_cpu_time": stopping_max_cpu_time,
        "dual_inf_tol": dual_inf_tol,
        "constr_viol_tol": constr_viol_tol,
        "compl_inf_tol": compl_inf_tol,
        # acceptable heuristic
        "acceptable_iter": int(acceptable_iter),
        "acceptable_tol": acceptable_tol,
        "acceptable_dual_inf_tol": acceptable_dual_inf_tol,
        "acceptable_constr_viol_tol": acceptable_constr_viol_tol,
        "acceptable_compl_inf_tol": acceptable_compl_inf_tol,
        "acceptable_obj_change_tol": acceptable_obj_change_tol,
        # bounds and more
        "diverging_iterates_tol": diverging_iterates_tol,
        "nlp_lower_bound_inf": nlp_lower_bound_inf,
        "nlp_upper_bound_inf": nlp_upper_bound_inf,
        "fixed_variable_treatment": fixed_variable_treatment,
        "dependency_detector": _convert_none_to_str(dependency_detector),
        "kappa_d": kappa_d,
        "bound_relax_factor": bound_relax_factor,
        "honor_original_bounds": honor_original_bounds,
        # scaling
        "nlp_scaling_method": _convert_none_to_str(nlp_scaling_method),
        "obj_scaling_factor": float(obj_scaling_factor),
        "nlp_scaling_max_gradient": float(nlp_scaling_max_gradient),
        "nlp_scaling_obj_target_gradient": float(nlp_scaling_obj_target_gradient),
        "nlp_scaling_constr_target_gradient": float(nlp_scaling_constr_target_gradient),
        "nlp_scaling_min_value": float(nlp_scaling_min_value),
        # initialization
        "bound_push": bound_push,
        "bound_frac": bound_frac,
        "slack_bound_push": slack_bound_push,
        "slack_bound_frac": slack_bound_frac,
        "constr_mult_init_max": float(constr_mult_init_max),
        "bound_mult_init_val": float(bound_mult_init_val),
        "bound_mult_init_method": bound_mult_init_method,
        # warm start
        "warm_start_bound_push": warm_start_bound_push,
        "warm_start_bound_frac": warm_start_bound_frac,
        "warm_start_slack_bound_push": warm_start_slack_bound_push,
        "warm_start_slack_bound_frac": warm_start_slack_bound_frac,
        "warm_start_mult_bound_push": warm_start_mult_bound_push,
        "warm_start_mult_init_max": warm_start_mult_init_max,
        "warm_start_target_mu": warm_start_target_mu,
        # more miscellaneous
        "option_file_name": option_file_name,
        # barrier parameter update
        "mu_target": float(mu_target),
        "mu_max_fact": float(mu_max_fact),
        "mu_max": float(mu_max),
        "mu_min": float(mu_min),
        "adaptive_mu_globalization": adaptive_mu_globalization,
        "adaptive_mu_kkterror_red_iters": adaptive_mu_kkterror_red_iters,
        "adaptive_mu_kkterror_red_fact": adaptive_mu_kkterror_red_fact,
        "filter_margin_fact": float(filter_margin_fact),
        "filter_max_margin": float(filter_max_margin),
        "adaptive_mu_monotone_init_factor": adaptive_mu_monotone_init_factor,
        "adaptive_mu_kkt_norm_type": adaptive_mu_kkt_norm_type,
        "mu_strategy": mu_strategy,
        "mu_oracle": mu_oracle,
        "fixed_mu_oracle": fixed_mu_oracle,
        "mu_init": mu_init,
        "barrier_tol_factor": float(barrier_tol_factor),
        "mu_linear_decrease_factor": mu_linear_decrease_factor,
        "mu_superlinear_decrease_power": mu_superlinear_decrease_power,
        "tau_min": tau_min,
        "sigma_max": float(sigma_max),
        "sigma_min": sigma_min,
        "quality_function_norm_type": quality_function_norm_type,
        "quality_function_centrality": _convert_none_to_str(
            quality_function_centrality
        ),
        "quality_function_balancing_term": _convert_none_to_str(
            quality_function_balancing_term
        ),
        "quality_function_max_section_steps": int(quality_function_max_section_steps),
        "quality_function_section_sigma_tol": quality_function_section_sigma_tol,
        "quality_function_section_qf_tol": quality_function_section_qf_tol,
        # linear search
        "line_search_method": line_search_method,
        "alpha_red_factor": alpha_red_factor,
        "accept_after_max_steps": accept_after_max_steps,
        "alpha_for_y": alpha_for_y,
        "alpha_for_y_tol": float(alpha_for_y_tol),
        "tiny_step_tol": tiny_step_tol,
        "tiny_step_y_tol": tiny_step_y_tol,
        "watchdog_shortened_iter_trigger": watchdog_shortened_iter_trigger,
        "watchdog_trial_iter_max": watchdog_trial_iter_max,
        "theta_max_fact": float(theta_max_fact),
        "theta_min_fact": theta_min_fact,
        "eta_phi": eta_phi,
        "delta": float(delta),
        "s_phi": s_phi,
        "s_theta": s_theta,
        "gamma_phi": gamma_phi,
        "gamma_theta": gamma_theta,
        "alpha_min_frac": alpha_min_frac,
        "max_soc": max_soc,
        "kappa_soc": kappa_soc,
        "obj_max_inc": float(obj_max_inc),
        "max_filter_resets": max_filter_resets,
        "filter_reset_trigger": filter_reset_trigger,
        "corrector_type": _convert_none_to_str(corrector_type),
        "corrector_compl_avrg_red_fact": float(corrector_compl_avrg_red_fact),
        "soc_method": soc_method,
        "nu_init": nu_init,
        "nu_inc": nu_inc,
        "rho": rho,
        "kappa_sigma": kappa_sigma,
        "recalc_y_feas_tol": recalc_y_feas_tol,
        "slack_move": slack_move,
        "constraint_violation_norm_type": constraint_violation_norm_type,
        # step calculation
        "min_refinement_steps": min_refinement_steps,
        "max_refinement_steps": max_refinement_steps,
        "residual_ratio_max": residual_ratio_max,
        "residual_ratio_singular": residual_ratio_singular,
        "residual_improvement_factor": float(residual_improvement_factor),
        "neg_curv_test_tol": float(neg_curv_test_tol),
        "max_hessian_perturbation": max_hessian_perturbation,
        "min_hessian_perturbation": min_hessian_perturbation,
        "perturb_inc_fact_first": float(perturb_inc_fact_first),
        "perturb_inc_fact": float(perturb_inc_fact),
        "perturb_dec_fact": float(perturb_dec_fact),
        "first_hessian_perturbation": float(first_hessian_perturbation),
        "jacobian_regularization_value": float(jacobian_regularization_value),
        "jacobian_regularization_exponent": float(jacobian_regularization_exponent),
        # restoration phase
        "expect_infeasible_problem_ctol": expect_infeasible_problem_ctol,
        "expect_infeasible_problem_ytol": expect_infeasible_problem_ytol,
        "soft_resto_pderror_reduction_factor": soft_resto_pderror_reduction_factor,
        "max_soft_resto_iters": max_soft_resto_iters,
        "required_infeasibility_reduction": float(required_infeasibility_reduction),
        "max_resto_iter": max_resto_iter,
        "resto_penalty_parameter": float(resto_penalty_parameter),
        "resto_proximity_weight": float(resto_proximity_weight),
        "bound_mult_reset_threshold": float(bound_mult_reset_threshold),
        "constr_mult_reset_threshold": float(constr_mult_reset_threshold),
        "resto_failure_feasibility_threshold": float(
            resto_failure_feasibility_threshold
        ),
        # hessian approximation
        "limited_memory_aug_solver": limited_memory_aug_solver,
        "limited_memory_max_history": limited_memory_max_history,
        "limited_memory_update_type": limited_memory_update_type,
        "limited_memory_initialization": limited_memory_initialization,
        "limited_memory_init_val": float(limited_memory_init_val),
        "limited_memory_init_val_max": limited_memory_init_val_max,
        "limited_memory_init_val_min": limited_memory_init_val_min,
        "limited_memory_max_skipping": limited_memory_max_skipping,
        "hessian_approximation": hessian_approximation,
        "hessian_approximation_space": hessian_approximation_space,
        # linear solver
        "linear_solver": linear_solver,
        **linear_solver_options,
        **converted_bool_to_str_options,
    }

    raw_res = cyipopt.minimize_ipopt(
        fun=criterion,
        x0=x,
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        jac=derivative,
        constraints=nonlinear_constraints,
        tol=convergence_ftol_rel,
        options=options,
    )

    res = process_scipy_result(raw_res)
    return res


def _convert_bool_to_str(var, name):
    """Convert input to either 'yes' or 'no' and check the output is yes or no.

    Args:
        var (str or bool): user input
        name (str): name of the variable.

    Returns:
        out (str): "yes" or "no".

    """
    if var is True:
        out = "yes"
    elif var is False:
        out = "no"
    else:
        out = var
    if out not in {"yes", "no"}:
        raise ValueError(
            f"{name} must be 'yes', 'no', True or False. You specified {var}."
        )
    return out


def _convert_none_to_str(var):
    out = "none" if var is None else var
    return out


def _get_scipy_bounds(lower_bounds, upper_bounds):
    # Scipy works with `None` instead of infinite values for unconstrained parameters
    # and requires a list of tuples for each parameter with lower and upper bound.
    bounds = np.column_stack([lower_bounds, upper_bounds])
    mask = ~np.isfinite(bounds)
    bounds = bounds.astype("object")
    bounds[mask] = None
    return list(map(tuple, bounds))
