"""Implement cyipopt's Interior Point Optimizer."""
import functools

from estimagic.config import IS_CYIPOPT_INSTALLED
from estimagic.optimization.algo_options import CONVERGENCE_RELATIVE_CRITERION_TOLERANCE
from estimagic.optimization.algo_options import STOPPING_MAX_ITERATIONS
from estimagic.optimization.scipy_optimizers import get_scipy_bounds
from estimagic.optimization.scipy_optimizers import process_scipy_result

try:
    import cyipopt
except ImportError:
    pass

DEFAULT_ALGO_INFO = {
    "primary_criterion_entry": "value",
    "parallelizes": False,
    "needs_scaling": False,
}


def ipopt(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    # convergence criteria
    convergence_relative_criterion_tolerance=CONVERGENCE_RELATIVE_CRITERION_TOLERANCE,
    dual_inf_tol=1.0,
    constr_viol_tol=0.0001,
    compl_inf_tol=0.0001,
    #
    s_max=100.0,
    mu_target=0.0,
    # stopping criteria
    stopping_max_iterations=STOPPING_MAX_ITERATIONS,
    stopping_max_wall_time_seconds=1e20,
    stopping_max_cpu_time=1e20,
    # acceptable heuristic
    acceptable_iter=15,
    acceptable_tol=1e-6,
    acceptable_dual_inf_tol=1e-10,
    acceptable_constr_viol_tol=0.01,
    acceptable_compl_inf_tol=0.01,
    acceptable_obj_change_tol=1e20,
    #
    diverging_iterates_tol=1e20,
    nlp_lower_bound_inf=-1e19,
    nlp_upper_bound_inf=1e19,
    fixed_variable_treatment="make_parameter",
    dependency_detector="none",
    dependency_detection_with_rhs="no",
    # bounds
    kappa_d=1e-5,
    bound_relax_factor=1e-8,
    honor_original_bounds="no",
    # derivatives
    check_derivatives_for_naninf="no",
    # not sure if we should support the following:
    jac_c_constant="no",
    jac_d_constant="no",
    hessian_constant="no",
    # initialization
    bound_push=0.01,
    bound_frac=0.01,
    slack_bound_push=0.01,
    slack_bound_frac=0.01,
    constr_mult_init_max=1000,
    bound_mult_init_val=1,
    bound_mult_init_method="constant",
    least_square_init_primal="no",
    least_square_init_duals="no",
    # warm start
    warm_start_init_point="no",
    warm_start_same_structure="no",
    warm_start_bound_push=0.001,
    warm_start_bound_frac=0.001,
    warm_start_slack_bound_push=0.001,
    warm_start_slack_bound_frac=0.001,
    warm_start_mult_bound_push=0.001,
    warm_start_mult_init_max=1e6,
    warm_start_entire_iterate="no",
    warm_start_target_mu=0.0,
    # miscellaneous
    option_file_name="",
    replace_bounds="no",
    skip_finalize_solution_call="no",
    timing_statistics="no",
    # barrier parameter update
    mu_max_fact=1000,
    mu_max=100_000,
    mu_min=1e-11,
    adaptive_mu_globalization="obj-constr-filter",
    adaptive_mu_kkterror_red_iters=4,
    adaptive_mu_kkterror_red_fact=0.9999,
    filter_margin_fact=1e-5,
    filter_max_margin=1,
    adaptive_mu_restore_previous_iterate="no",
    adaptive_mu_monotone_init_factor=0.8,
    adaptive_mu_kkt_norm_type="2-norm-squared",
    mu_strategy="monotone",
    mu_oracle="quality-function",
    fixed_mu_oracle="average_compl",
    mu_init=0.1,
    barrier_tol_factor=10,
    mu_linear_decrease_factor=0.2,
    mu_superlinear_decrease_power=1.5,
    mu_allow_fast_monotone_decrease="yes",
    tau_min=0.99,
    sigma_max=100,
    sigma_min=1e-6,
    quality_function_norm_type="2-norm-squared",
    quality_function_centrality="none",
    quality_function_balancing_term="none",
    quality_function_max_section_steps=8,
    quality_function_section_sigma_tol=0.01,
    quality_function_section_qf_tol=0.0,
):
    """Minimize a scalar function using the Interior Point Optimizer.

    This implementation of the Interior Point Optimizer (:cite:`Waechter2005`,
    :cite:`Waechter2005a`, :cite:`Waechter2005b`, :cite:`Nocedal2009`) relies on
    `cyipopt <https://cyipopt.readthedocs.io/en/latest/index.html>`_, a Python
    wrapper for the `Ipopt optimization package
    <https://coin-or.github.io/Ipopt/index.html>`_.

    There are two levels of termination criteria. If the usual "desired"
    tolerances (see tol, dual_inf_tol etc) are satisfied at an iteration, the
    algorithm immediately terminates with a success message. On the other hand,
    if the algorithm encounters "acceptable_iter" many iterations in a row that
    are considered "acceptable", it will terminate before the desired
    convergence tolerance is met. This is useful in cases where the algorithm
    might not be able to achieve the "desired" level of accuracy.

    - convergence.relative_criterion_tolerance (float): The algorithm terminates
      successfully, if the (scaled) non linear programming error becomes smaller
      than this value.

    - mu_target: Desired value of complementarity. Usually, the barrier
      parameter is driven to zero and the termination test for complementarity
      is measured with respect to zero complementarity. However, in some cases
      it might be desired to have Ipopt solve barrier problem for strictly
      positive value of the barrier parameter. In this case, the value of
      "mu_target" specifies the final value of the barrier parameter, and the
      termination tests are then defined with respect to the barrier problem for
      this value of the barrier parameter. The valid range for this real option
      is 0 ≤ mu_target and its default value is 0.
    - s_max (float): Scaling threshold for the NLP error.

    - stopping.max_iterations (int):  If the maximum number of iterations is
      reached, the optimization stops, but we do not count this as successful
      convergence. The difference to ``max_criterion_evaluations`` is that one
      iteration might need several criterion evaluations, for example in a line
      search or to determine if the trust region radius has to be shrunk.
    - stopping.max_wall_time_seconds (float): Maximum number of walltime clock
      seconds.
    - stopping.max_cpu_time (float): Maximum number of CPU seconds. A limit on
      CPU seconds that Ipopt can use to solve one problem. If during the
      convergence check this limit is exceeded, Ipopt will terminate with a
      corresponding message. The valid range for this real option is 0 <
      max_cpu_time and its default value is 1e20.

    - dual_inf_tol (float): Desired threshold for the dual infeasibility.
      Absolute tolerance on the dual infeasibility. Successful termination
      requires that the max-norm of the (unscaled) dual infeasibility is less
      than this threshold. The valid range for this real option is 0 <
      dual_inf_tol and its default value is 1.
    - constr_viol_tol (float): Desired threshold for the constraint and variable
      bound violation. Absolute tolerance on the constraint and variable bound
      violation. Successful termination requires that the max-norm of the
      (unscaled) constraint violation is less than this threshold. If option
      bound_relax_factor is not zero 0, then Ipopt relaxes given variable
      bounds. The value of constr_viol_tol is used to restrict the absolute
      amount of this bound relaxation. The valid range for this real option is 0
      < constr_viol_tol and its default value is 0.0001.
    - compl_inf_tol (float): Desired threshold for the complementarity
      conditions. Absolute tolerance on the complementarity. Successful
      termination requires that the max-norm of the (unscaled) complementarity
      is less than this threshold. The valid range for this real option is 0 <
      compl_inf_tol and its default value is 0.0001.

    - acceptable_iter (int): Number of "acceptable" iterates before triggering
      termination. If the algorithm encounters this many successive "acceptable"
      iterates (see above on the acceptable heuristic), it terminates, assuming
      that the problem has been solved to best possible accuracy given
      round-off. If it is set to zero, this heuristic is disabled. The valid
      range for this integer option is 0 ≤ acceptable_iter.
    - acceptable_tol (float):"Acceptable" convergence tolerance (relative).
      Determines which (scaled) overall optimality error is considered to be
      "acceptable". The valid range for this real option is 0 < acceptable_tol.
    - acceptable_dual_inf_tol (float):  "Acceptance" threshold for the dual
      infeasibility. Absolute tolerance on the dual infeasibility. "Acceptable"
      termination requires that the (max-norm of the unscaled) dual
      infeasibility is less than this threshold; see also acceptable_tol. The
      valid range for this real option is 0 < acceptable_dual_inf_tol and its
      default value is 10+10.
    - acceptable_constr_viol_tol (float): "Acceptance" threshold for the
      constraint violation. Absolute tolerance on the constraint violation.
      "Acceptable" termination requires that the max-norm of the (unscaled)
      constraint violation is less than this threshold; see also acceptable_tol.
      The valid range for this real option is 0 < acceptable_constr_viol_tol and
      its default value is 0.01.
    - acceptable_compl_inf_tol (float): "Acceptance" threshold for the
      complementarity conditions. Absolute tolerance on the complementarity.
      "Acceptable" termination requires that the max-norm of the (unscaled)
      complementarity is less than this threshold; see also acceptable_tol. The
      valid range for this real option is 0 < acceptable_compl_inf_tol and its
      default value is 0.01.
    - acceptable_obj_change_tol (float): "Acceptance" stopping criterion based
      on objective function change. If the relative change of the objective
      function (scaled by Max(1,|f(x)|)) is less than this value, this part of
      the acceptable tolerance termination is satisfied; see also
      acceptable_tol. This is useful for the quasi-Newton option, which has
      trouble to bring down the dual infeasibility. The valid range for this
      real option is 0 ≤ acceptable_obj_change_tol and its default value is
      10+20.

    - diverging_iterates_tol (float): Threshold for maximal value of primal
      iterates. If any component of the primal iterates exceeded this value (in
      absolute terms), the optimization is aborted with the exit message that
      the iterates seem to be diverging. The valid range for this real option is
      0 < diverging_iterates_tol and its default value is 10+20.
    - nlp_lower_bound_inf (float): any bound less or equal this value will be
      considered -inf (i.e. not lwer bounded). The valid range for this real
      option is unrestricted and its default value is -10+19.
    - nlp_upper_bound_inf (float): any bound greater or this value will be
      considered +inf (i.e. not upper bunded). The valid range for this real
      option is unrestricted and its default value is 10+19.
    - fixed_variable_treatment (str): Determines how fixed variables should be
      handled. The main difference between those options is that the starting
      point in the "make_constraint" case still has the fixed variables at their
      given values, whereas in the case "make_parameter(_nodual)" the functions
      are always evaluated with the fixed values for those variables. Also, for
      "relax_bounds", the fixing bound constraints are relaxed (according to"
      bound_relax_factor"). For all but "make_parameter_nodual", bound
      multipliers are computed for the fixed variables. The default value for
      this string option is "make_parameter". Possible values:
        - make_parameter: Remove fixed variable from optimization variables
        - make_parameter_nodual: Remove fixed variable from optimization
            variables and do not compute bound multipliers for fixed variables
        - make_constraint: Add equality constraints fixing variables
        - relax_bounds: Relax fixing bound constraints
    - dependency_detector (str): Indicates which linear solver should be used to
        detect lnearly dependent equality constraints. This is experimental and
        does not work well. The default value for this string option is "none".
        Possible values:
        - none: don't check; no extra work at beginning
        - mumps: use MUMPS
        - wsmp: use WSMP
        - ma28: use MA28
    - dependency_detection_with_rhs (str or bool): Indicates if the right hand
      sides of the constraints should be considered in addition to gradients
      during dependency detection. The default value for this string option is
      "no". Possible values: 'yes', 'no', True, False.

    - kappa_d (float): Weight for linear damping term (to handle one-sided
      bounds). See Section 3.7 in implementation paper. The valid range for this
      real option is 0 ≤ kappa_d and its default value is 10-05.
    - bound_relax_factor (float): Factor for initial relaxation of the bounds.
      Before start of the optimization, the bounds given by the user are
      relaxed. This option sets the factor for this relaxation. Additional, the
      constraint violation tolerance constr_viol_tol is used to bound the
      relaxation by an absolute value. If it is set to zero, then then bounds
      relaxation is disabled. See Eqn.(35) in implementation paper. Note that
      the constraint violation reported by Ipopt at the end of the solution
      process does not include violations of the original (non-relaxed) variable
      bounds. See also option honor_original_bounds. The valid range for this
      real option is 0 ≤ bound_relax_factor and its default value is 10-08.
    - honor_original_bounds (str or bool): Indicates whether final points should
      be projected into original bunds. Ipopt might relax the bounds during the
      optimization (see, e.g., option "bound_relax_factor"). This option
      determines whether the final point should be projected back into the
      user-provide original bounds after the optimization. Note that violations
      of constraints and complementarity reported by Ipopt at the end of the
      solution process are for the non-projected point. The default value for
      this string option is "no". Possible values: 'yes', 'no', True, False

    - check_derivatives_for_naninf (str): whether to check for NaN / inf in the
      derivative matrices. Activating this option will cause an error if an
      invalid number is detected in the constraint Jacobians or the Lagrangian
      Hessian. If this is not activated, the test is skipped, and the algorithm
      might proceed with invalid numbers and fail. If test is activated and an
      invalid number is detected, the matrix is written to output with
      print_level corresponding to J_MORE_DETAILED; so beware of large output!
      The default value for this string option is "no".
    - jac_c_constant (str or bool): Indicates whether to assume that all
      equality constraints are linear Activating this option will cause Ipopt to
      ask for the Jacobian of the equality constraints only once from the NLP
      and reuse this information later. The default value for this string option
      is "no". Possible values: yes, no, True, False.
    - jac_d_constant (str or bool): Indicates whether to assume that all
      inequality constraints are linear Activating this option will cause Ipopt
      to ask for the Jacobian of the inequality constraints only once from the
      NLP and reuse this information later. The default value for this string
      option is "no". Possible values: yes, no, True, False
    - hessian_constant: Indicates whether to assume the problem is a QP
      (quadratic objective, linear constraints). Activating this option will
      cause Ipopt to ask for the Hessian of the Lagrangian function only once
      from the NLP and reuse this information later. The default value for this
      string option is "no". Possible values: yes, no, True, False

    - bound_push (float): Desired minimum absolute distance from the initial
      point to bound. Determines how much the initial point might have to be
      modified in order to be sufficiently inside the bounds (together with
      "bound_frac"). (This is kappa_1 in Section 3.6 of implementation paper.)
      The valid range for this real option is 0 < bound_push and its default
      value is 0.01.
    - bound_frac (float): Desired minimum relative distance from the initial
      point to bound. Determines how much the initial point might have to be
      modified in order to be sufficiently inside the bounds (together with
      "bound_push"). (This is kappa_2 in Section 3.6 of implementation paper.)
      The valid range for this real option is 0 < bound_frac ≤ 0.5 and its
      default value is 0.01.
    - slack_bound_push (float): Desired minimum absolute distance from the
      initial slack to bound. Determines how much the initial slack variables
      might have to be modified in order to be sufficiently inside the
      inequality bounds (together with "slack_bound_frac"). (This is kappa_1 in
      Section 3.6 of implementation paper.) The valid range for this real option
      is 0 < slack_bound_push and its default value is 0.01.
    - slack_bound_frac (float): Desired minimum relative distance from the
      initial slack to bound. Determines how much the initial slack variables
      might have to be modified in order to be sufficiently inside the
      inequality bounds (together with "slack_bound_push"). (This is kappa_2 in
      Section 3.6 of implementation paper.) The valid range for this real option
      is 0 < slack_bound_frac ≤ 0.5 and its default value is 0.01.
    - constr_mult_init_max (float): Maximum allowed least-square guess of
      constraint multipliers. Determines how large the initial least-square
      guesses of the constraint multipliers are allowed to be (in max-norm). If
      the guess is larger than this value, it is discarded and all constraint
      multipliers are set to zero. This options is also used when initializing
      the restoration phase. By default, "resto.constr_mult_init_max" (the one
      used in RestoIterateInitializer) is set to zero. The valid range for this
      real option is 0 ≤ constr_mult_init_max and its default value is 1000.
    - bound_mult_init_val (float): Initial value for the bound multipliers. All
      dual variables corresponding to bound constraints are initialized to this
      value. The valid range for this real option is 0 < bound_mult_init_val and
      its default value is 1.
    - bound_mult_init_method (str): Initialization method for bound multipliers
      This option defines how the iterates for the bound multipliers are
      initialized. If "constant" is chosen, then all bound multipliers are
      initialized to the value of "bound_mult_init_val". If "mu-based" is
      chosen, the each value is initialized to the the value of "mu_init"
      divided by the corresponding slack variable. This latter option might be
      useful if the starting point is close to the optimal solution. The default
      value for this string option is "constant". Possible values:
        - "constant": set all bound multipliers to the value of
          bound_mult_init_val
        - "mu-based": initialize to mu_init/x_slack
    - least_square_init_primal (str or bool): Least square initialization of the
      primal variables. If set to yes, Ipopt ignores the user provided point and
      solves a least square problem for the primal variables (x and s) to fit
      the linearized equality and inequality constraints.This might be useful if
      the user doesn't know anything about the starting point, or for solving an
      LP or QP. The default value for this string option is "no".  Possible
      values:
        - "no": take user-provided point
        - "yes": overwrite user-provided point with least-square estimates
    - least_square_init_duals: Least square initialization of all dual variables
      If set to yes, Ipopt tries to compute least-square multipliers
      (considering ALL dual variables). If successful, the bound multipliers are
      possibly corrected to be at least bound_mult_init_val. This might be
      useful if the user doesn't know anything about the starting point, or for
      solving an LP or QP. This overwrites option "bound_mult_init_method". The
      default value for this string option is "no". Possible values:
        - "no": use bound_mult_init_val and least-square equality constraint
          multipliers
        - "yes": overwrite user-provided point with least-square estimates
    - warm_start_init_point (str or bool): Warm-start for initial point
      Indicates whether this optimization should use a warm start
      initialization, where values of primal and dual variables are given (e.g.,
      from a previous optimization of a related problem.) The default value for
      this string option is "no". Possible values:
        - "no" or False: do not use the warm start initialization
        - "yes" or True: use the warm start initialization
    - warm_start_same_structure (str or bool): Advanced feature! Indicates
      whether a problem with a structure identical t the previous one is to be
      solved. If enabled, then the algorithm assumes that an NLP is now to be
      solved whose structure is identical to one that already was considered
      (with the same NLP object). The default value for this string option is
      "no". Possible values: yes, no, True, False.
    - warm_start_bound_push (float): same as bound_push for the regular
      initializer. The valid range for this real option is 0 <
      warm_start_bound_push and its default value is 0.001.
    - warm_start_bound_frac (float): same as bound_frac for the regular
      initializer The valid range for this real option is 0 <
      warm_start_bound_frac ≤ 0.5 and its default value is 0.001.
    - warm_start_slack_bound_push (float): same as slack_bound_push for the
      regular initializer The valid range for this real option is 0 <
      warm_start_slack_bound_push and its default value is 0.001.
    - warm_start_slack_bound_frac (float): same as slack_bound_frac for the
      regular initializer The valid range for this real option is 0 <
      warm_start_slack_bound_frac ≤ 0.5 and its default value is 0.001.
    - warm_start_mult_bound_push (float): same as mult_bound_push for the
      regular initializer The valid range for this real option is 0 <
      warm_start_mult_bound_push and its default value is 0.001.
    - warm_start_mult_init_max (float): Maximum initial value for the equality
      multipliers. The valid range for this real option is unrestricted and its
      default value is 10+06.
    - warm_start_entire_iterate (str or bool): Tells algorithm whether to use
      the GetWarmStartIterate method in the NLP. The default value for this
      string option is "no".  Possible values:
        - "no": call GetStartingPoint in the NLP
        - "yes": call GetWarmStartIterate in the NLP
    - warm_start_target_mu (float): Advanced and experimental! The valid range
        for this real option is unrestricted and its default value is 0.

    - option_file_name (str): File name of options file. By default, the name of
      the Ipopt options file is "ipopt.opt" - or something else if specified in
      the IpoptApplication::Initialize call. If this option is set by
      SetStringValue BEFORE the options file is read, it specifies the name of
      the options file. It does not make any sense to specify this option within
      the options file. Setting this option to an empty string disables reading
      of an options file.
    - replace_bounds (bool or str): Whether all variable bounds should be
      replaced by inequality constraints. This option must be set for the
      inexact algorithm. The default value for this string option is "no".
      Possible values: "yes", "no", True, False.
    - skip_finalize_solution_call (str or bool): Whether a call to
      NLP::FinalizeSolution after optimization should be suppressed.    In some
      Ipopt applications, the user might want to call the FinalizeSolution
      method separately. Setting this option to "yes" will cause the
      IpoptApplication object to suppress the default call to that method. The
      default value for this string option is "no". Possible values: "yes",
      "no", True, False
    - timing_statistics (str or bool): Indicates whether to measure time spend
      in components of Ipopt and NLP evaluation.  The overall algorithm time is
      unaffected by this option. The default value for this string option is
      "no". Possible values: "yes", "no", True, False

    - mu_max_fact (float): Factor for initialization of maximum value for
        barrier parameter. This option determines the upper bound on the barrier
        parameter. This upper bound is computed as the average complementarity
        at the initial point times the value of this option. (Only used if
        option "mu_strategy" is chosen as "adaptive".) The valid range for this
        real option is 0 < mu_max_fact and its default value is 1000.
    - mu_max (float): Maximum value for barrier parameter. This option specifies
        an upper bound on the barrier parameter in the adaptive mu selection
        mode. If this option is set, it overwrites the effect of mu_max_fact.
        (Only used if option "mu_strategy" is chosen as "adaptive".) The valid
        range for this real option is 0 < mu_max and its default value is
        100000.
    - mu_min (float): Minimum value for barrier parameter. This option specifies
        the lower bound on the barrier parameter in the adaptive mu selection
        mode. By default, it is set to the minimum of 1e-11 and
        min("tol","compl_inf_tol")/("barrier_tol_factor"+1), which should be a
        reasonable value. (Only used if option "mu_strategy" is chosen as
        "adaptive".) The valid range for this real option is 0 < mu_min and its
        default value is 10-11.
    - adaptive_mu_globalization (str): Globalization strategy for the adaptive
        mu selection mode. To achieve global convergence of the adaptive
        version, the algorithm has to switch to the monotone mode
        (Fiacco-McCormick approach) when convergence does not seem to appear.
        This option sets the criterion used to decide when to do this switch.
        (Only used if option "mu_strategy" is chosen as "adaptive".) The default
        value for this string option is "obj-constr-filter". Possible values: -
        "kkt-error": nonmonotone decrease of kkt-error - "obj-constr-filter":
        2-dim filter for objective and constraint violation -
        "never-monotone-mode": disables globalization.
    - adaptive_mu_kkterror_red_iters (float): advanced feature! Maximum number
      of iterations requiring sufficient pogress. For the "kkt-error" based
      globalization strategy, sufficient progress must be made for
      "adaptive_mu_kkterror_red_iters" iterations. If this number of iterations
      is exceeded, the globalization strategy switches to the monotone mode. The
      valid range for this integer option is 0 ≤ adaptive_mu_kkterror_red_iters
      and its default value is 4.
    - adaptive_mu_kkterror_red_fact (float): advanced feature! Sufficient
      decrease factor for "kkt-error" gobalization strategy. For the "kkt-error"
      based globalization strategy, the error must decrease by this factor to be
      deemed sufficient decrease. The valid range for this real option is 0 <
      adaptive_mu_kkterror_red_fact < 1 and its default value is 0.9999.
    - filter_margin_fact (float): advanced feature! Factor determining width of
      margin for obj-constr-filter aaptive globalization strategy. When using
      the adaptive globalization strategy, "obj-constr-filter", sufficient
      progress for a filter entry is defined as follows: (new obj) < (filter
      obj) - filter_margin_fact*(new constr-viol) OR (new constr-viol) < (filter
      constr-viol) - filter_margin_fact*(new constr-viol). For the description
      of the "kkt-error-filter" option see "filter_max_margin". The valid range
      for this real option is 0 < filter_margin_fact < 1 and its default value
      is 10-05.
    - filter_max_margin (float): advanced feature! Maximum width of margin in
      obj-constr-filter adaptive globalization strategy. The valid range for
      this real option is 0 < filter_max_margin and its default value is 1.
    - adaptive_mu_restore_previous_iterate (str or bool): advanced feature!
      Indicates if the previous accepted iterate should be restored if the
      monotone mode is entered. When the globalization strategy for the adaptive
      barrier algorithm switches to the monotone mode, it can either start from
      the most recent iterate (no), or from the last iterate that was accepted
      (yes). The default value for this string option is "no". Possible values:
      "yes", "no", True, False
    - adaptive_mu_monotone_init_factor (float): advanced feature! Determines the
      initial value of the barrier parameter when switching to the monotone
      mode. When the globalization strategy for the adaptive barrier algorithm
      switches to the monotone mode and fixed_mu_oracle is chosen as
      "average_compl", the barrier parameter is set to the current average
      complementarity times the value of "adaptive_mu_monotone_init_factor". The
      valid range for this real option is 0 < adaptive_mu_monotone_init_factor
      and its default value is 0.8.
    - adaptive_mu_kkt_norm_type (advanced): Norm used for the KKT error in the
      adaptive mu globalization strategies. When computing the KKT error for the
      globalization strategies, the norm to be used is specified with this
      option. Note, this option is also used in the QualityFunctionMuOracle. The
      default value for this string option is "2-norm-squared". Possible values:
        - "1-norm": use the 1-norm (abs sum)
        - "2-norm-squared": use the 2-norm squared (sum of squares)
        - "max-norm": use the infinity norm (max)
        - "2-norm": use 2-norm
    - mu_strategy: Update strategy for barrier parameter. Determines which
      barrier parameter update strategy is to be used. The default value for
      this string option is "monotone". Possible values:
        - "monotone": use the monotone (Fiacco-McCormick) strategy
        - "adaptive": use the adaptive update strategy
    - mu_oracle: Oracle for a new barrier parameter in the adaptive strategy.
      Determines how a new barrier parameter is computed in each "free-mode"
      iteration of the adaptive barrier parameter strategy. (Only considered if
      "adaptive" is selected for option "mu_strategy"). The default value for
      this string option is "quality-function". Possible values:
        - "probing": Mehrotra's probing heuristic
        - "loqo": LOQO's centrality rule
        - "quality-function": minimize a quality function
    - fixed_mu_oracle (str): Oracle for the barrier parameter when switching to
      fixed mode. Determines how the first value of the barrier parameter should
      be computed when switching to the "monotone mode" in the adaptive
      strategy. (Only considered if "adaptive" is selected for option
      "mu_strategy".) The default value for this string option is
      "average_compl". Possible values:
        - "probing": Mehrotra's probing heuristic
        - "loqo": LOQO's centrality rule
        - "quality-function": minimize a quality function
        - "average_compl": base on current average complementarity
    - mu_init (float): Initial value for the barrier parameter. This option
      determines the initial value for the barrier parameter (mu). It is only
      relevant in the monotone, Fiacco-McCormick version of the algorithm.
      (i.e., if "mu_strategy" is chosen as "monotone") The valid range for this
      real option is 0 < mu_init and its default value is 0.1.
    - barrier_tol_factor (float): Factor for mu in barrier stop test. The
      convergence tolerance for each barrier problem in the monotone mode is the
      value of the barrier parameter times "barrier_tol_factor". This option is
      also used in the adaptive mu strategy during the monotone mode. This is
      kappa_epsilon in implementation paper. The valid range for this real
      option is 0 < barrier_tol_factor and its default value is 10.
    - mu_linear_decrease_factor (float): Determines linear decrease rate of
      barrier parameter. For the Fiacco-McCormick update procedure the new
      barrier parameter mu is obtained by taking the minimum of
      mu*"mu_linear_decrease_factor" and mu^"superlinear_decrease_power". This
      is kappa_mu in implementation paper. This option is also used in the
      adaptive mu strategy during the monotone mode. The valid range for this
      real option is 0 < mu_linear_decrease_factor < 1 and its default value is
      0.2.
    - mu_superlinear_decrease_power (float): Determines superlinear decrease
      rate of barrier parameter. For the Fiacco-McCormick update procedure the
      new barrier parameter mu is obtained by taking the minimum of
      mu*"mu_linear_decrease_factor" and mu^"superlinear_decrease_power". This
      is theta_mu in implementation paper. This option is also used in the
      adaptive mu strategy during the monotone mode. The valid range for this
      real option is 1 < mu_superlinear_decrease_power < 2 and its default value
      is 1.5.
    - mu_allow_fast_monotone_decrease (str or bool): Advanced feature! Allow
      skipping of barrier problem if barrier test i already met. The default
      value for this string option is "yes". Possible values:
        - "no": Take at least one iteration per barrier problem even if the
          barrier test is already met for the updated barrier parameter
        - "yes": Allow fast decrease of mu if barrier test it met
    - tau_min (float): Advanced feature! Lower bound on fraction-to-the-boundary
        parameter tau. This is tau_min in the implementation paper. This option
        is also used in the adaptive mu strategy during the monotone mode. The
        valid range for this real option is 0 < tau_min < 1 and its default
        value is 0.99.
    - sigma_max (float): Advanced feature! Maximum value of the centering
        parameter. This is the upper bound for the centering parameter chosen by
        the quality function based barrier parameter update. Only used if option
        "mu_oracle" is set to "quality-function". The valid range for this real
        option is 0 < sigma_max and its default value is 100.
    - sigma_min (float): Advanced feature! Minimum value of the centering
        parameter. This is the lower bound for the centering parameter chosen by
        the quality function based barrier parameter update. Only used if option
        "mu_oracle" is set to "quality-function". The valid range for this real
        option is 0 ≤ sigma_min and its default value is 10-06.
    - quality_function_norm_type (str): Advanced feature. Norm used for
      components of the quality function. Only used if option "mu_oracle" is set
      to "quality-function". The default value for this string option is
      "2-norm-squared". Possible values:
        - "1-norm": use the 1-norm (abs sum)
        - "2-norm-squared": use the 2-norm squared (sum of squares)
        - "max-norm": use the infinity norm (max)
        - "2-norm": use 2-norm
    - quality_function_centrality (str): Advanced feature. The penalty term for
      centrality that is included in qality function. This determines whether a
      term is added to the quality function to penalize deviation from
      centrality with respect to complementarity. The complementarity measure
      here is the xi in the Loqo update rule. Only used if option "mu_oracle" is
      set to "quality-function". The default value for this string option is
      "none". Possible values:
        - "none": no penalty term is added
        - "log": complementarity * the log of the centrality measure
        - "reciprocal": complementarity * the reciprocal of the centrality
          measure
        - "cubed-reciprocal": complementarity * the reciprocal of the centrality
          measure cubed
    - quality_function_balancing_term (str): Advanced feature. The balancing
      term included in the quality function for centrality. This determines
      whether a term is added to the quality function that penalizes situations
      where the complementarity is much smaller than dual and primal
      infeasibilities. Only used if option "mu_oracle" is set to
      "quality-function". The default value for this string option is "none".
      Possible values:
        - "none": no balancing term is added
        - "cubic": Max(0,Max(dual_inf,primal_inf)-compl)^3
    - quality_function_max_section_steps (int): Maximum number of search steps
      during direct search procedure determining the optimal centering
      parameter. The golden section search is performed for the quality function
      based mu oracle. Only used if option "mu_oracle" is set to
      "quality-function". The valid range for this integer option is 0 ≤
      quality_function_max_section_steps and its default value is 8.
    - quality_function_section_sigma_tol (float): advanced feature! Tolerance
      for the section search procedure determining the optimal centering
      parameter (in sigma space). The golden section search is performed for the
      quality function based mu oracle. Only used if option "mu_oracle" is set
      to "quality-function". The valid range for this real option is 0 ≤
      quality_function_section_sigma_tol < 1 and its default value is 0.01.
    - quality_function_section_qf_tol (float): advanced feature! Tolerance for
      the golden section search procedure determining the optimal centering
      parameter (in the function value space). The golden section search is
      performed for the quality function based mu oracle. Only used if option
      "mu_oracle" is set to "quality-function". The valid range for this real
      option is 0 ≤ quality_function_section_qf_tol < 1 and its default value is
      0.

    The following options are not supported:

        - num_linear_variables: since estimagic may reparametrize your problem
          and this changes the parameter problem, we do not support
          num_linear_variables.

        - scaling options (nlp_scaling_method, obj_scaling_factor,
          nlp_scaling_max_gradient, nlp_scaling_obj_target_gradient,
          nlp_scaling_constr_target_gradient, nlp_scaling_min_value)

    """
    if not IS_CYIPOPT_INSTALLED:
        raise NotImplementedError(
            "The cyipopt package is not installed and required for 'ipopt'. You can "
            "install the package with: `conda install -c conda-forge cyipopt`"
        )
    if acceptable_tol <= convergence_relative_criterion_tolerance:
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

    dependency_detection_with_rhs = convert_bool_to_str(
        dependency_detection_with_rhs, "dependency_detection_with_rhs"
    )
    dependency_detector = "none" if dependency_detector is None else dependency_detector
    if dependency_detector not in {"none", "mumps", "wsmp", "ma28"}:
        raise ValueError(
            "dependency_detector must be one of 'none', 'mumps', 'wsmp', 'ma28' or "
            f"None. You specified {dependency_detector}."
        )
    check_derivatives_for_naninf = convert_bool_to_str(
        check_derivatives_for_naninf, "check_derivatives_for_naninf"
    )
    jac_c_constant = convert_bool_to_str(jac_c_constant, "jac_c_constant")
    jac_d_constant = convert_bool_to_str(jac_d_constant, "jac_d_constant")
    hessian_constant = convert_bool_to_str(hessian_constant, "hessian_constant")
    if bound_mult_init_method not in {"constant", "mu-based"}:
        raise ValueError(
            f"You specified {bound_mult_init_method} as bound_mult_init_method. "
            "It must be 'constant' or 'mu-based'."
        )
    least_square_init_primal = convert_bool_to_str(
        least_square_init_primal, "least_square_init_primal"
    )
    least_square_init_duals = convert_bool_to_str(
        least_square_init_duals, "least_square_init_duals"
    )
    warm_start_init_point = convert_bool_to_str(
        warm_start_init_point, "warm_start_init_point"
    )
    warm_start_same_structure = convert_bool_to_str(
        warm_start_same_structure, "warm_start_same_structure"
    )
    warm_start_entire_iterate = convert_bool_to_str(
        warm_start_entire_iterate, "warm_start_entire_iterate"
    )
    replace_bounds = convert_bool_to_str(replace_bounds, "replace_bounds")
    skip_finalize_solution_call = convert_bool_to_str(
        skip_finalize_solution_call, "skip_finalize_solution_call"
    )
    timing_statistics = convert_bool_to_str(timing_statistics, "timing_statistics")
    adaptive_mu_restore_previous_iterate = convert_bool_to_str(
        adaptive_mu_restore_previous_iterate, "adaptive_mu_restore_previous_iterate"
    )
    mu_allow_fast_monotone_decrease = convert_bool_to_str(
        mu_allow_fast_monotone_decrease, "mu_allow_fast_monotone_decrease"
    )

    algo_info = DEFAULT_ALGO_INFO.copy()
    algo_info["name"] = "ipopt"

    gradient = functools.partial(
        criterion_and_derivative, task="derivative", algorithm_info=algo_info
    )

    func = functools.partial(
        criterion_and_derivative,
        task="criterion",
        algorithm_info=algo_info,
    )

    options = {
        "print_level": 0,  # disable verbosity
        "nlp_scaling_method": "none",  # disable scaling
        #
        "s_max": float(s_max),
        "max_iter": stopping_max_iterations,
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
        "dependency_detector": dependency_detector,
        "dependency_detection_with_rhs": dependency_detection_with_rhs,
        "kappa_d": kappa_d,
        "bound_relax_factor": bound_relax_factor,
        "honor_original_bounds": honor_original_bounds,
        # derivatives
        "check_derivatives_for_naninf": check_derivatives_for_naninf,
        "jac_c_constant": jac_c_constant,
        "jac_d_constant": jac_d_constant,
        "hessian_constant": hessian_constant,
        # initialization
        "bound_push": bound_push,
        "bound_frac": bound_frac,
        "slack_bound_push": slack_bound_push,
        "slack_bound_frac": slack_bound_frac,
        "constr_mult_init_max": float(constr_mult_init_max),
        "bound_mult_init_val": float(bound_mult_init_val),
        "bound_mult_init_method": bound_mult_init_method,
        "least_square_init_primal": least_square_init_primal,
        "least_square_init_duals": least_square_init_duals,
        # warm start
        "warm_start_init_point": warm_start_init_point,
        "warm_start_same_structure": warm_start_same_structure,
        "warm_start_bound_push": warm_start_bound_push,
        "warm_start_bound_frac": warm_start_bound_frac,
        "warm_start_slack_bound_push": warm_start_slack_bound_push,
        "warm_start_slack_bound_frac": warm_start_slack_bound_frac,
        "warm_start_mult_bound_push": warm_start_mult_bound_push,
        "warm_start_mult_init_max": warm_start_mult_init_max,
        "warm_start_entire_iterate": warm_start_entire_iterate,
        "warm_start_target_mu": warm_start_target_mu,
        # more miscellaneous
        "option_file_name": option_file_name,
        "replace_bounds": replace_bounds,
        "skip_finalize_solution_call": skip_finalize_solution_call,
        "timing_statistics": timing_statistics,
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
        "adaptive_mu_restore_previous_iterate": adaptive_mu_restore_previous_iterate,
        "adaptive_mu_monotone_init_factor": adaptive_mu_monotone_init_factor,
        "adaptive_mu_kkt_norm_type": adaptive_mu_kkt_norm_type,
        "mu_strategy": mu_strategy,
        "mu_oracle": mu_oracle,
        "fixed_mu_oracle": fixed_mu_oracle,
        "mu_init": mu_init,
        "barrier_tol_factor": float(barrier_tol_factor),
        "mu_linear_decrease_factor": mu_linear_decrease_factor,
        "mu_superlinear_decrease_power": mu_superlinear_decrease_power,
        "mu_allow_fast_monotone_decrease": mu_allow_fast_monotone_decrease,
        "tau_min": tau_min,
        "sigma_max": float(sigma_max),
        "sigma_min": sigma_min,
        "quality_function_norm_type": quality_function_norm_type,
        "quality_function_centrality": quality_function_centrality,
        "quality_function_balancing_term": quality_function_balancing_term,
        "quality_function_max_section_steps": int(quality_function_max_section_steps),
        "quality_function_section_sigma_tol": quality_function_section_sigma_tol,
        "quality_function_section_qf_tol": quality_function_section_qf_tol,
    }

    raw_res = cyipopt.minimize_ipopt(
        fun=func,
        x0=x,
        bounds=get_scipy_bounds(lower_bounds, upper_bounds),
        jac=gradient,
        constraints=(),
        tol=convergence_relative_criterion_tolerance,
        options=options,
    )

    res = process_scipy_result(raw_res)
    return res


def convert_bool_to_str(var, name):
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
