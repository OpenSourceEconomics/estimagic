(list_of_algorithms)=

# Optimizers

Check out {ref}`algorithms` to see how to select an algorithm and specify `algo_options`
when using `maximize` or `minimize`.

## Optimizers from scipy

(scipy-algorithms)=

optimagic supports most `scipy` algorithms and scipy is automatically installed when you
install optimagic.

```{eval-rst}
.. dropdown::  scipy_lbfgsb

    .. code-block::

        "scipy_lbfgsb"

    Minimize a scalar function of one or more variables using the L-BFGS-B algorithm.

    The optimizer is taken from scipy, which calls the Fortran code written by the
    original authors of the algorithm. The Fortran code includes the corrections
    and improvements that were introduced in a follow up paper.

    lbfgsb is a limited memory version of the original bfgs algorithm, that deals with
    lower and upper bounds via an active set approach.

    The lbfgsb algorithm is well suited for differentiable scalar optimization problems
    with up to several hundred parameters.

    It is a quasi-newton line search algorithm. At each trial point it evaluates the
    criterion function and its gradient to find a search direction. It then approximates
    the hessian using the stored history of gradients and uses the hessian to calculate
    a candidate step size. Then it uses a gradient based line search algorithm to
    determine the actual step length. Since the algorithm always evaluates the gradient
    and criterion function jointly, the user should provide a
    ``criterion_and_derivative`` function that exploits the synergies in the
    calculation of criterion and gradient.

    The lbfgsb algorithm is almost perfectly scale invariant. Thus, it is not necessary
    to scale the parameters.

    - **convergence.ftol_rel** (float): Stop when the relative improvement
      between two iterations is smaller than this. More formally, this is expressed as

    .. math::

        \frac{(f^k - f^{k+1})}{\\max{{|f^k|, |f^{k+1}|, 1}}} \leq
        \text{relative_criterion_tolerance}


    - **convergence.gtol_abs** (float): Stop if all elements of the projected
      gradient are smaller than this.
    - **stopping.maxfun** (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this as convergence.
    - **stopping.maxiter** (int): If the maximum number of iterations is reached,
      the optimization stops, but we do not count this as convergence.
    - **limited_memory_storage_length** (int): Maximum number of saved gradients used to approximate the hessian matrix.

```

```{eval-rst}
.. dropdown::  scipy_slsqp

    .. code-block::

        "scipy_slsqp"

    Minimize a scalar function of one or more variables using the SLSQP algorithm.

    SLSQP stands for Sequential Least Squares Programming.

    SLSQP is a line search algorithm. It is well suited for continuously
    differentiable scalar optimization problems with up to several hundred parameters.

    The optimizer is taken from scipy which wraps the SLSQP optimization subroutine
    originally implemented by :cite:`Kraft1988`.

    .. note::
        SLSQP's general nonlinear constraints are not supported yet by optimagic.

    - **convergence.ftol_abs** (float): Precision goal for the value of
      f in the stopping criterion.
    - **stopping.maxiter** (int): If the maximum number of iterations is reached,
      the optimization stops, but we do not count this as convergence.

```

```{eval-rst}
.. dropdown::  scipy_neldermead

    .. code-block::

      "scipy_neldermead"

    Minimize a scalar function using the Nelder-Mead algorithm.

    The Nelder-Mead algorithm is a direct search method (based on function comparison)
    and is often applied to nonlinear optimization problems for which derivatives are
    not known.
    Unlike most modern optimization methods, the Nelder–Mead heuristic can converge to
    a non-stationary point, unless the problem satisfies stronger conditions than are
    necessary for modern methods.

    Nelder-Mead is never the best algorithm to solve a problem but rarely the worst.
    Its popularity is likely due to historic reasons and much larger than its
    properties warrant.

    The argument `initial_simplex` is not supported by optimagic as it is not
    compatible with optimagic's handling of constraints.

    - **stopping.maxiter** (int): If the maximum number of iterations is reached, the optimization stops,
      but we do not count this as convergence.
    - **stopping.maxfun** (int): If the maximum number of function evaluation is reached,
      the optimization stops but we do not count this as convergence.
    - **convergence.xtol_abs** (float): Absolute difference in parameters between iterations
      that is tolerated to declare convergence. As no relative tolerances can be passed to Nelder-Mead,
      optimagic sets a non zero default for this.
    - **convergence.ftol_abs** (float): Absolute difference in the criterion value between
      iterations that is tolerated to declare convergence. As no relative tolerances can be passed to Nelder-Mead,
      optimagic sets a non zero default for this.
    - **adaptive** (bool): Adapt algorithm parameters to dimensionality of problem.
      Useful for high-dimensional minimization (:cite:`Gao2012`, p. 259-277). scipy's default is False.

```

```{eval-rst}
.. dropdown::  scipy_powell

   .. code-block::

       "scipy_powell"

   Minimize a scalar function using the modified Powell method.

    .. warning::
        In our benchmark using a quadratic objective function, the Powell algorithm
        did not find the optimum very precisely (less than 4 decimal places).
        If you require high precision, you should refine an optimum found with Powell
        with another local optimizer.

    The criterion function need not be differentiable.

    Powell's method is a conjugate direction method, minimizing the function by a
    bi-directional search in each parameter's dimension.

    The argument ``direc``, which is the initial set of direction vectors and which
    is part of the scipy interface is not supported by optimagic because it is
    incompatible with how optimagic handles constraints.

    - **convergence.xtol_rel (float)**: Stop when the relative movement between parameter
      vectors is smaller than this.
    - **convergence.ftol_rel** (float): Stop when the relative improvement between two
      iterations is smaller than this. More formally, this is expressed as

        .. math::

            \frac{(f^k - f^{k+1})}{\\max{{\{|f^k|, |f^{k+1}|, 1\}}}} \leq
            \text{relative_criterion_tolerance}

    - **stopping.maxfun** (int): If the maximum number of function evaluation is reached,
      the optimization stops but we do not count thisas convergence.
    - **stopping.maxiter** (int): If the maximum number of iterations is reached, the optimization stops,
      but we do not count this as convergence.

```

```{eval-rst}
.. dropdown::  scipy_bfgs

    .. code-block::

        "scipy_bfgs"

    Minimize a scalar function of one or more variables using the BFGS algorithm.

    BFGS stands for Broyden-Fletcher-Goldfarb-Shanno algorithm. It is a quasi-Newton
    method that can be used for solving unconstrained nonlinear optimization problems.

    BFGS is not guaranteed to converge unless the function has a quadratic Taylor
    expansion near an optimum. However, BFGS can have acceptable performance even
    for non-smooth optimization instances.

    - **convergence.gtol_abs** (float): Stop if all elements of the gradient are smaller than this.
    - **stopping.maxiter** (int): If the maximum number of iterations is reached, the optimization stops,
      but we do not count this as convergence.
    - **norm** (float): Order of the vector norm that is used to calculate the gradient's "score" that
      is compared to the gradient tolerance to determine convergence. Default is infinite which means that
      the largest entry of the gradient vector is compared to the gradient tolerance.

```

```{eval-rst}
.. dropdown::  scipy_conjugate_gradient

    .. code-block::

        "scipy_conjugate_gradient"

    Minimize a function using a nonlinear conjugate gradient algorithm.

    The conjugate gradient method finds functions' local optima using just the gradient.

    This conjugate gradient algorithm is based on that of Polak and Ribiere, detailed
    in :cite:`Nocedal2006`, pp. 120-122.

    Conjugate gradient methods tend to work better when:

      - the criterion has a unique global minimizing point, and no local minima or
        other stationary points.
      - the criterion is, at least locally, reasonably well approximated by a
        quadratic function.
      - the criterion is continuous and has a continuous gradient.
      - the gradient is not too large, e.g., has a norm less than 1000.
      - The initial guess is reasonably close to the criterion's global minimizer.

    - **convergence.gtol_abs** (float): Stop if all elements of the
      gradient are smaller than this.
    - **stopping.maxiter** (int): If the maximum number of iterations is reached,
      the optimization stops, but we do not count this as convergence.
    - **norm** (float): Order of the vector norm that is used to calculate the gradient's
      "score" that is compared to the gradient tolerance to determine convergence.
      Default is infinite which means that the largest entry of the gradient vector
      is compared to the gradient tolerance.

```

```{eval-rst}
.. dropdown::  scipy_newton_cg

    .. code-block::

        "scipy_newton_cg"

    Minimize a scalar function using Newton's conjugate gradient algorithm.

    .. warning::
        In our benchmark using a quadratic objective function, the truncated newton
        algorithm did not find the optimum very precisely (less than 4 decimal places).
        If you require high precision, you should refine an optimum found with Powell
        with another local optimizer.

    Newton's conjugate gradient algorithm uses an approximation of the Hessian to find
    the minimum of a function. It is practical for small and large problems
    (see :cite:`Nocedal2006`, p. 140).

    Newton-CG methods are also called truncated Newton methods. This function differs
    scipy_truncated_newton because

    - ``scipy_newton_cg``'s algorithm is written purely in Python using NumPy
      and scipy while ``scipy_truncated_newton``'s algorithm calls a C function.

    - ``scipy_newton_cg``'s algorithm is only for unconstrained minimization
      while ``scipy_truncated_newton``'s algorithm supports bounds.

    Conjugate gradient methods tend to work better when:

      - the criterion has a unique global minimizing point, and no local minima or
        other stationary points.
      - the criterion is, at least locally, reasonably well approximated by a
        quadratic function.
      - the criterion is continuous and has a continuous gradient.
      - the gradient is not too large, e.g., has a norm less than 1000.
      - The initial guess is reasonably close to the criterion's global minimizer.

    - **convergence.xtol_rel** (float): Stop when the relative movement
      between parameter vectors is smaller than this. Newton CG uses the average
      relative change in the parameters for determining the convergence.
    - **stopping.maxiter** (int): If the maximum number of iterations is reached,
      the optimization stops, but we do not count this as convergence.



```

```{eval-rst}
.. dropdown::  scipy_cobyla

  .. code-block::

      "scipy_cobyla"

  Minimize a scalar function of one or more variables using the COBYLA algorithm.

  COBYLA stands for Constrained Optimization By Linear Approximation.
  It is derivative-free and supports nonlinear inequality and equality constraints.

  .. note::
      Cobyla's general nonlinear constraints is not supported yet by optimagic.

  Scipy's implementation wraps the FORTRAN implementation of the algorithm.

  For more information on COBYLA see :cite:`Powell1994`, :cite:`Powell1998` and
  :cite:`Powell2007`.

  - **stopping.maxiter** (int): If the maximum number of iterations is reached,
    the optimization stops, but we do not count this as convergence.
  - **convergence.xtol_rel** (float): Stop when the relative movement
    between parameter vectors is smaller than this. In case of COBYLA this is
    a lower bound on the size of the trust region and can be seen as the
    required accuracy in the variables but this accuracy is not guaranteed.
  - **trustregion.initial_radius** (float): Initial value of the trust region radius.
    Since a linear approximation is likely only good near the current simplex,
    the linear program is given the further requirement that the solution,
    which will become the next evaluation point must be within a radius
    RHO_j from x_j. RHO_j only decreases, never increases. The initial RHO_j is
    the `trustregion.initial_radius`. In this way COBYLA's iterations behave
    like a trust region algorithm.

```

```{eval-rst}
.. dropdown::  scipy_truncated_newton

    .. code-block::

        "scipy_truncated_newton"

    Minimize a scalar function using truncated Newton algorithm.

    This function differs from scipy_newton_cg because

    - ``scipy_newton_cg``'s algorithm is written purely in Python using NumPy
      and scipy while ``scipy_truncated_newton``'s algorithm calls a C function.

    - ``scipy_newton_cg``'s algorithm is only for unconstrained minimization
      while ``scipy_truncated_newton``'s algorithm supports bounds.

    Conjugate gradient methods tend to work better when:

    - the criterion has a unique global minimizing point, and no local minima or
      other stationary points.
    - the criterion is, at least locally, reasonably well approximated by a
      quadratic function.
    - the criterion is continuous and has a continuous gradient.
    - the gradient is not too large, e.g., has a norm less than 1000.
    - The initial guess is reasonably close to the criterion's global minimizer.

    optimagic does not support the ``scale``  nor ``offset`` argument as they are not
    compatible with the way optimagic handles constraints. It also does not support
    ``messg_num`` which is an additional way to control the verbosity of the optimizer.

    - **func_min_estimate** (float): Minimum function value estimate. Defaults to 0.
    - **stopping.maxiter** (int): If the maximum number of iterations is reached,
      the optimization stops, but we do not count this as convergence.
    - **stopping.maxfun** (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this as
      convergence.
    - **convergence.xtol_abs** (float): Absolute difference in parameters
      between iterations after scaling that is tolerated to declare convergence.
    - **convergence.ftol_abs** (float): Absolute difference in the
      criterion value between iterations after scaling that is tolerated
      to declare convergence.
    - **convergence.gtol_abs** (float): Stop if the value of the
      projected gradient (after applying x scaling factors) is smaller than this.
      If convergence.gtol_abs < 0.0,
      convergence.gtol_abs is set to
      1e-2 * sqrt(accuracy).
    - **max_hess_evaluations_per_iteration** (int): Maximum number of hessian*vector
      evaluations per main iteration. If ``max_hess_evaluations == 0``, the
      direction chosen is ``- gradient``. If ``max_hess_evaluations < 0``,
      ``max_hess_evaluations`` is set to ``max(1,min(50,n/2))`` where n is the
      length of the parameter vector. This is also the default.
    - **max_step_for_line_search** (float): Maximum step for the line search.
      It may be increased during the optimization. If too small, it will be set
      to 10.0. By default we use scipy's default.
    - **line_search_severity** (float): Severity of the line search. If < 0 or > 1,
      set to 0.25. optimagic defaults to scipy's default.
    - **finitie_difference_precision** (float): Relative precision for finite difference
      calculations. If <= machine_precision, set to sqrt(machine_precision).
      optimagic defaults to scipy's default.
    - **criterion_rescale_factor** (float): Scaling factor (in log10) used to trigger
      criterion rescaling. If 0, rescale at each iteration. If a large value,
      never rescale. If < 0, rescale is set to 1.3. optimagic defaults to scipy's
      default.


```

```{eval-rst}
.. dropdown::  scipy_trust_constr

    .. code-block::

        "scipy_trust_constr"

    Minimize a scalar function of one or more variables subject to constraints.

    .. warning::
        In our benchmark using a quadratic objective function, the trust_constr
        algorithm did not find the optimum very precisely (less than 4 decimal places).
        If you require high precision, you should refine an optimum found with Powell
        with another local optimizer.

    .. note::
        Its general nonlinear constraints' handling is not supported yet by optimagic.

    It switches between two implementations depending on the problem definition.
    It is the most versatile constrained minimization algorithm
    implemented in SciPy and the most appropriate for large-scale problems.
    For equality constrained problems it is an implementation of Byrd-Omojokun
    Trust-Region SQP method described in :cite:`Lalee1998` and in :cite:`Conn2000`,
    p. 549. When inequality constraints  are imposed as well, it switches to the
    trust-region interior point method described in :cite:`Byrd1999`.
    This interior point algorithm in turn, solves inequality constraints by
    introducing slack variables and solving a sequence of equality-constrained
    barrier problems for progressively smaller values of the barrier parameter.
    The previously described equality constrained SQP method is
    used to solve the subproblems with increasing levels of accuracy
    as the iterate gets closer to a solution.

    It approximates the Hessian using the Broyden-Fletcher-Goldfarb-Shanno (BFGS)
    Hessian update strategy.

    - **convergence.gtol_abs** (float): Tolerance for termination
      by the norm of the Lagrangian gradient. The algorithm will terminate
      when both the infinity norm (i.e., max abs value) of the Lagrangian
      gradient and the constraint violation are smaller than the
      convergence.gtol_abs.
      For this algorithm we use scipy's gradient tolerance for trust_constr.
      This smaller tolerance is needed for the sum of squares tests to pass.
    - **stopping.maxiter** (int): If the maximum number of iterations is reached,
      the optimization stops, but we do not count this as convergence.
    - **convergence.xtol_rel** (float): Tolerance for termination by
      the change of the independent variable. The algorithm will terminate when
      the radius of the trust region used in the algorithm is smaller than the
      convergence.xtol_rel.
    - **trustregion.initial_radius** (float): Initial value of the trust region radius.
      The trust radius gives the maximum distance between solution points in
      consecutive iterations. It reflects the trust the algorithm puts in the
      local approximation of the optimization problem. For an accurate local
      approximation the trust-region should be large and for an approximation
      valid only close to the current point it should be a small one.
      The trust radius is automatically updated throughout the optimization
      process, with ``trustregion_initial_radius`` being its initial value.

```

```{eval-rst}
.. dropdown::  scipy_ls_dogbox

    .. code-block::

        "scipy_ls_dogbox"

    Minimize a nonlinear least squares problem using a rectangular trust region method.

    Typical use case is small problems with bounds. Not recommended for problems with
    rank-deficient Jacobian.

    The algorithm supports the following options:

    - **convergence.ftol_rel** (float): Stop when the relative
      improvement between two iterations is below this.
    - **convergence.gtol_rel** (float): Stop when the gradient,
      divided by the absolute value of the criterion function is smaller than this.
    - **stopping.maxfun** (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this as
      convergence.
    - **tr_solver** (str): Method for solving trust-region subproblems, relevant only
      for 'trf' and 'dogbox' methods.

      - 'exact' is suitable for not very large problems with dense
        Jacobian matrices. The computational complexity per iteration is
        comparable to a singular value decomposition of the Jacobian
        matrix.
      - 'lsmr' is suitable for problems with sparse and large Jacobian
        matrices. It uses the iterative procedure
        `scipy.sparse.linalg.lsmr` for finding a solution of a linear
        least-squares problem and only requires matrix-vector product
        evaluations.
        If None (default), the solver is chosen based on the type of Jacobian
        returned on the first iteration.
    - **tr_solver_options** (dict):  Keyword options passed to trust-region solver.

      - ``tr_solver='exact'``: `tr_options` are ignored.
      - ``tr_solver='lsmr'``: options for `scipy.sparse.linalg.lsmr`.

```

```{eval-rst}
.. dropdown::  scipy_ls_trf

    .. code-block::

        "scipy_ls_trf"

    Minimize a nonlinear least squares problem using a trustregion reflective method.

    Trust Region Reflective algorithm, particularly suitable for large sparse problems
    with bounds. Generally robust method.

    The algorithm supports the following options:

    - **convergence.ftol_rel** (float): Stop when the relative
      improvement between two iterations is below this.
    - **convergence.gtol_rel** (float): Stop when the gradient,
      divided by the absolute value of the criterion function is smaller than this.
    - **stopping.maxfun** (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this as
      convergence.
    - **tr_solver** (str): Method for solving trust-region subproblems, relevant only
      for 'trf' and 'dogbox' methods.

      - 'exact' is suitable for not very large problems with dense
        Jacobian matrices. The computational complexity per iteration is
        comparable to a singular value decomposition of the Jacobian
        matrix.
      - 'lsmr' is suitable for problems with sparse and large Jacobian
        matrices. It uses the iterative procedure
        `scipy.sparse.linalg.lsmr` for finding a solution of a linear
        least-squares problem and only requires matrix-vector product
        evaluations.
        If None (default), the solver is chosen based on the type of Jacobian
        returned on the first iteration.
    - **tr_solver_options** (dict):  Keyword options passed to trust-region solver.

      - ``tr_solver='exact'``: `tr_options` are ignored.
      - ``tr_solver='lsmr'``: options for `scipy.sparse.linalg.lsmr`.

```

```{eval-rst}
.. dropdown::  scipy_ls_lm

    .. code-block::

        "scipy_ls_lm"

    Minimize a nonlinear least squares problem using a Levenberg-Marquardt method.

    Does not handle bounds and sparse Jacobians. Usually the most efficient method for
    small unconstrained problems.

    The algorithm supports the following options:

    - **convergence.ftol_rel** (float): Stop when the relative
      improvement between two iterations is below this.
    - **convergence.gtol_rel** (float): Stop when the gradient,
      divided by the absolute value of the criterion function is smaller than this.
    - **stopping.maxfun** (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this as
      convergence.
    - **tr_solver** (str): Method for solving trust-region subproblems, relevant only
      for 'trf' and 'dogbox' methods.

      - 'exact' is suitable for not very large problems with dense
        Jacobian matrices. The computational complexity per iteration is
        comparable to a singular value decomposition of the Jacobian
        matrix.
      - 'lsmr' is suitable for problems with sparse and large Jacobian
        matrices. It uses the iterative procedure
        `scipy.sparse.linalg.lsmr` for finding a solution of a linear
        least-squares problem and only requires matrix-vector product
        evaluations.
        If None (default), the solver is chosen based on the type of Jacobian
        returned on the first iteration.
    - **tr_solver_options** (dict):  Keyword options passed to trust-region solver.

      - ``tr_solver='exact'``: `tr_options` are ignored.
      - ``tr_solver='lsmr'``: options for `scipy.sparse.linalg.lsmr`.

```

```{eval-rst}
.. dropdown::  scipy_basinhopping

    .. code-block::

        "scipy_basinhopping"

    Find the global minimum of a function using the basin-hopping algorithm which combines a global stepping algorithm with local minimization at each step.

    Basin-hopping is a two-phase method that combines a global stepping algorithm with local minimization at each step. Designed to mimic the natural process of energy minimization of clusters of atoms, it works well for similar problems with “funnel-like, but rugged” energy landscapes.

    This is mainly supported for completeness. Consider optimagic's built in multistart
    optimization for a similar approach that can run multiple optimizations in parallel,
    supports all local algorithms in optimagic (as opposed to just those from scipy)
    and allows for a better visualization of the multistart history.

    When provided the derivative is passed to the local minimization method.

    The algorithm supports the following options:

    - **local_algorithm** (str/callable): Any scipy local minimizer: valid options are.
      "Nelder-Mead". "Powell". "CG". "BFGS". "Newton-CG". "L-BFGS-B". "TNC". "COBYLA".
      "SLSQP". "trust-constr". "dogleg". "trust-ncg". "trust-exact". "trust-krylov".
      or a custom function for local minimization, default is "L-BFGS-B".
    - **n_local_optimizations**: (int) The number local optimizations. Default is 100 as
      in scipy's default.
    - **temperature**: (float) Controls the randomness in the optimization process.
      Higher the temperatures the larger jumps in function value will be accepted.
      Default is 1.0 as in scipy's default.
    - **stepsize**: (float) Maximum step size. Default is 0.5 as in scipy's default.
    - **local_algo_options**: (dict) Additional keyword arguments for the local
      minimizer. Check the documentation of the local scipy algorithms for details on
      what is supported.
    - **take_step**: (callable) Replaces the default step-taking routine. Default is
      None as in scipy's default.
    - **accept_test**: (callable) Define a test to judge the acception of steps. Default
      is None as in scipy's default.
    - **interval**: (int) Determined how often the step size is updated. Default is 50
      as in scipy's default.
    - **convergence.n_unchanged_iterations**: (int) Number of iterations the global
      minimum estimate stays the same to stops the algorithm. Default is None as in
      scipy's default.
    - **seed**: (None, int, numpy.random.Generator,numpy.random.RandomState)Default is
      None as in scipy's default.
    - **target_accept_rate**: (float) Adjusts the step size. Default is 0.5 as in scipy's default.
    - **stepwise_factor**: (float) Step size multiplier upon each step. Lies between (0,1), default is 0.9 as in scipy's default.

```

```{eval-rst}
.. dropdown::  scipy_brute

    .. code-block::

        "scipy_brute"

    Find the global minimum of a fuction over a given range by brute force.

    Brute force evaluates the criterion at each point and that is why better suited for problems with very few parameters.

    The start values are not actually used because the grid is only defined by bounds.
    It is still necessary for optimagic to infer the number and format of the
    parameters.

    Due to the parallelization, this algorithm cannot collect a history of parameters
    and criterion evaluations.

    The algorithm supports the following options:

    - **n_grid_points** (int):  the number of grid points to use for the brute force
      search. Default is 20 as in scipy.
    - **polishing_function** (callable):  Function to seek a more precise minimum near
      brute-force' best gridpoint taking brute-force's result at initial guess as a
      positional argument. Default is None providing no polishing.
    - **n_cores** (int): The number of cores on which the function is evaluated in
      parallel. Default 1.
    - **batch_evaluator** (str or callable). An optimagic batch evaluator. Default
      'joblib'.

```

```{eval-rst}
.. dropdown::  scipy_differential_evolution

    .. code-block::

        "scipy_differential_evolution"

    Find the global minimum of a multivariate function using differential evolution (DE). DE is a gradient-free method.

    Due to optimagic's general parameter format the integrality and vectorized
    arguments are not supported.

    The algorithm supports the following options:

    - **strategy** (str): Measure of quality to improve a candidate solution, can be one
      of the following keywords (default 'best1bin'.)
      - ‘best1bin’
      - ‘best1exp’
      - ‘rand1exp’
      - ‘randtobest1exp’
      - ‘currenttobest1exp’
      - ‘best2exp’
      - ‘rand2exp’
      - ‘randtobest1bin’
      - ‘currenttobest1bin’
      - ‘best2bin’
      - ‘rand2bin’
      - ‘rand1bin’

    - **stopping.maxiter** (int): The maximum number of criterion evaluations
      without polishing is(stopping.maxiter + 1) * population_size * number of
      parameters
    - **population_size_multiplier** (int): A multiplier setting the population size.
      The number of individuals in the population is population_size * number of
      parameters. The default 15.
    - **convergence.ftol_rel** (float): Default 0.01.
    - **mutation_constant** (float/tuple): The differential weight denoted by F in
      literature. Should be within 0 and 2.  The tuple form is used to specify
      (min, max) dithering which can help speed convergence.  Default is (0.5, 1).
    - **recombination_constant** (float): The crossover probability or CR in the
      literature determines the probability that two solution vectors will be combined
      to produce a new solution vector. Should be between 0 and 1. The default is 0.7.
    - **seed** (int): DE is stochastic. Define a seed for reproducability.
    - **polish** (bool): Uses scipy's L-BFGS-B for unconstrained problems and
      trust-constr for constrained problems to slightly improve the minimization.
      Default is True.
    - **sampling_method** (str/np.array): Specify the sampling method for the initial
      population. It can be one of the following options
      - "latinhypercube"
      - "sobol"
      - "halton"
      - "random"
      - an array specifying the initial population of shape (total population size,
      number of parameters). The initial population is clipped to bounds before use.
      Default is 'latinhypercube'

    - **convergence.ftol_abs** (float):
      CONVERGENCE_SECOND_BEST_ABSOLUTE_CRITERION_TOLERANCE
    - **n_cores** (int): The number of cores on which the function is evaluated in
      parallel. Default 1.
    - **batch_evaluator** (str or callable). An optimagic batch evaluator. Default
      'joblib'.

```

```{eval-rst}
.. dropdown::  scipy_shgo

    .. code-block::

        "scipy_shgo"

    Find the global minimum of a fuction using simplicial homology global optimization.

    The algorithm supports the following options:

    - **local_algorithm** (str): The local optimization algorithm to be used. Only
      COBYLA and SLSQP supports constraints. Valid options are
      "Nelder-Mead". "Powell". "CG". "BFGS". "Newton-CG". "L-BFGS-B". "TNC". "COBYLA".
      "SLSQP". "trust-constr". "dogleg". "trust-ncg". "trust-exact". "trust-krylov"
      or a custom function for local minimization, default is "L-BFGS-B".
    - **local_algo_options**: (dict) Additional keyword arguments for the local
      minimizer. Check the documentation of the local scipy algorithms for details on
      what is supported.
    - **n_sampling_points** (int): Specify the number of sampling points to construct
      the simplical complex.
    - **n_simplex_iterations** (int): Number of iterations to construct the simplical
      complex. Default is 1 as in scipy.
    - **sampling_method** (str/callable): The method to use for sampling the search
      space. Default 'simplicial'.
    - **max_sampling_evaluations** (int): The maximum number of evaluations of the
      criterion function in the sampling phase.
    - **convergence.minimum_criterion_value** (float): Specify the global minimum when
      it is known. Default is - np.inf. For maximization problems, flip the sign.
    - **convergence.minimum_criterion_tolerance** (float): Specify the relative error
      between the current best minimum and the supplied global criterion_minimum
      allowed. Default is scipy's default, 1e-4.
    - **stopping.maxiter** (int): The maximum number of iterations.
    - **stopping.maxfun** (int): The maximum number of criterion
      evaluations.
    - **stopping.max_processing_time** (int): The maximum time allowed for the
      optimization.
    - **minimum_homology_group_rank_differential** (int): The minimum difference in the
      rank of the homology group between iterations.
    - **symmetry** (bool): Specify whether the criterion contains symetric variables.
    - **minimize_every_iteration** (bool): Specify whether the gloabal sampling points
      are passed to the local algorithm in every iteration.
    - **max_local_minimizations_per_iteration** (int): The maximum number of local
      optimizations per iteration. Default False, i.e. no limit.
    - **infinity_constraints** (bool): Specify whether to save the sampling points
      outside the feasible domain. Default is True.

```

```{eval-rst}
.. dropdown::  scipy_dual_annealing

    .. code-block::

        "scipy_dual_annealing"

    Find the global minimum of a function using dual annealing for continuous variables.

    The algorithm supports the following options:

    - **stopping.maxiter** (int): Specify the maximum number of global searh
      iterations.
    - **local_algorithm** (str): The local optimization algorithm to be used. valid
      options are: "Nelder-Mead", "Powell", "CG", "BFGS", "Newton-CG", "L-BFGS-B",
      "TNC", "COBYLA", "SLSQP", "trust-constr", "dogleg", "trust-ncg", "trust-exact",
      "trust-krylov", Default "L-BFGS-B".
    - **local_algo_options**: (dict) Additional keyword arguments for the local
      minimizer. Check the documentation of the local scipy algorithms for details on
      what is supported.
    - **initial_temperature** (float): The temparature algorithm starts with. The higher values lead to a wider search space. The range is (0.01, 5.e4] and default is 5230.0.
    - **restart_temperature_ratio** (float): Reanneling starts when the algorithm is decreased to initial_temperature * restart_temperature_ratio. Default is 2e-05.
    - **visit** (float): Specify the thickness of visiting distribution's tails. Range is (1, 3] and default is scipy's default, 2.62.
    - **accept** (float): Controls the probability of acceptance. Range is (-1e4, -5] and default is scipy's default, -5.0. Smaller values lead to lower acceptance probability.
    - **stopping.maxfun** (int): soft limit for the number of criterion evaluations.
    - **seed** (int, None or RNG): Dual annealing is a stochastic process. Seed or
      random number generator. Default None.
    - **no_local_search** (bool): Specify whether to apply a traditional Generalized Simulated Annealing with no local search. Default is False.

```

```{eval-rst}
.. dropdown::  scipy_direct

    .. code-block::

        "scipy_direct"

    Find the global minimum of a function using dividing rectangles method. It is not necessary to provide an initial guess.

    The algorithm supports the following options:

    - **eps** (float): Specify the minimum difference of the criterion values between the current best hyperrectangle and the next potentially best hyperrectangle to be divided determining the trade off between global and local search. Default is 1e-6 differing from scipy's default 1e-4.
    - **stopping.maxfun** (int/None): Maximum number of criterion evaluations allowed. Default is None which caps the number of evaluations at 1000 * number of dimentions automatically.
    - **stopping.maxiter** (int): Maximum number of iterations allowed.
    - **locally_biased** (bool): Determine whether to use the locally biased variant of the algorithm DIRECT_L. Default is True.
    - **convergence.minimum_criterion_value** (float): Specify the global minimum when it is known. Default is minus infinity. For maximization problems, flip the sign.
    - **convergence.minimum_criterion_tolerance** (float): Specify the relative error between the current best minimum and the supplied global criterion_minimum allowed. Default is scipy's default, 1e-4.
    - **volume_hyperrectangle_tolerance** (float): Specify the smallest volume of the hyperrectangle containing the lowest criterion value allowed. Range is (0,1). Default is 1e-16.
    - **length_hyperrectangle_tolerance** (float): Depending on locally_biased it can refer to normalized side (True) or diagonal (False) length of the hyperrectangle containing the lowest criterion value. Range is (0,1). Default is scipy's default, 1e-6.

```

(own-algorithms)=

## Own optimizers

We implement a few algorithms from scratch. They are currently considered experimental.

```{eval-rst}
.. dropdown:: bhhh

    .. code-block::

        "bhhh"

    Minimize a likelihood function using the BHHH algorithm.

    BHHH (:cite:`Berndt1974`) can - and should ONLY - be used for minimizing
    (or maximizing) a likelihood. It is similar to the Newton-Raphson
    algorithm, but replaces the Hessian matrix with the outer product of the
    gradient. This approximation is based on the information matrix equality
    (:cite:`Halbert1982`) and is thus only vaid when minimizing (or maximizing)
    a likelihood.

    The criterion function :func:`func` should return a dictionary with
    at least the entry ``{"contributions": array_or_pytree}`` where ``array_or_pytree``
    contains the likelihood contributions of each individual.

    bhhh supports the following options:

    - **convergence.gtol_abs** (float): Stopping criterion for the
      gradient tolerance. Default is 1e-8.
    - **stopping.maxiter** (int): Maximum number of iterations.
      If reached, terminate. Default is 200.

```

```{eval-rst}
.. dropdown:: neldermead_parallel

    .. code-block::

        "neldermead_parallel"

    Minimize a function using the neldermead_parallel algorithm.

    This is a parallel Nelder-Mead algorithm following Lee D., Wiswall M., A parallel
    implementation of the simplex function minimization routine,
    Computational Economics, 2007.

    The algorithm was implemented by Jacek Barszczewski

    The algorithm supports the following options:

    - **init_simplex_method** (string or callable): Name of the method to create initial
      simplex or callable which takes as an argument initial value of parameters
      and returns initial simplex as j+1 x j array, where j is length of x.
      The default is "gao_han".
    - **n_cores** (int): Degree of parallization. The default is 1 (no parallelization).

    - **adaptive** (bool): Adjust parameters of Nelder-Mead algorithm to account
      for simplex size. The default is True.

    - **stopping.maxiter** (int): Maximum number of algorithm iterations.
      The default is STOPPING_MAX_ITERATIONS.

    - **convergence.ftol_abs** (float): maximal difference between
      function value evaluated on simplex points.
      The default is CONVERGENCE_SECOND_BEST_ABSOLUTE_CRITERION_TOLERANCE.

    - **convergence.xtol_abs** (float): maximal distance between points
      in the simplex. The default is CONVERGENCE_SECOND_BEST_ABSOLUTE_PARAMS_TOLERANCE.

    - **batch_evaluator** (string or callable): See :ref:`batch_evaluators` for
        details. Default "joblib".

```

```{eval-rst}
.. dropdown:: pounders

    .. code-block::

        "pounders"

    Minimize a function using the POUNDERS algorithm.

    POUNDERs (:cite:`Benson2017`, :cite:`Wild2015`, `GitHub repository
    <https://github.com/erdc/petsc4py>`_)

    can be a useful tool for economists who estimate structural models using
    indirect inference, because unlike commonly used algorithms such as Nelder-Mead,
    POUNDERs is tailored for minimizing a non-linear sum of squares objective function,
    and therefore may require fewer iterations to arrive at a local optimum than
    Nelder-Mead.

    The criterion function :func:`func` should return a dictionary with the following
    fields:

    1. ``"value"``: The sum of squared (potentially weighted) errors.
    2. ``"root_contributions"``: An array containing the root (weighted) contributions.

    Scaling the problem is necessary such that bounds correspond to the unit hypercube
    :math:`[0, 1]^n`. For unconstrained problems, scale each parameter such that unit
    changes in parameters result in similar order-of-magnitude changes in the criterion
    value(s).

    pounders supports the following options:


    - **convergence.gtol_abs**: Convergence tolerance for the
      absolute gradient norm. Stop if norm of the gradient is less than this.
      Default is 1e-8.
    - **convergence.gtol_rel**: Convergence tolerance for the
      relative gradient norm. Stop if norm of the gradient relative to the criterion
      value is less than this. Default is 1-8.
    - **convergence.gtol_scaled**: Convergence tolerance for the
      scaled gradient norm. Stop if norm of the gradient divided by norm of the
      gradient at the initial parameters is less than this.
      Disabled, i.e. set to False, by default.
    - **max_interpolation_points** (int): Maximum number of interpolation points.
      Default is `2 * n + 1`, where `n` is the length of the parameter vector.
    - **stopping.maxiter** (int): Maximum number of iterations.
      If reached, terminate. Default is 2000.
    - **trustregion_initial_radius (float)**: Delta, initial trust-region radius.
      0.1 by default.
    - **trustregion_minimal_radius** (float): Minimal trust-region radius.
      1e-6 by default.
    - **trustregion_maximal_radius** (float): Maximal trust-region radius.
      1e6 by default.
    - **trustregion_shrinking_factor_not_successful** (float): Shrinking factor of
      the trust-region radius in case the solution vector of the suproblem
      is not accepted, but the model is fully linear (i.e. "valid").
      Defualt is 0.5.
    - **trustregion_expansion_factor_successful** (float): Shrinking factor of
      the trust-region radius in case the solution vector of the suproblem
      is accepted. Default is 2.
    - **theta1** (float): Threshold for adding the current x candidate to the
      model. Function argument to find_affine_points(). Default is 1e-5.
    - **theta2** (float): Threshold for adding the current x candidate to the model.
      Argument to get_interpolation_matrices_residual_model(). Default is 1e-4.
    - **trustregion_threshold_successful** (float): First threshold for accepting the
      solution vector of the subproblem as the best x candidate. Default is 0.
    - **trustregion_threshold_very_successful** (float): Second threshold for accepting
      the solution vector of the subproblem as the best x candidate. Default is 0.1.
    - **c1** (float): Treshold for accepting the norm of our current x candidate.
      Function argument to find_affine_points() for the case where input array
      *model_improving_points* is zero.
    - **c2** (int): Treshold for accepting the norm of our current x candidate.
      Equal to 10 by default. Argument to *find_affine_points()* in case
      the input array *model_improving_points* is not zero.
    - **trustregion_subproblem_solver** (str): Solver to use for the trust-region
      subproblem. Two internal solvers are supported:
      - "bntr": Bounded Newton Trust-Region (default, supports bound constraints)
      - "gqtpar": (does not support bound constraints)
    - **trustregion_subsolver_options** (dict): Options dictionary containing
      the stopping criteria for the subproblem. It takes different keys depending
      on the type of subproblem solver used. With the exception of the stopping criterion
      "maxiter", which is always included.

      If the subsolver "bntr" is used, the dictionary also contains the tolerance levels
      "gtol_abs", "gtol_rel", and "gtol_scaled". Moreover, the "conjugate_gradient_method"
      can be provided. Available conjugate gradient methods are:
      - "cg". In this case, two additional stopping criteria are "gtol_abs_cg" and "gtol_rel_cg"
      - "steihaug-toint"
      - "trsbox" (default)

      If the subsolver "gqtpar" is employed, the two stopping criteria are
      "k_easy" and "k_hard".

      None of the dictionary keys need to be specified by default, but can be.
    - **batch_evaluator** (str or callable): Name of a pre-implemented batch evaluator
      (currently "joblib" and "pathos_mp") or callable with the same interface
      as the optimagic batch_evaluators. Default is "joblib".
    - **n_cores (int)**: Number of processes used to parallelize the function
      evaluations. Default is 1.

```

(tao-algorithms)=

## Optimizers from the Toolkit for Advanced Optimization (TAO)

We wrap the pounders algorithm from the Toolkit of Advanced optimization. To use it you
need to have [petsc4py](https://pypi.org/project/petsc4py/) installed.

```{eval-rst}
.. dropdown::  tao_pounders

    .. code-block::

        "tao_pounders"

    Minimize a function using the POUNDERs algorithm.

    POUNDERs (:cite:`Benson2017`, :cite:`Wild2015`, `GitHub repository
    <https://github.com/erdc/petsc4py>`_)

    can be a useful tool for economists who estimate structural models using
    indirect inference, because unlike commonly used algorithms such as Nelder-Mead,
    POUNDERs is tailored for minimizing a non-linear sum of squares objective function,
    and therefore may require fewer iterations to arrive at a local optimum than
    Nelder-Mead.

    The criterion function :func:`func` should return a dictionary with the following
    fields:

    1. ``"value"``: The sum of squared (potentially weighted) errors.
    2. ``"root_contributions"``: An array containing the root (weighted) contributions.

    Scaling the problem is necessary such that bounds correspond to the unit hypercube
    :math:`[0, 1]^n`. For unconstrained problems, scale each parameter such that unit
    changes in parameters result in similar order-of-magnitude changes in the criterion
    value(s).

    POUNDERs has several convergence criteria. Let :math:`X` be the current parameter
    vector, :math:`X_0` the initial parameter vector, :math:`g` the gradient, and
    :math:`f` the criterion function.

    ``absolute_gradient_tolerance`` stops the optimization if the norm of the gradient
    falls below :math:`\epsilon`.

    .. math::

        ||g(X)|| < \epsilon

    ``relative_gradient_tolerance`` stops the optimization if the norm of the gradient
    relative to the criterion value falls below :math:`epsilon`.

    .. math::

        \frac{||g(X)||}{|f(X)|} < \epsilon

    ``scaled_gradient_tolerance`` stops the optimization if the norm of the gradient is
    lower than some fraction :math:`epsilon` of the norm of the gradient at the initial
    parameters.

    .. math::

        \frac{||g(X)||}{||g(X0)||} < \epsilon

    - **convergence.gtol_abs** (float): Stop if norm of gradient is less than this.
      If set to False the algorithm will not consider convergence.gtol_abs.
    - **convergence.gtol_rel** (float): Stop if relative norm of gradient is less
      than this. If set to False the algorithm will not consider
      convergence.gtol_rel.
    - **convergence.scaled_gradient_tolerance** (float): Stop if scaled norm of gradient is smaller
      than this. If set to False the algorithm will not consider
      convergence.scaled_gradient_tolerance.
    - **trustregion.initial_radius** (float): Initial value of the trust region radius.
      It must be :math:`> 0`.
    - **stopping.maxiter** (int): Alternative Stopping criterion.
      If set the routine will stop after the number of specified iterations or
      after the step size is sufficiently small. If the variable is set the
      default criteria will all be ignored.


```

(nag-algorithms)=

## Optimizers from the Numerical Algorithms Group (NAG)

We wrap two algorithms from the numerical algorithms group. To use them, you need to
install each of them separately:

- `pip install DFO-LS`
- `pip install Py-BOBYQA`

```{eval-rst}
.. dropdown::  nag_dfols

    .. code-block::

        "nag_dfols"

    Minimize a function with least squares structure using DFO-LS.

    The DFO-LS algorithm :cite:`Cartis2018b` is designed to solve the nonlinear
    least-squares minimization problem (with optional bound constraints).
    Remember to cite :cite:`Cartis2018b` when using DF-OLS in addition to optimagic.

    .. math::

        \min_{x\in\mathbb{R}^n}  &\quad  f(x) := \sum_{i=1}^{m}r_{i}(x)^2 \\
        \text{s.t.} &\quad  \text{lower_bounds} \leq x \leq \text{upper_bounds}

    The :math:`r_{i}` are called root contributions in optimagic.

    DFO-LS is a derivative-free optimization algorithm, which means it does not require
    the user to provide the derivatives of f(x) or :math:`r_{i}(x)`, nor does it
    attempt to estimate them internally (by using finite differencing, for instance).

    There are two main situations when using a derivative-free algorithm
    (such as DFO-LS) is preferable to a derivative-based algorithm (which is the vast
    majority of least-squares solvers):

    1. If the residuals are noisy, then calculating or even estimating their derivatives
       may be impossible (or at least very inaccurate). By noisy, we mean that if we
       evaluate :math:`r_{i}(x)` multiple times at the same value of x, we get different
       results. This may happen when a Monte Carlo simulation is used, for instance.

    2. If the residuals are expensive to evaluate, then estimating derivatives
       (which requires n evaluations of each :math:`r_{i}(x)` for every point of
       interest x) may be prohibitively expensive. Derivative-free methods are designed
       to solve the problem with the fewest number of evaluations of the criterion as
       possible.

    To read the detailed documentation of the algorithm `click here
    <https://numericalalgorithmsgroup.github.io/dfols/>`_.

    There are four possible convergence criteria:

    1. when the lower trust region radius is shrunk below a minimum
       (``convergence.minimal_trustregion_radius_tolerance``).

    2. when the improvements of iterations become very small
       (``convergence.slow_progress``). This is very similar to
       ``relative_criterion_tolerance`` but ``convergence.slow_progress`` is more
       general allowing to specify not only the threshold for convergence but also
       a period over which the improvements must have been very small.

    3. when a sufficient reduction to the criterion value at the start parameters
       has been reached, i.e. when
       :math:`\frac{f(x)}{f(x_0)} \leq
       \text{convergence.ftol_scaled}`

    4. when all evaluations on the interpolation points fall within a scaled version of
       the noise level of the criterion function. This is only applicable if the
       criterion function is noisy. You can specify this criterion with
       ``convergence.noise_corrected_criterion_tolerance``.

    DF-OLS supports resetting the optimization and doing a fast start by
    starting with a smaller interpolation set and growing it dynamically.
    For more information see `their detailed documentation
    <https://numericalalgorithmsgroup.github.io/dfols/>`_ and :cite:`Cartis2018b`.

    - **clip_criterion_if_overflowing** (bool): see :ref:`algo_options`.
      convergence.minimal_trustregion_radius_tolerance (float): see
      :ref:`algo_options`.
    - **convergence.noise_corrected_criterion_tolerance** (float): Stop when the
      evaluations on the set of interpolation points all fall within this factor
      of the noise level.
      The default is 1, i.e. when all evaluations are within the noise level.
      If you want to not use this criterion but still flag your
      criterion function as noisy, set this tolerance to 0.0.

      .. warning::
          Very small values, as in most other tolerances don't make sense here.

    - **convergence.ftol_scaled** (float):
      Terminate if a point is reached where the ratio of the criterion value
      to the criterion value at the start params is below this value, i.e. if
      :math:`f(x_k)/f(x_0) \leq
      \text{convergence.ftol_scaled}`. Note this is
      deactivated unless the lowest mathematically possible criterion value (0.0)
      is actually achieved.
    - **convergence.slow_progress** (dict): Arguments for converging when the evaluations
      over several iterations only yield small improvements on average, see
      see :ref:`algo_options` for details.
    - **initial_directions (str)**: see :ref:`algo_options`.
    - **interpolation_rounding_error** (float): see :ref:`algo_options`.
    - **noise_additive_level** (float): Used for determining the presence of noise
      and the convergence by all interpolation points being within noise level.
      0 means no additive noise. Only multiplicative or additive is supported.
    - **noise_multiplicative_level** (float): Used for determining the presence of noise
      and the convergence by all interpolation points being within noise level.
      0 means no multiplicative noise. Only multiplicative or additive is
      supported.
    - **noise_n_evals_per_point** (callable): How often to evaluate the criterion
      function at each point.
      This is only applicable for criterion functions with noise,
      when averaging multiple evaluations at the same point produces a more
      accurate value.
      The input parameters are the ``upper_trustregion_radius`` (:math:`\Delta`),
      the ``lower_trustregion_radius`` (:math:`\rho`),
      how many iterations the algorithm has been running for, ``n_iterations``
      and how many resets have been performed, ``n_resets``.
      The function must return an integer.
      Default is no averaging (i.e.
      ``noise_n_evals_per_point(...) = 1``).
    - **random_directions_orthogonal** (bool): see :ref:`algo_options`.
    - **stopping.maxfun** (int): see :ref:`algo_options`.
    - **threshold_for_safety_step** (float): see :ref:`algo_options`.
    - **trustregion.expansion_factor_successful** (float): see :ref:`algo_options`.
    - **trustregion.expansion_factor_very_successful** (float): see :ref:`algo_options`.
    - **trustregion.fast_start_options** (dict): see :ref:`algo_options`.
    - **trustregion.initial_radius** (float): Initial value of the trust region radius.
    - **trustregion.method_to_replace_extra_points (str)**: If replacing extra points in
      successful iterations, whether to use geometry improving steps or the
      momentum method. Can be "geometry_improving" or "momentum".
    - **trustregion.n_extra_points_to_replace_successful** (int): The number of extra
      points (other than accepting the trust region step) to replace. Useful when
      ``trustregion.n_interpolation_points > len(x) + 1``.
    - **trustregion.n_interpolation_points** (int): The number of interpolation points to
      use. The default is :code:`len(x) + 1`. If using resets, this is the
      number of points to use in the first run of the solver, before any resets.
    - **trustregion.precondition_interpolation** (bool): see :ref:`algo_options`.
    - **trustregion.shrinking_factor_not_successful** (float): see :ref:`algo_options`.
    - **trustregion.shrinking_factor_lower_radius** (float): see :ref:`algo_options`.
    - **trustregion.shrinking_factor_upper_radius** (float): see :ref:`algo_options`.
    - **trustregion.threshold_successful** (float): Share of the predicted improvement
      that has to be achieved for a trust region iteration to count as successful.
    - **trustregion.threshold_very_successful** (float): Share of the predicted
      improvement that has to be achieved for a trust region iteration to count
      as very successful.

```

```{eval-rst}
.. dropdown::  nag_pybobyqa

    .. code-block::

        "nag_pybobyqa"

    Minimize a function using the BOBYQA algorithm.

    BOBYQA (:cite:`Powell2009`, :cite:`Cartis2018`, :cite:`Cartis2018a`) is a
    derivative-free trust-region method. It is designed to solve nonlinear local
    minimization problems.

    Remember to cite :cite:`Powell2009` and :cite:`Cartis2018` when using pybobyqa in
    addition to optimagic. If you take advantage of the ``seek_global_optimum`` option,
    cite :cite:`Cartis2018a` additionally.

    There are two main situations when using a derivative-free algorithm like BOBYQA
    is preferable to derivative-based algorithms:

    1. The criterion function is not deterministic, i.e. if we evaluate the criterion
       function multiple times at the same parameter vector we get different results.

    2. The criterion function is very expensive to evaluate and only finite differences
       are available to calculate its derivative.

    The detailed documentation of the algorithm can be found `here
    <https://numericalalgorithmsgroup.github.io/pybobyqa/>`_.

    There are four possible convergence criteria:

    1. when the trust region radius is shrunk below a minimum. This is
       approximately equivalent to an absolute parameter tolerance.

    2. when the criterion value falls below an absolute, user-specified value,
       the optimization terminates successfully.

    3. when insufficient improvements have been gained over a certain number of
       iterations. The (absolute) threshold for what constitutes an insufficient
       improvement, how many iterations have to be insufficient and with which
       iteration to compare can all be specified by the user.

    4. when all evaluations on the interpolation points fall within a scaled version of
       the noise level of the criterion function. This is only applicable if the
       criterion function is noisy.

    - **clip_criterion_if_overflowing** (bool): see :ref:`algo_options`.
    - **convergence.criterion_value** (float): Terminate successfully if
      the criterion value falls below this threshold. This is deactivated
      (i.e. set to -inf) by default.
    - **convergence.minimal_trustregion_radius_tolerance** (float): Minimum allowed
      value of the trust region radius, which determines when a successful
      termination occurs.
    - **convergence.noise_corrected_criterion_tolerance** (float): Stop when the
      evaluations on the set of interpolation points all fall within this
      factor of the noise level.
      The default is 1, i.e. when all evaluations are within the noise level.
      If you want to not use this criterion but still flag your
      criterion function as noisy, set this tolerance to 0.0.

      .. warning::
          Very small values, as in most other tolerances don't make sense here.

    - **convergence.slow_progress** (dict): Arguments for converging when the evaluations
      over several iterations only yield small improvements on average, see
      see :ref:`algo_options` for details.
    - **initial_directions** (str)``: see :ref:`algo_options`.
    - **interpolation_rounding_error** (float): see :ref:`algo_options`.
    - **noise_additive_level** (float): Used for determining the presence of noise
      and the convergence by all interpolation points being within noise level.
      0 means no additive noise. Only multiplicative or additive is supported.
    - **noise_multiplicative_level** (float): Used for determining the presence of noise
      and the convergence by all interpolation points being within noise level.
      0 means no multiplicative noise. Only multiplicative or additive is
      supported.
    - **noise_n_evals_per_point** (callable): How often to evaluate the criterion
      function at each point.
      This is only applicable for criterion functions with noise,
      when averaging multiple evaluations at the same point produces a more
      accurate value.
      The input parameters are the ``upper_trustregion_radius`` (``delta``),
      the ``lower_trustregion_radius`` (``rho``),
      how many iterations the algorithm has been running for, ``n_iterations``
      and how many resets have been performed, ``n_resets``.
      The function must return an integer.
      Default is no averaging (i.e. ``noise_n_evals_per_point(...) = 1``).
    - **random_directions_orthogonal** (bool): see :ref:`algo_options`.
    - **seek_global_optimum** (bool): whether to apply the heuristic to escape local
      minima presented in :cite:`Cartis2018a`. Only applies for noisy criterion
      functions.
    - **stopping.maxfun** (int): see :ref:`algo_options`.
    - **threshold_for_safety_step** (float): see :ref:`algo_options`.
    - **trustregion.expansion_factor_successful** (float): see :ref:`algo_options`.
    - **trustregion.expansion_factor_very_successful** (float): see :ref:`algo_options`.
    - **trustregion.initial_radius** (float): Initial value of the trust region radius.
    - **trustregion.minimum_change_hession_for_underdetermined_interpolation** (bool):
      Whether to solve the underdetermined quadratic interpolation problem by
      minimizing the Frobenius norm of the Hessian, or change in Hessian.
    - **trustregion.n_interpolation_points** (int): The number of interpolation points to
      use. With $n=len(x)$ the default is $2n+1$ if the criterion is not noisy.
      Otherwise, it is set to $(n+1)(n+2)/2)$.

      Larger values are particularly useful for noisy problems.
      Py-BOBYQA requires

      .. math::
          n + 1 \leq \text{trustregion.n_interpolation_points} \leq (n+1)(n+2)/2.
    - **trustregion.precondition_interpolation** (bool): see :ref:`algo_options`.
    - **trustregion.reset_options** (dict): Options for resetting the optimization,
      see :ref:`algo_options` for details.
    - **trustregion.shrinking_factor_not_successful** (float): see :ref:`algo_options`.
    - **trustregion.shrinking_factor_upper_radius** (float): see :ref:`algo_options`.
    - **trustregion.shrinking_factor_lower_radius** (float): see :ref:`algo_options`.
    - **trustregion.threshold_successful** (float): see :ref:`algo_options`.
    - **trustregion.threshold_very_successful** (float): see :ref:`algo_options`.



```

(pygmo-algorithms)=

## PYGMO2 Optimizers

Please cite {cite}`Biscani2020` in addition to optimagic when using pygmo. optimagic
supports the following [pygmo2](https://esa.github.io/pygmo2) optimizers.

```{eval-rst}
.. dropdown::  pygmo_gaco

    .. code-block::

        "pygmo_gaco"

    Minimize a scalar function using the generalized ant colony algorithm.

    The version available through pygmo is an generalized version of the
    original ant colony algorithm proposed by :cite:`Schlueter2009`.

    This algorithm can be applied to box-bounded problems.

    Ant colony optimization is a class of optimization algorithms modeled on the
    actions of an ant colony. Artificial "ants" (e.g. simulation agents) locate
    optimal solutions by moving through a parameter space representing all
    possible solutions. Real ants lay down pheromones directing each other to
    resources while exploring their environment. The simulated "ants" similarly
    record their positions and the quality of their solutions, so that in later
    simulation iterations more ants locate better solutions.

    The generalized ant colony algorithm generates future generations of ants by
    using a multi-kernel gaussian distribution based on three parameters (i.e.,
    pheromone values) which are computed depending on the quality of each
    previous solution. The solutions are ranked through an oracle penalty
    method.

    - **population_size** (int): Size of the population. If None, it's twice the
      number of parameters but at least 64.
    - **batch_evaluator** (str or Callable): Name of a pre-implemented batch
      evaluator (currently 'joblib' and 'pathos_mp') or Callable with the same
      interface as the optimagic batch_evaluators. See :ref:`batch_evaluators`.
    - **n_cores** (int): Number of cores to use.
    - **seed** (int): seed used by the internal random number generator.
    - **discard_start_params** (bool): If True, the start params are not guaranteed
      to be part of the initial population. This saves one criterion function
      evaluation that cannot be done in parallel with other evaluations. Default
      False.

    - **stopping.maxiter** (int): Number of generations to evolve.
    - **kernel_size** (int): Number of solutions stored in the solution archive.
    - **speed_parameter_q** (float): This parameter manages the convergence speed
      towards the found minima (the smaller the faster). In the pygmo
      documentation it is referred to as $q$. It must be positive and can be
      larger than 1. The default is 1.0 until **threshold** is reached. Then it
      is set to 0.01.
    - **oracle** (float): oracle parameter used in the penalty method.
    - **accuracy** (float): accuracy parameter for maintaining a minimum penalty
      function's values distances.
    - **threshold** (int): when the iteration counter reaches the threshold the
      convergence speed is set to 0.01 automatically. To deactivate this effect
      set the threshold to stopping.maxiter which is the largest allowed
      value.
    - **speed_of_std_values_convergence** (int): parameter that determines the
      convergence speed of the standard deviations. This must be an integer
      (`n_gen_mark` in pygmo and pagmo).
    - **stopping.max_n_without_improvements** (int): if a positive integer is
      assigned here, the algorithm will count the runs without improvements, if
      this number exceeds the given value, the algorithm will be stopped.
    - **stopping.maxfun** (int): maximum number of function
      evaluations.
    - **focus** (float): this parameter makes the search for the optimum greedier
      and more focused on local improvements (the higher the greedier). If the
      value is very high, the search is more focused around the current best
      solutions. Values larger than 1 are allowed.
    - **cache** (bool): if True, memory is activated in the algorithm for multiple calls.

```

```{eval-rst}
.. dropdown::  pygmo_bee_colony

    .. code-block::

        "pygmo_bee_colony"

    Minimize a scalar function using the artifical bee colony algorithm.

    The Artificial Bee Colony Algorithm was originally proposed by
    :cite:`Karaboga2007`. The implemented version of the algorithm is proposed
    in :cite:`Mernik2015`. The algorithm is only suited for bounded parameter
    spaces.

    - **stopping.maxiter** (int): Number of generations to evolve.
    - **seed** (int): seed used by the internal random number generator.
    - **discard_start_params** (bool): If True, the start params are not guaranteed
      to be part of the initial population. This saves one criterion function
      evaluation that cannot be done in parallel with other evaluations. Default
      False.
    - **max_n_trials** (int): Maximum number of trials for abandoning a source.
      Default is 1.
    - **population_size** (int): Size of the population. If None, it's twice the
      number of parameters but at least 20.
```

```{eval-rst}
.. dropdown::  pygmo_de

    .. code-block::

        "pygmo_de"

    Minimize a scalar function using the differential evolution algorithm.

    Differential Evolution is a heuristic optimizer originally presented in
    :cite:`Storn1997`. The algorithm is only suited for bounded parameter
    spaces.

    - **population_size** (int): Size of the population. If None, it's twice the
      number of parameters but at least 10.
    - **seed** (int): seed used by the internal random number generator.
    - **discard_start_params** (bool): If True, the start params are not guaranteed
      to be part of the initial population. This saves one criterion function
      evaluation that cannot be done in parallel with other evaluations. Default
      False.
    - **stopping.maxiter** (int): Number of generations to evolve.
    - **weight_coefficient** (float): Weight coefficient. It is denoted by $F$ in
      the main paper and must lie in [0, 2]. It controls the amplification of
      the differential variation $(x_{r_2, G} - x_{r_3, G})$.
    - **crossover_probability** (float): Crossover probability.
    - **mutation_variant (str or int)**: code for the mutation variant to create a
      new candidate individual. The default is . The following are available:

        - "best/1/exp" (1, when specified as int)
        - "rand/1/exp" (2, when specified as int)
        - "rand-to-best/1/exp" (3, when specified as int)
        - "best/2/exp" (4, when specified as int)
        - "rand/2/exp" (5, when specified as int)
        - "best/1/bin" (6, when specified as int)
        - "rand/1/bin" (7, when specified as int)
        - "rand-to-best/1/bin" (8, when specified as int)
        - "best/2/bin" (9, when specified as int)
        - "rand/2/bin" (10, when specified as int)
    - **convergence.criterion_tolerance**: stopping criteria on the criterion
      tolerance. Default is 1e-6. It is not clear whether this is the absolute
      or relative criterion tolerance.
    - **convergence.xtol_rel**: stopping criteria on the x
      tolerance. In pygmo the default is 1e-6 but we use our default value of
      1e-5.
```

```{eval-rst}
.. dropdown::  pygmo_sea

    .. code-block::

        "pygmo_sea"

    Minimize a scalar function using the (N+1)-ES simple evolutionary algorithm.

    This algorithm represents the simplest evolutionary strategy, where a population of
    $\lambda$ individuals at each generation produces one offspring by mutating its best
    individual uniformly at random within the bounds. Should the offspring be better
    than the worst individual in the population it will substitute it.

    See :cite:`Oliveto2007`.

    The algorithm is only suited for bounded parameter spaces.

    - **population_size** (int): Size of the population. If None, it's twice the number of
      parameters but at least 10.
    - **seed** (int): seed used by the internal random number generator.
    - **discard_start_params** (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.
    - **stopping.maxiter** (int): number of generations to consider. Each generation
      will compute the objective function once.

```

```{eval-rst}
.. dropdown::  pygmo_sga

    .. code-block::

        "pygmo_sga"

    Minimize a scalar function using a simple genetic algorithm.

    A detailed description of the algorithm can be found `in the pagmo2 documentation
    <https://esa.github.io/pagmo2/docs/cpp/algorithms/sga.html>`_.

    See also :cite:`Oliveto2007`.

    - **population_size** (int): Size of the population. If None, it's twice the number of
      parameters but at least 64.
    - **seed** (int): seed used by the internal random number generator.
    - **discard_start_params** (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.
    - **stopping.maxiter** (int): Number of generations to evolve.
    - **crossover_probability** (float): Crossover probability.
    - **crossover_strategy** (str): the crossover strategy. One of “exponential”,“binomial”,
      “single” or “sbx”. Default is "exponential".
    - **eta_c** (float): distribution index for “sbx” crossover. This is an inactive
      parameter if other types of crossovers are selected. Can be in [1, 100].
    - **mutation_probability** (float): Mutation probability.
    - **mutation_strategy** (str): Mutation strategy. Must be "gaussian", "polynomial" or
      "uniform". Default is "polynomial".
    - **mutation_polynomial_distribution_index** (float): Must be in [0, 1]. Default is 1.
    - **mutation_gaussian_width** (float): Must be in [0, 1]. Default is 1.
    - **selection_strategy (str)**: Selection strategy. Must be "tournament" or "truncated".
    - **selection_truncated_n_best** (int): number of best individuals to use in the
      "truncated" selection mechanism.
    - **selection_tournament_size** (int): size of the tournament in the "tournament"
      selection mechanism. Default is 1.
```

```{eval-rst}
.. dropdown::  pygmo_sade

    .. code-block::

        "pygmo_sade"

    Minimize a scalar function using Self-adaptive Differential Evolution.

    The original Differential Evolution algorithm (pygmo_de) can be significantly
    improved introducing the idea of parameter self-adaptation.

    Many different proposals have been made to self-adapt both the crossover and the
    F parameters of the original differential evolution algorithm. pygmo's
    implementation supports two different mechanisms. The first one, proposed by
    :cite:`Brest2006`, does not make use of the differential evolution operators to
    produce new values for the weight coefficient $F$ and the crossover probability
    $CR$ and, strictly speaking, is thus not self-adaptation, rather parameter control.
    The resulting differential evolution variant is often referred to as jDE.
    The second variant is inspired by the ideas introduced by :cite:`Elsayed2011` and
    uses a variaton of the selected DE operator to produce new $CR$ anf $F$ parameters
    for each individual. This variant is referred to iDE.

    - **population_size** (int): Size of the population. If None, it's twice the number of
      parameters but at least 64.
    - **seed** (int): seed used by the internal random number generator.
    - **discard_start_params** (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.
    - jde (bool): Whether to use the jDE self-adaptation variant to control the $F$ and
      $CR$ parameter. If True jDE is used, else iDE.
    - **stopping.maxiter** (int): Number of generations to evolve.
    - **mutation_variant** (int or str): code for the mutation variant to create a new
      candidate individual. The default is "rand/1/exp". The first ten are the
      classical mutation variants introduced in the orginal DE algorithm, the remaining
      ones are, instead, considered in the work by :cite:`Elsayed2011`.
      The following are available:

        - "best/1/exp" or 1
        - "rand/1/exp" or 2
        - "rand-to-best/1/exp" or 3
        - "best/2/exp" or 4
        - "rand/2/exp" or 5
        - "best/1/bin" or 6
        - "rand/1/bin" or 7
        - "rand-to-best/1/bin" or 8
        - "best/2/bin" or 9
        - "rand/2/bin" or 10
        - "rand/3/exp" or 11
        - "rand/3/bin" or 12
        - "best/3/exp" or 13
        - "best/3/bin" or 14
        - "rand-to-current/2/exp" or 15
        - "rand-to-current/2/bin" or 16
        - "rand-to-best-and-current/2/exp" or 17
        - "rand-to-best-and-current/2/bin" or 18

    - **keep_adapted_params** (bool):  when true the adapted parameters $CR$ anf $F$ are
      not reset between successive calls to the evolve method. Default is False.
    - ftol (float): stopping criteria on the x tolerance.
    - xtol (float): stopping criteria on the f tolerance.


```

```{eval-rst}
.. dropdown::  pygmo_cmaes

    .. code-block::

        "pygmo_cmaes"

    Minimize a scalar function using the Covariance Matrix Evolutionary Strategy.

    CMA-ES is one of the most successful algorithm, classified as an Evolutionary
    Strategy, for derivative-free global optimization. The version supported by
    optimagic is the version described in :cite:`Hansen2006`.

    In contrast to the pygmo version, optimagic always sets force_bounds to True. This
    avoids that ill defined parameter values are evaluated.

    - **population_size** (int): Size of the population. If None, it's twice the number of
      parameters but at least 64.
    - **seed** (int): seed used by the internal random number generator.
    - **discard_start_params** (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.

    - **stopping.maxiter** (int): Number of generations to evolve.
    - **backward_horizon** (float): backward time horizon for the evolution path. It must
      lie betwen 0 and 1.
    - **variance_loss_compensation** (float): makes partly up for the small variance loss in
      case the indicator is zero. `cs` in the MATLAB Code of :cite:`Hansen2006`. It must
      lie between 0 and 1.
    - **learning_rate_rank_one_update** (float): learning rate for the rank-one update of
      the covariance matrix. `c1` in the pygmo and pagmo documentation. It must lie
      between 0 and 1.
    - **learning_rate_rank_mu_update** (float): learning rate for the rank-mu update of the
      covariance matrix. `cmu` in the pygmo and pagmo documentation. It must lie between
      0 and 1.
    - **initial_step_size** (float): initial step size, :math:`\sigma^0` in the original
      paper.
    - **ftol** (float): stopping criteria on the x tolerance.
    - **xtol** (float): stopping criteria on the f tolerance.
    - **keep_adapted_params** (bool):  when true the adapted parameters are not reset
      between successive calls to the evolve method. Default is False.

```

```{eval-rst}
.. dropdown::  pygmo_simulated_annealing

    .. code-block::

        "pygmo_simulated_annealing"

    Minimize a function with the simulated annealing algorithm.

    This version of the simulated annealing algorithm is, essentially, an iterative
    random search procedure with adaptive moves along the coordinate directions. It
    permits uphill moves under the control of metropolis criterion, in the hope to avoid
    the first local minima encountered. This version is the one proposed in
    :cite:`Corana1987`.

    .. note: When selecting the starting and final temperature values it helps to think
        about the tempertaure as the deterioration in the objective function value that
        still has a 37% chance of being accepted.

    - **population_size** (int): Size of the population. If None, it's twice the number of
      parameters but at least 64.
    - **seed** (int): seed used by the internal random number generator.
    - **discard_start_params** (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.
    - **start_temperature** (float): starting temperature. Must be > 0.
    - **end_temperature** (float): final temperature. Our default (0.01) is lower than in
      pygmo and pagmo. The final temperature must be positive.
    - **n_temp_adjustments** (int): number of temperature adjustments in the annealing
      schedule.
    - **n_range_adjustments** (int): number of adjustments of the search range performed at
      a constant temperature.
    - **bin_size** (int): number of mutations that are used to compute the acceptance rate.
    - **start_range** (float): starting range for mutating the decision vector. It must lie
      between 0 and 1.
```

```{eval-rst}
.. dropdown::  pygmo_pso

    .. code-block::

        "pygmo_pso"

    Minimize a scalar function using Particle Swarm Optimization.

    Particle swarm optimization (PSO) is a population based algorithm inspired by the
    foraging behaviour of swarms. In PSO each point has memory of the position where it
    achieved the best performance xli (local memory) and of the best decision vector
    :math:`x^g` in a certain neighbourhood, and uses this information to update its
    position.

    For a survey on particle swarm optimization algorithms, see :cite:`Poli2007`.

    Each particle determines its future position :math:`x_{i+1} = x_i + v_i` where

    .. math:: v_{i+1} = \omega (v_i + \eta_1 \cdot \mathbf{r}_1 \cdot (x_i - x^{l}_i) +
        \eta_2 \cdot \mathbf{r}_2 \cdot (x_i - x^g))

    - **population_size** (int): Size of the population. If None, it's twice the number of
      parameters but at least 10.
    - **seed** (int): seed used by the internal random number generator.
    - **discard_start_params** (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.
    - **stopping.maxiter** (int): Number of generations to evolve.

    - **omega** (float): depending on the variant chosen, :math:`\omega` is the particles'
      inertia weight or the construction coefficient. It must lie between 0 and 1.
    - **force_of_previous_best** (float): :math:`\eta_1` in the equation above. It's the
      magnitude of the force, applied to the particle’s velocity, in the direction of
      its previous best position. It must lie between 0 and 4.
    - **force_of_best_in_neighborhood** (float): :math:`\eta_2` in the equation above. It's
      the magnitude of the force, applied to the particle’s velocity, in the direction
      of the best position in its neighborhood. It must lie between 0 and 4.
    - **max_velocity** (float): maximum allowed particle velocity as fraction of the box
      bounds. It must lie between 0 and 1.
    - **algo_variant (int or str)**: algorithm variant to be used:
        - 1 or "canonical_inertia": Canonical (with inertia weight)
        - 2 or "social_and_cog_rand": Same social and cognitive rand.
        - 3 or "all_components_rand": Same rand. for all components
        - 4 or "one_rand": Only one rand.
        - 5 or "canonical_constriction": Canonical (with constriction fact.)
        - 6 or "fips": Fully Informed (FIPS)

    - **neighbor_definition (int or str)**: swarm topology that defines each particle's
      neighbors that is to be used:

        - 1 or "gbest"
        - 2 or "lbest"
        - 3 or "Von Neumann"
        - 4 or "Adaptive random"

    - **neighbor_param** (int): the neighbourhood parameter. If the lbest topology is
      selected (neighbor_definition=2), it represents each particle's indegree (also
      outdegree) in the swarm topology. Particles have neighbours up to a radius of k =
      neighbor_param / 2 in the ring. If the Randomly-varying neighbourhood topology is
      selected (neighbor_definition=4), it represents each particle’s maximum outdegree
      in the swarm topology. The minimum outdegree is 1 (the particle always connects
      back to itself). If neighbor_definition is 1 or 3 this parameter is ignored.
    - **keep_velocities** (bool): when true the particle velocities are not reset between
      successive calls to `evolve`.
```

```{eval-rst}
.. dropdown::  pygmo_pso_gen

    .. code-block::

        "pygmo_pso_gen"

    Minimize a scalar function with generational Particle Swarm Optimization.

    Particle Swarm Optimization (generational) is identical to pso, but does update the
    velocities of each particle before new particle positions are computed (taking into
    consideration all updated particle velocities). Each particle is thus evaluated on
    the same seed within a generation as opposed to the standard PSO which evaluates
    single particle at a time. Consequently, the generational PSO algorithm is suited
    for stochastic optimization problems.

    For a survey on particle swarm optimization algorithms, see :cite:`Poli2007`.

    Each particle determines its future position :math:`x_{i+1} = x_i + v_i` where

    .. math:: v_{i+1} = \omega (v_i + \eta_1 \cdot \mathbf{r}_1 \cdot (x_i - x^{l}_i) +
        \eta_2 \cdot \mathbf{r}_2 \cdot (x_i - x^g))

    - **population_size** (int): Size of the population. If None, it's twice the number of
      parameters but at least 10.
    - **batch_evaluator (str or Callable)**: Name of a pre-implemented batch evaluator
      (currently 'joblib' and 'pathos_mp') or Callable with the same interface as the
      optimagic batch_evaluators. See :ref:`batch_evaluators`.
    - **n_cores** (int): Number of cores to use.
    - **seed** (int): seed used by the internal random number generator.
    - **discard_start_params** (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.
    - **stopping.maxiter** (int): Number of generations to evolve.

    - **omega** (float): depending on the variant chosen, :math:`\omega` is the particles'
      inertia weight or the constructuion coefficient. It must lie between 0 and 1.
    - **force_of_previous_best** (float): :math:`\eta_1` in the equation above. It's the
      magnitude of the force, applied to the particle’s velocity, in the direction of
      its previous best position. It must lie between 0 and 4.
    - **force_of_best_in_neighborhood** (float): :math:`\eta_2` in the equation above. It's
      the magnitude of the force, applied to the particle’s velocity, in the direction
      of the best position in its neighborhood. It must lie between 0 and 4.
    - **max_velocity** (float): maximum allowed particle velocity as fraction of the box
      bounds. It must lie between 0 and 1.
    - **algo_variant** (int): code of the algorithm's variant to be used:

        - 1 or "canonical_inertia": Canonical (with inertia weight)
        - 2 or "social_and_cog_rand": Same social and cognitive rand.
        - 3 or "all_components_rand": Same rand. for all components
        - 4 or "one_rand": Only one rand.
        - 5 or "canonical_constriction": Canonical (with constriction fact.)
        - 6 or "fips": Fully Informed (FIPS)

    - **neighbor_definition** (int): code for the swarm topology that defines each
      particle's neighbors that is to be used:

        - 1 or "gbest"
        - 2 or "lbest"
        - 3 or "Von Neumann"
        - 4 or "Adaptive random"

    - **neighbor_param** (int): the neighbourhood parameter. If the lbest topology is
      selected (neighbor_definition=2), it represents each particle's indegree (also
      outdegree) in the swarm topology. Particles have neighbours up to a radius of k =
      neighbor_param / 2 in the ring. If the Randomly-varying neighbourhood topology is
      selected (neighbor_definition=4), it represents each particle’s maximum outdegree
      in the swarm topology. The minimum outdegree is 1 (the particle always connects
      back to itself). If neighbor_definition is 1 or 3 this parameter is ignored.
    - **keep_velocities** (bool): when true the particle velocities are not reset between
      successive calls to `evolve`.
```

```{eval-rst}
.. dropdown::  pygmo_mbh

    .. code-block::

        "pygmo_mbh"

    Minimize a scalar function using generalized Monotonic Basin Hopping.

    Monotonic basin hopping, or simply, basin hopping, is an algorithm rooted in the
    idea of mapping the objective function $f(x_0)$ into the local minima found starting
    from $x_0$. This simple idea allows a substantial increase of efficiency in solving
    problems, such as the Lennard-Jones cluster or the MGA-1DSM interplanetary
    trajectory problem that are conjectured to have a so-called funnel structure.

    See :cite:`Wales1997` for the paper introducing the basin hopping idea for a
    Lennard-Jones cluster optimization.

    pygmo provides an original generalization of this concept resulting in a
    meta-algorithm that operates on a population. When a population containing a single
    individual is used the original method is recovered.

    - **population_size** (int): Size of the population. If None, it's twice the number of
      parameters but at least 250.
    - **seed** (int): seed used by the internal random number generator.
    - **discard_start_params** (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.
    - **inner_algorithm** (pygmo.algorithm): an pygmo algorithm or a user-defined algorithm,
      either C++ or Python. If None the `pygmo.compass_search` algorithm will be used.
    - **stopping.max_inner_runs_without_improvement** (int): consecutive runs of the inner
      algorithm that need to result in no improvement for mbh to stop.
    - **perturbation** (float): the perturbation to be applied to each component.
```

```{eval-rst}
.. dropdown::  pygmo_xnes

    .. code-block::

        "pygmo_xnes"

    Minimize a scalar function using Exponential Evolution Strategies.

    Exponential Natural Evolution Strategies is an algorithm closely related to CMAES
    and based on the adaptation of a gaussian sampling distribution via the so-called
    natural gradient. Like CMAES it is based on the idea of sampling new trial vectors
    from a multivariate distribution and using the new sampled points to update the
    distribution parameters. Naively this could be done following the gradient of the
    expected fitness as approximated by a finite number of sampled points. While this
    idea offers a powerful lead on algorithmic construction it has some major drawbacks
    that are solved in the so-called Natural Evolution Strategies class of algorithms by
    adopting, instead, the natural gradient. xNES is one of the most performing variants
    in this class.

    See :cite:`Glasmachers2010` and the `pagmo documentation on xNES
    <https://esa.github.io/pagmo2/docs/cpp/algorithms/xnes.html#_CPPv4N5pagmo4xnesE>`_
    for details.

    - **population_size** (int): Size of the population. If None, it's twice the number of
      parameters but at least 64.
    - **seed** (int): seed used by the internal random number generator.
    - **discard_start_params** (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.
    - **stopping.maxiter** (int): Number of generations to evolve.

    - **learning_rate_mean_update** (float): learning rate for the mean update
      (:math:`\eta_\mu`). It must be between 0 and 1 or None.
    - **learning_rate_step_size_update** (float): learning rate for the step-size update. It
      must be between 0 and 1 or None.
    - **learning_rate_cov_matrix_update** (float): learning rate for the covariance matrix
      update. It must be between 0 and 1 or None.
    - **initial_search_share** (float): share of the given search space that will be
      initally searched. It must be between 0 and 1. Default is 1.
    - **ftol** (float): stopping criteria on the x tolerance.
    - **xtol** (float): stopping criteria on the f tolerance.
    - **keep_adapted_params** (bool): when true the adapted parameters are not reset between
      successive calls to the evolve method. Default is False.
```

```{eval-rst}
.. dropdown::  pygmo_gwo

    .. code-block::

        "pygmo_gwo"

    Minimize a scalar function usinng the Grey Wolf Optimizer.

    The grey wolf optimizer was proposed by :cite:`Mirjalili2014`. The pygmo
    implementation that is wrapped by optimagic is pased on the pseudo code provided in
    that paper.

    This algorithm is a classic example of a highly criticizable line of search that led
    in the first decades of our millenia to the development of an entire zoo of
    metaphors inspiring optimzation heuristics. In our opinion they, as is the case for
    the grey wolf optimizer, are often but small variations of already existing
    heuristics rebranded with unnecessray and convoluted biological metaphors. In the
    case of GWO this is particularly evident as the position update rule is shokingly
    trivial and can also be easily seen as a product of an evolutionary metaphor or a
    particle swarm one. Such an update rule is also not particulary effective and
    results in a rather poor performance most of times.

    - **population_size** (int): Size of the population. If None, it's twice the number of
      parameters but at least 64.
    - **seed** (int): seed used by the internal random number generator.
    - **discard_start_params** (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.
    - **stopping.maxiter** (int): Number of generations to evolve.

```

```{eval-rst}
.. dropdown::  pygmo_compass_search

    .. code-block::

        "pygmo_compass_search"

    Minimize a scalar function using compass search.

    The algorithm is described in :cite:`Kolda2003`.

    It is considered slow but reliable. It should not be used for stochastic problems.

    - **population_size** (int): Size of the population. Even though the algorithm is not
      population based the population size does affect the results of the algorithm.
    - **seed** (int): seed used by the internal random number generator.
    - **discard_start_params** (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.
    - **stopping.maxfun** (int): maximum number of function evaluations.
    - **start_range** (float): the start range. Must be in (0, 1].
    - **stop_range** (float): the stop range. Must be in (0, start_range].
    - **reduction_coeff** (float): the range reduction coefficient. Must be in (0, 1).
```

```{eval-rst}
.. dropdown::  pygmo_ihs

    .. code-block::

        "pygmo_ihs"

    Minimize a scalar function using the improved harmony search algorithm.

    Improved harmony search (IHS) was introduced by :cite:`Mahdavi2007`.
    IHS supports stochastic problems.

    - **population_size** (int): Size of the population. If None, it's twice the number of
      parameters.
    - **seed** (int): seed used by the internal random number generator.
    - **discard_start_params** (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.
    - **stopping.maxiter** (int): Number of generations to evolve.
    - **choose_from_memory_probability** (float): probability of choosing from memory
      (similar to a crossover probability).
    - **min_pitch_adjustment_rate** (float): minimum pitch adjustment rate. (similar to a
      mutation rate). It must be between 0 and 1.
    - **max_pitch_adjustment_rate** (float): maximum pitch adjustment rate. (similar to a
      mutation rate). It must be between 0 and 1.
    - **min_distance_bandwidth** (float): minimum distance bandwidth. (similar to a mutation
      width). It must be positive.
    - **max_distance_bandwidth** (float): maximum distance bandwidth. (similar to a mutation
      width).
```

```{eval-rst}
.. dropdown::  pygmo_de1220

    .. code-block::

        "pygmo_de1220"

    Minimize a scalar function using Self-adaptive Differential Evolution, pygmo flavor.

    See `the PAGMO documentation for details
    <https://esa.github.io/pagmo2/docs/cpp/algorithms/de1220.html>`_.

    - **population_size** (int): Size of the population. If None, it's twice the number of
      parameters but at least 64.
    - **seed** (int): seed used by the internal random number generator.
    - **discard_start_params** (bool): If True, the start params are not guaranteed to be
      part of the initial population. This saves one criterion function evaluation that
      cannot be done in parallel with other evaluations. Default False.
    - **jde** (bool): Whether to use the jDE self-adaptation variant to control the $F$ and
      $CR$ parameter. If True jDE is used, else iDE.
    - **stopping.maxiter** (int): Number of generations to evolve.
    - **allowed_variants** (array-like object): allowed mutation variants (can be codes
      or strings). Each code refers to one mutation variant to create a new candidate
      individual. The first ten refer to the classical mutation variants introduced in
      the original DE algorithm, the remaining ones are, instead, considered in the work
      by :cite:`Elsayed2011`. The default is ["rand/1/exp", "rand-to-best/1/exp",
      "rand/1/bin", "rand/2/bin", "best/3/exp", "best/3/bin", "rand-to-current/2/exp",
      "rand-to-current/2/bin"]. The following are available:

        - 1 or "best/1/exp"
        - 2 or "rand/1/exp"
        - 3 or "rand-to-best/1/exp"
        - 4 or "best/2/exp"
        - 5 or "rand/2/exp"
        - 6 or "best/1/bin"
        - 7 or "rand/1/bin"
        - 8 or "rand-to-best/1/bin"
        - 9 or "best/2/bin"
        - 10 or "rand/2/bin"
        - 11 or "rand/3/exp"
        - 12 or "rand/3/bin"
        - 13 or "best/3/exp"
        - 14 or "best/3/bin"
        - 15 or "rand-to-current/2/exp"
        - 16 or "rand-to-current/2/bin"
        - 17 or "rand-to-best-and-current/2/exp"
        - 18 or "rand-to-best-and-current/2/bin"

    - **keep_adapted_params** (bool):  when true the adapted parameters $CR$ anf $F$ are not
      reset between successive calls to the evolve method. Default is False.
    - **ftol** (float): stopping criteria on the x tolerance.
    - **xtol** (float): stopping criteria on the f tolerance.

```

(ipopt-algorithm)=

## The Interior Point Optimizer (ipopt)

optimagic's support for the Interior Point Optimizer ({cite}`Waechter2005`,
{cite}`Waechter2005a`, {cite}`Waechter2005b`, {cite}`Nocedal2009`) is built on
[cyipopt](https://cyipopt.readthedocs.io/en/latest/index.html), a Python wrapper for the
[Ipopt optimization package](https://coin-or.github.io/Ipopt/index.html).

To use ipopt, you need to have
[cyipopt installed](https://cyipopt.readthedocs.io/en/latest/index.html)
(`conda install cyipopt`).

```{eval-rst}
.. dropdown:: ipopt

    .. code-block::

        "ipopt"

    Minimize a scalar function using the Interior Point Optimizer.

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

    The options are analogous to the ones in the `ipopt documentation
    <https://coin-or.github.io/Ipopt/OPTIONS.html#>`_ with the exception of the
    linear solver options which are here bundled into a dictionary. Any argument
    that takes "yes" and "no" in the ipopt documentation can also be passed as a
    `True` and `False`, respectively. and any option that accepts "none" in
    ipopt accepts a Python `None`.

    The following options are not supported:
      - `num_linear_variables`: since optimagic may reparametrize your problem
        and this changes the parameter problem, we do not support this option.
      - derivative checks
      - print options.


    - **convergence.ftol_rel** (float): The algorithm
      terminates successfully, if the (scaled) non linear programming error
      becomes smaller than this value.

    - **mu_target** (float): Desired value of complementarity. Usually, the barrier
      parameter is driven to zero and the termination test for complementarity
      is measured with respect to zero complementarity. However, in some cases
      it might be desired to have Ipopt solve barrier problem for strictly
      positive value of the barrier parameter. In this case, the value of
      "mu_target" specifies the final value of the barrier parameter, and the
      termination tests are then defined with respect to the barrier problem for
      this value of the barrier parameter. The valid range for this real option
      is 0 ≤ mu_target  and its default value is 0.

    - **s_max** (float): Scaling threshold for the NLP error.

    - **stopping.maxiter** (int):  If the maximum number of iterations is
      reached, the optimization stops, but we do not count this as successful
      convergence. The difference to ``max_criterion_evaluations`` is that one
      iteration might need several criterion evaluations, for example in a line
      search or to determine if the trust region radius has to be shrunk.
    - **stopping.max_wall_time_seconds** (float): Maximum number of walltime clock seconds.
    - **stopping.max_cpu_time** (float): Maximum number of CPU seconds.
      A limit on CPU seconds that Ipopt can use to solve one problem.
      If during the convergence check this limit is exceeded, Ipopt will
      terminate with a corresponding message. The valid range for this
      real option is 0 < max_cpu_time and its default value is :math:`1e+20` .

    - **dual_inf_tol** (float): Desired threshold for the dual infeasibility.
      Absolute tolerance on the dual infeasibility. Successful termination
      requires that the max-norm of the (unscaled) dual infeasibility is less
      than this threshold. The valid range for this real option is 0 <
      dual_inf_tol and its default value is 1.
    - **constr_viol_tol** (float): Desired threshold for the constraint and bound
      violation. Absolute tolerance on the constraint and variable bound
      violation. Successful termination requires that the max-norm of the
      (unscaled) constraint violation is less than this threshold.
      If option ``bound_relax_factor``  is not zero 0, then Ipopt relaxes given variable bounds.
      The value of constr_viol_tol is used to restrict the absolute amount of this bound
      relaxation. The valid range for this real option is 0 < constr_viol_tol
      and its default value is 0.0001.
    - **compl_inf_tol** (float): Desired threshold for the complementarity conditions.
      Absolute tolerance on the complementarity. Successful termination
      requires that the max-norm of the (unscaled) complementarity is
      less than this threshold. The valid range for this real option is
      0 < text{compl_inf_tol and its default is 0.0001.
    - **acceptable_iter** (int): Number of "acceptable" iterates before termination.
      If the algorithm encounters this many successive "acceptable"
      iterates (see above on the acceptable heuristic), it terminates, assuming
      that the problem has been solved to best possible accuracy given
      round-off. If it is set to zero, this heuristic is disabled. The valid
      range for this integer option is 0 ≤ acceptable_iter.
    - **acceptable_tol** (float):"Acceptable" convergence tolerance (relative).
      Determines which (scaled) overall optimality error is considered to be "acceptable".
      The valid range for this real option is 0 < acceptable_tol.
    - **acceptable_dual_inf_tol** (float):  "Acceptance" threshold for the dual
      infeasibility. Absolute tolerance on the dual infeasibility. "Acceptable"
      termination requires that the (max-norm of the unscaled) dual
      infeasibility is less than this threshold; see also  ``acceptable_tol`` . The
      valid range for this real option is 0 < acceptable_dual_inf_tol and its
      default value is :math:`1e+10.`
    - **acceptable_constr_viol_tol** (float): "Acceptance" threshold for the constraint violation.
      Absolute tolerance on the constraint violation.
      "Acceptable" termination requires that the max-norm
      of the (unscaled) constraint violation is less than this threshold; see
      also  ``acceptable_tol`` . The valid range for this real option is 0 <
      acceptable_constr_viol_tol and its default value is 0.01.
    - **acceptable_compl_inf_tol** (float): "Acceptance" threshold for the
      complementarity conditions. Absolute tolerance on the complementarity.
      "Acceptable" termination requires that the max-norm of the (unscaled)
      complementarity is less than this threshold; see also  ``acceptable_tol`` . The
      valid range for this real option is 0 < text{acceptable_compl_inf_tol and its
      default value is 0.01.
    - **acceptable_obj_change_tol** (float): "Acceptance" stopping criterion based on
      objective function change. If the relative
      change of the objective function (scaled by :math:`max(1,|f(x)|)` ) is less than
      this value, this part of the acceptable tolerance termination is
      satisfied; see also  ``acceptable_tol`` . This is useful for the quasi-Newton
      option, which has trouble to bring down the dual infeasibility. The valid
      range for this real option is 0 ≤ acceptable_obj_change_tol and its
      default value is :math:`1e+20` .

    - **diverging_iterates_tol** (float): Threshold for maximal value of primal iterates.
      If any component of the primal iterates exceeded this value (in
      absolute terms), the optimization is aborted with the exit message that
      the iterates seem to be diverging. The valid range for this real option is
      0 < diverging_iterates_tol and its default value is :math:`1e+20` .
    - **nlp_lower_bound_inf** (float): any bound less or equal this value will be
      considered -inf (i.e. not lwer bounded). The valid range for this real
      option is unrestricted and its default value is :math:`-1e+19` .
    - **nlp_upper_bound_inf** (float): any bound greater or this value will be
      considered :math:`+\inf` (i.e. not upper bunded). The valid range for this real
      option is unrestricted and its default value is :math:`1e+19` .
    - **fixed_variable_treatment (str)**: Determines how fixed variables should be
      handled. The main difference between those options is that the starting
      point in the "make_constraint" case still has the fixed variables at their
      given values, whereas in the case "make_parameter(_nodual)" the functions
      are always evaluated with the fixed values for those variables. Also, for
      "relax_bounds", the fixing bound constraints are relaxed (according to
      ``bound_relax_factor`` ). For all but "make_parameter_nodual", bound
      multipliers are computed for the fixed variables. The default value for
      this string option is "make_parameter". Possible values:

             - "make_parameter": Remove fixed variable from optimization variables
             - "make_parameter_nodual": Remove fixed variable from optimization
               variables and do not compute bound multipliers for fixed variables
             - "make_constraint": Add equality constraints fixing variables
             - "relax_bounds": Relax fixing bound constraints
    - **dependency_detector (str)**: Indicates which linear solver
      should be used to detect linearly dependent equality constraints. This is
      experimental and does not work well. The default value for this string
      option is "none". Possible values:

            - "none" or None: don't check; no extra work at beginning
            - "mumps": use MUMPS
            - "wsmp": use WSMP
            - "ma28": use MA28
    - **dependency_detection_with_rhs (str or bool)**: Indicates if the right hand
      sides of the constraints should be considered in addition to gradients
      during dependency detection. The default value for this string option is
      "no". Possible values: 'yes', 'no', True, False.

    - **kappa_d** (float): Weight for linear damping term (to handle one-sided bounds).
      See Section 3.7 in implementation paper. The valid range for this
      real option is 0 ≤ kappa_d and its default value is :math:`1e-05` .
    - **bound_relax_factor** (float): Factor for initial relaxation of the bounds.
      Before start of the optimization, the bounds given by the user are
      relaxed. This option sets the factor for this relaxation. Additional, the
      constraint violation tolerance  ``constr_viol_tol``  is used to bound the
      relaxation by an absolute value. If it is set to zero, then then bounds
      relaxation is disabled. See Eqn.(35) in implementation paper. Note that
      the constraint violation reported by Ipopt at the end of the solution
      process does not include violations of the original (non-relaxed) variable
      bounds. See also option honor_original_bounds. The valid range for this
      real option is 0 ≤ bound_relax_factor  and its default value is :math:`1e-08` .
    - **honor_original_bounds** (str or bool): Indicates whether final points should
      be projected into original bunds. Ipopt might relax the bounds during the
      optimization (see, e.g., option  ``bound_relax_factor`` ). This option
      determines whether the final point should be projected back into the
      user-provide original bounds after the optimization. Note that violations
      of constraints and complementarity reported by Ipopt at the end of the
      solution process are for the non-projected point. The default value for
      this string option is "no". Possible values: 'yes', 'no', True, False

    - **check_derivatives_for_naninf (str)**: whether to check for NaN / inf in the
      derivative matrices.
      Activating this option will cause an error if an
      invalid number is detected in the constraint Jacobians or the Lagrangian
      Hessian. If this is not activated, the test is skipped, and the algorithm
      might proceed with invalid numbers and fail. If test is activated and an
      invalid number is detected, the matrix is written to output with
      print_level corresponding to J_MORE_DETAILED; so beware of large output!
      The default value for this string option is "no".
    - **jac_c_constant (str or bool)**: Indicates whether to assume that all equality
      constraints are linear Activating this option will cause Ipopt to ask
      for the Jacobian of the equality constraints only once from the NLP and
      reuse this information later. The default value for this string option
      is "no". Possible values: yes, no, True, False.
    - **jac_d_constant (str or bool)**: Indicates whether to
      assume that all inequality constraints are linear Activating this option
      will cause Ipopt to ask for the Jacobian of the inequality constraints
      only once from the NLP and reuse this information later. The default value
      for this string option is "no". Possible values: yes, no, True, False
    - **hessian_constant (str or bool)**: Indicates whether to assume the problem is a QP
      (quadratic objective, linear constraints). Activating this option will
      cause Ipopt to ask for the Hessian of the Lagrangian function only once
      from the NLP and reuse this information later. The default value for this
      string option is "no". Possible values: yes, no, True, False.

    - **nlp_scaling_method (str)**: Select the technique used for scaling the NLP.
      Selects the technique used for scaling the problem internally before it is
      solved. For user-scaling, the parameters come from the NLP. If you are
      using AMPL, they can be specified through suffixes ("scaling_factor") The
      default value for this string option is "gradient-based". Possible values:

            - "none": no problem scaling will be performed - "user-scaling": scaling
              parameters will come from the user - "gradient-based":
              scale the problem so the maximum gradient at the starting point is
              ``nlp_scaling_max_gradient`` .
            - "equilibration-based": scale the problem so that first derivatives are
              of order 1 at random points (uses Harwell routine MC19)
    - **obj_scaling_factor** (float): Scaling factor for the objective function.
      This option sets a scaling factor for the objective function. The
      scaling is seen internally by Ipopt but the unscaled objective is
      reported in the console output. If additional scaling parameters are
      computed (e.g. user-scaling or gradient-based), both factors are
      multiplied. If this value is chosen to be negative, Ipopt will maximize
      the objective function instead of minimizing it. The valid range for
      this real option is unrestricted and its default value is 1.
    - **nlp_scaling_max_gradient** (float): Maximum gradient after NLP scaling.
      This is the gradient scaling cut-off. If the maximum gradient is above
      this value, then gradient based scaling will be performed. Scaling
      parameters are calculated to scale the maximum gradient back to this
      value. (This is g_max in Section 3.8 of the implementation paper.) Note:
      This option is only used if  ``nlp_scaling_method``  is chosen as
      "gradient-based". The valid range for this real option is :math:`0 <
      \text{nlp_scaling_max_gradient}` and its default value is 100.
    - **nlp_scaling_obj_target_gradient** (float): advanced! Target value for
      objective function gradient size. If a positive number is chosen, the
      scaling factor for the objective function is computed so that the
      gradient has the max norm of the given size at the starting point. This
      overrides  ``nlp_scaling_max_gradient``  for the objective function. The valid
      range for this real option is 0 ≤ nlp_scaling_obj_target_gradient and
      its default value is 0.
    - **nlp_scaling_constr_target_gradient** (float): Min value of gradient-based
      scaling values.
      This is the lower bound for the scaling factors computed by
      gradient-based scaling method. If
      some derivatives of some functions are huge, the scaling factors will
      otherwise become very small, and the (unscaled) final constraint
      violation, for example, might then be significant. Note: This option is
      only used if  ``nlp_scaling_method`` is chosen as "gradient-based". The
      valid range for this real option is 0 ≤ nlp_scaling_min_value and its
      default value is :math:`1e-08`.
    - **nlp_scaling_min_value** (float): Minimum value of
      gradient-based scaling values. This is the lower bound for the scaling
      factors computed by gradient-based scaling method. If some derivatives
      of some functions are huge, the scaling factors will otherwise become
      very small, and the (unscaled) final constraint violation, for example,
      might then be significant. Note: This option is only used if
      ``nlp_scaling_method`` is chosen as "gradient-based". The valid range for
      this real option is 0 ≤ nlp_scaling_min_value and its default value is
      :math:`1e-08`.

    - **bound_push** (float): Desired minimum absolute distance from the initial
      point to bound. Determines how much the initial point might have to be
      modified in order to be sufficiently inside the bounds (together with
      ``bound_frac`` ). (This is kappa_1 in Section 3.6 of implementation paper.)
      The valid range for this real option is 0 < bound_push and its default
      value is 0.01.
    - **bound_frac** (float): Desired minimum relative distance
      from the initial point to bound. Determines how much the initial point
      might have to be modified in order to be sufficiently inside the bounds
      (together with "bound_push"). (This is kappa_2 in Section 3.6 of
      implementation paper.) The valid range for this real option is 0 <
      bound_frac ≤ 0.5 and its default value is 0.01.
    - **slack_bound_push** (float): Desired minimum absolute distance from the
      initial slack to bound. Determines how much the initial slack
      variables might have to be modified in order to be sufficiently inside the inequality bounds
      (together with  ``slack_bound_frac`` ). (This is kappa_1 in Section 3.6 of
      implementation paper.) The valid range for this real option is 0 <
      slack_bound_push and its default value is 0.01.
    - **slack_bound_frac** (float): Desired minimum relative distance from the
      initial slack to bound. Determines how much the initial slack
      variables might have to be modified in order to be sufficiently inside the inequality bounds
      (together with  ``slack_bound_push`` ). (This is kappa_2 in Section 3.6 of
      implementation paper.) The valid range for this real option is 0 <
      slack_bound_frac ≤ 0.5 and its default value is 0.01.
    - **constr_mult_init_max** (float): Maximum allowed least-square guess of
      constraint multipliers. Determines how large the initial least-square
      guesses of the constraint multipliers are allowed to be (in max-norm).
      If the guess is larger than this value, it is discarded and all
      constraint multipliers are set to zero. This options is also used when
      initializing the restoration phase. By default,
      "resto.constr_mult_init_max" (the one used in RestoIterateInitializer)
      is set to zero. The valid range for this real option is 0 ≤
      constr_mult_init_max and its default value is 1000.
    - **bound_mult_init_val** (float): Initial value for the bound multipliers.
      All dual variables corresponding to bound constraints are initialized
      to this value. The valid range for this real option is
      0 < bound_mult_init_val and its default value is 1.
    - **bound_mult_init_method (str)**: Initialization method
      for bound multipliers This option defines how the iterates for the bound
      multipliers are initialized. If "constant" is chosen, then all bound
      multipliers are initialized to the value of  ``bound_mult_init_val``. If
      "mu-based" is chosen, the each value is initialized to the the value of
      "mu_init" divided by the corresponding slack variable. This latter
      option might be useful if the starting point is close to the optimal
      solution. The default value for this string option is "constant".
      Possible values:

            - "constant": set all bound multipliers to the value of  ``bound_mult_init_val``
            - "mu-based": initialize to mu_init/x_slack
    - **least_square_init_primal (str or bool)**:
      Least square initialization of the primal variables. If set to
      yes, Ipopt ignores the user provided point and solves a least square
      problem for the primal variables (x and s) to fit the linearized
      equality and inequality constraints.This might be useful if the user
      doesn't know anything about the starting point, or for solving an LP or
      QP. The default value for this string option is "no".  Possible values:

            - "no": take user-provided point
            - "yes": overwrite user-provided point with least-square estimates
    - **least_square_init_duals (str or bool)**: Least square
      initialization of all dual variables If set to yes, Ipopt tries to
      compute least-square multipliers (considering ALL dual variables). If
      successful, the bound multipliers are possibly corrected to be at
      least  ``bound_mult_init_val`` . This might be useful if the user doesn't
      know anything about the starting point, or for solving an LP or QP.
      This overwrites option  ``bound_mult_init_method`` . The default value for
      this string option is "no". Possible values:

            - "no": use  ``bound_mult_init_val``  and least-square equality constraint multipliers
            - "yes": overwrite user-provided point with least-square estimates
    - **warm_start_init_point (str or bool)**: Warm-start for initial point
      Indicates whether this optimization should use a warm start
      initialization, where values of primal and dual variables are given
      (e.g., from a previous optimization of a related problem.) The default
      value for this string option is "no". Possible values:

            - "no" or False: do not use the warm start initialization
            - "yes" or True: use the warm start initialization
    - **warm_start_same_structure (str or bool)**:
      Advanced feature! Indicates whether a problem with a structure
      identical t the previous one is to be solved. If enabled, then the
      algorithm assumes that an NLP is now to be solved whose structure is
      identical to one that already was considered (with the same NLP
      object). The default value for this string option is "no". Possible
      values: yes, no, True, False.
    - **warm_start_bound_push** (float): same as
      ``bound_push`` for the regular initializer. The valid range for this real
      option is 0 < warm_start_bound_push and its default value is 0.001.
    - **warm_start_bound_frac** (float): same as  ``bound_frac``  for the regular
      initializer The valid range for this real option is 0 <
      warm_start_bound_frac ≤ 0.5 and its default value is 0.001.
    - **warm_start_slack_bound_push** (float): same as  ``slack_bound_push``  for the
      regular initializer The valid range for this real option is 0 <
      warm_start_slack_bound_push and its default value is 0.001.
    - **warm_start_slack_bound_frac** (float): same as  ``slack_bound_frac``  for the
      regular initializer The valid range for this real option is 0 <
      warm_start_slack_bound_frac ≤ 0.5 and its default value is 0.001.
    - **warm_start_mult_bound_push** (float): same as  ``mult_bound_push``  for the
      regular initializer The valid range for this real option is 0 <
      warm_start_mult_bound_push and its default value is 0.001.
    - **warm_start_mult_init_max** (float): Maximum initial value for the
      equality multipliers. The valid range for this real option is
      unrestricted and its default value is :math:`1e+06` .
    - **warm_start_entire_iterate (str or bool)**: Tells algorithm whether to use the GetWarmStartIterate
      method in the NLP. The default value for this string option is "no".
      Possible values:

            - "no": call GetStartingPoint in the NLP
            - "yes": call GetWarmStartIterate in the NLP
    - **warm_start_target_mu** (float): Advanced and experimental! The valid range
      for this real option is unrestricted and its default value is 0.

    - **option_file_name (str)**: File name of options file. By default, the name
      of the Ipopt options file is "ipopt.opt" - or something else if
      specified in the IpoptApplication::Initialize call. If this option is
      set by SetStringValue BEFORE the options file is read, it specifies the
      name of the options file. It does not make any sense to specify this
      option within the options file. Setting this option to an empty string
      disables reading of an options file.
    - **replace_bounds (bool or str)**:
      Whether all variable bounds should be replaced by inequality
      constraints. This option must be set for the inexact algorithm. The
      default value for this string option is "no". Possible values: "yes",
      "no", True, False.
    - **skip_finalize_solution_call (str or bool)**: Whether a
      call to NLP::FinalizeSolution after optimization should be suppressed.
      In some Ipopt applications, the user might want to call the
      FinalizeSolution method separately. Setting this option to "yes" will
      cause the IpoptApplication object to suppress the default call to that
      method. The default value for this string option is "no". Possible
      values: "yes", "no", True, False
    - **timing_statistics (str or bool)**:
      Indicates whether to measure time spend in components of Ipopt and NLP
      evaluation.  The overall algorithm time is unaffected by this option.
      The default value for this string option is "no". Possible values:
      "yes", "no", True, False

    - **mu_max_fact** (float): Factor for initialization of maximum value for
      barrier parameter. This option determines the upper bound on the barrier
      parameter. This upper bound is computed as the average complementarity
      at the initial point times the value of this option. (Only used if
      option "mu_strategy" is chosen as "adaptive".) The valid range for this
      real option is 0 < mu_max_fact and its default value is 1000.
    - **mu_max** (float): Maximum value for barrier parameter. This option specifies an
      upper bound on the barrier parameter in the adaptive mu selection mode.
      If this option is set, it overwrites the effect of mu_max_fact. (Only
      used if option "mu_strategy" is chosen as "adaptive".) The valid range
      for this real option is 0 < mu_max and its default value is
      100000.
    - **mu_min** (float): Minimum value for barrier parameter. This option
      specifies the lower bound on the barrier parameter in the adaptive mu
      selection mode. By default, it is set to the minimum of :math:`1e-11`  and
      min( ``tol`` , ``compl_inf_tol`` )/( ``barrier_tol_factor`` +1), which should be a
      reasonable value. (Only used if option  ``mu_strategy``  is chosen as
      "adaptive".) The valid range for this real option is 0 < mu_min and its
      default value is :math:`1e-11` .
    - **adaptive_mu_globalization (str)**: Globalization
      strategy for the adaptive mu selection mode. To achieve global
      convergence of the adaptive version, the algorithm has to switch to the
      monotone mode (Fiacco-McCormick approach) when convergence does not seem
      to appear. This option sets the criterion used to decide when to do this
      switch. (Only used if option "mu_strategy" is chosen as "adaptive".) The
      default value for this string option is "obj-constr-filter". Possible
      values:

            - "kkt-error": nonmonotone decrease of kkt-error
            - "obj-constr-filter": 2-dim filter for objective and constraint violation
            - "never-monotone-mode": disables globalization.
    - **adaptive_mu_kkterror_red_iters** (float): advanced feature! Maximum
      number of iterations requiring sufficient progress. For the
      "kkt-error" based globalization strategy, sufficient progress must be
      made for "adaptive_mu_kkterror_red_iters" iterations. If this number
      of iterations is exceeded, the globalization strategy switches to the
      monotone mode. The valid range for this integer option is 0 ≤
      adaptive_mu_kkterror_red_iters and its default value is 4.
    - **adaptive_mu_kkterror_red_fact** (float): advanced feature! Sufficient
      decrease factor for "kkt-error" globalization strategy. For the
      "kkt-error" based globalization strategy, the error must decrease by
      this factor to be deemed sufficient decrease. The valid range for this
      real option is 0 < adaptive_mu_kkterror_red_fact < 1 and its default
      value is 0.9999.
    - **filter_margin_fact** (float): advanced feature! Factor
      determining width of margin for obj-constr-filter adaptive
      globalization strategy. When using the adaptive globalization
      strategy, "obj-constr-filter", sufficient progress for a filter entry
      is defined as follows: (new obj) < (filter obj) -
      filter_margin_fact*(new constr-viol) OR (new constr-viol) < (filter
      constr-viol) - filter_margin_fact*(new constr-viol). For the
      description of the "kkt-error-filter" option see  ``filter_max_margin`` .
      The valid range for this real option is 0 < filter_margin_fact < 1 and
      its default value is :math:`10-05` .
    - **filter_max_margin** (float): advanced
      feature! Maximum width of margin in obj-constr-filter adaptive
      globalization strategy. The valid range for this real option is 0 <
      filter_max_margin and its default value is 1.
    - **adaptive_mu_restore_previous_iterate (str or bool)**: advanced feature!
      Indicates if the previous accepted iterate should be restored if the
      monotone mode is entered. When the globalization strategy for the
      adaptive barrier algorithm switches to the monotone mode, it can
      either start from the most recent iterate (no), or from the last
      iterate that was accepted (yes). The default value for this string
      option is "no". Possible values: "yes", "no", True, False
    - **adaptive_mu_monotone_init_factor** (float): advanced feature! Determines
      the initial value of the barrier parameter when switching to the
      monotone mode. When the globalization strategy for the adaptive
      barrier algorithm switches to the monotone mode and fixed_mu_oracle is
      chosen as "average_compl", the barrier parameter is set to the current
      average complementarity times the value of
      "adaptive_mu_monotone_init_factor". The valid range for this real
      option is 0 < adaptive_mu_monotone_init_factor and its default value
      is 0.8.
    - **adaptive_mu_kkt_norm_type (str)**: advanced! Norm used for the KKT
      error in the adaptive mu globalization strategies. When computing the
      KKT error for the globalization strategies, the norm to be used is
      specified with this option. Note, this option is also used in the
      QualityFunctionMuOracle. The default value for this string option is
      "2-norm-squared". Possible values:

            - "1-norm": use the 1-norm (abs sum)
            - "2-norm-squared": use the 2-norm squared (sum of squares)
            - "max-norm": use the infinity norm (max)
            - "2-norm": use 2-norm
    - **mu_strategy (str)**: Update strategy for barrier
      parameter. Determines which barrier parameter update strategy is to be
      used. The default value for this string option is "monotone". Possible values:

            - "monotone": use the monotone (Fiacco-McCormick) strategy
            - "adaptive": use the adaptive update strategy
    - **mu_oracle (str)**: Oracle for a new barrier parameter in the adaptive strategy.
      Determines how a new barrier parameter is computed in each "free-mode" iteration of the
      adaptive barrier parameter strategy. (Only considered if "adaptive" is
      selected for option "mu_strategy"). The default value for this string
      option is "quality-function". Possible values:

            - "probing": Mehrotra's probing heuristic
            - "loqo": LOQO's centrality rule
            - "quality-function": minimize a quality function
    - **fixed_mu_oracle (str)**:
      Oracle for the barrier parameter when switching to fixed mode.
      Determines how the first value of the barrier parameter should be
      computed when switching to the "monotone mode" in the adaptive
      strategy. (Only considered if "adaptive" is selected for option
      "mu_strategy".) The default value for this string option is
      "average_compl". Possible values:

            - "probing": Mehrotra's probing heuristic
            - "loqo": LOQO's centrality rule
            - "quality-function": minimize a quality function
            - "average_compl": base on current average complementarity
    - **mu_init** (float): Initial value for the barrier parameter. This option
      determines the initial value for the barrier parameter (mu). It is
      only relevant in the monotone, Fiacco-McCormick version of the
      algorithm. (i.e., if "mu_strategy" is chosen as "monotone") The valid
      range for this real option is 0 < mu_init and its default value is 0.1.
    - **barrier_tol_factor** (float): Factor for mu in barrier stop test.
      The convergence tolerance for each barrier problem in the monotone
      mode is the value of the barrier parameter times "barrier_tol_factor".
      This option is also used in the adaptive mu strategy during the
      monotone mode. This is kappa_epsilon in implementation paper. The
      valid range for this real option is 0 < barrier_tol_factor and its
      default value is 10.
    - **mu_linear_decrease_factor** (float): Determines
      linear decrease rate of barrier parameter. For the Fiacco-McCormick
      update procedure the new barrier parameter mu is obtained by taking
      the minimum of mu*"mu_linear_decrease_factor" and
      mu^"superlinear_decrease_power". This is kappa_mu in implementation
      paper. This option is also used in the adaptive mu strategy during the
      monotone mode. The valid range for this real option is 0 <
      mu_linear_decrease_factor < 1 and its default value is 0.2.
    - **mu_superlinear_decrease_power** (float): Determines superlinear decrease
      rate of barrier parameter. For the Fiacco-McCormick update procedure
      the new barrier parameter mu is obtained by taking the minimum of
      mu*"mu_linear_decrease_factor" and mu^"superlinear_decrease_power".
      This is theta_mu in implementation paper. This option is also used in
      the adaptive mu strategy during the monotone mode. The valid range for
      this real option is 1 < mu_superlinear_decrease_power < 2 and its
      default value is 1.5.
    - **mu_allow_fast_monotone_decrease (str or bool)**:
      Advanced feature! Allow skipping of barrier problem if barrier test i
      already met. The default value for this string option is "yes".
      Possible values:

            - "no": Take at least one iteration per barrier problem even if the
              barrier test is already met for the updated barrier parameter
            - "yes": Allow fast decrease of mu if barrier test it met
    - **tau_min** (float): Advanced feature! Lower bound on fraction-to-the-boundary
      parameter tau. This is tau_min in the implementation paper. This
      option is also used in the adaptive mu strategy during the monotone
      mode. The valid range for this real option is 0 < tau_min < 1 and its
      default value is 0.99.
    - **sigma_max** (float): Advanced feature! Maximum
      value of the centering parameter. This is the upper bound for the
      centering parameter chosen by the quality function based barrier
      parameter update. Only used if option "mu_oracle" is set to
      "quality-function". The valid range for this real option is 0 <
      sigma_max and its default value is 100.

    - **sigma_min** (float): Advanced
      feature! Minimum value of the centering parameter. This is the lower
      bound for the centering parameter chosen by the quality function based
      barrier parameter update. Only used if option "mu_oracle" is set to
      "quality-function". The valid range for this real option is 0 ≤
      sigma_min and its default value is :math:`10-06` .
    - **quality_function_norm_type (str)**: Advanced feature.
      Norm used for components of the quality
      function. Only used if option "mu_oracle" is set to
      "quality-function". The default value for this string option is
      "2-norm-squared". Possible values:

            - "1-norm": use the 1-norm (abs sum)
            - "2-norm-squared": use the 2-norm squared (sum of squares)
            - "max-norm": use the infinity norm (max)
            - "2-norm": use 2-norm
    - **quality_function_centrality (str)**: Advanced
      feature. The penalty term for centrality that is included in quality
      function. This determines whether a term is added to the quality
      function to penalize deviation from centrality with respect to
      complementarity. The complementarity measure here is the xi in the
      Loqo update rule. Only used if option "mu_oracle" is set to
      "quality-function". The default value for this string option is
      "none". Possible values:

            - "none": no penalty term is added
            - "log": complementarity * the log of the centrality measure
            - "reciprocal": complementarity * the reciprocal of the centrality
              measure
            - "cubed-reciprocal": complementarity * the reciprocal of the centrality
              measure cubed
    - **quality_function_balancing_term (str)**: Advanced
      feature. The balancing term included in the quality function for
      centrality. This determines whether a term is added to the quality
      function that penalizes situations where the complementarity is much
      smaller than dual and primal infeasibilities. Only used if option
      "mu_oracle" is set to "quality-function". The default value for this
      string option is "none". Possible values:

            - "none": no balancing term is adde
            - "cubic":  :math:`max(0,\max(\text{dual_inf},\text{primal_inf})-\text{compl})^3`
    - **quality_function_max_section_steps** (int): Maximum number of search
      steps during direct search procedure determining the optimal centering
      parameter. The golden section search is performed for the quality
      function based mu oracle. Only used if option "mu_oracle" is set to
      "quality-function". The valid range for this integer option is 0 ≤
      quality_function_max_section_steps and its default value is 8.
    - **quality_function_section_sigma_tol** (float): advanced feature!
      Tolerance for the section search procedure determining the optimal
      centering parameter (in sigma space). The golden section search is
      performed for the quality function based mu oracle. Only used if
      option "mu_oracle" is set to "quality-function". The valid range for
      this real option is 0 ≤ quality_function_section_sigma_tol < 1 and its
      default value is 0.01.
    - **quality_function_section_qf_tol** (float):
      advanced feature! Tolerance for the golden section search procedure
      determining the optimal centering parameter (in the function value
      space). The golden section search is performed for the quality
      function based mu oracle. Only used if option "mu_oracle" is set to
      "quality-function". The valid range for this real option is 0 ≤
      quality_function_section_qf_tol < 1 and its default value is 0.

    - **line_search_method (str)**: Advanced feature. Globalization method used in
      backtracking line search. Only the "filter" choice is officially
      supported. But sometimes, good results might be obtained with the other
      choices. The default value for this string option is "filter". Possible values:

             - "filter": Filter method
             - "cg-penalty": Chen-Goldfarb penalty function
             - "penalty": Standard penalty function
    - **alpha_red_factor** (float): Advanced feature.
      Fractional reduction of the trial step size
      in the backtracking lne search. At every step of the backtracking line
      search, the trial step size is reduced by this factor. The valid range
      for this real option is 0 < alpha_red_factor < 1 and its default value
      is 0.5.
    - **accept_every_trial_step (str or bool)**: Always accept the first
      trial step. Setting this option to "yes" essentially disables the line
      search and makes the algorithm take aggressive steps, without global
      convergence guarantees. The default value for this string option is
      "no". Possible values: "yes", "no", True, False.
    - **accept_after_max_steps** (float): advanced feature.
      Accept a trial point after maximal this
      number of steps een if it does not satisfy line search conditions.
      Setting this to -1 disables this option. The valid range for this
      integer option is -1 ≤ accept_after_max_steps and its default value is -1.
    - **alpha_for_y (str)**: Method to determine the step size for constraint
      multipliers (alpha_y) . The default value for this string option is
      "primal". Possible values:

            - "primal": use primal step size
            - "bound-mult": use step size for the bound multipliers (good for LPs)
            - "min": use the min of primal and bound multipliers
            - "max": use the max of primal and bound multipliers
            - "full": take a full step of size one
            - "min-dual-infeas": choose step size minimizing new dual infeasibility
            - "safer-min-dual-infeas": like "min_dual_infeas", but safeguarded by
              "min" and "max"
            - "primal-and-full": use the primal step size, and full step if
              delta_x <= alpha_for_y_tol
            - "dual-and-full": use the dual step size, and full step if
              delta_x <= alpha_for_y_tol
            - "acceptor": Call LSAcceptor to get step size for y
    - **alpha_for_y_tol** (float): Tolerance for
      switching to full equality multiplier steps. This is only relevant if
      "alpha_for_y" is chosen "primal-and-full" or "dual-and-full". The step
      size for the equality constraint multipliers is taken to be one if the
      max-norm of the primal step is less than this tolerance. The valid range
      for this real option is 0 ≤ alpha_for_y_tol and its default value is 10.
    - **tiny_step_tol** (float): Advanced feature. Tolerance for detecting
      numerically insignificant steps. If the search direction in the primal
      variables (x and s) is, in relative terms for each component, less than
      this value, the algorithm accepts the full step without line search. If
      this happens repeatedly, the algorithm will terminate with a
      corresponding exit message. The default value is 10 times machine
      precision. The valid range for this real option is 0 ≤ tiny_step_tol and
      its default value is 2.22045 · :math:`1e-15`.
    - **tiny_step_y_tol** (float): Advanced
      feature. Tolerance for quitting because of numerically insignificant
      steps. If the search direction in the primal variables (x and s) is, in
      relative terms for each component, repeatedly less than tiny_step_tol,
      and the step in the y variables is smaller than this threshold, the
      algorithm will terminate. The valid range for this real option is 0 ≤
      tiny_step_y_tol and its default value is 0.01.

    - **watchdog_shortened_iter_trigger** (int): Number of shortened iterations
      that trigger the watchdog. If the number of successive iterations in
      which the backtracking line search did not accept the first trial point
      exceeds this number, the watchdog procedure is activated. Choosing "0"
      here disables the watchdog procedure. The valid range for this integer
      option is 0 ≤ watchdog_shortened_iter_trigger and its default value is
      10.
    - **watchdog_trial_iter_max** (int): Maximum number of watchdog
      iterations. This option determines the number of trial iterations
      allowed before the watchdog procedure is aborted and the algorithm
      returns to the stored point. The valid range for this integer option
      is 1 ≤ watchdog_trial_iter_max and its default value is 3.
      theta_max_fact (float): Advanced feature. Determines upper bound for
      constraint violation in the filter. The algorithmic parameter
      theta_max is determined as theta_max_fact times the maximum of 1 and
      the constraint violation at initial point. Any point with a
      constraint violation larger than theta_max is unacceptable to the
      filter (see Eqn. (21) in the implementation paper). The valid range
      for this real option is 0 < theta_max_fact and its default value is
      10000.
    - **theta_min_fact** (float): advanced feature. Determines
      constraint violation threshold in the switching rule. The
      algorithmic parameter theta_min is determined as
      theta_min_fact times the maximum of 1 and the constraint
      violation at initial point. The switching rules treats an
      iteration as an h-type iteration whenever the current
      constraint violation is larger than theta_min (see paragraph
      before Eqn. (19) in the implementation paper). The valid
      range for this real option is 0 < theta_min_fact and its
      default value is 0.0001.
    - **eta_phi** (float): advanced!
      Relaxation factor in the Armijo condition. See Eqn. (20) in
      the implementation paper. The valid range for this real
      option is 0 < eta_phi < 0.5 and its default value is :math:`1e-08`.
    - **delta** (float): advanced! Multiplier for constraint violation
      in the switching rule. See Eqn. (19) in the implementation
      paper. The valid range for this real option is 0 < delta and
      its default value is 1.
    - **s_phi** (float): advanced! Exponent for
      linear barrier function model in the switching rule. See Eqn.
      (19) in the implementation paper. The valid range for this
      real option is 1 < s_phi and its default value is 2.3.
    - **s_theta** (float): advanced! Exponent for current constraint
      violation in the switching rule. See Eqn. (19) in the
      implementation paper. The valid range for this real option is
      1 < s_theta and its default value is 1.1.
    - **gamma_phi** (float):
      advanced! Relaxation factor in the filter margin for the
      barrier function. See Eqn. (18a) in the implementation paper.
      The valid range for this real option is 0 < gamma_phi < 1 and
      its default value is :math:`1e-08`.
    - **gamma_theta** (float): advanced!
      Relaxation factor in the filter margin for the constraint
      violation. See Eqn. (18b) in the implementation paper. The
      valid range for this real option is 0 < gamma_theta < 1 and
      its default value is :math:`1e-05`.
    - **alpha_min_frac** (float): advanced!
      Safety factor for the minimal step size (before switching to
      restoration phase). This is gamma_alpha in Eqn. (20) in the
      implementation paper. The valid range for this real option is
      0 < alpha_min_frac < 1 and its default value is 0.05.
    - **max_soc** (int): Maximum number of second order correction trial steps
      at each iteration. Choosing 0 disables the second order
      corrections. This is p^{max} of Step A-5.9 of Algorithm A in
      the implementation paper. The valid range for this integer
      option is 0 ≤ max_soc and its default value is 4.
    - **kappa_soc** (float): advanced! Factor in the sufficient reduction rule
      for second order correction. This option determines how much
      a second order correction step must reduce the constraint
      violation so that further correction steps are attempted. See
      Step A-5.9 of Algorithm A in the implementation paper. The
      valid range for this real option is 0 < kappa_soc and its
      default value is 0.99.
    - **obj_max_inc** (float): advanced!
      Determines the upper bound on the acceptable increase of
      barrier objective function. Trial points are rejected if they
      lead to an increase in the barrier objective function by more
      than obj_max_inc orders of magnitude. The valid range for
      this real option is 1 < obj_max_inc and its default value is 5.
    - **max_filter_resets** (int): advanced! Maximal allowed number
      of filter resets. A positive number enables a heuristic
      that resets the filter, whenever in more than
      "filter_reset_trigger" successive iterations the last
      rejected trial steps size was rejected because of the
      filter. This option determine the maximal number of resets
      that are allowed to take place. The valid range for this
      integer option is 0 ≤ max_filter_resets and its default
      value is 5.
    - **filter_reset_trigger** (int): Advanced! Number
      of iterations that trigger the filter reset. If the filter
      reset heuristic is active and the number of successive
      iterations in which the last rejected trial step size was
      rejected because of the filter, the filter is reset. The
      valid range for this integer option is 1 ≤
      filter_reset_trigger and its default value is 5.
    - **corrector_type (str)**: advanced! The type of corrector steps that should
      be taken. If "mu_strategy" is "adaptive", this option determines what
      kind of corrector steps should be tried. Changing this option is
      experimental. The default value for this string option is "none".
      Possible values:

        - "none" or None: no corrector
        - "affine": corrector step towards mu=0
        - "primal-dual": corrector step towards current mu
    - **skip_corr_if_neg_curv (str or bool)**: advanced!
      Whether to skip the corrector step in negative curvature
      iteration. The corrector step is not tried if negative curvature has been
      encountered during the computation of the search direction in the current
      iteration. This option is only used if "mu_strategy" is "adaptive".
      Changing this option is experimental. The default value for this string
      option is "yes". Possible values: "yes", "no", True, False.
    - **skip_corr_in_monotone_mode (str or bool)**: Advanced! Whether to skip the
      corrector step during monotone brrier parameter mode. The corrector step
      is not tried if the algorithm is currently in the monotone mode (see also
      option "barrier_strategy"). This option is only used if "mu_strategy" is
      "adaptive". Changing this option is experimental. The default value for
      this string option is "yes". Possible values: "yes", "no", True, False
    - **corrector_compl_avrg_red_fact** (int): advanced! Complementarity tolerance
      factor for accepting corrector step. This option determines the factor by
      which complementarity is allowed to increase for a corrector step to be
      accepted. Changing this option is experimental. The valid range for this
      real option is 0 < corrector_compl_avrg_red_fact and its default value is
      1.
    - **soc_method** (int): Ways to apply second order correction. This option
      determines the way to apply second order correction, 0 is the method
      described in the implementation paper. 1 is the modified way which adds
      alpha on the rhs of x and s rows. Officially, the valid range for this
      integer option is 0 ≤ soc_method ≤ 1 and its default value is 0 but only 0
      and 1 are allowed.

    - **nu_init** (float): advanced! Initial value of the penalty parameter. The
      valid range for this real option is 0 < nu_init and its default value is
      :math:`1e-06`.
    - **nu_inc** (float): advanced! Increment of the penalty parameter. The
      valid range for this real option is 0 < nu_inc and its default value is
      0.0001.
    - **rho** (float): advanced! Value in penalty parameter update formula.
      The valid range for this real option is 0 < rho < 1 and its default value
      is 0.1.
    - **kappa_sigma** (float): advanced! Factor limiting the deviation of
      dual variables from primal estimates. If the dual variables deviate from
      their primal estimates, a correction is performed. See Eqn. (16) in the
      implementation paper. Setting the value to less than 1 disables the
      correction. The valid range for this real option is 0 < kappa_sigma and
      its default value is :math:`1e+10`.
    - **recalc_y (str or bool)**: Tells the algorithm to
      recalculate the equality and inequality multipliers as least square
      estimates. This asks the algorithm to recompute the multipliers, whenever
      the current infeasibility is less than recalc_y_feas_tol. Choosing yes
      might be helpful in the quasi-Newton option. However, each recalculation
      requires an extra factorization of the linear system. If a limited memory
      quasi-Newton option is chosen, this is used by default. The default value
      for this string option is "no". Possible values:

          - "no" or False: use the Newton step to update the multipliers
          - "yes" or True: use least-square multiplier
    - **estimates recalc_y_feas_tol** (float): Feasibility threshold for
      recomputation of multipliers. If recalc_y is chosen and the current
      infeasibility is less than this value, then the multipliers are
      recomputed. The valid range for this real option is 0 < recalc_y_feas_tol
      and its default value is :math:`1e-06`.
    - **slack_move** (float): advanced! Correction
      size for very small slacks. Due to numerical issues or the lack of an
      interior, the slack variables might become very small. If a slack becomes
      very small compared to machine precision, the corresponding bound is moved
      slightly. This parameter determines how large the move should be. Its
      default value is mach_eps^{3/4}. See also end of Section 3.5 in
      implementation paper - but actual implementation might be somewhat
      different. The valid range for this real option is 0 ≤ slack_move and its
      default value is 1.81899 · :math:`1e-12`.
    - **constraint_violation_norm_type (str)**: advanced!
      Norm to be used for the constraint violation in te line search.
      Determines which norm should be used when the algorithm computes the
      constraint violation in the line search. The default value for this string
      option is "1-norm". Possible values:

          - "1-norm": use the 1-norm
          - "2-norm": use the 2-norm
          - "max-norm": use the infinity norm

    - **mehrotra_algorithm (str or bool)**: Indicates whether to do Mehrotra's
      predictor-corrector algorithm. If enabled, line search is disabled and the
      (unglobalized) adaptive mu strategy is chosen with the "probing" oracle,
      and "corrector_type=affine" is used without any safeguards; you should not
      set any of those options explicitly in addition. Also, unless otherwise
      specified, the values of  ``bound_push`` ,  ``bound_frac`` , and
      ``bound_mult_init_val`` are set more aggressive, and sets
      "alpha_for_y=bound_mult". The Mehrotra's predictor-corrector algorithm
      works usually very well for LPs and convex QPs. The default value for this
      string option is "no". Possible values: "yes", "no", True, False.
    - **fast_step_computation (str or bool)**: Indicates if the linear system should
      be solved quickly. If enabled, the algorithm assumes that the linear
      system that is solved to obtain the search direction is solved
      sufficiently well. In that case, no residuals are computed to verify the
      solution and the computation of the search direction is a little faster.
      The default value for this string option is "no". Possible values: "yes",
      "no", True, False.
    - **min_refinement_steps** (int): Minimum number of iterative
      refinement steps per linear system solve. Iterative refinement (on the
      full asymmetric system) is performed for each right hand side. This
      option determines the minimum number of iterative refinements (i.e. at
      least "min_refinement_steps" iterative refinement steps are enforced per
      right hand side.) The valid range for this integer option is 0 ≤
      min_refinement_steps and its default value is 1.
    - **max_refinement_steps** (int): Maximum number of iterative refinement
      steps per linear system
      solve. Iterative refinement (on the full unsymmetric system) is performed
      for each right hand side. This option determines the maximum number of
      iterative refinement steps. The valid range for this integer option is 0 ≤
      max_refinement_steps and its default value is 10.
    - **residual_ratio_max** (float): advanced! Iterative refinement tolerance.
      Iterative refinement is
      performed until the residual test ratio is less than this tolerance (or
      until "max_refinement_steps" refinement steps are performed). The valid
      range for this real option is 0 < residual_ratio_max and its default value
      is :math:`1e-10`.
    - **residual_ratio_singular** (float): advanced! Threshold for
      declaring linear system singular after filed iterative refinement. If the
      residual test ratio is larger than this value after failed iterative
      refinement, the algorithm pretends that the linear system is singular. The
      valid range for this real option is 0 < residual_ratio_singular and its
      default value is :math:`1e-05`.
    - **residual_improvement_factor** (float): advanced!
      Minimal required reduction of residual test ratio in iterative refinement.
      If the improvement of the residual test ratio made by one iterative
      refinement step is not better than this factor, iterative refinement is
      aborted. The valid range for this real option is 0 <
      residual_improvement_factor and its default value is 1.

    - **neg_curv_test_tol** (float): Tolerance for heuristic to ignore wrong
      inertia. If nonzero, incorrect inertia in the augmented system is ignored,
      and Ipopt tests if the direction is a direction of positive curvature.
      This tolerance is alpha_n in the paper by :cite:`Chiang2014` and it
      determines when the direction is considered to be sufficiently positive. A
      value in the range of [1e-12, 1e-11] is recommended. The valid range for
      this real option is 0 ≤ neg_curv_test_tol and its default value is 0.
    - **neg_curv_test_reg (str or bool)**: Whether to do the curvature test with the
      primal regularization (see :cite:`Chiang2014`). The default value for
      this string option is "yes". Possible values:

          - "yes" or True: use primal regularization with the
            inertia-free curvature test
          - "no" or False: use original IPOPT approach, in which the
            primal regularization is ignored
    - **max_hessian_perturbation** (float): Maximum value of regularization
      parameter for handling negative curvature. In order to guarantee that the
      search directions are indeed proper descent directions, Ipopt requires
      that the inertia of the (augmented) linear system for the step computation
      has the correct number of negative and positive eigenvalues. The idea is
      that this guides the algorithm away from maximizers and makes Ipopt more
      likely converge to first order optimal points that are minimizers. If the
      inertia is not correct, a multiple of the identity matrix is added to the
      Hessian of the Lagrangian in the augmented system. This parameter gives
      the maximum value of the regularization parameter. If a regularization of
      that size is not enough, the algorithm skips this iteration and goes to
      the restoration phase. This is delta_w^max in the implementation paper.
      The valid range for this real option is 0 < max_hessian_perturbation and
      its default value is :math:`1e+20`.
    - **min_hessian_perturbation** (float): Smallest
      perturbation of the Hessian block. The size of the perturbation of the
      Hessian block is never selected smaller than this value, unless no
      perturbation is necessary. This is delta_w^min in implementation paper.
      The valid range for this real option is 0 ≤ min_hessian_perturbation and
      its default value is :math:`1e-20`.
    - **perturb_inc_fact_first** (float): Increase
      factor for x-s perturbation for very first perturbation. The factor by
      which the perturbation is increased when a trial value was not sufficient
      - this value is used for the computation of the very first perturbation
      and allows a different value for the first perturbation than that used
      for the remaining perturbations. This is bar_kappa_w^+ in the
      implementation paper. The valid range for this real option is 1 <
      perturb_inc_fact_first and its default value is 100.
    - **perturb_inc_fact** (float): Increase factor for x-s perturbation. The factor
      by which the perturbation is increased when a trial value was not
      sufficient - this value is used for the computation of all
      perturbations except for
      the first. This is kappa_w^+ in the implementation paper. The valid
      range for this real option is 1 < perturb_inc_fact and its default value
      is 8.
    - **perturb_dec_fact** (float): Decrease factor for x-s perturbation.
      The factor by which the perturbation is decreased when a trial value is
      deduced from the size of the most recent successful perturbation. This
      is kappa_w^- in the implementation paper. The valid range for this real
      option is 0 < perturb_dec_fact < 1 and its default value is 0.333333.
    - **first_hessian_perturbation** (float): Size of first x-s perturbation
      tried. The first value tried for the x-s perturbation in the inertia
      correction scheme. This is delta_0 in the implementation paper. The
      valid range for this real option is 0 < first_hessian_perturbation and
      its default value is 0.0001.
    - **jacobian_regularization_value** (float): Size
      of the regularization for rank-deficient constraint Jacobians. This is
      bar delta_c in the implementation paper. The valid range for this real
      option is 0 ≤ jacobian_regularization_value and its default value is
      :math:`1e-08`.
    - **jacobian_regularization_exponent** (float): advanced! Exponent for
      mu in the regularization for rnk-deficient constraint Jacobians. This is
      kappa_c in the implementation paper. The valid range for this real
      option is 0 ≤ jacobian_regularization_exponent and its default value is
      0.25.
    - **perturb_always_cd (str or bool)**: advanced! Active permanent
      perturbation of constraint linearization. Enabling this option leads to
      using the delta_c and delta_d perturbation for the computation of every
      search direction. Usually, it is only used when the iteration matrix is
      singular. The default value for this string option is "no". Possible
      values: "yes", "no", True, False.

    - **expect_infeasible_problem (str or bool)**: Enable heuristics to quickly
      detect an infeasible problem. This options is meant to activate
      heuristics that may speed up the infeasibility determination if you
      expect that there is a good chance for the problem to be infeasible. In
      the filter line search procedure, the restoration phase is called more
      quickly than usually, and more reduction in the constraint violation is
      enforced before the restoration phase is left. If the problem is square,
      this option is enabled automatically. The default value for this string
      option is "no". Possible values: "yes", "no", True, False.
    - **expect_infeasible_problem_ctol** (float): Threshold for disabling
      "expect_infeasible_problem" option. If the constraint violation becomes
      smaller than this threshold, the "expect_infeasible_problem" heuristics
      in the filter line search are disabled. If the problem is square, this
      options is set to 0. The valid range for this real option is 0 ≤
      expect_infeasible_problem_ctol and its default value is 0.001.
    - **expect_infeasible_problem_ytol** (float): Multiplier threshold for
      activating "xpect_infeasible_problem" option. If the max norm of the
      constraint multipliers becomes larger than this value and
      "expect_infeasible_problem" is chosen, then the restoration phase is
      entered. The valid range for this real option is 0 <
      expect_infeasible_problem_ytol and its default value is :math:`1e+08`.
    - **start_with_resto (str or bool)**: Whether to switch to restoration phase
      in first iteration.Setting this option to "yes" forces the algorithm to
      switch to the feasibility restoration phase in the first iteration. If
      the initial point is feasible, the algorithm will abort with a failure.
      The default value for this string option is "no". Possible values:
      "yes", "no", True, False
    - **soft_resto_pderror_reduction_factor** (float):
      Required reduction in primal-dual error in the soft restoration phase.
      The soft restoration phase attempts to reduce the primal-dual error with
      regular steps. If the damped primal-dual step (damped only to satisfy
      the fraction-to-the-boundary rule) is not decreasing the primal-dual
      error by at least this factor, then the regular restoration phase is
      called. Choosing "0" here disables the soft restoration phase. The valid
      range for this real option is 0 ≤ soft_resto_pderror_reduction_factor
      and its default value is 0.9999.
    - **max_soft_resto_iters** (int): advanced!
      Maximum number of iterations performed successively in soft rstoration
      phase. If the soft restoration phase is performed for more than so many
      iterations in a row, the regular restoration phase is called. The valid
      range for this integer option is 0 ≤ max_soft_resto_iters and its
      default value is 10.
    - **required_infeasibility_reduction** (float): Required
      reduction of infeasibility before leaving restoration phase. The
      restoration phase algorithm is performed, until a point is found that is
      acceptable to the filter and the infeasibility has been reduced by at
      least the fraction given by this option. The valid range for this real
      option is 0 ≤ required_infeasibility_reduction < 1 and its default value
      is 0.9.
    - **max_resto_iter** (int): advanced! Maximum number of successive
      iterations in restoration phase.The algorithm terminates with an error
      message if the number of iterations successively taken in the
      restoration phase exceeds this number. The valid range for this integer
      option is 0 ≤ max_resto_iter and its default value is 3000000.
    - **evaluate_orig_obj_at_resto_trial (str or bool)**: Determines if the
      original objective function should be evaluated at restoration phase
      trial points. Enabling this option makes the restoration phase algorithm
      evaluate the objective function of the original problem at every trial
      point encountered during the restoration phase, even if this value is
      not required. In this way, it is guaranteed that the original objective
      function can be evaluated without error at all accepted iterates;
      otherwise the algorithm might fail at a point where the restoration
      phase accepts an iterate that is good for the restoration phase problem,
      but not the original problem. On the other hand, if the evaluation of
      the original objective is expensive, this might be costly. The default
      value for this string option is "yes". Possible values: "yes", "no",
      True, False
    - **resto_penalty_parameter** (float): advanced! Penalty parameter
      in the restoration phase objective function. This is the parameter rho in
      equation (31a) in the Ipopt implementation paper. The valid range for
      this real option is 0 < resto_penalty_parameter and its default value is
      1000.
    - **resto_proximity_weight** (float): advanced! Weighting factor for the
      proximity term in restoration pase objective. This determines how
      the parameter zeta in equation (29a) in the implementation paper
      is computed. zeta here is resto_proximity_weight*sqrt(mu), where
      mu is the current barrier parameter. The valid range for this real
      option is 0 ≤ resto_proximity_weight and its default value is 1.
    - **bound_mult_reset_threshold** (float): Threshold for resetting bound
      multipliers after the restoration pase. After returning from the
      restoration phase, the bound multipliers are updated with a Newton
      step for complementarity. Here, the change in the primal variables
      during the entire restoration phase is taken to be the
      corresponding primal Newton step. However, if after the update the
      largest bound multiplier exceeds the threshold specified by this
      option, the multipliers are all reset to 1.
      The valid range for this real option is 0 ≤ bound_mult_reset_threshold
      and its default value is 1000.
    - **constr_mult_reset_threshold** (float):
      Threshold for resetting equality and inequality multipliers ater
      restoration phase. After returning from the restoration phase, the
      constraint multipliers are recomputed by a least square estimate. This
      option triggers when those least-square estimates should be ignored.
      The valid range for this real option is 0 ≤ constr_mult_reset_threshold
      and its default value is 0.
    - **resto_failure_feasibility_threshold** (float): advanced!
      Threshold for primal infeasibility to declare failure
      of restoration phase. If the restoration phase is terminated because of
      the "acceptable" termination criteria and the primal infeasibility is
      smaller than this value, the restoration phase is declared to have
      failed. The default value is actually 1e2*tol, where tol is the general
      termination tolerance. The valid range for this real option is 0 ≤
      resto_failure_feasibility_threshold and its default value is 0.

    - **limited_memory_aug_solver (str)**: advanced! Strategy for solving the
      augmented system for low-rank Hessian.
      The default value for this string option is "sherman-morrison".
      Possible values:

          - "sherman-morrison": use Sherman-Morrison formula
          - "extended": use an extended augmented system
    - **limited_memory_max_history** (int): Maximum size of the history for the
      limited quasi-Newton Hessian approximation. This option determines the
      number of most recent iterations that are taken into account for the
      limited-memory quasi-Newton approximation. The valid range for this
      integer option is 0 ≤ limited_memory_max_history and
      its default value is 6.
    - **limited_memory_update_type (str)**: Quasi-Newton update formula for the
      limited memory quasi-Newton approximation. The default value for this
      string option is "bfgs". Possible values:

          - "bfgs": BFGS update (with skipping)
          - "sr1": SR1 (not working well)
    - **limited_memory_initialization (str)**:
      Initialization strategy for the limited memory quasi-Newton
      aproximation. Determines how the diagonal Matrix B_0 as the first term in
      the limited memory approximation should be computed. The default value for
      this string option is "scalar1". Possible values:

          - "scalar1": sigma = s^Ty/s^Ts
          - "scalar2": sigma = y^Ty/s^Ty
          - "scalar3": arithmetic average of scalar1 and scalar2
          - "scalar4": geometric average of scalar1 and scalar2
          - "constant": sigma = limited_memory_init_val
    - **limited_memory_init_val** (float): Value for B0 in low-rank update. The
      starting matrix in the low rank update, B0, is chosen to be this multiple
      of the identity in the first iteration (when no updates have been
      performed yet), and is constantly chosen as this value, if
      "limited_memory_initialization" is "constant". The valid range for this
      real option is 0 < limited_memory_init_val and its default value is 1.
    - **limited_memory_init_val_max** (float): Upper bound on value for B0 in
      low-rank update. The starting matrix in the low rank update, B0, is chosen
      to be this multiple of the identity in the first iteration (when no
      updates have been performed yet), and is constantly chosen as this value,
      if "limited_memory_initialization" is "constant". The valid range for this
      real option is 0 < limited_memory_init_val_max and its default value is
      :math:`1e+08`.
    - **limited_memory_init_val_min** (float): Lower bound on value for B0 in
      low-rank update. The starting matrix in the low rank update, B0, is chosen
      to be this multiple of the identity in the first iteration (when no
      updates have been performed yet), and is constantly chosen as this value,
      if "limited_memory_initialization" is "constant". The valid range for this
      real option is 0 < limited_memory_init_val_min and its default value is
      :math:`1e-08`.
    - **limited_memory_max_skipping** (int): Threshold for successive
      iterations where update is skipped. If the update is skipped more than
      this number of successive iterations, the quasi-Newton approximation is
      reset. The valid range for this integer option is 1 ≤
      limited_memory_max_skipping and its default value is 2.
    - **limited_memory_special_for_resto (str or bool)**: Determines if the
      quasi-Newton updates should be special dring the restoration phase. Until
      Nov 2010, Ipopt used a special update during the restoration phase, but it
      turned out that this does not work well. The new default uses the regular
      update procedure and it improves results. If for some reason you want to
      get back to the original update, set this option to "yes". The default
      value for this string option is "no". Possible values: "yes", "no", True,
      False.
    - **hessian_approximation (str)**: Indicates what Hessian information is
      to be used. This determines which kind of information for the Hessian of
      the Lagrangian function is used by the algorithm. The default value for
      this string option is "limited-memory". Possible values: - "exact": Use
      second derivatives provided by the NLP. - "limited-memory": Perform a
      limited-memory quasi-Newton approximation
    - **hessian_approximation_space (str)**: advanced!
      Indicates in which subspace the Hessian information is to
      be approximated. The default value for this string option is
      "nonlinear-variables". Possible values: - "nonlinear-variables": only in
      space of nonlinear variables. - "all-variables": in space of all variables
      (without slacks)
    - **linear_solver (str)**: Linear solver used for step
      computations. Determines which linear algebra package is to be used for
      the solution of the augmented linear system (for obtaining the search
      directions). The default value for this string option is "ma27". Possible
      values:

          - "mumps" (use the Mumps package, default)
          - "ma27" (load the Harwell routine MA27 from library at runtime)
          - "ma57" (load the Harwell routine MA57 from library at runtime)
          - "ma77" (load the Harwell routine HSL_MA77 from library at runtime)
          - "ma86" (load the Harwell routine MA86 from library at runtime)
          - "ma97" (load the Harwell routine MA97 from library at runtime)
          - "pardiso" (load the Pardiso package from pardiso-project.org
            from user-provided library at runtime)
          - "custom" (use custom linear solver (expert use))
    - **linear_solver_options** (dict or None): dictionary with the
      linear solver options, possibly including `linear_system_scaling`,
      `hsllib` and `pardisolib`. See the `ipopt documentation for details
      <https://coin-or.github.io/Ipopt/OPTIONS.html>`_. The linear solver
      options are not automatically converted to float at the moment.]

```

(fides-algorithm)=

## The Fides Optimizer

optimagic supports the
[Fides Optimizer](https://fides-optimizer.readthedocs.io/en/latest). To use Fides, you
need to have [the fides package](https://github.com/fides-dev/fides) installed
(`pip install fides>=0.7.4`, make sure you have at least 0.7.1).

```{eval-rst}
.. dropdown:: fides

  .. code-block::

      "fides"

  `Fides <https://fides-optimizer.readthedocs.io/en/latest>`_ implements an Interior
  Trust Region Reflective for boundary costrained optimization problems based on the
  papers :cite:`Coleman1994` and :cite:`Coleman1996`. Accordingly, Fides is named after
  the Roman goddess of trust and reliability. In contrast to other optimizers, Fides
  solves the full trust-region subproblem exactly, which can yields higher quality
  proposal steps, but is computationally more expensive. This makes Fides particularly
  attractive for optimization problems with objective functions that are computationally
  expensive to evaluate and the computational cost of solving the trust-region
  subproblem is negligible.

  - **hessian_update_strategy** (str): Hessian Update Strategy to employ. You can provide
    a lowercase or uppercase string or a
    fides.hession_approximation.HessianApproximation class instance. FX, SSM, TSSM and
    GNSBFGS are not supported by optimagic. The available update strategies are:

      - **bb**: Broydens "bad" method as introduced :cite:`Broyden1965`.
      - **bfgs**: Broyden-Fletcher-Goldfarb-Shanno update strategy.
      - **bg**: Broydens "good" method as introduced in :cite:`Broyden1965`.
      - You can use a general BroydenClass Update scheme using the Broyden class from
        `fides.hessian_approximation`. This is a generalization of BFGS/DFP methods
        where the parameter :math:`phi` controls the convex combination between the
        two. This is a rank 2 update strategy that preserves positive-semidefiniteness
        and symmetry (if :math:`\phi \in [0,1]`). It is described in
        :cite:`Nocedal1999`, Chapter 6.3.
      - **dfp**: Davidon-Fletcher-Powell update strategy.
      - **sr1**: Symmetric Rank 1 update strategy as described in :cite:`Nocedal1999`,
        Chapter 6.2.

  - **convergence.ftol_abs** (float): absolute convergence criterion
    tolerance. This is only the interpretation of this parameter if the relative
    criterion tolerance is set to 0. Denoting the absolute criterion tolerance by
    :math:`\alpha` and the relative criterion tolerance by :math:`\beta`, the
    convergence condition on the criterion improvement is
    :math:`|f(x_k) - f(x_{k-1})| < \alpha + \beta \cdot |f(x_{k-1})|`
  - **convergence.ftol_rel** (float): relative convergence criterion
    tolerance. This is only the interpretation of this parameter if the absolute
    criterion tolerance is set to 0 (as is the default). Denoting the absolute
    criterion tolerance by :math:`\alpha` and the relative criterion tolerance by
    :math:`\beta`, the convergence condition on the criterion improvement is
    :math:`|f(x_k) - f(x_{k-1})| < \alpha + \beta \cdot |f(x_{k-1})|`
  - **convergence.xtol_abs** (float): The optimization terminates
    successfully when the step size falls below this number, i.e. when
    :math:`||x_{k+1} - x_k||` is smaller than this tolerance.
  - **convergence.gtol_abs** (float): The optimization terminates
    successfully when the gradient norm is less or equal than this tolerance.
  - **convergence.gtol_rel** (float): The optimization terminates
    successfully when the norm of the gradient divided by the absolute function value
    is less or equal to this tolerance.

  - **stopping.maxiter** (int): maximum number of allowed iterations.
  - **stopping.max_seconds** (int): maximum number of walltime seconds, deactivated by
    default.

  - **trustregion.initial_radius** (float): Initial trust region radius. Default is 1.
  - **trustregion.stepback_strategy** (str): search refinement strategy if proposed step
    reaches a parameter bound. The default is "truncate". The available options are:

      - "reflect": recursive reflections at boundary.
      - "reflect_single": single reflection at boundary.
      - "truncate": truncate step at boundary and re-solve the restricted subproblem
      - "mixed": mix reflections and truncations

  - **trustregion.subspace_dimension** (str): Subspace dimension in which the subproblem
    will be solved. The default is "2D". The following values are available:

      - "2D": Two dimensional Newton/Gradient subspace
      - "full": full dimensionality
      - "scg": Conjugated Gradient subspace via Steihaug's method

  - **trustregion.max_stepback_fraction** (float): Stepback parameter that controls how
    close steps are allowed to get to the boundary. It is the maximal fraction of a
    step to take if full step would reach breakpoint.

  - **trustregion.decrease_threshold** (float): Acceptance threshold for trust region
    ratio. The default is 0.25 (:cite:`Nocedal2006`). The radius is decreased if the
    trust region ratio is below this value. This is denoted by :math:`\\mu` in
    algorithm 4.1 in :cite:`Nocedal2006`.
  - **trustregion.increase_threshold** (float): Threshold for the trust region radius
    ratio above which the trust region radius can be increased. This is denoted by
    :math:`\eta` in algorithm 4.1 in :cite:`Nocedal2006`. The default is 0.75
    (:cite:`Nocedal2006`).
  - **trustregion.decrease_factor** (float): factor by which trust region radius will be
    decreased in case it is decreased. This is denoted by :math:`\gamma_1` in
    algorithm 4.1 in :cite:`Nocedal2006` and its default is 0.25.
  - **trustregion.increase_factor** (float): factor by which trust region radius will be
    increase in case it is increase. This is denoted by :math:`\gamma_2` in algorithm
    4.1 in :cite:`Nocedal2006` and its default is 2.0.

  - **trustregion.refine_stepback** (bool): whether to refine stepbacks via optimization.
    Default is False.
  - **trustregion.scaled_gradient_as_possible_stepback** (bool): whether the scaled
    gradient should be added to the set of possible stepback proposals. Default is
    False.

```

## The NLOPT Optimizers (nlopt)

optimagic supports the following [NLOPT](https://nlopt.readthedocs.io/en/latest/)
algorithms. Please add the
[appropriate citations](https://nlopt.readthedocs.io/en/latest/Citing_NLopt/) in
addition to optimagic when using an NLOPT algorithm. To install nlopt run
`conda install nlopt`.

```{eval-rst}
.. dropdown:: nlopt_bobyqa

    .. code-block::

        "nlopt_bobyqa"

    Minimize a scalar function using the BOBYQA algorithm.

    The implementation is derived from the BOBYQA subroutine of M. J. D. Powell.

    The algorithm performs derivative free bound-constrained optimization using
    an iteratively constructed quadratic approximation for the objective function.
    Due to its use of quadratic appoximation, the algorithm may perform poorly
    for objective functions that are not twice-differentiable.

    For details see :cite:`Powell2009`.

    - **convergence.xtol_rel** (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - **convergence.xtol_abs** (float): Stop when the absolute movement
      between parameter vectors is smaller than this.
    - **convergence.ftol_rel** (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - **convergence.ftol_abs** (float): Stop when the change of the
      criterion function between two iterations is smaller than this.
    - **stopping.maxfun** (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this
      as convergence.
```

```{eval-rst}
.. dropdown:: nlopt_neldermead

    .. code-block::

        "nlopt_neldermead"

    Minimize a scalar function using the Nelder-Mead simplex algorithm.

    The basic algorithm is described in :cite:`Nelder1965`.

    The difference between the nlopt implementation an the original implementation is
    that the nlopt version supports bounds. This is done by moving all new points that
    would lie outside the bounds exactly on the bounds.

    - **convergence.xtol_rel** (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - **convergence.xtol_abs** (float): Stop when the absolute movement
      between parameter vectors is smaller than this.
    - **convergence.ftol_rel** (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - **convergence.ftol_abs** (float): Stop when the change of the
      criterion function between two iterations is smaller than this.
    - **stopping.maxfun** (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this
      as convergence.
```

```{eval-rst}
.. dropdown:: nlopt_praxis

    .. code-block::

        "nlopt_praxis"

    Minimize a scalar function using principal-axis method.

    This is a gradient-free local optimizer originally described in :cite:`Brent1972`.
    It assumes quadratic form of the optimized function and repeatedly updates a set of conjugate
    search directions.

    The algorithm is not invariant to scaling of the objective function and may
    fail under its certain rank-preserving transformations (e.g., will lead to
    a non-quadratic shape of the objective function).

    The algorithm is not determenistic and it is not possible to achieve
    detereminancy via seed setting.

    The algorithm failed on a simple benchmark function with finite parameter bounds.
    Passing arguments `lower_bounds` and `upper_bounds` has been disabled for this
    algorithm.

    The difference between the nlopt implementation an the original implementation is
    that the nlopt version supports bounds. This is done by returning infinity (Inf)
    when the constraints are violated. The implementation of bound constraints
    is achieved at the const of significantly reduced speed of convergence.
    In case of bounded constraints, this method is dominated by `nlopt_bobyqa`
    and `nlopt_cobyla`.

    - **convergence.xtol_rel** (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - **convergence.xtol_abs** (float): Stop when the absolute movement
      between parameter vectors is smaller than this.
    - **convergence.ftol_rel** (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - **convergence.ftol_abs** (float): Stop when the change of the
      criterion function between two iterations is smaller than this.
    - **stopping.maxfun** (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this
      as convergence.

```

```{eval-rst}
.. dropdown:: nlopt_cobyla

    .. code-block::

        "nlopt_cobyla"

    Minimize a scalar function using the cobyla method.

    The alggorithm is derived from Powell's Constrained Optimization BY Linear
    Approximations (COBYLA) algorithm. It is a derivative-free optimizer with
    nonlinear inequality and equality constrains, described in :cite`Powell1994`.

    It constructs successive linear approximations of the objective function and
    constraints via a simplex of n+1 points (in n dimensions), and optimizes these
    approximations in a trust region at each step.

    The the nlopt implementation differs from the original implementation in a
    a few ways:
    - Incorporates all of the NLopt termination criteria.
    - Adds explicit support for bound constraints.
    - Allows the algorithm to increase the trust-reion radius if the predicted
    imptoovement was approximately right and the simplex is satisfactory.
    - Pseudo-randomizes simplex steps in the algorithm, aimproving robustness by
    avoiding accidentally taking steps that don't improve conditioning, preserving
    the deterministic nature of the algorithm.
    - Supports unequal initial-step sizes in the different parameters.


    - **convergence.xtol_rel** (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - **convergence.xtol_abs** (float): Stop when the absolute movement
      between parameter vectors is smaller than this.
    - **convergence.ftol_rel** (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - **convergence.ftol_abs** (float): Stop when the change of the
      criterion function between two iterations is smaller than this.
    - **stopping.maxfun** (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this
      as convergence.
```

```{eval-rst}
.. dropdown:: nlopt_sbplx

    .. code-block::

        "nlopt_sbplx"

    Minimize a scalar function using the "Subplex" algorithm.

    The alggorithm is a reimplementation of  Tom Rowan's "Subplex" algorithm.
    See :cite:`Rowan1990`.
    Subplex is a variant of Nedler-Mead that uses Nedler-Mead on a sequence of
    subspaces. It is climed to be more efficient and robust than the original
    Nedler-Mead algorithm.

    The difference between this re-implementation and the original algorithm
    of Rowan, is that it explicitly supports bound constraints providing big
    improvement in the case where the optimum lies against one of the constraints.

    - **convergence.xtol_rel** (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - **convergence.xtol_abs** (float): Stop when the absolute movement
      between parameter vectors is smaller than this.
    - **convergence.ftol_rel** (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - **convergence.ftol_abs** (float): Stop when the change of the
      criterion function between two iterations is smaller than this.
    - **stopping.maxfun** (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this
      as convergence.
```

```{eval-rst}
.. dropdown:: nlopt_newuoa

    .. code-block::

        "nlopt_newuoa"

    Minimize a scalar function using the NEWUOA algorithm.

    The algorithm is derived from the NEWUOA subroutine of M.J.D Powell which
    uses iteratively constructed quadratic approximation of the objctive fucntion
    to perform derivative-free unconstrained optimization. Fore more details see:
    :cite:`Powell2004`.

    The algorithm in `nlopt` has been modified to support bound constraints. If all
    of the bound constraints are infinite, this function calls the `nlopt.LN_NEWUOA`
    optimizers for uncsonstrained optimization. Otherwise, the `nlopt.LN_NEWUOA_BOUND`
    optimizer for constrained problems.

    `NEWUOA` requires the dimension n of the parameter space to be `≥ 2`, i.e. the
    implementation does not handle one-dimensional optimization problems.

    - **convergence.xtol_rel** (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - **convergence.xtol_abs** (float): Stop when the absolute movement
      between parameter vectors is smaller than this.
    - **convergence.ftol_rel** (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - **convergence.ftol_abs** (float): Stop when the change of the
      criterion function between two iterations is smaller than this.
    - **stopping.maxfun** (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this
      as convergence.
```

```{eval-rst}
.. dropdown:: nlopt_tnewton

    .. code-block::

        "nlopt_tnewton"

    Minimize a scalar function using the "TNEWTON" algorithm.

    The alggorithm is based on a Fortran implementation of a preconditioned
    inexact truncated Newton algorithm written by Prof. Ladislav Luksan.

    Truncated Newton methods are a set of algorithms designed to solve large scale
    optimization problems. The algorithms use (inaccurate) approximations of the
    solutions to Newton equations, using conjugate gradient methodds, to handle the
    expensive calculations of derivatives during each iteration.

    Detailed description of algorithms is given in :cite:`Dembo1983`.

    - **convergence.xtol_rel** (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - **convergence.xtol_abs** (float): Stop when the absolute movement
      between parameter vectors is smaller than this.
    - **convergence.ftol_rel** (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - **convergence.ftol_abs** (float): Stop when the change of the
      criterion function between two iterations is smaller than this.
    - **stopping.maxfun** (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this
      as convergence.
```

```{eval-rst}
.. dropdown:: nlopt_lbfgs

    .. code-block::

        "nlopt_lbfgs"

    Minimize a scalar function using the "LBFGS" algorithm.

    The alggorithm is based on a Fortran implementation of low storage BFGS algorithm
    written by Prof. Ladislav Luksan.

    LFBGS is an approximation of the original Broyden–Fletcher–Goldfarb–Shanno algorithm
    based on limited use of memory. Memory efficiency is obtained by preserving a limi-
    ted number (<10) of past updates of candidate points and gradient values and using
    them to approximate the hessian matrix.

    Detailed description of algorithms is given in :cite:`Nocedal1989`, :cite:`Nocedal1980`.

    - **convergence.xtol_rel** (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - **convergence.xtol_abs** (float): Stop when the absolute movement
      between parameter vectors is smaller than this.
    - **convergence.ftol_rel** (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - **convergence.ftol_abs** (float): Stop when the change of the
      criterion function between two iterations is smaller than this.
    - **stopping.maxfun** (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this
      as convergence.
```

```{eval-rst}
.. dropdown:: nlopt_ccsaq

    .. code-block::

        "nlopt_ccsaq"

    Minimize a scalar function using CCSAQ algorithm.

    CCSAQ uses the quadratic variant of the conservative convex separable approximation.
    The algorithm performs gradient based local optimization with equality (but not
    inequality) constraints. At each candidate point x, a quadratic approximation
    to the criterion faunction is computed using the value of gradient at point x. A
    penalty term is incorporated to render optimizaion convex and conservative. The
    algorithm is "globally convergent" in the sense that it is guaranteed to con-
    verge to a local optimum from any feasible starting point.

    The implementation is based on CCSA algorithm described in :cite:`Svanberg2002`.

    - **convergence.xtol_rel** (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - **convergence.xtol_abs** (float): Stop when the absolute movement
      between parameter vectors is smaller than this.
    - **convergence.ftol_rel** (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - **convergence.ftol_abs** (float): Stop when the change of the
      criterion function between two iterations is smaller than this.
    - **stopping.maxfun** (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this
      as convergence.
```

```{eval-rst}
.. dropdown:: nlopt_mma

    .. code-block::

        "nlopt_mma"

    Minimize a scalar function using the method of moving asymptotes (MMA).

    The implementation is based on an algorithm described in :cite:`Svanberg2002`.

    The algorithm performs gradient based local optimization with equality (but
    not inequality) constraints. At each candidate point x, an approximation to the
    criterion faunction is computed using the value of gradient at point x. A quadratic
    penalty term is incorporated to render optimizaion convex and conservative. The
    algorithm is "globally convergent" in the sense that it is guaranteed to con-
    verge to a local optimum from any feasible starting point.


    - **convergence.xtol_rel** (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - **convergence.xtol_abs** (float): Stop when the absolute movement
      between parameter vectors is smaller than this.
    - **convergence.ftol_rel** (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - **convergence.ftol_abs** (float): Stop when the change of the
      criterion function between two iterations is smaller than this.
    - **stopping.maxfun** (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this
      as convergence.
```

```{eval-rst}
.. dropdown:: nlopt_var

    .. code-block::

        "nlopt_var"

    Minimize a scalar function limited memory switching variable-metric method.

    The algorithm relies on saving only limited number M of past updates of the
    gradient to approximate the inverse hessian. The large is M, the more memory is
    consumed

    Detailed explanation of the algorithm, including its two variations of  rank-2 and
    rank-1 methods can be found in the following paper :cite:`Vlcek2006` .

    - **convergence.xtol_rel** (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - **convergence.xtol_abs** (float): Stop when the absolute movement
      between parameter vectors is smaller than this.
    - **convergence.ftol_rel** (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - **convergence.ftol_abs** (float): Stop when the change of the
      criterion function between two iterations is smaller than this.
    - **stopping.maxfun** (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this
      as convergence.
    - **rank_1_update** (bool): Whether I rank-1 or rank-2 update is used.
```

```{eval-rst}
.. dropdown:: nlopt_slsqp

    .. code-block::

        "nlopt_slsqp"

    Optimize a scalar function based on SLSQP method.

    SLSQP solves gradient based nonlinearly constrained optimization problems.
    The algorithm treats the optimization problem as a sequence of constrained
    least-squares problems.

    The implementation is based on the procedure described in :cite:`Kraft1988`
    and :cite:`Kraft1994` .

    - **convergence.xtol_rel** (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - **convergence.xtol_abs** (float): Stop when the absolute movement
      between parameter vectors is smaller than this.
    - **convergence.ftol_rel** (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - **convergence.ftol_abs** (float): Stop when the change of the
      criterion function between two iterations is smaller than this.
    - **stopping.maxfun** (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this
      as convergence.
```

```{eval-rst}
.. dropdown:: nlopt_direct

    .. code-block::

        "nlopt_direct"

    Optimize a scalar function based on DIRECT method.

    DIRECT is the DIviding RECTangles algorithm for global optimization, described
    in :cite:`Jones1993` .

    Variations of the algorithm include locally biased routines (distinguished by _L
    suffix) that prove to be more efficients for functions that have few local minima.
    See the following for the DIRECT_L variant :cite:`Gablonsky2001` .

    Locally biased algorithms can be implmented both with deterministic and random
    (distinguished by _RAND suffix) search algorithm.

    Finally, both original and locally biased variants can be implemented with and
    without the rescaling of the bound constraints.

    Boolean arguments `locally_biased`, 'random_search', and 'unscaled_bouds' can be
    set to `True` or `False` to determine which method is run. The comprehensive list
    of available methods are:
    - "DIRECT"
    - "DIRECT_L"
    - "DIRECT_L_NOSCAL"
    - "DIRECT_L_RAND"
    - "DIRECT_L_RAND_NOSCAL"
    - "DIRECT_RAND"

    - **convergence.xtol_rel** (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - **convergence.xtol_abs** (float): Stop when the absolute movement
      between parameter vectors is smaller than this.
    - **convergence.ftol_rel** (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - **convergence.ftol_abs** (float): Stop when the change of the
      criterion function between two iterations is smaller than this.
    - **stopping.maxfun** (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this
      as convergence.
    - **locally_biased** (bool): Whether the "L" version of the algorithm is selected.
    - **random_search** (bool): Whether the randomized version of the algorithm is selected.
    - **unscaled_bounds** (bool): Whether the "NOSCAL" version of the algorithm is selected.
```

```{eval-rst}
.. dropdown:: nlopt_esch

    .. code-block::

        "nlopt_esch"

    Optimize a scalar function using the ESCH algorithm.

    ESCH is an evolutionary algorithm that supports bound constraints only. Specifi
    cally, it does not support nonlinear constraints.

    More information on this method can be found in
    :cite:`DaSilva2010` , :cite:`DaSilva2010a` , :cite:`Beyer2002`  and :cite:`Vent1975` .

    - **convergence.xtol_rel** (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - **convergence.xtol_abs** (float): Stop when the absolute movement
      between parameter vectors is smaller than this.
    - **convergence.ftol_rel** (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - **convergence.ftol_abs** (float): Stop when the change of the
      criterion function between two iterations is smaller than this.
    - **stopping.maxfun** (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this
      as convergence.
```

```{eval-rst}
.. dropdown:: nlopt_isres

    .. code-block::

        "nlopt_isres"

    Optimize a scalar function using the ISRES algorithm.

    ISRES is an implementation of "Improved Stochastic Evolution Strategy"
    written for solving optimization problems with non-linear constraints. The
    algorithm is supposed to be a global method, in that it has heuristics to
    avoid local minima. However, no convergence proof is available.

    The original method and a refined version can be found, respecively, in
    :cite:`PhilipRunarsson2005` and :cite:`Thomas2000` .


    - **convergence.xtol_rel** (float):  Stop when the relative
      movement between parameter vectors is smaller than this.
    - **convergence.xtol_abs** (float): Stop when the absolute
      movement between parameter vectors is smaller than this.
    - **convergence.ftol_rel** (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - **convergence.ftol_abs** (float): Stop when the change of
      the criterion function between two iterations is smaller than this.
    - **stopping.maxfun** (int): If the maximum number of
      function evaluation is reached, the optimization stops but we do not count
      this as convergence.
```

```{eval-rst}
.. dropdown:: nlopt_crs2_lm

    .. code-block::

        "nlopt_crs2_lm"

    Optimize a scalar function using the CRS2_LM algorithm.

    This implementation of controlled random search method with local mutation is based
    on :cite:`Kaelo2006` .

    The original CRS method is described in :cite:`Price1978`  and :cite:`Price1983` .

    CRS class of algorithms starts with random population of points and evolves the
    points "randomly". The size of the initial population can be set via the param-
    meter population_size. If the user doesn't specify a value, it is set to the nlopt
    default of 10*(n+1).

    - **convergence.xtol_rel** (float):  Stop when the relative movement
      between parameter vectors is smaller than this.
    - **convergence.xtol_abs** (float): Stop when the absolute movement
      between parameter vectors is smaller than this.
    - **convergence.ftol_rel** (float): Stop when the relative
      improvement between two iterations is smaller than this.
    - **convergence.ftol_abs** (float): Stop when the change of the
      criterion function between two iterations is smaller than this.
    - **stopping.maxfun** (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this as
      convergence.
    - **population_size** (int): Size of the population. If None, it's set to be
      10 * (number of parameters + 1).
```

## References

```{eval-rst}
.. bibliography:: refs.bib
    :labelprefix: algo_
    :filter: docname in docnames
    :style: unsrt
```
