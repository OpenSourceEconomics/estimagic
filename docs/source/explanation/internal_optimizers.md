(internal_optimizer_interface)=

# Internal optimizers for optimagic

optimagic provides a large collection of optimization algorithm that can be used by
passing the algorithm name as `algorithm` into `maximize` or `minimize`. Advanced users
can also use optimagic with their own algorithm, as long as it conforms with the
internal optimizer interface.

The advantages of using the algorithm with optimagic over using it directly are:

- optimagic turns an unconstrained optimizer into constrained ones.
- You can use logging.
- You get great error handling for exceptions in the criterion function or gradient.
- You get a parallelized and customizable numerical gradient if the user did not provide
  a closed form gradient.
- You can compare your optimizer with all the other optimagic optimizers by changing
  only one line of code.

All of this functionality is achieved by transforming a more complicated user provided
problem into a simpler problem and then calling "internal optimizers" to solve the
transformed problem.

## The internal optimizer interface

An internal optimizer is a a function that minimizes a criterion function and fulfills a
few conditions. In our experience, it is not hard to wrap any optimizer into this
interface. The mandatory conditions for an internal optimizer function are:

1. It is decorated with the `mark_minimizer` decorator and thus carries information that
   tells optimagic how to use the internal optimizer.

1. It uses the standard names for the arguments that describe the optimization problem:

   - criterion: for the criterion function
   - x: for the start parameters in form of a 1d numpy array
   - derivative: for the first derivative of the criterion function
   - criterion_and_derivative: for a function that evaluates the criterion and its first
     derivative jointly
   - lower_bounds: for lower bounds in form of a 1d numpy array
   - upper_bounds: for upper bounds in form of a 1d numpy array
   - nonlinear_constraints: for nonlinear constraints in form a list of dictionaries

   Of course, algorithms that do not need a certain argument (e.g. unbounded or
   derivative free ones) do not need those arguments at all.

1. All other arguments have default values.

(internal-optimizer-output)=

## Output of internal optimizers

After convergence or when another stopping criterion is achieved the internal optimizer
should return a dictionary with the following entries:

- solution_x: The best parameter achieved so far
- solution_criterion: The value of the criterion at solution_x. This can be a scalar or
  dictionary.
- n_fun_evals: The number of criterion evaluations.
- n_jac_evals: The number of derivative evaluations.
- n_iterations: The number of iterations
- success: True if convergence was achieved
- message: A string with additional information.

If some of the entries are missing, they will automatically be filled with `None` and no
errors are raised. Nevertheless, you should try to return as much information as
possible.

(naming-conventions)=

## Naming conventions for algorithm specific arguments

Many optimizers have similar but slightly different names for arguments that configure
the convergence criteria, other stopping conditions, and so on. We try to harmonize
those names and their default values where possible.

Since some optimizers support many tuning parameters we group some of them by the first
part of their name (e.g. all convergence criteria names start with `convergence`). See
{ref}`list_of_algorithms` for the signatures of the provided internal optimizers.

The preferred default values can be imported from `optimagic.optimization.algo_options`
which are documented in {ref}`algo_options`. If you add a new optimizer to optimagic you
should only deviate from them if you have good reasons.

Note that a complete harmonization is not possible nor desirable, because often
convergence criteria that clearly are the same are implemented slightly different for
different optimizers. However, complete transparency is possible and we try to document
the exact meaning of all options for all optimizers.

## Algorithms that parallelize

Algorithms can evaluate the criterion function in parallel. To make such a parallel
algorithm fully compatible with optimagic (including history collection and benchmarking
functionality), the following conditions need to be fulfilled:

- The algorithm has an argument called `n_cores` which determines how many cores are
  used for the parallelization.
- The algorithm has an argument called `batch_evaluator` and all parallelization is done
  using a built-in or user provided batch evaluator.

Moreover, we strongly suggest to comply with the following convention:

- The algorithm has an argument called `batch_size` which is an integer that is greater
  or equal to `n_cores`. Setting the `batch_size` larger than n_cores, allows to
  simulate how the algorithm would behave with `n_cores=batch_size` but only uses
  `n_cores` cores. This allows to simulate / benchmark the parallelizability of an
  algorithm even if no parallel hardware is available.

If the mandatory conditions are not fulfilled, the algorithm should disable all history
collection by using `mark_minimizer(..., disable_history=True)`.

## Nonlinear constraints

optimagic can pass nonlinear constraints to the internal optimizer. The internal
interface for nonlinear constraints is as follows.

A nonlinear constraint is a `list` of `dict` 's, where each `dict` represents a group of
constraints. In each group the constraint function can potentially be multi-dimensional.
We distinguish between equality and inequality constraints, which is signalled by a dict
entry `type` that takes values `"eq"` and `"ineq"`. The constraint function, which takes
as input an internal parameter vector, is stored under the entry `fun`, while the
Jacobian of that function is stored at `jac`. The tolerance for the constraints is
stored under `tol`. At last, the number of constraints in each group is specified under
`n_constr`. An example list with one constraint that would be passed to the internal
optimizer is given by

```
constraints = [
    {
        "type": "ineq",
        "n_constr": 1,
        "tol": 1e-5,
        "fun": lambda x: x**3,
        "jac": lambda x: 3 * x**2,
    }
]
```

**Equality.** Internal equality constraints assume that the constraint is met when the
function is zero. That is

$$
0 = g(x) \in \mathbb{R}^m .
$$

**Inequality.** Internal inequality constraints assume that the constraint is met when
the function is greater or equal to zero. That is

$$
0 \leq g(x) \in \mathbb{R}^m .
$$

## Other conventions

- Internal optimizer are functions and should thus adhere to python naming conventions,
  for functions (i.e. only consist of lowercase letters and individual words should be
  separated by underscores). For optimizers that are implemented in many packages (e.g.
  Nelder Mead or BFGS), the name of the original package in which it was implemented has
  to be part of the name.
- All arguments of an internal optimizer should actually be used. In particular, if an
  optimizer does not support bounds it should not have `lower_bounds` and `upper_bounds`
  as arguments; derivative free optimizers should not have `derivative` or
  `criterion_and_derivative` as arguments, etc.
