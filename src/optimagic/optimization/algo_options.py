import numpy as np

CONVERGENCE_FTOL_REL = 2e-9
"""float: Stop when the relative improvement between two iterations is below this.

    The exact definition of relative improvement depends on the optimizer and should
    be documented there. To disable it, set it to 0.

    The default value is inspired by scipy L-BFGS-B defaults, but rounded.

"""

CONVERGENCE_FTOL_ABS = 0
"""float: Stop when the absolute improvement between two iterations is below this.

    Disabled by default because it is very problem specific.

"""

CONVERGENCE_GTOL_ABS = 1e-5
"""float: Stop when the gradient are smaller than this.

    For some algorithms this criterion refers to all entries, for others to some norm.

    For bound constrained optimizers this typically refers to a projected gradient.
    The exact definition should be documented for each optimizer.

    The default is the same as scipy. To disable it, set it to zero.

"""

CONVERGENCE_GTOL_REL = 1e-8
"""float: Stop when the gradient, divided by the absolute value of the criterion
    function is smaller than this. For some algorithms this criterion refers to
    all entries, for others to some norm.For bound constrained optimizers this
    typically refers to a projected gradient. The exact definition should be documented
    for each optimizer. To disable it, set it to zero.

"""

CONVERGENCE_GTOL_SCALED = 1e-8
"""float: Stop when all entries (or for some algorithms the norm) of the gradient,
    divided by the norm of the gradient at start parameters is smaller than this.
    For bound constrained optimizers this typically refers to a projected gradient.
    The exact definition should be documented for each optimizer.
    To disable it, set it to zero.

"""

CONVERGENCE_XTOL_REL = 1e-5
"""float: Stop when the relative change in parameters is smaller than this.
    The exact definition of relative change and whether this refers to the maximum
    change or the average change depends on the algorithm and should be documented
    there. To disable it, set it to zero. The default is the same as in scipy.

"""

CONVERGENCE_XTOL_ABS = 0
"""float: Stop when the absolute change in parameters between two iterations is smaller
    than this. Whether this refers to the maximum change or the average change depends
    on the algorithm and should be documented there.

    Disabled by default because it is very problem specific. To enable it, set it to a
    value larger than zero.

"""


STOPPING_MAXFUN = 1_000_000
"""int:
    If the maximum number of function evaluation is reached, the optimization stops
    but we do not count this as successful convergence. The function evaluations used
    to evaluate a numerical gradient do not count for this.

"""


STOPPING_MAXFUN_GLOBAL = 1_000
"""int:
    If the maximum number of function evaluation is reached, the optimization stops
    but we do not count this as successful convergence. The function evaluations used
    to evaluate a numerical gradient do not count for this. Set to a lower number than
    STOPPING_MAX_CRITERION_EVALUATIONS for global optimizers.

"""


STOPPING_MAXITER = 1_000_000
"""int:
    If the maximum number of iterations is reached, the
    optimization stops, but we do not count this as successful convergence.
    The difference to ``max_criterion_evaluations`` is that one iteration might
    need several criterion evaluations, for example in a line search or to determine
    if the trust region radius has to be shrunk.

"""

CONVERGENCE_SECOND_BEST_FTOL_ABS = 1e-08
"""float: absolute criterion tolerance optimagic requires if no other stopping
criterion apart from max iterations etc. is available
this is taken from scipy (SLSQP's value, smaller than Nelder-Mead).

"""

CONVERGENCE_SECOND_BEST_XTOL_ABS = 1e-08
"""float: The absolute parameter tolerance optimagic requires if no other stopping
criterion apart from max iterations etc. is available. This is taken from pybobyqa.

"""


MAX_LINE_SEARCH_STEPS = 20
"""int: Inspired by scipy L-BFGS-B."""

LIMITED_MEMORY_STORAGE_LENGTH = 10
"""int: Taken from scipy L-BFGS-B."""


CONSTRAINTS_ABSOLUTE_TOLERANCE = 1e-5
"""float: Allowed tolerance of the equality and inequality constraints for values to be
considered 'feasible'.

"""


def get_population_size(population_size, x, lower_bound=10):
    """Default population size for genetic algorithms."""
    if population_size is None:
        population_size = int(np.clip(10 * (len(x) + 1), lower_bound, np.inf))
    else:
        population_size = int(population_size)
    return population_size
