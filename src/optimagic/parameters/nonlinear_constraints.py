import itertools
from dataclasses import asdict, replace
from functools import partial

import numpy as np
import pandas as pd
from pybaum import tree_flatten, tree_just_flatten, tree_unflatten

from optimagic.differentiation.derivatives import first_derivative
from optimagic.exceptions import InvalidConstraintError, InvalidFunctionError
from optimagic.optimization.algo_options import CONSTRAINTS_ABSOLUTE_TOLERANCE
from optimagic.parameters.block_trees import block_tree_to_matrix
from optimagic.parameters.tree_registry import get_registry


def process_nonlinear_constraints(
    nonlinear_constraints,
    params,
    bounds,
    converter,
    numdiff_options,
    skip_checks,
):
    """Process and prepare nonlinear constraints for internal use.

    Args:
        nonlinear_constraints (list[dict]): List of dictionaries, each representing a
            nonlinear constraint.
        params (pandas): A pytree containing the parameters with respect to which the
            criterion is optimized. Examples are a numpy array, a pandas Series,
            a DataFrame with "value" column, a float and any kind of (nested) dictionary
            or list containing these elements. See :ref:`params` for examples.
        bounds (Bounds): Bounds object containing information on the bounds of the
            parameters. See :ref:`bounds` for details.
        converter (Converter): NamedTuple with methods to convert between internal and
            external parameters, derivatives and function outputs.
        numdiff_options (NumdiffOptions): Options for numerical derivatives. See
            :ref:`first_derivative` for details. Note that the default method is changed
            to "forward" for speed reasons.
        skip_checks (bool): Whether checks on the inputs are skipped. This makes the
            optimization faster, especially for very fast constraint functions. Default
            False.

    Returns:
        list[dict]: List of processed constraints.

    """
    # do checks first to fail fast
    constraint_evals = []
    for _constraint in nonlinear_constraints:
        _eval = _check_validity_and_return_evaluation(_constraint, params, skip_checks)
        constraint_evals.append(_eval)

    processed = []
    for _constraint, _eval in zip(
        nonlinear_constraints, constraint_evals, strict=False
    ):
        _processed_constraint = _process_nonlinear_constraint(
            _constraint,
            constraint_eval=_eval,
            params=params,
            bounds=bounds,
            converter=converter,
            numdiff_options=numdiff_options,
        )
        processed.append(_processed_constraint)

    return processed


def _process_nonlinear_constraint(
    c, constraint_eval, params, bounds, converter, numdiff_options
):
    """Process a single nonlinear constraint."""

    # ==================================================================================
    # Process selector and evaluate functions if necessary
    # ==================================================================================

    external_selector = _process_selector(c)  # functional selector

    constraint_func = c["func"]

    if constraint_eval is None:
        selected = external_selector(params)
        constraint_eval = constraint_func(selected)

    if bounds is not None:
        constraint_bounds = replace(
            bounds,
            lower=external_selector(bounds.lower),
            upper=external_selector(bounds.upper),
        )
    else:
        constraint_bounds = None

    _n_constr = len(np.atleast_1d(constraint_eval))

    # ==================================================================================
    # Consolidate and transform jacobian
    # ==================================================================================

    # process numdiff_options for numerical derivative

    if "derivative" in c:
        if not callable(c["derivative"]):
            msg = "Jacobian of constraints needs to be callable."
            raise ValueError(msg)
        jacobian = c["derivative"]
    else:
        # use finite-differences if no closed-form jacobian is defined
        def jacobian(p):
            return first_derivative(
                constraint_func,
                p,
                bounds=constraint_bounds,
                error_handling="raise_strict",
                **asdict(numdiff_options),
            ).derivative

    # To define the internal Jacobian we need to know which parameters enter the
    # contraint function.
    selection_indices, n_params = _get_selection_indices(params, external_selector)

    def _internal_jacobian(x):
        """Return Jacobian of constraint at internal parameters.

        The constraint function is written to be evaluated on a selection of the
        external parameters. The optimizer, however, only works on internal parameters.
        These can be significantly different from the external parameters, due to
        optimagic's reparametrization features. In this function we compute the Jacobian
        of the constraint at the internal parameters using information on the Jacobian
        of the constraint at the selected external parameters.

        """
        params = converter.params_from_internal(x)
        selected = external_selector(params)
        jac = jacobian(selected)
        jac_matrix = block_tree_to_matrix(jac, constraint_eval, selected)
        jac_extended = _extend_jacobian(jac_matrix, selection_indices, n_params)
        jac_internal = converter.derivative_to_internal(
            jac_extended, x, jac_is_flat=True
        )
        return np.atleast_2d(jac_internal)

    # ==================================================================================
    # Transform constraint function and derive bounds
    # ==================================================================================
    _type = "eq" if "value" in c else "ineq"

    if _type == "eq":
        # ==============================================================================
        # Equality constraints
        #
        # We define the internal constraint function to be satisfied if it is equal
        # to zero, by subtracting the fixed value.

        _value = np.atleast_1d(np.array(c["value"], dtype=float))

        def internal_constraint_func(x):
            params = converter.params_from_internal(x)
            select = external_selector(params)
            out = np.atleast_1d(constraint_func(select)) - _value
            return out

        jacobian_from_internal = _internal_jacobian
        n_constr = _n_constr

    else:
        # ==============================================================================
        # Inequality constraints
        #
        # We define the internal constraint function to be satisfied if it is
        # greater than or equal to zero (positivity constraint). If the bounds already
        # satify this condition we do not change anything, otherwise we need to perform
        # a transformation.

        def _internal_constraint_func(x):
            params = converter.params_from_internal(x)
            select = external_selector(params)
            return np.atleast_1d(constraint_func(select))

        lower_bounds = c.get("lower_bounds", 0)
        upper_bounds = c.get("upper_bounds", np.inf)

        transformation = _get_transformation(lower_bounds, upper_bounds)

        internal_constraint_func = _compose_funcs(
            _internal_constraint_func, transformation["func"]
        )

        jacobian_from_internal = _compose_funcs(
            _internal_jacobian, transformation["derivative"]
        )

        n_constr = 2 * _n_constr if transformation["name"] == "stack" else _n_constr

    internal_constr = {
        "n_constr": n_constr,
        "type": _type,
        "fun": internal_constraint_func,  # internal name for 'func'
        "jac": jacobian_from_internal,  # internal name for 'derivative'
        "tol": c.get("tol", CONSTRAINTS_ABSOLUTE_TOLERANCE),
    }

    return internal_constr


def equality_as_inequality_constraints(nonlinear_constraints):
    """Return constraints where equality constraints are converted to inequality."""
    constraints = [_equality_to_inequality(c) for c in nonlinear_constraints]
    return constraints


def _equality_to_inequality(c):
    """Transform a single constraint.

    An equality constaint g(x) = 0 can be transformed to two inequality constraints
    using (g(x), -g(x)) >= 0. Hence, the number of constraints doubles, and the
    constraint functions itself as well as the derivative need to be updated.

    """
    if c["type"] == "eq":

        def transform(x, func):
            value = func(x)
            return np.concatenate((value, -value), axis=0)

        out = {
            "fun": partial(transform, func=c["fun"]),
            "jac": partial(transform, func=c["jac"]),
            "n_constr": 2 * c["n_constr"],
            "tol": c["tol"],
            "type": "ineq",
        }
    else:
        out = c
    return out


def vector_as_list_of_scalar_constraints(nonlinear_constraints):
    """Return constraints where vector constraints are converted to scalar constraints.

    This is necessary for internal optimizers that only support scalar constraints.

    """
    list_of_constraints_lists = [
        _vector_to_list_of_scalar(c) for c in nonlinear_constraints
    ]
    constraints = list(itertools.chain.from_iterable(list_of_constraints_lists))
    return constraints


def _vector_to_list_of_scalar(constraint):
    if constraint["n_constr"] > 1:
        out = []
        for k in range(constraint["n_constr"]):
            c = constraint.copy()
            fun, jac = _get_components(constraint["fun"], constraint["jac"], idx=k)
            c["fun"] = fun
            c["jac"] = jac
            c["n_constr"] = 1
            out.append(c)
    else:
        out = [constraint]
    return out


def _get_components(fun, jac, idx):
    """Return function and derivative for a single component of a vector function.

    Args:
        fun (callable): Function that returns a vector.
        jac (callable): Derivative of the function that returns a matrix.
        idx (int): Index of the component.

    Returns:
        callable: Component function at index idx.
        callable: Jacobian of the component function.

    """
    fun_component = lambda x: fun(x)[idx]
    jac_component = lambda x: jac(x)[idx]
    return fun_component, jac_component


# ======================================================================================
# Helper Functions
# ======================================================================================


def _process_selector(c):
    if "selector" in c:
        selector = c["selector"]
    elif "loc" in c:

        def selector(params):
            return params.loc[c["loc"]]

    elif "query" in c:

        def selector(params):
            return params.query(c["query"])

    else:
        selector = _identity
    return selector


def _compose_funcs(f, g):
    return lambda x: g(f(x))


def _identity(x):
    return x


# ======================================================================================
# Jacobian helper functions
# ======================================================================================


def _extend_jacobian(jac_mat, selection_indices, n_params):
    """Extend Jacobian on selected parameters to full params.

    Jacobian of constraints is defined on a selection of the parameters, however, we
    need the Jacobian on the full params. Since the Jacobian is trivially zero at the
    non-selected params we can simply fill a zero matrix.

    """
    jac_extended = np.zeros((jac_mat.shape[0], n_params))
    jac_extended[:, selection_indices] = jac_mat
    return jac_extended


def _get_selection_indices(params, selector):
    """Get index of selected flat params and number of flat params."""
    registry = get_registry(extended=True)
    flat_params, params_treedef = tree_flatten(params, registry=registry)
    n_params = len(flat_params)
    indices = np.arange(n_params, dtype=int)
    params_indices = tree_unflatten(params_treedef, indices, registry=registry)
    selected = selector(params_indices)
    selection_indices = np.array(
        tree_just_flatten(selected, registry=registry), dtype=int
    )
    return selection_indices, n_params


# ======================================================================================
# Transformation helper functions
# ======================================================================================


def _get_transformation(lower_bounds, upper_bounds):
    """Get transformation given bounds.

    The internal inequality constraint is defined as h(x) >= 0. However, the user can
    specify: a <= g(x) <= b. To get the internal represenation we need to transform the
    constraint.

    """
    transformation_type = _get_transformation_type(lower_bounds, upper_bounds)

    if transformation_type == "identity":
        transformer = {"func": _identity, "derivative": _identity}
    elif transformation_type == "subtract_lb":
        transformer = {
            "func": lambda v: v - lower_bounds,
            "derivative": _identity,
        }
    elif transformation_type == "stack":
        transformer = {
            "func": lambda v: np.concatenate(
                (v - lower_bounds, upper_bounds - v), axis=0
            ),
            "derivative": lambda v: np.concatenate((v, -v), axis=0),
        }
    transformer["name"] = transformation_type
    return transformer


def _get_transformation_type(lower_bounds, upper_bounds):
    lb_is_zero = not np.count_nonzero(lower_bounds)
    ub_is_inf = np.all(np.isposinf(upper_bounds))

    if lb_is_zero and ub_is_inf:
        # the external constraint is already in the correct format
        _transformation_type = "identity"
    elif ub_is_inf:
        # the external constraint can be transformed by subtraction
        _transformation_type = "subtract_lb"
    else:
        # the external constraint can only be transformed by duplication (stacking)
        _transformation_type = "stack"
    return _transformation_type


# ======================================================================================
# Checks
# ======================================================================================


def _check_validity_and_return_evaluation(c, params, skip_checks):
    """Check that nonlinear constraints are valid.

    Returns:
        constaint_eval: Evaluation of constraint at params, if skip_checks if False,
            else None.

    """
    # ==================================================================================
    # check functions
    # ==================================================================================

    if "func" not in c:
        raise InvalidConstraintError(
            "Constraint needs to have entry 'fun', representing the constraint "
            "function."
        )
    if not callable(c["func"]):
        raise InvalidConstraintError(
            "Entry 'fun' in nonlinear constraints has be callable."
        )

    if "derivative" in c and not callable(c["derivative"]):
        raise InvalidConstraintError(
            "Entry 'jac' in nonlinear constraints has be callable."
        )

    # ==================================================================================
    # check bounds
    # ==================================================================================

    is_equality_constraint = "value" in c

    if is_equality_constraint:
        if "lower_bounds" in c or "upper_bounds" in c:
            raise InvalidConstraintError(
                "Only one of 'value' or ('lower_bounds', 'upper_bounds') can be "
                "passed to a nonlinear constraint."
            )

    if not is_equality_constraint:
        if "lower_bounds" not in c and "upper_bounds" not in c:
            raise InvalidConstraintError(
                "For inequality constraint at least one of ('lower_bounds', "
                "'upper_bounds') has to be passed to the nonlinear constraint."
            )

    if "lower_bounds" in c and "upper_bounds" in c:
        if not np.all(np.array(c["lower_bounds"]) <= np.array(c["upper_bounds"])):
            raise InvalidConstraintError(
                "If lower bounds need to less than or equal to upper bounds."
            )

    # ==================================================================================
    # check selector
    # ==================================================================================

    if "selector" in c:
        if not callable(c["selector"]):
            raise InvalidConstraintError(
                f"'selector' entry needs to be callable in constraint {c}."
            )
        else:
            try:
                c["selector"](params)
            except Exception as e:
                raise InvalidFunctionError(
                    "Error when calling 'selector' function on params in constraint "
                    f" {c}"
                ) from e

    elif "loc" in c:
        if not isinstance(params, (pd.Series, pd.DataFrame)):
            raise InvalidConstraintError(
                "params needs to be pd.Series or pd.DataFrame to use 'loc' selector in "
                f"in consrtaint {c}."
            )
        try:
            params.loc[c["loc"]]
        except (KeyError, IndexError) as e:
            raise InvalidConstraintError("'loc' string is invalid.") from e

    elif "query" in c:
        if not isinstance(params, pd.DataFrame):
            raise InvalidConstraintError(
                "params needs to be pd.DataFrame to use 'query' selector in "
                f"constraints {c}."
            )
        try:
            params.query(c["query"])
        except Exception as e:
            raise InvalidConstraintError(
                f"'query' string is invalid in constraint {c}."
            ) from e

    # ==================================================================================
    # check that constraints can be evaluated
    # ==================================================================================

    constraint_eval = None

    if not skip_checks:
        selector = _process_selector(c)

        try:
            constraint_eval = c["func"](selector(params))
        except Exception as e:
            raise InvalidFunctionError(
                f"Error when evaluating function of constraint {c}."
            ) from e

    return constraint_eval
