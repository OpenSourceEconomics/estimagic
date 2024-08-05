"""Functions to consolidate user provided constraints.

Consolidation means that redundant constraints are dropped and other constraints are
collected in meaningful bundles.

Check the module docstring of process_constraints for naming conventions.

"""

import numpy as np
import pandas as pd

from optimagic.exceptions import InvalidConstraintError
from optimagic.utilities import number_of_triangular_elements_to_dimension


def consolidate_constraints(
    constraints, parvec, lower_bounds, upper_bounds, param_names
):
    """Consolidate constraints with each other and remove redundant ones.

    Args:
        constraints (list): List with constraint dictionaries. It is assumed that
            the selectors are already processed, increasing and decreasing
            constraints have been rewritten as linear constraints and
            pairwise_equality constraints have been rewritten as equality constraints.
        parvec (np.ndarray): 1d numpy array with parameters.
        lower_bounds (np.ndarray): 1d numpy array with lower_bounds
        upper_bounds (np.ndarray): 1d numpy array wtih upper_bounds
        param_names (list): Names of parameters. Used for error messages.

    Returns:
        list: This contains processed version of all
            constraints that require an actual kernel transformation. The information
            on all other constraints is subsumed in pp.
        dict: Dict of 1d numpy arrays with information about non-transforming
            constraints.

    """
    raw_eq, other_constraints = _split_constraints(constraints, "equality")
    equality_constraints = _consolidate_equality_constraints(raw_eq)

    fixed_constraints, other_constraints = _split_constraints(
        other_constraints, "fixed"
    )
    fixed_value = _consolidate_fixes_with_equality_constraints(
        fixed_constraints, equality_constraints, parvec
    )

    constr_info = {
        "fixed_values": fixed_value,
        "is_fixed_to_value": np.isfinite(fixed_value),
    }

    other_constraints = [
        c
        for c in other_constraints
        if not constr_info["is_fixed_to_value"][c["index"]].all()
    ]

    (
        other_constraints,
        lower_bounds,
        upper_bounds,
    ) = simplify_covariance_and_sdcorr_constraints(
        constraints=other_constraints,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        is_fixed_to_value=constr_info["is_fixed_to_value"],
        fixed_value=constr_info["fixed_values"],
    )

    lower_bounds, upper_bounds = _consolidate_bounds_with_equality_constraints(
        equality_constraints,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    constr_info["lower_bounds"] = lower_bounds
    constr_info["upper_bounds"] = upper_bounds

    (
        other_constraints,
        post_replacements,
        is_fixed_to_other,
    ) = _plug_equality_constraints_into_selectors(
        equality_constraints, other_constraints, n_params=len(parvec)
    )

    constr_info["post_replacements"] = post_replacements
    constr_info["is_fixed_to_other"] = is_fixed_to_other

    linear_constraints, other_constraints = _split_constraints(
        other_constraints, "linear"
    )

    if len(linear_constraints) > 0:
        linear_constraints = _consolidate_linear_constraints(
            params_vec=parvec,
            linear_constraints=linear_constraints,
            constr_info=constr_info,
            param_names=param_names,
        )

    constraints = other_constraints + linear_constraints

    return constraints, constr_info


def _consolidate_equality_constraints(equality_constraints):
    """Consolidate equality constraints as far as possible.

    Since equality is a transitive conditions we can consolidate any two equality
    constraints that have at least one parameter in common into one condition. Besides
    being faster, this also ensures that the result remains unchanged if equality
    constraints are are split into several different constraints or if they are
    specified in a different order.

    The index in the consolidated equality constraints is sorted in the same order
    as the index of params. This is no problem because order is irrelevant for
    equality constraints.

    Args:
        equality_constraints (list): List of dictionaries where each dictionary is a
            constraint. It is assumed that the selectors were already processed.

    Returns:
        list: List of consolidated equality constraints.

    """
    candidates = [constr["index"] for constr in equality_constraints]
    # drop constraints that just restrict one parameter to be equal to itself
    candidates = [c for c in candidates if len(c) >= 2]
    merged = _join_overlapping_lists(candidates)
    consolidated = [{"index": sorted(index), "type": "equality"} for index in merged]

    return consolidated


def _join_overlapping_lists(candidates):
    """Bundle all candidates with with non-empty intersection.

    Args:
        candidates (list): List of potentially overlapping lists.

    Returns:
        bundles (list): List of lists where all overlapping lists have been joined
            and sorted.

    """
    bundles = []

    while len(candidates) > 0:
        new_candidates = _unite_first_with_all_intersecting_elements(candidates)
        if len(candidates) == len(new_candidates):
            bundles.append(sorted(new_candidates[0]))
            candidates = candidates[1:]
        else:
            candidates = new_candidates

    return bundles


def _unite_first_with_all_intersecting_elements(indices):
    """Helper function to bundle overlapping indices.

    Args:
        indices (list): A list lists with indices.

    """
    first = set(indices[0])
    new_first = first
    new_others = []
    for idx in indices[1:]:
        if len(first.intersection(idx)) > 0:
            new_first = new_first.union(idx)
        else:
            new_others.append(idx)

    return [new_first, *new_others]


def _consolidate_fixes_with_equality_constraints(
    fixed_constraints, equality_constraints, parvec
):
    """Consolidate fixes with equality constraints.

    If any equality constrained parameter is fixed, all of the parameters that are
    equal to it have to be fixed to the same value.

    Args:
        fixed_constraints (list): List of constrains of type "fixed".
        equality_constraints (list): List of constraints of type "equality".
        parvec (np.ndarray): 1d numpy array with parameters.

    Returns:
        fixed_value (pd.Series): Series with the fixed value for all parameters that
            are fixed and np.nan everywhere else. Has the same index as params.

    """
    fixed_value = np.full(len(parvec), np.nan)
    for fix in fixed_constraints:
        fixed_value[fix["index"]] = fix.get("value", parvec[fix["index"]])

    for eq in equality_constraints:
        if np.isfinite(fixed_value[eq["index"]]).any():
            valcounts = _unique_values(fixed_value[eq["index"]])
            assert (
                len(valcounts) == 1
            ), "Equality constrained parameters cannot be fixed to different values."
            fixed_value[eq["index"]] = valcounts[0]

    return fixed_value


def _consolidate_bounds_with_equality_constraints(
    equality_constraints, lower_bounds, upper_bounds
):
    """Consolidate bounds with equality constraints.

    Check that there are no incompatible bounds on equality constrained parameters and
    set the bounds for equal parameters to the strictest bound encountered on any of
    them.

    Args:
        equality_constraints (list): List of constraints of type "equality".
        lower_bounds (np.ndarray): Lower bounds for parameters.
        upper_bounds (np.ndarray): Upper bounds for parameters.

    Returns:
        np.ndarray: 1d array with lower bounds
        np.ndarray: 1d array with upper bounds

    """
    lower = lower_bounds.copy()
    upper = upper_bounds.copy()
    for eq in equality_constraints:
        lower[eq["index"]] = lower[eq["index"]].max()
        upper[eq["index"]] = upper[eq["index"]].min()

    return lower, upper


def _split_constraints(constraints, type_):
    """Split list of constraints in two list.

    The first list contains all constraints of type and the second the rest.

    """
    filtered = [c for c in constraints if c["type"] == type_]
    rest = [c for c in constraints if c["type"] != type_]
    return filtered, rest


def simplify_covariance_and_sdcorr_constraints(
    constraints,
    lower_bounds,
    upper_bounds,
    is_fixed_to_value,
    fixed_value,
):
    """Enforce covariance and sdcorr constraints by bounds if possible.

    This is possible if the dimension is <= 2 or all covariances are fexd to 0.

    """
    cov_constraints, others = _split_constraints(constraints, "covariance")
    sdcorr_constraints, others = _split_constraints(others, "sdcorr")
    to_simplify = cov_constraints + sdcorr_constraints
    lower = lower_bounds.copy()
    upper = upper_bounds.copy()

    not_simplifyable = []
    for constr in to_simplify:
        dim = number_of_triangular_elements_to_dimension(len(constr["index"]))
        if constr["type"] == "covariance":
            diag_positions = [0, *np.cumsum(range(2, dim + 1)).tolist()]
            diag_indices = np.array(constr["index"])[diag_positions].tolist()
            off_indices = [i for i in constr["index"] if i not in diag_positions]
        if constr["type"] == "sdcorr":
            diag_indices = constr["index"][:dim]
            off_indices = constr["index"][dim:]

        uncorrelated = False
        if is_fixed_to_value[off_indices].all():
            if (fixed_value[off_indices] == 0).all():
                uncorrelated = True

        if uncorrelated:
            lower[diag_indices] = np.maximum(0, lower[diag_indices])
        elif dim <= 2 and constr["type"] == "sdcorr":
            lower[diag_indices] = np.maximum(0, lower[diag_indices])
            lower[off_indices] = -1
            upper[off_indices] = 1
        else:
            not_simplifyable.append(constr)

    return others + not_simplifyable, lower, upper


def _plug_equality_constraints_into_selectors(
    equality_constraints, other_constraints, n_params
):
    """Rewrite all constraint in terms of free parameters.

    Only one parameter from a set of equality constrained parameters will actually
    be free. Which one is not important. We take the one with the lowest iloc.

    Then all other constraints have to be rewritten in terms of the free parameters.
    Once that is done, redundant constraints can be filtered out.

    Args:
        equality_constraints (list): List of constraints of type "equality".
        other_constraints (list): All other constraints.
        n_params (int): Number of parameters.

    Returns:
        list: List of processed non-equality constraints.
        np.ndarray: post_replacements
        np.ndarray: is_fixed_to_other

    """
    is_equal_to = np.full(n_params, -1)
    for eq in equality_constraints:
        is_equal_to[sorted(eq["index"])[1:]] = sorted(eq["index"])[0]
    post_replacements = is_equal_to.astype(int)
    is_fixed_to_other = is_equal_to >= 0
    helper = pd.Series(post_replacements)
    replace_dict = helper[helper >= 0].to_dict()

    plugged_in = []
    for constr in other_constraints:
        new = constr.copy()
        new["index"] = pd.Series(constr["index"]).replace(replace_dict).tolist()
        plugged_in.append(new)

    linear_constraints, others = _split_constraints(plugged_in, "linear")

    pc = []
    for constr in others:
        if not _is_redundant(constr, pc):
            pc.append(constr)

    pc += linear_constraints

    return pc, post_replacements, is_fixed_to_other


def _consolidate_linear_constraints(
    params_vec, linear_constraints, constr_info, param_names
):
    """Consolidate linear constraints.

    Consolidation entails the following steps:
    - Plugging fixes and equality constraints into the linear constraints
    - Collect weights of those constraints that overlap into weight DataFrames
    - Collect corresponding right hand sides (bounds or values) in DataFrames
    - Express box constraints of parameters involved in linear constraints as
      additional linear constraints.
    - Rescale the weights for easier detection of linear dependence
    - Drop redundant constraints
    - Check compatibility of constraints
    - Construct a list of consolidated constraint dictionaries that contain
        all matrices needed for the kernel transformations.

    Args:
        params_vec (np.ndarray): 1d numpy array wtih parameters
        linear_constraints (list): Linear constraints that already have processed
            weights and selector fields.
        constr_info (dict): Dict with information about constraints.
        param_names (list): Parameter names. Used for error messages.

    Returns:
        list: Processed and consolidated linear constraints.

    """
    weights, right_hand_side = _transform_linear_constraints_to_pandas_objects(
        linear_constraints, n_params=len(params_vec)
    )

    weights = _plug_equality_constraints_into_linear_weights(
        weights, constr_info["post_replacements"]
    )
    weights, right_hand_side = _plug_fixes_into_linear_weights_and_rhs(
        weights,
        right_hand_side,
        constr_info["is_fixed_to_value"],
        constr_info["fixed_values"],
    )

    involved_parameters = [set(w[w != 0].index) for _, w in weights.iterrows()]

    bundled_indices = _join_overlapping_lists(involved_parameters)

    pc = []
    for involved_parameters in bundled_indices:
        w = weights[involved_parameters][
            (weights[involved_parameters] != 0).any(axis=1)
        ].copy(deep=True)
        rhs = right_hand_side.loc[w.index].copy(deep=True)
        w, rhs = _express_bounds_as_linear_constraints(
            w, rhs, constr_info["lower_bounds"], constr_info["upper_bounds"]
        )
        w, rhs = _rescale_linear_constraints(w, rhs)
        w, rhs = _drop_redundant_linear_constraints(w, rhs)
        _check_consolidated_weights(w, param_names=param_names)
        to_internal, from_internal = _get_kernel_transformation_matrices(w)
        constr = {
            "index": list(w.columns),
            "type": "linear",
            "to_internal": to_internal,
            "from_internal": from_internal,
            "right_hand_side": rhs,
        }
        pc.append(constr)

    return pc


def _transform_linear_constraints_to_pandas_objects(linear_constranits, n_params):
    """Collect information from the linear constraint dictionaries into pandas objects.

    Args:
        linear_constraints (list): List of constraint of type "linear".
        n_params (int): number of parameters.

    Returns:
        weights (pd.DataFrame): DataFrame with one row per constraint and one column
            per parameter. Columns names are the ilocs of the parameters in params.
        rhs (pd.DataFrame): DataFrame with the columns "value", "lower_bound" and
            "upper_bound" that collects the right hand sides of the constraints.

    """
    all_weights, all_values, all_lbs, all_ubs = [], [], [], []
    for constr in linear_constranits:
        all_weights.append(constr["weights"])
        all_values.append(constr.get("value", np.nan))
        all_lbs.append(constr.get("lower_bound", -np.inf))
        all_ubs.append(constr.get("upper_bound", np.inf))

    weights = pd.concat(all_weights, axis=1).T.reset_index()
    weights = weights.reindex(columns=np.arange(n_params)).fillna(0)
    values = pd.Series(all_values, name="value")
    lbs = pd.Series(all_lbs, name="lower_bound")
    ubs = pd.Series(all_ubs, name="upper_bound")
    rhs = pd.concat([values, lbs, ubs], axis=1)

    return weights, rhs


def _plug_equality_constraints_into_linear_weights(weights, post_replacements):
    """Sum the weights of equality constrained parameters.

    The sum of the weights is then the new weight of the equality constrained parameter
    that is actually free. The weights of the other parameters are set to zero.

    Args:
        weights (pd.DataFrame): Weight matrices for linear constraints.
        post_replacements (pd.Series): The _post_replacements column of pp.

    Returns:
        plugged_weights (pd.DataFrame)

    """
    w = weights.T
    plugged_iloc = pd.Series(post_replacements)
    plugged_iloc = plugged_iloc.where(plugged_iloc >= 0, np.arange(len(plugged_iloc)))
    w["plugged_iloc"] = plugged_iloc

    plugged_weights = w.groupby("plugged_iloc").sum()
    plugged_weights = plugged_weights.reindex(w.index).fillna(0).T

    return plugged_weights


def _plug_fixes_into_linear_weights_and_rhs(
    weights, rhs, is_fixed_to_value, fixed_value
):
    """Set weights of fixed parameters to 0 and adjust right hand sides accordingly.

    Args:
        weights (pd.DataFrame): Weight matrix for linear constraint.
        rhs (pd.DataFrame): Right hand side of the linear constraint.
        is_fixed_to_value (pd.Series): The _is_fixed_to_value column of pp.
        fixed_value (pd.Series): The _fixed_value column of pp.

    Returns:
        new_weights (pd.DataFrame)
        new_rhs (pd.DataFrame)

    """
    ilocs = np.arange(len(fixed_value))
    fixed_ilocs = ilocs[is_fixed_to_value].tolist()
    new_rhs = rhs.copy()
    new_weights = weights.copy()

    if len(fixed_ilocs) > 0:
        fixed_values = fixed_value[fixed_ilocs]
        fixed_contribution = weights[fixed_ilocs] @ fixed_values
        for column in ["lower_bound", "upper_bound", "value"]:
            new_rhs[column] = new_rhs[column] - fixed_contribution
        for i in fixed_ilocs:
            new_weights[i] = 0

    return new_weights, new_rhs


def _express_bounds_as_linear_constraints(weights, rhs, lower, upper):
    """Express bounds of linearly constrained params as linear constraint.

    In general it is easier to keep bounds separately from the constraints
    but in the case of linearly constrained parameters we need to express them as
    additional linear constraints to check compatibility and to choose the correct
    reparametrization.

    Args:
        weights (pd.DataFrame): The weight matrix of the linear constraint.
        rhs (pd.DataFrame): The right hand side of the linear constraint.
        lower (np.ndarray): Lower bounds.
        upper (np.ndarray): Upper bounds.

    Returns:
        extended_weights (pd.DataFrame)
        extended_rhs (pd.DataFrame)

    """
    additional_pc = []
    for i in weights.columns:
        new = {}
        if np.isfinite(lower[i]):
            new["lower_bound"] = lower[i]
        if np.isfinite(upper[i]):
            new["upper_bound"] = upper[i]
        if new != {}:
            new["weights"] = pd.Series([1], name="w", index=[i])
            additional_pc.append(new)

    if len(additional_pc) > 0:
        new_weights, new_rhs = _transform_linear_constraints_to_pandas_objects(
            additional_pc, len(lower)
        )
        new_weights = new_weights[weights.columns]

        extended_weights = pd.concat([weights, new_weights]).reset_index(drop=True)
        extended_rhs = pd.concat([rhs, new_rhs]).reset_index(drop=True)
    else:
        extended_weights, extended_rhs = weights, rhs

    return extended_weights, extended_rhs


def _rescale_linear_constraints(weights, rhs):
    """Rescale rows in weights such that the first nonzero element equals one.

    This will make it easier to detect redundant rows.

    Args:
        weights (pd.DataFrame): The weight matrix of the linear constraint.
        rhs (pd.DataFrame): The right hand side of the linear constraint.

    Returns:
        new_weights (pd.DataFrame)
        new_rhs (pd.DataFrame)

    """
    first_nonzero = weights.replace(0, np.nan).bfill(axis=1).iloc[:, 0]
    scaling_factor = 1 / first_nonzero.to_numpy().reshape(-1, 1)
    new_weights = scaling_factor * weights
    scaled_rhs = scaling_factor * rhs
    new_rhs = scaled_rhs.copy()
    new_rhs["lower_bound"] = scaled_rhs["lower_bound"].where(
        scaling_factor.flatten() > 0, scaled_rhs["upper_bound"]
    )
    new_rhs["upper_bound"] = scaled_rhs["upper_bound"].where(
        scaling_factor.flatten() > 0, scaled_rhs["lower_bound"]
    )

    return new_weights, new_rhs


def _drop_redundant_linear_constraints(weights, rhs):
    """Drop linear constraints that are implied by other linear constraints.

    This is not yet very smart. We just check for linearly dependent weights.

    Args:
        weights (pd.DataFrame): The weight matrix of the linear constraint.
        rhs (pd.DataFrame): The right hand side of the linear constraint.

    Returns:
        new_weights (pd.DataFrame)
        new_rhs (pd.DataFrame)

    """
    weights["dupl_group"] = weights.groupby(list(weights.columns)).ngroup()
    rhs["dupl_group"] = weights["dupl_group"]
    weights.set_index("dupl_group", inplace=True)

    new_weights = weights.drop_duplicates()

    def _consolidate_fix(x):
        vc = x.value_counts(dropna=True)
        if len(vc) == 0:
            return np.nan
        elif len(vc) == 1:
            return vc.index[0]
        else:
            raise ValueError

    ub = rhs.groupby("dupl_group")["upper_bound"].min()
    lb = rhs.groupby("dupl_group")["lower_bound"].max()
    fix = rhs.groupby("dupl_group")["value"].apply(_consolidate_fix)

    # remove the bounds for fixed parameters
    ub = ub.where(fix.isnull(), np.inf)
    lb = lb.where(fix.isnull(), -np.inf)

    new_rhs = pd.concat(
        [lb, ub, fix], axis=1, names=["lower_bound", "upper_bound", "value"]
    )
    new_rhs = new_rhs.reindex(new_weights.index)

    return new_weights, new_rhs


def _check_consolidated_weights(weights, param_names):
    """Check the rank condition on the linear weights."""
    n_constraints, n_params = weights.shape

    msg_too_many = (
        "Too many linear constraints. There can be at most as many linear constraints"
        "as involved parameters with non-zero weights.\n"
    )

    msg_rank = "The weights for linear constraints must be linearly independent.\n"

    msg_general = (
        "The error occurred for constraints on the following parameters:\n{}\n with "
        "weighting matrix:\n{}\nIt is possible that you did not specify those "
        "constraints as linear constraints but as bounds, fixes, increasing or "
        "decreasing constraints."
    )
    relevant_names = [param_names[i] for i in weights.columns]

    if n_constraints > n_params:
        raise InvalidConstraintError(
            msg_too_many + msg_general.format(relevant_names, weights)
        )

    if np.linalg.matrix_rank(weights) < n_constraints:
        raise InvalidConstraintError(
            msg_rank + msg_general.format(relevant_names, weights)
        )


def _get_kernel_transformation_matrices(weights):
    """Construct the m matrix for the kernel transformations.

    See :ref:`linear_constraint_implementation` for details.

    Args:
        weights (pd.DataFrame): Weight matrix of a linear constraint.

    """
    n_constraints, n_params = weights.shape

    identity = np.eye(n_params)

    i = 0
    filled_weights = weights
    while len(filled_weights) < n_params:
        candidate = np.vstack([identity[i], filled_weights])
        if np.linalg.matrix_rank(candidate) == len(candidate):
            filled_weights = candidate
        i += 1

    k = n_params - n_constraints

    filled_weights[:k] = filled_weights[:k][::-1]

    to_internal = filled_weights
    from_internal = np.linalg.inv(to_internal)

    return to_internal, from_internal


def _is_redundant(candidate, others):
    """Check if a constraint is redundant given other constraints.

    Applicable to all but linear constraints.

    """
    assert candidate["type"] != "linear"
    if len(others) == 0:
        is_redundant = False
    else:
        same_type, _ = _split_constraints(others, candidate["type"])
        duplicates = [c for c in same_type if c["index"] == candidate["index"]]
        is_redundant = len(duplicates) > 0

    return is_redundant


def _unique_values(arr, dropna=True):
    if dropna:
        arr = arr[np.isfinite(arr)]
    return list(set(arr.tolist()))
