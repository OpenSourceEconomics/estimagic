"""Process the user provided pc for use during the optimization.

The main purpose of this module is to convert the user provided pc into
inputs for fast reparametrization functions. In the process, the pc are
checked and consolidated. Consolidation means that redundant pc are dropped
and other pc are collected in meaningful bundles.

To improve readability, the actual code for checking and consolidation are in separate
modules.

The calls to the check functions are not in one place but scattered over the module.
This is because we want to do each check as soon as it becomes possible, which allows
to write error messages that are closely tied to the pc as written by the
users and not some transformed version thereof. However, some checks can only be done
after the consolidation.


The challenge in making this module readable is that after each function is applied
the list of pc and the params dataframe will be slightly different, but it
is not possible to reflect all of the changes in meaningful names because thy would
get way too long. We chose the following conventions:

As soon as a list of pc or a params DataFrame is different from what the
user provided they are called pc (for pc) and pp (for processed
params) respectively. If all pc of a certain type, say linear, are collected
in a this collection is called pc_linear.

If only few columns of processed params are used in a function, it is better to
pass them as Series, to make the flow of information more explicit.

"""
import warnings

import numpy as np
import pandas as pd

from estimagic.optimization.check_constraints import check_constraints_are_satisfied
from estimagic.optimization.check_constraints import check_fixes_and_bounds
from estimagic.optimization.check_constraints import check_for_incompatible_overlaps
from estimagic.optimization.check_constraints import check_types
from estimagic.optimization.consolidate_constraints import consolidate_constraints
from estimagic.optimization.utilities import number_of_triangular_elements_to_dimension


def process_constraints(constraints, params):
    """Process, consolidate and check constraints.

    Args:
        constraints (list): List of dictionaries where each dictionary is a constraint.
        params (pd.DataFrame): see :ref:`params`.

    Returns:

        pc (list): A processed version of those constraints
            that entail actual transformations and not just fixing parameters.
        pp (pd.DataFrame): Processed params. A copy of params with additional columns:
            - _internal_lower:
              Lower bounds for the internal parameter vector. Those are derived from
              the original lower bounds and additional bounds implied by other
              constraints.
            - _internal_upper: As _internal_lower but for upper bounds.
            - _internal_free: Boolean column that is true for those parameters over
              which the optimizer will actually optimize.
            - _pre_replacements: The j_th element indicates the position of the internal
              parameter that has to be copied into the j_th position of the external
              parameter vector when reparametrizing from_internal, before any
              transformations are applied. Negative if no element has to be copied.
            - _post_replacements: As pre_replacements, but applied after the
              transformations are done.
            - _internal_fixed_value: Contains transformed versions of the fixed values
              that will become equal to the external fixed values after the
              kernel transformations are applied.
            - _is_fixed_to_value: True for parameters that are fixed to a value
            - _is_fixed_to_other: True for parameters that are fixed to another
              parameter

    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="indexing past lexsort depth may impact performance."
        )
        params = params.copy()
        pc = _apply_constraint_killers(constraints)
        check_types(pc)
        # selectors have to be processed before anything else happens to the params
        pc = _process_selectors(pc, params)
        if "lower_bound" not in params.columns:
            params["lower_bound"] = -np.inf
        if "upper_bound" not in params.columns:
            params["upper_bound"] = np.inf
        pc = _replace_pairwise_equality_by_equality(pc)
        pc = _process_linear_weights(pc, params)
        check_constraints_are_satisfied(pc, params)
        pc = _replace_increasing_and_decreasing_by_linear(pc)
        pc = _process_linear_weights(pc, params)
        pc, pp = consolidate_constraints(pc, params)
        check_for_incompatible_overlaps(pp, pc)
        check_fixes_and_bounds(pp, pc)

        int_lower, int_upper = _create_internal_bounds(
            pp.lower_bound, pp.upper_bound, pc
        )
        pp["_internal_lower"] = int_lower
        pp["_internal_upper"] = int_upper
        pp["_internal_free"] = _create_internal_free(
            pp._is_fixed_to_value, pp._is_fixed_to_other, pc
        )
        pp["_pre_replacements"] = _create_pre_replacements(pp._internal_free)
        pp["_internal_fixed_value"] = _create_internal_fixed_value(pp._fixed_value, pc)

        return pc, pp


def _apply_constraint_killers(constraints):
    """Filter out constraints that have a killer."""
    killers, real_constraints = set(), []
    for constr in constraints:
        if "kill" in constr and len(constr) == 1:
            killers.add(constr["kill"])
        else:
            real_constraints.append(constr)

    survivors = []
    for constr in real_constraints:
        if constr.get("id", None) not in killers:
            survivors.append(constr)
        killers.discard(constr.get("id", None))

    if killers:
        raise KeyError(f"You try to kill non-existing constraints with ids: {killers}")

    return survivors


def _process_selectors(constraints, params):
    """Convert the query and loc field of the constraint into position based indices.

    Args:
        constraints (list): List of dictionaries where each dictionary is a constraint.
        params (pd.DataFrame): see :ref:`params`.

    Returns:
        pc (list): The resulting constraint dictionaries contain a new entry
            called 'index' that consists of the positions of the selected parameters.
            If the selected parameters are consecutive entries, the value corresponding
            to 'index' is a list of positions.

    """
    pc = []

    for constr in constraints:
        new_constr = constr.copy()

        if constr["type"] != "pairwise_equality":
            locs = [constr["loc"]] if "loc" in constr else []
            queries = [constr["query"]] if "query" in constr else []
        else:
            locs = new_constr.pop("locs", [])
            queries = new_constr.pop("queries", [])

        positions = pd.Series(data=np.arange(len(params)), index=params.index)

        indices = []
        for loc in locs:
            index = positions.loc[loc].astype(int).tolist()
            index = [index] if not isinstance(index, list) else index
            assert len(set(index)) == len(index), "Duplicates in loc are not allowed."
            indices.append(index)
        for query in queries:
            loc = params.query(query).index
            index = positions.loc[loc].astype(int).tolist()
            index = [index] if not isinstance(index, list) else index
            indices.append(index)

        if constr["type"] == "pairwise_equality":
            assert (
                len(indices) >= 2
            ), "Select at least 2 sets of parameters for pairwise equality constraint!"
            length = len(indices[0])
            for index in indices:
                assert len(index) == length, (
                    "All sets of parameters in pairwise_equality pc must have "
                    "the same length."
                )
            new_constr["indices"] = indices
        else:
            assert (
                len(indices) == 1
            ), "Either loc or query can be in constraint but not both."
            new_constr["index"] = indices[0]
        pc.append(new_constr)

    return pc


def _replace_pairwise_equality_by_equality(pc):
    """Rewrite pairwise equality constraints to equality constraints.

    Args:
        pc (list): List of dictionaries where each dictionary is a constraint.
            It is assumed that the selectors in constraints were already processed.

    Returns:
        pc (list): List of processed constraints.

    """
    pairwise_constraints = [c for c in pc if c["type"] == "pairwise_equality"]
    pc = [c for c in pc if c["type"] != "pairwise_equality"]
    for constr in pairwise_constraints:
        equality_constraints = []
        for elements in zip(*constr["indices"]):
            equality_constraints.append({"index": list(elements), "type": "equality"})
        pc += equality_constraints

    return pc


def _process_linear_weights(pc, params):
    """Harmonize the weights of linear constraints.

    Args:
        pc (list): Constraints where the selector have already been processed.
        params (pd.DataFrame): see :ref:`params`.

    Returns:
        processed (list): Constraints where all weights are Series.

    """
    processed = []
    for constr in pc:
        if constr["type"] == "linear":
            raw_weights = constr["weights"]
            params_subset = params.iloc[constr["index"]]

            msg = f"Weights must be same length as selected parameters: {params_subset}"
            if isinstance(raw_weights, pd.Series):
                weights = raw_weights.loc[params_subset.index].to_numpy()
            elif isinstance(raw_weights, (np.ndarray, list, tuple)):
                if len(raw_weights) != len(params_subset):
                    raise ValueError(msg)
                weights = np.asarray(raw_weights)
            elif isinstance(raw_weights, (float, int)):
                weights = np.full(len(params_subset), float(raw_weights))
            else:
                raise TypeError(f"Invalid type for linear weights {type(raw_weights)}.")

            new_constr = constr.copy()
            weights_sr = pd.Series(weights, index=params_subset.index)
            new_constr["weights"] = weights_sr
            processed.append(new_constr)
        else:
            processed.append(constr)

    return processed


def _replace_increasing_and_decreasing_by_linear(pc):
    """Write increasing and decreasing constraints as linear constraints.

    Args:
        pc (list): Constraints where the selectors have already been processed.

    Returns:
        processed (list)

    """
    increasing_ilocs, other_constraints = [], []

    for constr in pc:
        if constr["type"] == "increasing":
            increasing_ilocs.append(constr["index"])
        elif constr["type"] == "decreasing":
            increasing_ilocs.append(constr["index"][::-1])
        else:
            other_constraints.append(constr)

    linear_constraints = []
    for iloc in increasing_ilocs:
        for smaller, larger in zip(iloc, iloc[1:]):
            linear_constr = {
                "index": [smaller, larger],
                "type": "linear",
                "weights": np.array([-1, 1]),
                "lower_bound": 0,
            }
            linear_constraints.append(linear_constr)

    processed = linear_constraints + other_constraints
    return processed


def _create_internal_bounds(lower, upper, pc):
    """Create columns with bounds for the internal parameter vector.

    The columns have the length of the external params and will be reduced later.

    Args:
        lower (pd.Series): Processed and consolidated external lower bounds.
        upper (pd.Series): Processed and consolidated external upper bounds.
        pc (pd.DataFrame): Processed and consolidated constraints.

    Returns:
        int_lower (pd.Series): Lower bound of internal parameters.
        int_upper (pd.Series): Upper bound of internal parameters.

    """
    int_lower, int_upper = lower.copy(), upper.copy()

    for constr in pc:
        if constr["type"] in ["covariance", "sdcorr"]:
            # Note that the diagonal positions are the same for covariance and sdcorr
            # because the internal params contains the Cholesky factor of the implied
            # covariance matrix in both cases.
            dim = number_of_triangular_elements_to_dimension(len(constr["index"]))
            diag_positions = [0] + np.cumsum(range(2, dim + 1)).tolist()
            diag_indices = np.array(constr["index"])[diag_positions].tolist()
            bd = constr.get("bounds_distance", 0)
            bd = np.sqrt(bd) if constr["type"] == "covariance" else bd
            int_lower.iloc[diag_indices] = np.maximum(int_lower.iloc[diag_indices], bd)
        elif constr["type"] == "probability":
            int_lower.iloc[constr["index"]] = 0
        elif constr["type"] == "linear":
            int_lower.iloc[constr["index"]] = -np.inf
            int_upper.iloc[constr["index"]] = np.inf
            int_lower.update(constr["right_hand_side"]["lower_bound"])
            int_upper.update(constr["right_hand_side"]["upper_bound"])
        else:
            raise TypeError("Invalid constraint type {}".format(constr["type"]))

    return int_lower, int_upper


def _create_internal_free(is_fixed_to_value, is_fixed_to_other, pc):
    """Boolean Series that is true for parameters over which the optimizer optimizes.

    Args:
        is_fixed_to_value (pd.Series): The _is_fixed_to_value column of pp.
        is_fixed_to_other (pd.Series): The _is_fixed_to_other column of pp.

    Returns:
        int_free (pd.Series)
    """
    int_fixed = is_fixed_to_value | is_fixed_to_other

    for constr in pc:
        if constr["type"] == "probability":
            int_fixed.iloc[constr["index"][-1]] = True
        elif constr["type"] == "linear":
            int_fixed.iloc[constr["index"]] = False
            int_fixed.update(constr["right_hand_side"]["value"].notnull())
            # dtype gets messed up by update
            int_fixed = int_fixed.astype(bool)

    int_free = ~int_fixed

    return int_free


def _create_pre_replacements(internal_free):
    """Series with internal position of parameters.

    The j_th element indicates the position of the internal parameter that has to be
    copied into the j_th position of the external parameter vector when reparametrizing
    from_internal, before any transformations are applied. Negative if no element has
    to be copied.

    This will be used to copy the free internal parameters into a parameter vector
    that has the same length as all params.

    Args:
        internal_free (pd.Series): The _internal_free column of the processed params.

    """
    pre_replacements = (
        internal_free.replace(False, np.nan).cumsum().subtract(1).fillna(-1).astype(int)
    )

    return pre_replacements


def _create_internal_fixed_value(fixed_value, pc):
    """Pandas Series containing the values to which internal parameters are fixe.

    This contains additional fixes used to enforce other constraints and (potentially
    transformed) user specified fixed values.

    Args:
        fixed_value (pd.Series): The (external) _fixed_value column of pp.
        pc (list): Processed and consolidated params.

    """
    int_fix = fixed_value.copy()
    for constr in pc:
        if constr["type"] == "probability":
            int_fix.iloc[constr["index"][-1]] = 1
        elif constr["type"] in ["covariance", "sdcorr"]:
            int_fix.iloc[constr["index"][0]] = np.sqrt(int_fix.iloc[constr["index"][0]])
        elif constr["type"] == "linear":
            int_fix.iloc[constr["index"]] = np.nan
            int_fix.update(constr["right_hand_side"]["value"])

    return int_fix
