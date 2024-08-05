"""Process the user provided pc for use during the optimization.

The main purpose of this module is to convert the user provided constraints into inputs
for fast reparametrization functions. In the process, the constraints are checked and
consolidated. Consolidation means that redundant constraints are dropped and other
constraints are collected in meaningful bundles.

To improve readability, the actual code for checking and consolidation are in separate
modules.

Calls to functions doing checking are scattered across the module. This is in order to
perform each check as soon as it becomes possible, which allows errors to be raised at a
point where constraints still look similar to what users wrote. However, some checks can
only be done after consolidation.

"""

import numpy as np
import pandas as pd

from optimagic.parameters.check_constraints import (
    check_constraints_are_satisfied,
    check_fixes_and_bounds,
    check_for_incompatible_overlaps,
    check_types,
)
from optimagic.parameters.consolidate_constraints import consolidate_constraints
from optimagic.utilities import number_of_triangular_elements_to_dimension


def process_constraints(
    constraints,
    params_vec,
    lower_bounds,
    upper_bounds,
    param_names,
):
    """Process, consolidate and check constraints.

    Args:
        constraints (list): List of constraints where the fields that select parameters
            have already been consolidated into an ``"index"`` field that selects
            the same parameters from the flattened_parameter vector.
        params_vec (np.ndarray): Flattened version of params.
        lower_bounds (np.ndarray): Lower bounds for params_vec.
        upper_bounds (np.ndarray): Upper bounds for params_vec.
        param_names (list): Names of the flattened parameters. Only used to produce
            good error messages.

    Returns:

        transformations (list): A processed version of those constraints
            that entail actual transformations and not just fixing parameters.
        constr_info (dict): Dict of 1d numpy arrays of length n_params (or None) with
            information that is needed for the reparametrizations.
            - lower_bounds: Lower bounds for the internal parameter vector. Those are
              derived from the original lower bounds and additional bounds implied by
              other constraints.
            - upper_bounds: As lower_bounds but for upper bounds.
            - internal_free: Boolean column that is true for those parameters over
              which the optimizer will actually optimize.
            - pre_replacements: The j_th element indicates the position of the internal
              parameter that has to be copied into the j_th position of the external
              parameter vector when reparametrizing from_internal, before any
              transformations are applied. Negative if no element has to be copied.
            - post_replacements: As pre_replacements, but applied after the
              transformations are done.
            - internal_fixed_values: Contains transformed versions of the fixed values
              that will become equal to the external fixed values after the
              kernel transformations are applied.
              parameter

    """
    params_vec = params_vec.copy()
    check_types(constraints)

    constraints = _replace_pairwise_equality_by_equality(constraints)
    constraints = _process_linear_weights(constraints)
    check_constraints_are_satisfied(constraints, params_vec, param_names)
    constraints = _replace_increasing_and_decreasing_by_linear(constraints)
    constraints = _process_linear_weights(constraints)

    transformations, constr_info = consolidate_constraints(
        constraints=constraints,
        parvec=params_vec,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        param_names=param_names,
    )

    check_for_incompatible_overlaps(transformations, param_names)
    check_fixes_and_bounds(constr_info, transformations, param_names)

    is_fixed_to_value = constr_info.pop("is_fixed_to_value")
    is_fixed_to_other = constr_info.pop("is_fixed_to_other")
    int_lower, int_upper = _create_internal_bounds(
        constr_info["lower_bounds"], constr_info["upper_bounds"], transformations
    )
    constr_info["internal_free"] = _create_internal_free(
        is_fixed_to_value=is_fixed_to_value,
        is_fixed_to_other=is_fixed_to_other,
        constraints=transformations,
    )
    constr_info["lower_bounds"] = int_lower[constr_info["internal_free"]]
    constr_info["upper_bounds"] = int_upper[constr_info["internal_free"]]

    constr_info["pre_replacements"] = _create_pre_replacements(
        constr_info["internal_free"]
    )

    constr_info["internal_fixed_values"] = _create_internal_fixed_value(
        constr_info["fixed_values"], transformations
    )

    del constr_info["fixed_values"]

    return transformations, constr_info


def _replace_pairwise_equality_by_equality(constraints):
    """Rewrite pairwise equality constraints to equality constraints.

    Args:
        constraints (list): List of dictionaries where each dictionary is a constraint.
            It is assumed that the selectors in constraints were already processed.

    Returns:
        list: List of processed constraints.

    """
    pairwise_constraints = [c for c in constraints if c["type"] == "pairwise_equality"]
    constraints = [c for c in constraints if c["type"] != "pairwise_equality"]
    for constr in pairwise_constraints:
        equality_constraints = [
            {"index": list(elements), "type": "equality"}
            for elements in zip(*constr["indices"], strict=False)
        ]
        constraints += equality_constraints

    return constraints


def _process_linear_weights(constraints):
    """Harmonize the weights of linear constraints.

    Args:
        constraints (list): Constraints where the selectors have already been processed.

    Returns:
        list: Constraints where all weights are Series.

    """
    processed = []
    for constr in constraints:
        if constr["type"] == "linear":
            raw_weights = constr["weights"]

            if isinstance(raw_weights, (np.ndarray, list, tuple, pd.Series)):
                if len(raw_weights) != len(constr["index"]):
                    msg = (
                        f"weights of length {len(raw_weights)} could not be aligned "
                        f"with selected parameters of length {len(constr['index'])}."
                    )
                    raise ValueError(msg)
                weights = np.asarray(raw_weights)
            elif isinstance(raw_weights, (float, int)):
                weights = np.full(len(constr["index"]), float(raw_weights))
            else:
                raise TypeError(f"Invalid type for linear weights {type(raw_weights)}.")

            new_constr = constr.copy()
            weights_sr = pd.Series(weights, index=constr["index"])
            new_constr["weights"] = weights_sr
            processed.append(new_constr)
        else:
            processed.append(constr)

    return processed


def _replace_increasing_and_decreasing_by_linear(constraints):
    """Write increasing and decreasing constraints as linear constraints.

    Args:
        constraints (list): Constraints where the selectors have already been processed.

    Returns:
        list: Processed constraints.

    """
    increasing_ilocs, other_constraints = [], []

    for constr in constraints:
        if constr["type"] == "increasing":
            increasing_ilocs.append(constr["index"])
        elif constr["type"] == "decreasing":
            increasing_ilocs.append(constr["index"][::-1])
        else:
            other_constraints.append(constr)

    linear_constraints = []
    for iloc in increasing_ilocs:
        for smaller, larger in zip(iloc, iloc[1:], strict=False):
            linear_constr = {
                "index": [smaller, larger],
                "type": "linear",
                "weights": np.array([-1, 1]),
                "lower_bound": 0,
            }
            linear_constraints.append(linear_constr)

    processed = linear_constraints + other_constraints
    return processed


def _create_internal_bounds(lower, upper, constraints):
    """Create bounds for the internal parameter vector.

    The resulting arrays have the length of the flat external params and will be reduced
    later.

    Args:
        lower (np.ndarray): Processed and consolidated external lower bounds.
        upper (np.ndarray): Processed and consolidated external upper bounds.
        constraints (pd.DataFrame): Processed and consolidated constraints.

    Returns:
        int_lower (np.ndarray): Lower bound of internal parameters.
        int_upper (np.ndarray): Upper bound of internal parameters.

    """
    int_lower, int_upper = lower.copy(), upper.copy()

    for constr in constraints:
        if constr["type"] in ["covariance", "sdcorr"]:
            # Note that the diagonal positions are the same for covariance and sdcorr
            # because the internal params contains the Cholesky factor of the implied
            # covariance matrix in both cases.
            dim = number_of_triangular_elements_to_dimension(len(constr["index"]))
            diag_positions = [0, *np.cumsum(range(2, dim + 1)).tolist()]
            diag_indices = np.array(constr["index"])[diag_positions].tolist()
            bd = constr.get("bounds_distance", 0)
            bd = np.sqrt(bd) if constr["type"] == "covariance" else bd
            int_lower[diag_indices] = np.maximum(int_lower[diag_indices], bd)
        elif constr["type"] == "probability":
            int_lower[constr["index"]] = 0
        elif constr["type"] == "linear":
            int_lower[constr["index"]] = -np.inf
            int_upper[constr["index"]] = np.inf
            relevant_index = constr["index"][-len(constr["right_hand_side"]) :]
            int_lower[relevant_index] = constr["right_hand_side"]["lower_bound"]
            int_upper[relevant_index] = constr["right_hand_side"]["upper_bound"]
        else:
            raise TypeError("Invalid constraint type {}".format(constr["type"]))

    return int_lower, int_upper


def _create_internal_free(is_fixed_to_value, is_fixed_to_other, constraints):
    """Boolean array that is True for parameters over which the optimizer optimizes.

    Args:
        is_fixed_to_value (np.ndarray): boolean array
        is_fixed_to_other (np.ndarray): boolean array

    Returns:
        np.ndarray

    """
    int_fixed = is_fixed_to_value | is_fixed_to_other

    for constr in constraints:
        if constr["type"] == "probability":
            int_fixed[constr["index"][-1]] = True
        elif constr["type"] == "linear":
            int_fixed[constr["index"]] = False
            relevant_index = constr["index"][-len(constr["right_hand_side"]) :]
            int_fixed[relevant_index] = np.isfinite(constr["right_hand_side"]["value"])

    int_free = ~int_fixed

    return int_free


def _create_pre_replacements(internal_free):
    """Create an array with internal position of parameters.

    The j_th element indicates the position of the internal parameter that has to be
    copied into the j_th position of the external parameter vector when reparametrizing
    from_internal, before any transformations are applied. Negative if no element has
    to be copied.

    This will be used to copy the free internal parameters into a parameter vector
    that has the same length as all params.

    Args:
        internal_free (np.ndarray): The _internal_free column of the processed params.

    """
    pre_replacements = np.full(len(internal_free), -1)
    pre_replacements[internal_free] = np.arange(internal_free.sum())

    return pre_replacements


def _create_internal_fixed_value(fixed_values, constraints):
    """Create and array with the values to which internal parameters are fixed.

    This contains additional fixes used to enforce other constraints and (potentially
    transformed) user specified fixed values.

    Args:
        fixed_value (np.ndarray): The (external) _fixed_value column of pp.
        constraints (list): Processed and consolidated params.

    """
    int_fix = fixed_values.copy()
    for constr in constraints:
        if constr["type"] == "probability":
            int_fix[constr["index"][-1]] = 1
        elif constr["type"] in ["covariance", "sdcorr"]:
            int_fix[constr["index"][0]] = np.sqrt(int_fix[constr["index"][0]])
        elif constr["type"] == "linear":
            int_fix[constr["index"]] = np.nan
            relevant_index = constr["index"][-len(constr["right_hand_side"]) :]
            int_fix[relevant_index] = constr["right_hand_side"]["value"].to_numpy()

    return int_fix
