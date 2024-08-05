"""Check compatibility of pc with each other and with bounds and fixes.

See the module docstring of process_constraints for naming conventions.

"""

from functools import partial

import numpy as np
import pandas as pd

from optimagic.exceptions import InvalidConstraintError, InvalidParamsError
from optimagic.utilities import cov_params_to_matrix, sdcorr_params_to_matrix


def check_constraints_are_satisfied(flat_constraints, param_values, param_names):
    """Check that params satisfies all constraints.

    This should be called before the more specialized constraints are rewritten to
    linear constraints in order to get better error messages!

    We let the checks pass if all "values" are np.nan. This way `process_constraints`
    can be used on empty params DataFrames which is useful to construct templates for
    start parameters that can be filled out by the user.

    Args:
        pc (list): List of constraints with processed selectors.
        params (pd.DataFrame): See :ref:`params`

    Raises:
        ValueError if constraints are not satisfied.

    """
    # skip check if all parameters are NaN
    if not np.isfinite(param_values).any():
        return

    for constr in flat_constraints:
        typ = constr["type"]
        subset = param_values[constr["index"]]

        report = []

        _msg = partial(_get_message, constr, param_names)

        if typ == "covariance":
            cov = cov_params_to_matrix(subset)
            e, _ = np.linalg.eigh(cov)
            if not np.all(e > -1e-8):
                report.append(_msg())
        elif typ == "sdcorr":
            cov = sdcorr_params_to_matrix(subset)
            e, _ = np.linalg.eigh(cov)
            if not np.all(e > -1e-8):
                report.append(_msg())
        elif typ == "probability":
            if not np.isclose(subset.sum(), 1, rtol=0.01):
                explanation = "Probabilities do not sum to 1."
                report.append(_msg(explanation))
            if np.any(subset < 0):
                explanation = "There are negative Probabilities."
                report.append(_msg(explanation))
            if np.any(subset > 1):
                explanation = "There are probabilities larger than 1."
                report.append(_msg(explanation))
        elif typ == "fixed":
            if "value" in constr and not np.allclose(subset, constr["value"]):
                explanation = (
                    "Fixing parameters to different values than their start values "
                    "was allowed in earlier versions of optimagic but is "
                    "forbidden now. "
                )
                report.append(_msg(explanation))
        elif typ == "increasing":
            if np.any(np.diff(subset) < 0):
                report.append(_msg())
        elif typ == "decreasing":
            if np.any(np.diff(subset) > 0):
                report.append(_msg())
        elif typ == "linear":
            wsum = subset.dot(constr["weights"])
            if "lower_bound" in constr and wsum < constr["lower_bound"]:
                explanation = "Lower bound of linear constraint is violated."
                report.append(_msg(explanation))
            elif "upper_bound" in constr and wsum > constr["upper_bound"]:
                explanation = "Upper bound of linear constraint violated"
                report.append(_msg(explanation))
            elif "value" in constr and not np.isclose(wsum, constr["value"]):
                explanation = "Equality condition of linear constraint violated"
                report.append(_msg(explanation))
        elif typ == "equality":
            if len(set(subset.tolist())) > 1:
                report.append(_msg())

        report = "\n".join(report)
        if report != "":
            raise InvalidParamsError(f"Violated constraint at start params:\n{report}")


def _get_message(constraint, param_names, explanation=""):
    start = (
        f"A constraint of type '{constraint['type']}' is not fulfilled in params, "
        "please make sure that it holds for the starting values. The problem arose "
        "because:"
    )

    if explanation:
        explanation = f" {explanation.rstrip('. ')}. "

    names = [param_names[i] for i in constraint["index"]]

    end = (
        f"The names of the involved parameters are:\n{names}\n"
        "The relevant constraint is:\n"
        f"{constraint}."
    )

    msg = start + explanation + end
    return msg


def check_types(constraints):
    """Check that no invalid constraint types are requested.

    Args:
        constraints (list): List of constraints.

    Raises:
        TypeError if invalid constraint types are encountered

    """
    valid_types = {
        "covariance",
        "sdcorr",
        "linear",
        "probability",
        "increasing",
        "decreasing",
        "equality",
        "pairwise_equality",
        "fixed",
    }
    for constr in constraints:
        if constr["type"] not in valid_types:
            raise InvalidConstraintError(
                "Invalid constraint_type: {}".format(constr["type"]),
            )


def check_for_incompatible_overlaps(transformations, parnames):
    """Check that there are no overlaps between constraints that transform parameters.

    Since the constraints are already consolidated such that only those that transform
    a parameter are left and all equality constraints are already plugged in, this
    boils down to checking that no parameter appears more than once.

    Args:
        constr_info (dict): Dict of 1d numpy arrays with info about constraints.
        transformations (list): Processed transforming constraints.
        parnames (list): List of parameter names.

    """
    all_indices = []
    for constr in transformations:
        all_indices += constr["index"]

    msg = (
        "Transforming constraints such as 'covariance', 'sdcorr', 'probability' "
        "and 'linear' cannot overlap. This includes overlaps induced by equality "
        "constraints. This was violated for the following parameters:\n{}"
    )

    if len(set(all_indices)) < len(all_indices):
        unique, counts = np.unique(all_indices, return_counts=True)
        invalid_indices = unique[counts >= 2]
        invalid_names = [parnames[i] for i in invalid_indices]

        raise InvalidConstraintError(msg.format(invalid_names))


def check_fixes_and_bounds(constr_info, transformations, parnames):
    """Check fixes.

    Warn the user if he fixes a parameter to a value even though that parameter has
    a different non-nan value in params

    check that fixes are compatible with other constraints.

    Args:
        constr_info (dict): Dict of 1d numpy arrays with info about constraints.
        transformations (list): Processed transforming constraints.
        parnames (list): List of parameter names.

    """
    constr_info = constr_info.copy()
    constr_info["index"] = parnames

    prob_msg = (
        "{} constraints are incompatible with fixes or bounds. "
        "This is violated for:\n{}"
    )

    cov_msg = (
        "{} constraints are incompatible with fixes or bounds except for the first "
        "parameter. This is violated for:\n{}"
    )

    for constr in transformations:
        if constr["type"] in ["covariance", "sdcorr"]:
            subset = _iloc(dictionary=constr_info, positions=constr["index"][1:])
            if subset["is_fixed_to_value"].any():
                problematic = subset["index"][subset["is_fixed_to_value"]]
                raise InvalidConstraintError(
                    cov_msg.format(constr["type"], problematic)
                )
            finite_bounds = np.isfinite(subset["lower_bounds"]) | np.isfinite(
                subset["upper_bounds"]
            )
            if finite_bounds.any():
                problematic = subset["index"][finite_bounds]
                raise InvalidConstraintError(
                    prob_msg.format(constr["type"], problematic)
                )
        elif constr["type"] == "probability":
            subset = _iloc(dictionary=constr_info, positions=constr["index"])
            if subset["is_fixed_to_value"].any():
                problematic = subset["index"][subset["is_fixed_to_value"]]
                raise InvalidConstraintError(
                    prob_msg.format(constr["type"], problematic)
                )
            finite_bounds = np.isfinite(subset["lower_bounds"]) | np.isfinite(
                subset["upper_bounds"]
            )
            if finite_bounds.any():
                problematic = subset["index"][finite_bounds]
                raise InvalidConstraintError(
                    prob_msg.format(constr["type"], problematic)
                )

    is_invalid = constr_info["lower_bounds"] >= constr_info["upper_bounds"]
    if is_invalid.any():
        info = pd.DataFrame(
            {
                "names": parnames[is_invalid],
                "lower_bounds": constr_info["lower_bounds"][is_invalid],
                "upper_bounds": constr_info["upper_bounds"][is_invalid],
            }
        )

        msg = (
            "lower_bound must be strictly smaller than upper_bound. "
            f"This is violated for:\n{info}"
        )

        raise InvalidConstraintError(msg)


def _iloc(dictionary, positions):
    """Substitute function for DataFrame.iloc. that works for a dictionary of arrays.

    It creates a subset of the input dictionary based on the
    index values in the info list, and returns this subset as
    a dictionary with numpy arrays.

    Args:
        dictionary (dict): Dictionary of arrays.
        position (list): List, slice or array of indices.

    """
    subset = {}
    for key, value in dictionary.items():
        if isinstance(value, list) and not isinstance(positions, slice):
            subset[key] = [value[i] for i in positions]
        else:
            subset[key] = value[positions]

    return subset
