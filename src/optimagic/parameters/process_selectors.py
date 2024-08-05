import warnings
from collections import Counter

import numpy as np
import pandas as pd
from pybaum import tree_just_flatten

from optimagic.exceptions import InvalidConstraintError
from optimagic.parameters.tree_registry import get_registry


def process_selectors(constraints, params, tree_converter, param_names):
    """Process and harmonize the selector fields of constraints.

    By selector fields we mean loc, locs, query, queries, selector and selectors
    entries in constraints.

    The processed selector fields are called "index" and are integer numpy arrays with
    positions of parameters in a flattened parameter vector.

    Args:
        constraints (list): User provided constraints.
        params (pytree): User provided params.
        tree_converter (TreeConverter): NamedTuple with methods to convert between
            flattend and unflattend parameters.
        param_names (list): Names of flattened parameters. Used for error messages.

    Returns:
        list: List of constraints with additional "index" entry.

    """
    # fast path
    if constraints in (None, []):
        return []

    if isinstance(constraints, dict):
        constraints = [constraints]

    registry = get_registry(extended=True)
    n_params = len(tree_converter.params_flatten(params))
    helper = tree_converter.params_unflatten(np.arange(n_params))
    params_case = _get_params_case(params)
    flat_constraints = []
    for constr in constraints:
        selector_case = _get_selector_case(constr)
        field = _get_selection_field(
            constraint=constr,
            selector_case=selector_case,
            params_case=params_case,
        )
        evaluator = _get_selection_evaluator(
            field=field,
            constraint=constr,
            params_case=params_case,
            registry=registry,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
                selected = evaluator(helper)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            msg = (
                "An error occurred when trying to select parameters for the following "
                f"constraint:\n{constr}"
            )
            raise InvalidConstraintError(msg) from e

        if selector_case == "one selector":
            if np.isscalar(selected):
                selected = [selected]
            _fail_if_duplicates(selected, constr, param_names)
            selected = np.array(selected).astype(int)
        else:
            selected = [[sel] if np.isscalar(sel) else sel for sel in selected]
            _fail_if_selections_are_incompatible(selected, constr)
            for sel in selected:
                _fail_if_duplicates(sel, constr, param_names)
            selected = [np.array(sel).astype(int) for sel in selected]

        new_constr = constr.copy()
        if selector_case == "one selector":
            new_constr["index"] = selected
        else:
            new_constr["indices"] = selected

        if selector_case == "one selector":
            if len(new_constr["index"]) > 0:
                flat_constraints.append(new_constr)
        else:
            if len(new_constr["indices"][0]) > 0:
                flat_constraints.append(new_constr)
    return flat_constraints


def _get_selection_field(constraint, selector_case, params_case):
    """Get the relevant selection field of a constraint."""
    selector_case = _get_selector_case(constraint)

    valid = {
        "multiple selectors": {
            "dataframe": {"locs", "queries", "selectors"},
            "numpy array": {"locs", "selectors"},
            "pytree": {"selectors"},
            "series": {"locs", "selectors"},
        },
        "one selector": {
            "dataframe": {"loc", "query", "selector"},
            "numpy array": {"loc", "selector"},
            "pytree": {"selector"},
            "series": {"loc", "selector"},
        },
    }

    valid = valid[selector_case][params_case]

    present = set(constraint).intersection(valid)

    if not present:
        msg = (
            "No valid parameter selection field in constraint. Valid selection fields "
            f"are {valid}. The constraint is:\n{constraint}"
        )
        raise InvalidConstraintError(msg)
    elif len(present) > 1:
        msg = (
            f"Too many parameter selection fields in constraint: {present}. "
            "Constraints must have exactly one parameter selection field. The "
            f"constraint was:\n{constraint}"
        )
        raise InvalidConstraintError(msg)

    field = list(present)[0]
    return field


def _get_selection_evaluator(field, constraint, params_case, registry):
    if field == "selector":

        def evaluator(params):
            raw = constraint["selector"](params)
            flat = tree_just_flatten(raw, registry=registry)
            return flat

    elif field == "selectors":

        def evaluator(params):
            raw = [sel(params) for sel in constraint["selectors"]]
            flat = [tree_just_flatten(r, registry=registry) for r in raw]
            return flat

    elif field == "loc":
        if params_case == "dataframe":

            def evaluator(params):
                return params.loc[constraint["loc"], "value"].tolist()

        else:

            def evaluator(params):
                return params[constraint["loc"]].tolist()

    elif field == "locs":
        if params_case == "dataframe":

            def evaluator(params):
                return [params.loc[lo, "value"].tolist() for lo in constraint["locs"]]

        else:

            def evaluator(params):
                return [params[lo].tolist() for lo in constraint["locs"]]

    elif field == "query":

        def evaluator(params):
            return params.query(constraint["query"])["value"].tolist()

    elif field == "queries":

        def evaluator(params):
            return [params.query(q)["value"].tolist() for q in constraint["queries"]]

    else:
        raise ValueError(f"Invalid parameter selection field: {field}")

    return evaluator


def _get_params_case(params):
    if isinstance(params, pd.DataFrame) and "value" in params:
        params_case = "dataframe"
    elif isinstance(params, pd.Series):
        params_case = "series"
    elif isinstance(params, np.ndarray):
        params_case = "numpy array"
    else:
        params_case = "pytree"
    return params_case


def _get_selector_case(constraint):
    if constraint["type"] == "pairwise_equality":
        selector_case = "multiple selectors"
    else:
        selector_case = "one selector"
    return selector_case


def _fail_if_duplicates(selected, constraint, param_names):
    duplicates = _find_duplicates(selected)
    if duplicates:
        names = [param_names[i] for i in duplicates]
        msg = (
            "Error while processing constraints. There are duplicates in selected "
            "parameters. The parameters that were selected more than once are "
            f"{names}. The problematic constraint is:\n{constraint}"
        )
        raise InvalidConstraintError(msg)


def _fail_if_selections_are_incompatible(selected, constraint):
    if len(selected) <= 1:
        msg = (
            "pairwise equality constraints require mutliple sets of selected "
            "parameters but there is just one in the following constraint:\n"
            f"{constraint}"
        )
        raise InvalidConstraintError(msg)
    lengths = [len(sel) for sel in selected]
    if len(set(lengths)) != 1:
        msg = (
            "All sets of selected parameters for pairwise equality constraints need "
            f"to have the same length. You have lengths {lengths} in constraint:\n"
            f"{constraint}"
        )
        raise InvalidConstraintError(msg)


def _find_duplicates(list_):
    return [item for item, count in Counter(list_).items() if count > 1]
