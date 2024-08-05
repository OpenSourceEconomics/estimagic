from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from pybaum import leaf_names, tree_map
from pybaum import tree_just_flatten as tree_leaves
from scipy.optimize import Bounds as ScipyBounds

from optimagic.exceptions import InvalidBoundsError
from optimagic.parameters.tree_registry import get_registry
from optimagic.typing import PyTree, PyTreeRegistry


@dataclass(frozen=True)
class Bounds:
    lower: PyTree | None = None
    upper: PyTree | None = None
    soft_lower: PyTree | None = None
    soft_upper: PyTree | None = None


def pre_process_bounds(
    bounds: None | Bounds | ScipyBounds | Sequence[tuple[float, float]],
) -> Bounds | None:
    """Convert all valid types of specifying bounds to optimagic.Bounds.

    This just harmonizes multiple ways of specifying bounds into a single format.
    It does not check that bounds are valid or compatible with params.

    Args:
        bounds: The user provided bounds.

    Returns:
        The bounds in the optimagic format.

    Raises:
        InvalidBoundsError: If bounds cannot be processed, e.g. because they do not have
            the correct type.

    """
    if isinstance(bounds, ScipyBounds):
        bounds = Bounds(lower=bounds.lb, upper=bounds.ub)
    elif isinstance(bounds, Bounds) or bounds is None:
        pass
    else:
        try:
            bounds = _process_bounds_sequence(bounds)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            raise InvalidBoundsError(
                f"Invalid bounds of type: {type(bounds)}. Bounds must be "
                "optimagic.Bounds, scipy.optimize.Bounds or a Sequence of tuples with "
                "lower and upper bounds."
            ) from e
    return bounds


def _process_bounds_sequence(bounds: Sequence[tuple[float, float]]) -> Bounds:
    lower = np.full(len(bounds), -np.inf)
    upper = np.full(len(bounds), np.inf)

    for i, (lb, ub) in enumerate(bounds):
        if lb is not None:
            lower[i] = lb
        if ub is not None:
            upper[i] = ub
    return Bounds(lower=lower, upper=upper)


def get_internal_bounds(
    params: PyTree,
    bounds: Bounds | None = None,
    registry: PyTreeRegistry | None = None,
    add_soft_bounds: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create consolidated and flattened bounds for params.

    If params is a DataFrame with value column, the user provided bounds are
    extended with bounds from the params DataFrame.

    If no bounds are available the entry is set to minus np.inf for the lower bound and
    np.inf for the upper bound.

    The bounds provided in `bounds` override bounds provided in params if both are
    specified (in the case where params is a DataFrame with bounds as a column).

    Args:
        params: The parameter pytree.
        bounds: The lower and upper bounds.
        registry: pybaum registry.
        add_soft_bounds: If True, the element-wise maximum (minimum) of the lower and
            soft_lower (upper and soft_upper) bounds are taken. If False, the lower
            (upper) bounds are returned.

    Returns:
        Consolidated and flattened lower_bounds.
        Consolidated and flattened upper_bounds.

    """
    bounds = Bounds() if bounds is None else bounds

    fast_path = _is_fast_path(
        params=params,
        bounds=bounds,
        add_soft_bounds=add_soft_bounds,
    )
    if fast_path:
        return _get_fast_path_bounds(
            params=params,
            bounds=bounds,
        )

    registry = get_registry(extended=True) if registry is None else registry
    n_params = len(tree_leaves(params, registry=registry))

    # Fill leaves with np.nan. If params contains a data frame with bounds as a column,
    # that column is NOT overwritten (as long as an extended registry is used).
    nan_tree = tree_map(lambda leaf: np.nan, params, registry=registry)  # noqa: ARG005

    lower_flat = _update_bounds_and_flatten(nan_tree, bounds.lower, kind="lower_bound")
    upper_flat = _update_bounds_and_flatten(nan_tree, bounds.upper, kind="upper_bound")

    if len(lower_flat) != n_params:
        raise InvalidBoundsError("lower_bounds do not match dimension of params.")
    if len(upper_flat) != n_params:
        raise InvalidBoundsError("upper_bounds do not match dimension of params.")

    lower_flat[np.isnan(lower_flat)] = -np.inf
    upper_flat[np.isnan(upper_flat)] = np.inf

    if add_soft_bounds:
        lower_flat_soft = _update_bounds_and_flatten(
            nan_tree, bounds.soft_lower, kind="soft_lower_bound"
        )
        lower_flat_soft[np.isnan(lower_flat_soft)] = -np.inf
        lower_flat = np.maximum(lower_flat, lower_flat_soft)

        upper_flat_soft = _update_bounds_and_flatten(
            nan_tree, bounds.soft_upper, kind="soft_upper_bound"
        )
        upper_flat_soft[np.isnan(upper_flat_soft)] = np.inf
        upper_flat = np.minimum(upper_flat, upper_flat_soft)

    if (lower_flat > upper_flat).any():
        msg = "Invalid bounds. Some lower bounds are larger than upper bounds."
        raise InvalidBoundsError(msg)

    return lower_flat, upper_flat


def _update_bounds_and_flatten(
    nan_tree: PyTree, bounds: PyTree, kind: str
) -> NDArray[np.float64]:
    """Flatten bounds array and update it with bounds from params.

    Args:
        nan_tree: Pytree with the same structure as params, filled with nans.
        bounds: The candidate bounds to be updated and flattened.
        kind: One of "lower_bound", "upper_bound", "soft_lower_bound",
            "soft_upper_bound".

    Returns:
        np.ndarray: The updated and flattened bounds.

    """
    registry = get_registry(extended=True, data_col=kind)
    flat_nan_tree = tree_leaves(nan_tree, registry=registry)

    if bounds is not None:
        registry = get_registry(extended=True)
        flat_bounds = tree_leaves(bounds, registry=registry)

        seperator = 10 * "$"
        params_names = leaf_names(nan_tree, registry=registry, separator=seperator)
        bounds_names = leaf_names(bounds, registry=registry, separator=seperator)

        flat_nan_dict = dict(zip(params_names, flat_nan_tree, strict=False))

        invalid = {"names": [], "bounds": []}  # type: ignore
        for bounds_name, bounds_leaf in zip(bounds_names, flat_bounds, strict=False):
            # if a bounds leaf is None we treat it as saying the the corresponding
            # subtree of params has no bounds.
            if bounds_leaf is not None:
                if bounds_name in flat_nan_dict:
                    flat_nan_dict[bounds_name] = bounds_leaf
                else:
                    invalid["names"].append(bounds_name)
                    invalid["bounds"].append(bounds_leaf)

        if invalid["bounds"]:
            msg = (
                f"{kind} could not be matched to params pytree. The bounds "
                f"{invalid['bounds']} with names {invalid['names']} are not part of "
                "params."
            )
            raise InvalidBoundsError(msg)

        flat_nan_tree = list(flat_nan_dict.values())

    updated = np.array(flat_nan_tree, dtype=np.float64)
    return updated


def _is_fast_path(params: PyTree, bounds: Bounds, add_soft_bounds: bool) -> bool:
    out = True
    if add_soft_bounds:
        out = False

    if not _is_1d_array(params):
        out = False

    for bound in bounds.lower, bounds.upper:
        if not (_is_1d_array(bound) or bound is None):
            out = False
    return out


def _is_1d_array(candidate: Any) -> bool:
    return isinstance(candidate, np.ndarray) and candidate.ndim == 1


def _get_fast_path_bounds(
    params: PyTree, bounds: Bounds
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if bounds.lower is None:
        # faster than np.full
        lower_bounds = np.array([-np.inf] * len(params))
    else:
        lower_bounds = bounds.lower.astype(float)

    if bounds.upper is None:
        # faster than np.full
        upper_bounds = np.array([np.inf] * len(params))
    else:
        upper_bounds = bounds.upper.astype(float)

    if (lower_bounds > upper_bounds).any():
        msg = "Invalid bounds. Some lower bounds are larger than upper bounds."
        raise InvalidBoundsError(msg)

    return lower_bounds, upper_bounds
