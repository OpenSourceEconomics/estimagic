"""Functions to convert between array and block-tree representations of a matrix."""

import numpy as np
import pandas as pd
from pybaum import tree_flatten, tree_unflatten
from pybaum import tree_just_flatten as tree_leaves

from optimagic.parameters.tree_registry import get_registry


def matrix_to_block_tree(matrix, outer_tree, inner_tree):
    """Convert a matrix (2-dimensional array) to block-tree.

    A block tree most often arises when one applies an operation to a function that maps
    between two trees. For certain functions this results in a 2-dimensional data array.
    Two main examples are the Jacobian of the function f : inner_tree -> outer_tree,
    which results in a block tree structure, or the covariance matrix of a tree, in
    which case outer_tree = inner_tree.

    Args:
        matrix (numpy.ndarray): 2d representation of the block tree. Has shape (m, n).
        outer_tree: A pytree. If flattened to scalars has length m.
        inner_tree: A pytree. If flattened to scalars has length n.

    Returns:
        block_tree: A (block) pytree.

    """
    _check_dimensions_matrix(matrix, outer_tree, inner_tree)

    flat_outer, treedef_outer = tree_flatten(outer_tree)
    flat_inner, treedef_inner = tree_flatten(inner_tree)

    flat_outer_np = [_convert_to_numpy(leaf, only_pandas=True) for leaf in flat_outer]
    flat_inner_np = [_convert_to_numpy(leaf, only_pandas=True) for leaf in flat_inner]

    shapes_outer = [np.shape(a) for a in flat_outer_np]
    shapes_inner = [np.shape(a) for a in flat_inner_np]

    block_bounds_outer = np.cumsum([int(np.prod(s)) for s in shapes_outer[:-1]])
    block_bounds_inner = np.cumsum([int(np.prod(s)) for s in shapes_inner[:-1]])

    blocks = []
    for leaf_outer, s1, submat in zip(
        flat_outer,
        shapes_outer,
        np.split(matrix, block_bounds_outer, axis=0),
        strict=False,
    ):
        row = []
        for leaf_inner, s2, block_values in zip(
            flat_inner,
            shapes_inner,
            np.split(submat, block_bounds_inner, axis=1),
            strict=False,
        ):
            raw_block = block_values.reshape((*s1, *s2))
            block = _convert_raw_block_to_pandas(raw_block, leaf_outer, leaf_inner)
            row.append(block)

        blocks.append(row)

    block_tree = tree_unflatten(
        treedef_outer, [tree_unflatten(treedef_inner, row) for row in blocks]
    )

    return block_tree


def hessian_to_block_tree(hessian, f_tree, params_tree):
    """Convert a Hessian array to block-tree format.

    Remark: In comparison to Jax we need this formatting function because we calculate
    the second derivative using second-order finite differences. Jax computes the
    second derivative by applying their jacobian function twice, which produces the
    desired block-tree shape of the Hessian automatically. If we apply our first
    derivative function twice we get the same block-tree shape.

    Args:
        hessian (np.ndarray): The Hessian, 2- or 3-dimensional array representation of
            the resulting block-tree.
        f_tree (pytree): The function evaluated at params_tree.
        params_tree (pytree): The params_tree.

    Returns:
        hessian_block_tree (pytree): The pytree

    """
    _check_dimensions_hessian(hessian, f_tree, params_tree)

    if hessian.ndim == 2:
        hessian = hessian[np.newaxis]

    flat_f, treedef_f = tree_flatten(f_tree)
    flat_p, treedef_p = tree_flatten(params_tree)

    flat_f_np = [_convert_to_numpy(leaf, only_pandas=True) for leaf in flat_f]
    flat_p_np = [_convert_to_numpy(leaf, only_pandas=True) for leaf in flat_p]

    shapes_f = [np.shape(a) for a in flat_f_np]
    shapes_p = [np.shape(a) for a in flat_p_np]

    block_bounds_f = np.cumsum([int(np.prod(s)) for s in shapes_f[:-1]])
    block_bounds_p = np.cumsum([int(np.prod(s)) for s in shapes_p[:-1]])

    sub_block_trees = []
    for s0, subarr in zip(
        shapes_f, np.split(hessian, block_bounds_f, axis=0), strict=False
    ):
        blocks = []
        for leaf_outer, s1, submat in zip(
            flat_p, shapes_p, np.split(subarr, block_bounds_p, axis=1), strict=False
        ):
            row = []
            for leaf_inner, s2, block_values in zip(
                flat_p, shapes_p, np.split(submat, block_bounds_p, axis=2), strict=False
            ):
                _shape = [k for k in (*s0, *s1, *s2) if k != 1]
                raw_block = block_values.reshape(_shape)
                block = _convert_raw_block_to_pandas(raw_block, leaf_outer, leaf_inner)
                row.append(block)
            blocks.append(row)
        block_tree = tree_unflatten(
            treedef_p, [tree_unflatten(treedef_p, row) for row in blocks]
        )
        sub_block_trees.append(block_tree)

    hessian_block_tree = tree_unflatten(treedef_f, sub_block_trees)
    return hessian_block_tree


def block_tree_to_matrix(block_tree, outer_tree, inner_tree):
    """Convert a block tree to a matrix.

    A block tree most often arises when one applies an operation to a function that maps
    between two trees. Two main examples are the Jacobian of the function f : inner_tree
    -> outer_tree, which results in a block tree structure, or the covariance matrix of
    a tree, in which case outer_tree = inner_tree.

    Args:
        block_tree: A (block) pytree, must match dimensions of outer_tree and inner_tree
        outer_tree: A pytree.
        inner_tree: A pytree.

    Returns:
        matrix (np.ndarray): 2d array containing information stored in block_tree.

    """
    flat_outer = tree_leaves(outer_tree)
    flat_inner = tree_leaves(inner_tree)
    flat_block_tree = tree_leaves(block_tree)

    flat_outer_np = [_convert_to_numpy(leaf, only_pandas=True) for leaf in flat_outer]
    flat_inner_np = [_convert_to_numpy(leaf, only_pandas=True) for leaf in flat_inner]

    size_outer = [np.size(a) for a in flat_outer_np]
    size_inner = [np.size(a) for a in flat_inner_np]

    n_blocks_outer = len(size_outer)
    n_blocks_inner = len(size_inner)

    block_rows_raw = [
        flat_block_tree[n_blocks_inner * i : n_blocks_inner * (i + 1)]
        for i in range(n_blocks_outer)
    ]

    block_rows = []
    for s1, row in zip(size_outer, block_rows_raw, strict=False):
        shapes = [(s1, s2) for s2 in size_inner]
        row_np = [_convert_to_numpy(leaf, only_pandas=False) for leaf in row]
        row_reshaped = _reshape_list(row_np, shapes)
        row_concatenated = np.concatenate(row_reshaped, axis=1)
        block_rows.append(row_concatenated)

    matrix = np.concatenate(block_rows, axis=0)

    _check_dimensions_matrix(matrix, flat_outer, flat_inner)
    return matrix


def block_tree_to_hessian(block_hessian, f_tree, params_tree):
    """Convert a block tree to a Hessian array.

    Remark: In comparison to Jax we need this formatting function because we calculate
    the second derivative using second-order finite differences. Jax computes the
    second derivative by applying their jacobian function twice, which produces the
    desired block-tree shape of the Hessian automatically. If we apply our first
    derivative function twice we get the same block-tree shape.

    Args:
        block_hessian: A (block) pytree, must match dimensions of f_tree and params_tree
        f_tree (pytree): The function evaluated at params_tree.
        params_tree (pytree): The params_tree.

    Returns:
        matrix (np.ndarray): 2d array containing information stored in block_tree.

    """
    flat_f = tree_leaves(f_tree)
    flat_p = tree_leaves(params_tree)
    flat_block_tree = tree_leaves(block_hessian)

    flat_f_np = [_convert_to_numpy(leaf, only_pandas=True) for leaf in flat_f]
    flat_p_np = [_convert_to_numpy(leaf, only_pandas=True) for leaf in flat_p]

    size_f = [np.size(a) for a in flat_f_np]
    size_p = [np.size(a) for a in flat_p_np]

    n_blocks_f = len(size_f)
    n_blocks_p = len(size_p)

    outer_blocks = [
        flat_block_tree[(n_blocks_p**2) * i : (n_blocks_p**2) * (i + 1)]
        for i in range(n_blocks_f)
    ]

    inner_matrices = []
    for outer_block_dim, list_inner_blocks in zip(size_f, outer_blocks, strict=False):
        block_rows_raw = [
            list_inner_blocks[n_blocks_p * i : n_blocks_p * (i + 1)]
            for i in range(n_blocks_p)
        ]
        block_rows = []
        for s1, row in zip(size_p, block_rows_raw, strict=False):
            shapes = [(outer_block_dim, s1, s2) for s2 in size_p]
            row_np = [_convert_to_numpy(leaf, only_pandas=False) for leaf in row]
            row_np_3d = [leaf[np.newaxis] if leaf.ndim < 3 else leaf for leaf in row_np]
            row_reshaped = _reshape_list(row_np_3d, shapes)
            row_concatenated = np.concatenate(row_reshaped, axis=2)
            block_rows.append(row_concatenated)

        inner_matrix = np.concatenate(block_rows, axis=1)
        inner_matrices.append(inner_matrix)

    hessian = np.concatenate(inner_matrices, axis=0)
    _check_dimensions_hessian(hessian, f_tree, params_tree)
    return hessian


def _convert_to_numpy(obj, only_pandas=True):
    if only_pandas:
        out = _convert_pandas_objects_to_numpy(obj)
    else:
        out = np.asarray(obj)
    return out


def _convert_pandas_objects_to_numpy(obj):
    if not isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj
    elif isinstance(obj, pd.Series):
        out = obj.to_numpy()
    elif "value" in obj.columns:
        out = obj["value"].to_numpy()
    else:
        out = obj.to_numpy()
    return out


def _convert_raw_block_to_pandas(raw_block, leaf_outer, leaf_inner):
    if np.ndim(raw_block) not in (1, 2):
        return raw_block

    if not _is_pd_object(leaf_outer) and not _is_pd_object(leaf_inner):
        return raw_block

    index1 = None if not _is_pd_object(leaf_outer) else leaf_outer.index
    index2 = None if not _is_pd_object(leaf_inner) else leaf_inner.index

    # can only happen if one leaf is a scalar and the other a pandas
    # object that is interpreted as one-dimensional. We want to convert
    # the block to a series wtih the index of the pandas object
    if np.ndim(raw_block) == 1:
        out = pd.Series(raw_block, index=_select_non_none(index1, index2))

    # can happen in two cases
    elif np.ndim(raw_block) == 2:
        # case 1: one leaf is scalar and the other is a DataFrame
        # without value column. We want to convert the block to a DataFrame
        # with same index and columns as original DataFrame
        if np.isscalar(leaf_outer) or np.isscalar(leaf_inner):
            if np.isscalar(leaf_outer):
                index, columns = leaf_inner.index, leaf_inner.columns
            elif np.isscalar(leaf_inner):
                index, columns = leaf_outer.index, leaf_outer.columns
            out = pd.DataFrame(raw_block, index=index, columns=columns)
        # case 2: both 1d Data structures and at least one of them is
        # a pandas object. We want to convert the result to a DataFrame
        # with index=index1 and columns=index2
        else:
            out = pd.DataFrame(raw_block, index=index1, columns=index2)

    return out


def _select_non_none(first, second):
    if first is None and second is None:
        raise ValueError()
    elif first is not None and second is not None:
        raise ValueError()
    elif first is None:
        out = second
    elif second is None:
        out = first
    return out


def _reshape_list(list_to_reshape, shapes):
    """Reshape list of numpy.ndarray according to list of shapes.

    Args:
        list_to_reshape (list): List containing numpy.ndarray's.
        shapes (list): List of shape tuples.

    Returns:
        reshaped (list): List containing the reshaped numpy.ndarray's.

    """
    if len(list_to_reshape) != len(shapes):
        raise ValueError("Arguments must have the same number of elements.")
    reshaped = [
        a.reshape(shape) for a, shape in zip(list_to_reshape, shapes, strict=False)
    ]
    return reshaped


def _is_pd_object(obj):
    return isinstance(obj, (pd.Series, pd.DataFrame))


def _check_dimensions_matrix(matrix, outer_tree, inner_tree):
    extended_registry = get_registry(extended=True)
    flat_outer = tree_leaves(outer_tree, registry=extended_registry)
    flat_inner = tree_leaves(inner_tree, registry=extended_registry)

    if matrix.shape[0] != len(flat_outer):
        raise ValueError("First dimension of matrix does not match that of outer_tree.")
    if matrix.shape[1] != len(flat_inner):
        raise ValueError(
            "Second dimension of matrix does not match that of inner_tree."
        )


def _check_dimensions_hessian(hessian, f_tree, params_tree):
    extended_registry = get_registry(extended=True)
    flat_f = tree_leaves(f_tree, registry=extended_registry)
    flat_p = tree_leaves(params_tree, registry=extended_registry)

    if len(flat_f) == 1:
        # consider only dimensions with non trivial size (larger than 1)
        relevant_hessian_shape = tuple(k for k in hessian.shape if k != 1)

        if len(relevant_hessian_shape) == 0 and len(flat_p) != 1:
            # scalar f and scalar params -> scalar hessian
            raise ValueError("Hessian dimension does not match those of params.")

        if len(relevant_hessian_shape) == 2:
            # scalar f and vector params -> matrix hessian
            if relevant_hessian_shape != (len(flat_p), len(flat_p)):
                raise ValueError("Hessian dimension does not match those of params.")

        if len(relevant_hessian_shape) > 2:
            raise ValueError("Hessian must be 0- or 2-d if f is scalar-valued.")
    else:
        if hessian.ndim != 3:
            raise ValueError("Hessian must be 3d if f is multidimensional.")
        if hessian.shape[0] != len(flat_f):
            raise ValueError("First Hessian dimension does not match that of f.")
        if hessian.shape[1:] != (len(flat_p), len(flat_p)):
            raise ValueError(
                "Last two Hessian dimensions do not match those of params."
            )
