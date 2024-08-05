from functools import partial

import numpy as np
from pybaum import tree_just_flatten

from optimagic.parameters.tree_registry import get_registry


def get_history_arrays(history, direction):
    parhist = history["params"]
    is_flat = (
        len(parhist) > 0 and isinstance(parhist[0], np.ndarray) and parhist[0].ndim == 1
    )
    if is_flat:
        to_internal = lambda x: x.tolist()
    else:
        registry = get_registry(extended=True)
        to_internal = partial(tree_just_flatten, registry=registry)

    critvals = np.array(history["criterion"])

    params = np.array([to_internal(p) for p in history["params"]])

    runtimes = np.array(history["runtime"])

    if direction == "minimize":
        monotone = np.minimum.accumulate(critvals)
        is_accepted = critvals <= monotone
    elif direction == "maximize":
        monotone = np.maximum.accumulate(critvals)
        is_accepted = critvals >= monotone

    out = {
        "criterion": critvals,
        "params": params,
        "runtimes": runtimes,
        "monotone_criterion": monotone,
        "is_accepted": is_accepted,
    }
    return out
