"""Test the pc processing."""

import numpy as np
import pandas as pd
import pytest
from optimagic.exceptions import InvalidConstraintError
from optimagic.parameters.bounds import Bounds
from optimagic.parameters.constraint_tools import check_constraints
from optimagic.parameters.process_constraints import (
    _replace_pairwise_equality_by_equality,
)


def test_replace_pairwise_equality_by_equality():
    constr = {"indices": [[0, 1], [2, 3]], "type": "pairwise_equality"}

    expected = [
        {"index": [0, 2], "type": "equality"},
        {"index": [1, 3], "type": "equality"},
    ]

    calculated = _replace_pairwise_equality_by_equality([constr])

    assert calculated == expected


def test_empty_constraints_work():
    params = pd.DataFrame()
    params["value"] = np.arange(5)
    params["bla"] = list("abcde")

    constraints = [{"query": "bla == 'blubb'", "type": "equality"}]

    check_constraints(params, constraints)


def test_to_many_bounds_in_increasing_constraint_raise_good_error():
    with pytest.raises(InvalidConstraintError):
        check_constraints(
            params=np.arange(3),
            bounds=Bounds(lower=np.arange(3) - 1),
            constraints={"loc": [0, 1, 2], "type": "increasing"},
        )
