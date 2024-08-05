"""Test optimizations with logging in a temporary database.

- Only minimize
- Only dict criterion
- scipy_lbfgsb and scipy_ls_dogbox
- with and without derivatives

"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.examples.criterion_functions import (
    sos_dict_criterion,
    sos_dict_derivative,
)
from optimagic.exceptions import TableExistsError
from optimagic.optimization.optimize import minimize
from optimagic.parameters.tree_registry import get_registry
from pybaum import tree_just_flatten


def flexible_sos_ls(params):
    return {"root_contributions": params}


algorithms = ["scipy_lbfgsb", "scipy_ls_dogbox"]
derivatives = [None, sos_dict_derivative]
params = [pd.DataFrame({"value": np.arange(3)}), np.arange(3), {"a": 1, "b": 2, "c": 3}]

test_cases = []
for algo in algorithms:
    for p in params:
        test_cases.append((algo, p))


@pytest.mark.parametrize("algorithm, params", test_cases)
def test_optimization_with_valid_logging(algorithm, params):
    res = minimize(
        flexible_sos_ls,
        params=params,
        algorithm=algorithm,
        logging="logging.db",
    )
    registry = get_registry(extended=True)
    flat = np.array(tree_just_flatten(res.params, registry=registry))
    aaae(flat, np.zeros(3))


def test_optimization_with_existing_exsting_database():
    minimize(
        sos_dict_criterion,
        pd.Series([1, 2, 3], name="value").to_frame(),
        algorithm="scipy_lbfgsb",
        logging="logging.db",
        log_options={"if_database_exists": "raise"},
    )

    with pytest.raises(FileExistsError):
        minimize(
            sos_dict_criterion,
            pd.Series([1, 2, 3], name="value").to_frame(),
            algorithm="scipy_lbfgsb",
            logging="logging.db",
            log_options={"if_database_exists": "raise"},
        )


def test_optimization_with_existing_exsting_table():
    minimize(
        sos_dict_criterion,
        pd.Series([1, 2, 3], name="value").to_frame(),
        algorithm="scipy_lbfgsb",
        logging="logging.db",
        log_options={"if_database_exists": "raise"},
    )

    with pytest.raises(TableExistsError):
        minimize(
            sos_dict_criterion,
            pd.Series([1, 2, 3], name="value").to_frame(),
            algorithm="scipy_lbfgsb",
            logging="logging.db",
            log_options={"if_table_exists": "raise"},
        )
