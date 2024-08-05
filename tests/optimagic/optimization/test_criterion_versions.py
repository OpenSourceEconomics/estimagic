"""Test different versions of specifying a criterion functions.

Here we want to take:
- Few representative algorithms (derivative based, derivative free, least squares)
- One basic criterion function (sum of squares)
- Many criterion versions (dict output, pandas output, scalar output)

"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.config import IS_DFOLS_INSTALLED
from optimagic.examples.criterion_functions import (
    sos_dict_criterion,
    sos_dict_criterion_with_pd_objects,
    sos_scalar_criterion,
)
from optimagic.exceptions import InvalidFunctionError
from optimagic.optimization.optimize import minimize

algorithms = ["scipy_lbfgsb", "scipy_neldermead"]
if IS_DFOLS_INSTALLED:
    algorithms.append("nag_dfols")

ls_algorithms = {"nag_dfols"}


criterion_functions = {
    "sos_dict_criterion": sos_dict_criterion,
    "sos_scalar_criterion": sos_scalar_criterion,
    "sos_dict_criterion_with_pd_objects": sos_dict_criterion_with_pd_objects,
}


valid_cases = []
invalid_cases = []
for algo in algorithms:
    for name, crit in criterion_functions.items():
        if algo in ls_algorithms and "dict" not in name:
            invalid_cases.append((crit, algo))
        else:
            valid_cases.append((crit, algo))


@pytest.mark.parametrize("criterion, algorithm", valid_cases)
def test_valid_criterion_versions(criterion, algorithm):
    start_params = pd.DataFrame()
    start_params["value"] = [1, 2, 3]
    res = minimize(
        fun=criterion,
        params=start_params,
        algorithm=algorithm,
    )

    aaae(res.params["value"].to_numpy(), np.zeros(3), decimal=4)


@pytest.mark.parametrize("criterion, algorithm", invalid_cases)
def test_invalid_criterion_versions(criterion, algorithm):
    start_params = pd.DataFrame()
    start_params["value"] = [1, 2, 3]

    with pytest.raises(InvalidFunctionError):
        minimize(
            fun=criterion,
            params=start_params,
            algorithm=algorithm,
        )
