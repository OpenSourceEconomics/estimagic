import itertools

import numpy as np
import pandas as pd
import pytest
from estimagic.msm_covs import cov_optimal, cov_robust
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.utilities import get_rng
from pandas.testing import assert_frame_equal

rng = get_rng(seed=1234)

jac_np = rng.uniform(size=(10, 5))
jac_pd = pd.DataFrame(jac_np)

moments_cov_np = rng.uniform(size=(10, 10)) + np.eye(10) * 2.5
moments_cov_pd = pd.DataFrame(moments_cov_np)

test_cases = itertools.product([jac_np, jac_pd], [moments_cov_np, moments_cov_pd])


@pytest.mark.parametrize("jac, moments_cov", test_cases)
def test_cov_robust_and_cov_optimal_are_equivalent_in_special_case(jac, moments_cov):
    weights = np.linalg.inv(moments_cov)
    if isinstance(moments_cov, pd.DataFrame):
        weights = pd.DataFrame(
            weights, index=moments_cov.index, columns=moments_cov.columns
        )

    sandwich = cov_robust(jac, weights, moments_cov)
    optimal = cov_optimal(jac, weights)

    if isinstance(sandwich, pd.DataFrame):
        assert_frame_equal(sandwich, optimal)

    else:
        aaae(sandwich, optimal)
