import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from estimagic.optimization.utilities import chol_params_to_lower_triangular_matrix
from estimagic.optimization.utilities import cov_matrix_to_params
from estimagic.optimization.utilities import cov_matrix_to_sdcorr_params
from estimagic.optimization.utilities import cov_params_to_matrix
from estimagic.optimization.utilities import cov_to_sds_and_corr
from estimagic.optimization.utilities import dimension_to_number_of_triangular_elements
from estimagic.optimization.utilities import number_of_triangular_elements_to_dimension
from estimagic.optimization.utilities import robust_cholesky
from estimagic.optimization.utilities import sdcorr_params_to_matrix
from estimagic.optimization.utilities import sdcorr_params_to_sds_and_corr
from estimagic.optimization.utilities import sds_and_corr_to_cov


def test_chol_params_to_lower_triangular_matrix():
    calculated = chol_params_to_lower_triangular_matrix(pd.Series([1, 2, 3]))
    expected = np.array([[1, 0], [2, 3]])
    aaae(calculated, expected)


def test_cov_params_to_matrix():
    params = np.array([1, 0.1, 2, 0.2, 0.22, 3])
    expected = np.array([[1, 0.1, 0.2], [0.1, 2, 0.22], [0.2, 0.22, 3]])
    calculated = cov_params_to_matrix(params)
    aaae(calculated, expected)


def test_cov_matrix_to_params():
    expected = np.array([1, 0.1, 2, 0.2, 0.22, 3])
    cov = np.array([[1, 0.1, 0.2], [0.1, 2, 0.22], [0.2, 0.22, 3]])
    calculated = cov_matrix_to_params(cov)
    aaae(calculated, expected)


def test_sdcorr_params_to_sds_and_corr():
    sdcorr_params = pd.Series([1, 2, 3, 0.1, 0.2, 0.3])
    exp_corr = np.array([[1, 0.1, 0.2], [0.1, 1, 0.3], [0.2, 0.3, 1]])
    exp_sds = np.array([1, 2, 3])
    calc_sds, calc_corr = sdcorr_params_to_sds_and_corr(sdcorr_params)
    aaae(calc_sds, exp_sds)
    aaae(calc_corr, exp_corr)


def test_sdcorr_params_to_matrix():
    sds = np.sqrt([1, 2, 3])
    corrs = [0.07071068, 0.11547005, 0.08981462]
    params = np.hstack([sds, corrs])
    expected = np.array([[1, 0.1, 0.2], [0.1, 2, 0.22], [0.2, 0.22, 3]])
    calculated = sdcorr_params_to_matrix(params)
    aaae(calculated, expected)


def test_cov_matrix_to_sdcorr_params():
    sds = np.sqrt([1, 2, 3])
    corrs = [0.07071068, 0.11547005, 0.08981462]
    expected = np.hstack([sds, corrs])
    cov = np.array([[1, 0.1, 0.2], [0.1, 2, 0.22], [0.2, 0.22, 3]])
    calculated = cov_matrix_to_sdcorr_params(cov)
    aaae(calculated, expected)


def test_sds_and_corr_to_cov():
    sds = [1, 2, 3]
    corr = np.ones((3, 3)) * 0.2
    corr[np.diag_indices(3)] = 1
    calculated = sds_and_corr_to_cov(sds, corr)
    expected = np.array([[1.0, 0.4, 0.6], [0.4, 4.0, 1.2], [0.6, 1.2, 9.0]])
    aaae(calculated, expected)


def test_cov_to_sds_and_corr():
    cov = np.array([[1.0, 0.4, 0.6], [0.4, 4.0, 1.2], [0.6, 1.2, 9.0]])
    calc_sds, calc_corr = cov_to_sds_and_corr(cov)
    exp_sds = [1, 2, 3]
    exp_corr = np.ones((3, 3)) * 0.2
    exp_corr[np.diag_indices(3)] = 1
    aaae(calc_sds, exp_sds)
    aaae(calc_corr, exp_corr)


def test_number_of_triangular_elements_to_dimension():
    inputs = [6, 10, 15, 21]
    expected = [3, 4, 5, 6]
    for inp, exp in zip(inputs, expected):
        assert number_of_triangular_elements_to_dimension(inp) == exp


def test_dimension_to_number_of_triangular_elements():
    inputs = [3, 4, 5, 6]
    expected = [6, 10, 15, 21]
    for inp, exp in zip(inputs, expected):
        assert dimension_to_number_of_triangular_elements(inp) == exp


def random_cov(dim, seed):
    np.random.seed(seed)

    num_elements = int(dim * (dim + 1) / 2)
    chol = np.zeros((dim, dim))
    chol[np.tril_indices(dim)] = np.random.uniform(size=num_elements)
    cov = chol @ chol.T
    zero_positions = np.random.choice(range(dim), size=int(dim / 5), replace=False)
    for pos in zero_positions:
        cov[:, pos] = 0
        cov[pos] = 0
    return cov


seeds = [58822, 3181, 98855, 44002, 47631, 97741, 10655, 4600, 1151, 58189]
dims = [8] * 6 + [10, 12, 15, 20]


@pytest.mark.parametrize("dim, seed", zip(dims, seeds))
def test_robust_cholesky_with_zero_variance(dim, seed):
    cov = random_cov(dim, seed)
    chol = robust_cholesky(cov)
    aaae(chol.dot(chol.T), cov)
    assert (chol[np.triu_indices(len(cov), k=1)] == 0).all()


def test_robust_cholesky_with_extreme_cases():
    for cov in [np.ones((5, 5)), np.zeros((5, 5))]:
        chol = robust_cholesky(cov)
        aaae(chol.dot(chol.T), cov)
