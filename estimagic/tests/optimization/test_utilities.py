import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from estimagic.optimization.utilities import cov_matrix_to_params
from estimagic.optimization.utilities import cov_matrix_to_sdcorr_params
from estimagic.optimization.utilities import cov_params_to_matrix
from estimagic.optimization.utilities import dimension_to_number_of_triangular_elements
from estimagic.optimization.utilities import index_element_to_string
from estimagic.optimization.utilities import number_of_triangular_elements_to_dimension
from estimagic.optimization.utilities import robust_cholesky
from estimagic.optimization.utilities import sdcorr_params_to_matrix


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


def test_index_element_to_string():
    inputs = [(("a", "b", 1),), (["bla", 5, 6], "~"), ("bla", "*")]
    expected = ["a_b_1", "bla~5~6", "bla"]
    for inp, exp in zip(inputs, expected):
        assert index_element_to_string(*inp) == exp


def random_cov(dim, seed):
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
def test_robust_cholesky(dim, seed):
    cov = random_cov(dim, seed)
    chol = robust_cholesky(cov)
    aaae(chol.dot(chol.T), cov)
    assert (chol[np.triu_indices(len(cov), k=1)] == 0).all()
