import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.config import IS_JAX_INSTALLED
from optimagic.utilities import (
    calculate_trustregion_initial_radius,
    chol_params_to_lower_triangular_matrix,
    cov_matrix_to_params,
    cov_matrix_to_sdcorr_params,
    cov_params_to_matrix,
    cov_to_sds_and_corr,
    dimension_to_number_of_triangular_elements,
    get_rng,
    hash_array,
    isscalar,
    number_of_triangular_elements_to_dimension,
    propose_alternatives,
    read_pickle,
    robust_cholesky,
    robust_inverse,
    sdcorr_params_to_matrix,
    sdcorr_params_to_sds_and_corr,
    sds_and_corr_to_cov,
    to_pickle,
)

if IS_JAX_INSTALLED:
    import jax.numpy as jnp


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
    for inp, exp in zip(inputs, expected, strict=False):
        assert number_of_triangular_elements_to_dimension(inp) == exp


def test_dimension_to_number_of_triangular_elements():
    inputs = [3, 4, 5, 6]
    expected = [6, 10, 15, 21]
    for inp, exp in zip(inputs, expected, strict=False):
        assert dimension_to_number_of_triangular_elements(inp) == exp


def random_cov(dim, seed):
    rng = np.random.default_rng(seed)

    num_elements = int(dim * (dim + 1) / 2)
    chol = np.zeros((dim, dim))
    chol[np.tril_indices(dim)] = rng.uniform(size=num_elements)
    cov = chol @ chol.T
    zero_positions = rng.choice(range(dim), size=int(dim / 5), replace=False)
    for pos in zero_positions:
        cov[:, pos] = 0
        cov[pos] = 0
    return cov


seeds = [58822, 3181, 98855, 44002, 47631, 97741, 10655, 4600, 1151, 58189]
dims = [8] * 6 + [10, 12, 15, 20]


@pytest.mark.parametrize("dim, seed", zip(dims, seeds, strict=False))
def test_robust_cholesky_with_zero_variance(dim, seed):
    cov = random_cov(dim, seed)
    chol = robust_cholesky(cov)
    aaae(chol.dot(chol.T), cov)
    assert (chol[np.triu_indices(len(cov), k=1)] == 0).all()


def test_robust_cholesky_with_extreme_cases():
    for cov in [np.ones((5, 5)), np.zeros((5, 5))]:
        chol = robust_cholesky(cov)
        aaae(chol.dot(chol.T), cov)


def test_robust_inverse_nonsingular():
    mat = np.eye(3) + 0.2
    expected = np.linalg.inv(mat)
    calculated = robust_inverse(mat)
    aaae(calculated, expected)


def test_robust_inverse_singular():
    mat = np.zeros((5, 5))
    expected = np.zeros((5, 5))
    with pytest.warns(UserWarning, match="LinAlgError"):
        calculated = robust_inverse(mat)
    aaae(calculated, expected)


def test_hash_array():
    arr1 = np.arange(4)[::2]
    arr2 = np.array([0, 2])

    arr3 = np.array([0, 3])
    assert hash_array(arr1) == hash_array(arr2)
    assert hash_array(arr1) != hash_array(arr3)


def test_initial_trust_radius_small_x():
    x = np.array([0.01, 0.01])
    expected = 0.1
    res = calculate_trustregion_initial_radius(x)
    assert expected == pytest.approx(res, abs=1e-8)


def test_initial_trust_radius_large_x():
    x = np.array([20.5, 10])
    expected = 2.05
    res = calculate_trustregion_initial_radius(x)
    assert expected == pytest.approx(res, abs=1e-8)


def test_pickling(tmp_path):
    a = [1, 2, 3]
    path = tmp_path / "bla.pkl"
    to_pickle(a, path)
    b = read_pickle(path)
    assert a == b


SCALARS = [1, 2.0, np.pi, np.array(1), np.array(2.0), np.array(np.pi), np.nan]


@pytest.mark.parametrize("element", SCALARS)
def test_isscalar_true(element):
    assert isscalar(element) is True


NON_SCALARS = [np.arange(3), {"a": 1}, [1, 2, 3]]


@pytest.mark.parametrize("element", NON_SCALARS)
def test_isscalar_false(element):
    assert isscalar(element) is False


@pytest.mark.skipif(not IS_JAX_INSTALLED, reason="Needs jax.")
def tets_isscalar_jax_true():
    x = jnp.arange(3)
    element = x @ x
    assert isscalar(element) is True


@pytest.mark.skipif(not IS_JAX_INSTALLED, reason="Needs jax.")
def test_isscalar_jax_false():
    element = jnp.arange(3)
    assert isscalar(element) is False


TEST_CASES = [
    0,
    1,
    10,
    1000000,
    None,
    np.random.default_rng(),
    np.random.Generator(np.random.MT19937()),
]


@pytest.mark.parametrize("seed", TEST_CASES)
def test_get_rng_correct_input(seed):
    rng = get_rng(seed)
    assert isinstance(rng, np.random.Generator)


TEST_CASES = [0.1, "a", object(), lambda x: x**2]


@pytest.mark.parametrize("seed", TEST_CASES)
def test_get_rng_wrong_input(seed):
    with pytest.raises(TypeError):
        get_rng(seed)


def test_propose_alternatives():
    possibilities = ["scipy_lbfgsb", "scipy_slsqp", "nlopt_lbfgsb"]
    inputs = [["scipy_L-BFGS-B", 1], ["L-BFGS-B", 2]]
    expected = [["scipy_slsqp"], ["scipy_slsqp", "scipy_lbfgsb"]]
    for inp, exp in zip(inputs, expected, strict=False):
        assert propose_alternatives(inp[0], possibilities, number=inp[1]) == exp
