import numpy as np
import pytest
from estimagic.optimization.tranquilo.poisedness import _reshape_coef_to_square_terms
from estimagic.optimization.tranquilo.poisedness import get_poisedness_constant
from estimagic.optimization.tranquilo.poisedness import improve_poisedness
from estimagic.optimization.tranquilo.poisedness import lagrange_poly_matrix
from numpy.testing import assert_array_almost_equal as aaae


def evaluate_scalar_model(x, intercept, linear_terms, square_terms):
    return intercept + linear_terms.T @ x + 0.5 * x.T @ square_terms @ x


# ======================================================================================
# Improve poisedness
# ======================================================================================

TEST_CASES = [
    (
        "sphere",
        5,
        [
            5324.240935366314,
            36.87996947175511,
            11.090857556966462,
            1.3893207179888898,
            1.0016763267639168,
        ],
    ),
    (
        "cube",
        10,
        [
            10648.478006222356,
            49.998826793338836,
            13.145227394549012,
            1.0313287779903457,
            1.008398336326099,
            1.0306831620836225,
            1.0019247733166188,
            1.0044418474330754,
            1.0024393102571791,
            1.0017007017773365,
        ],
    ),
]


@pytest.mark.parametrize("shape, maxiter, expected", TEST_CASES)
def test_improve_poisedness(shape, maxiter, expected):
    sample = np.array(
        [
            [-0.98, -0.96],
            [-0.96, -0.98],
            [0, 0],
            [0.98, 0.96],
            [0.96, 0.98],
            [0.94, 0.94],
        ]
    )

    _, got_lambdas = improve_poisedness(sample=sample, shape=shape, maxiter=maxiter)

    aaae(got_lambdas, expected, decimal=2)


# ======================================================================================
# Lambda poisedness constant
# ======================================================================================

TEST_CASES = [
    (
        np.array(
            [
                [-0.98, -0.96],
                [-0.96, -0.98],
                [0, 0],
                [0.98, 0.96],
                [0.96, 0.98],
                [0.94, 0.94],
            ]
        ),
        5324,
    ),
    (
        np.array(
            [
                [-0.98, -0.96],
                [-0.96, -0.98],
                [0, 0],
                [0.98, 0.96],
                [0.96, 0.98],
                [0.707, -0.707],
            ]
        ),
        36.88,
    ),
    (
        np.array(
            [
                [-0.967, 0.254],
                [-0.96, -0.98],
                [0, 0],
                [0.98, 0.96],
                [-0.199, 0.979],
                [0.707, -0.707],
            ]
        ),
        1.001,
    ),
]


@pytest.mark.parametrize("sample, expected", TEST_CASES)
def test_poisedness_scaled_precise(sample, expected):
    """Test cases are taken from :cite:`Conn2009` p. 99."""

    got, *_ = get_poisedness_constant(sample, shape="sphere")
    assert np.allclose(got, expected, rtol=1e-2)


TEST_CASES = [
    (
        np.array(
            [
                [0.848, 0.528],
                [-0.96, -0.98],
                [0, 0],
                [0.98, 0.96],
                [-0.96, -0.98],
                [0.707, -0.707],
            ]
        ),
        15.66,
    ),
    (
        np.array(
            [
                [-0.848, 0.528],
                [-0.96, -0.98],
                [0, 0],
                [0.98, 0.96],
                [-0.89, 0.996],
                [0.707, -0.707],
            ]
        ),
        1.11,
    ),
    (
        np.array(
            [
                [-0.967, 0.254],
                [-0.96, -0.98],
                [0, 0],
                [0.98, 0.96],
                [-0.89, 0.996],
                [0.707, -0.707],
            ]
        ),
        1.01,
    ),
]


@pytest.mark.xfail(reason="Imprecise results, but expected decrease in lambda.")
@pytest.mark.parametrize("sample, expected", TEST_CASES)
def test_poisedness_scaled_imprecise(sample, expected):
    """Test cases are taken from :cite:`Conn2009` p. 99."""

    got, *_ = get_poisedness_constant(sample, shape="sphere")
    assert np.allclose(got, expected, rtol=1e-2)


TEST_CASES = [
    (
        np.array(
            [
                [0.524, 0.0006],
                [0.032, 0.323],
                [0.187, 0.890],
                [0.5, 0.5],
                [0.982, 0.368],
                [0.774, 0.918],
            ]
        ),
        1,
    )
]


@pytest.mark.parametrize("sample, expected", TEST_CASES)
def test_poisedness_unscaled_precise(sample, expected):
    """This test case is taken from :cite:`Conn2009` p. 45."""
    n_params = sample.shape[1]

    radius = 0.5
    center = 0.5 * np.ones(n_params)
    sample_centered = (sample - center) / radius

    got, *_ = get_poisedness_constant(sample_centered, shape="sphere")
    assert np.allclose(got, expected, rtol=1e-2)


# ======================================================================================
# Lagrange polynomials
# ======================================================================================

TEST_CASES = [
    (
        np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2], [0.5, 0.5]]),
        np.array(
            [
                [
                    1,
                    -1.5,
                    -1.5,
                    1,
                    1,
                    1,
                ],
                [
                    0,
                    5 / 3,
                    -1 / 3,
                    -1.64705882e00,
                    -7.64705882e-01,
                    3.52941176e-01,
                ],
                [
                    0,
                    -1 / 3,
                    5 / 3,
                    3.52941176e-01,
                    -7.64705882e-01,
                    -1.64705882e00,
                ],
                [
                    0,
                    -5 / 12,
                    1 / 12,
                    9.11764706e-01,
                    -5.88235294e-02,
                    -8.82352941e-02,
                ],
                [
                    -0,
                    -1 / 6,
                    -1 / 6,
                    1.76470588e-01,
                    1.11764706e00,
                    1.76470588e-01,
                ],
                [
                    0,
                    1 / 12,
                    -5 / 12,
                    -8.82352941e-02,
                    -5.88235294e-02,
                    9.11764706e-01,
                ],
                [
                    0,
                    2 / 3,
                    2 / 3,
                    -7.05882353e-01,
                    -4.70588235e-01,
                    -7.05882353e-01,
                ],
            ]
        ),
        np.array([1, 0.84, 0.84, 0.99, 0.96, 0.99, 0.37]),
    )
]


@pytest.mark.parametrize("sample, expected_lagrange_mat, expected_critval", TEST_CASES)
def test_lagrange_poly_matrix(sample, expected_lagrange_mat, expected_critval):
    """This test case is taken from :cite:`Conn2009` p. 62."""
    sample = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2], [0.5, 0.5]])
    n_params = sample.shape[1]

    lagrange_mat = lagrange_poly_matrix(sample)
    aaae(lagrange_mat, expected_lagrange_mat)

    for idx, lagrange_poly in enumerate(lagrange_mat):
        intercept = lagrange_poly[0]
        linear_terms = lagrange_poly[1 : n_params + 1]
        _coef_square_terms = lagrange_poly[n_params + 1 :]
        square_terms = _reshape_coef_to_square_terms(_coef_square_terms, n_params)

        got = evaluate_scalar_model(sample[idx], intercept, linear_terms, square_terms)
        aaae(got, expected_critval[idx], decimal=2)
