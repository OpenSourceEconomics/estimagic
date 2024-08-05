"""Functions and derivatives thereof to transform external and internal params.

Remarks on the mathematical notation:
-------------------------------------

We let :math:`X` denote the Cholesky factor of some covariance matrix :math:`S`.
That is :math:`X X^\top = S`. We write :math:`\text{vec}(A)` for the column-wise
vectorization of the matrix :math:`A` and we write :math:`\text{vech}(A)` for
the row-wise half vectorization of :math:`A`. We denote the elimination
matrix by :math:`L`, which fulfills :math:`L \text{vec}(A) = \text{vech}(A)`.
For lower-triangular matrices :math:`A` we define the "lower-triangular"
duplication matrix :math:`D`, which is not to be confused with the standard
duplication matrix, and fulfills :math:`D \text{vech}(A) = \text{vec}(A)`. At
last we define the so called commutation matrix :math:`K` which is given by the
property that :math:`K \text{vec}(A) = \text{vec}(A^\top)`.

Remarks on reference literature:
--------------------------------

The solutions on how to compute the jacobians implemented here can be found
using matrix calculus. See for example 'Matrix Differential Calculus with
Applications in Statistics and Econometrics' by Magnus and Neudecker. In
specific cases we refer to posts on math.stackexchange.com.

.. rubric:: References

.. _post_mathoverflow:
   https://google.github.io/styleguide/pyguide.html

"""

import numpy as np

from optimagic.utilities import (
    chol_params_to_lower_triangular_matrix,
    cov_matrix_to_sdcorr_params,
    cov_params_to_matrix,
    dimension_to_number_of_triangular_elements,
    robust_cholesky,
    sdcorr_params_to_matrix,
)


def covariance_to_internal(external_values, constr):
    """Do a cholesky reparametrization."""
    cov = cov_params_to_matrix(external_values)
    chol = robust_cholesky(cov)
    return chol[np.tril_indices(len(cov))]


def covariance_to_internal_jacobian(external_values, constr):
    r"""Jacobian of ``covariance_to_internal``.

    For reference see docstring of ``jacobian_covariance_from_internal``. In
    comparison to that function, however, here we want to differentiate the
    reverse graph
                external --> cov --> cholesky --> internal

    Again use the vectors :math:`c` and :math:`x` to denote the external and
    internal values, respectively. To solve for the jacobian we make use of the
    identity

    .. math::
        \frac{\mathrm{d}x}{\mathrm{d}c} = (\frac{\mathrm{d}c}{\mathrm{d}x})^{-1}

    Args:
        external_values (np.ndarray): Row-wise half-vectorized covariance matrix

    Returns:
        deriv: The Jacobian matrix.

    """
    cov = cov_params_to_matrix(external_values)
    chol = robust_cholesky(cov)

    internal = chol[np.tril_indices(len(chol))]

    deriv = covariance_from_internal_jacobian(internal, constr=None)
    deriv = np.linalg.pinv(deriv)
    return deriv


def covariance_from_internal(internal_values, constr):
    """Undo a cholesky reparametrization."""
    chol = chol_params_to_lower_triangular_matrix(internal_values)
    cov = chol @ chol.T
    return cov[np.tril_indices(len(chol))]


def covariance_from_internal_jacobian(internal_values, constr):
    r"""Jacobian of ``covariance_from_internal``.

    The following result is motivated by https://tinyurl.com/y4pbfxst, which is
    shortly presented again here. For notation see the explaination at the
    beginning of the module.


    Explaination of the result
    --------------------------

    We want to differentiate the graph
                internal --> cholesky --> cov --> external

    Define :math:`x' := \text{vec}(X)` and :math:`c' := \text{vec}(S)`, where
    :math:`X` denotes the Cholesky factor of the covariance matrix :math:`S`.
    We then first differentiate the part "cholesky --> cov" using the result
    stated in the tinyurl above to get

    .. math::
        J' := \frac{\mathrm{d}c'}{\mathrm{d}x'} = (I + K)(X \otimes I) \,,

    where :math:`K` denotes the commutation matrix. Using this intermediate
    result we can compute the jacobian as

    .. math:: \frac{\mathrm{d}c}{\mathrm{d}x} = L J' D \,,

    where :math:`c := \text{external}` and :math:`x := \text{internal}`.

    Args:
        internal_values (np.ndarray): Cholesky factors stored in an "internal"
            format.

    Returns:
        deriv: The Jacobian matrix.

    """
    chol = chol_params_to_lower_triangular_matrix(internal_values)
    dim = len(chol)

    K = _commutation_matrix(dim)
    L = _elimination_matrix(dim)

    left = np.eye(dim**2) + K
    right = np.kron(chol, np.eye(dim))

    intermediate = left @ right

    deriv = L @ intermediate @ L.T
    return deriv


def sdcorr_to_internal(external_values, constr):
    """Convert sdcorr to cov and do a cholesky reparametrization."""
    cov = sdcorr_params_to_matrix(external_values)
    chol = robust_cholesky(cov)
    return chol[np.tril_indices(len(cov))]


def sdcorr_to_internal_jacobian(external_values, constr):
    r"""Derivative of ``sdcorr_to_internal``.

    For reference see docstring of ``jacobian_sdcorr_from_internal``. In
    comparison to that function, however, here we want to differentiate the
    reverse graph

     external --> mod. corr-mat --> corr-mat --> cov --> cholesky --> internal

    Again use the vectors :math:`p` and :math:`x` to denote the external and
    internal values, respectively. To solve for the jacobian we make use of the
    identity

    .. math::
        \frac{\mathrm{d}x}{\mathrm{d}p} = (\frac{\mathrm{d}p}{\mathrm{d}x})^{-1}

    Args:
        external_values (np.ndarray): Row-wise half-vectorized modified correlation
            matrix.

    Returns:
        deriv: The Jacobian matrix.

    """
    cov = sdcorr_params_to_matrix(external_values)
    chol = robust_cholesky(cov)

    internal = chol[np.tril_indices(len(chol))]

    deriv = sdcorr_from_internal_jacobian(internal, constr=None)
    deriv = np.linalg.pinv(deriv)
    return deriv


def sdcorr_from_internal(internal_values, constr):
    """Undo a cholesky reparametrization."""
    chol = chol_params_to_lower_triangular_matrix(internal_values)
    cov = chol @ chol.T
    return cov_matrix_to_sdcorr_params(cov)


def sdcorr_from_internal_jacobian(internal_values, constr):
    r"""Derivative of ``sdcorr_from_internal``.

    The following result is motivated by https://tinyurl.com/y6ytlyd9; however
    since the question was formulated with an error the result here is adjusted
    slightly. In particular, in the answer by user 'greg', the matrix :math:`A`
    should have been defined as :math:`A = \text{diag}(||x_1||, \dots, ||x_n||)`
    , where :math:`||x_i||` denotes the euclidian norm of the the i-th row of
    :math:`X` (the Cholesky factor). For notation see the explaination at the
    beginning of the module or the question on the tinyurl. The variable names
    in this function are chosen to be consistent with the tinyurl link.

    Explaination on the result
    --------------------------

    We want to differentiate the graph

     internal --> cholesky --> cov --> corr-mat --> mod. corr-mat --> external

    where mod. corr-mat denotes the modified correlation matrix which has the
    standard deviations stored on its diagonal. Let :math:`x := \text{internal}`
    and :math:`p := \text{external}`. Then we want to compute the quantity

    .. math:: \frac{\mathrm{d} p}{\mathrm{d} x} .

    As before we consider an intermediate result first. Namely we define
    :math:`A` as above, :math:`V := A^{-1}` and :math:`P := V S V + A - I`. The
    attentive reader might now notice that :math:`P` is the modified correlation
    matrix. At last we write :math:`x' := \text{vec}(X)` and
    :math:`p' := \text{vec}(P)`. Using the result stated in the tinyurl above,
    adjusted for the different matrix :math:`A`, we can compute the quantity
    :math:`(\mathrm{d} p'/ \mathrm{d} x')`.

    Finally, since we can define transformation matrices :math:`T` and :math:`L`
    to get :math:`p = T p'` and :math:`x = L x'` (where :math:`L` denotes the
    elimination matrix with corresponding duplication matrix :math:`D`), we can
    get our final result as

    .. math::
        \frac{\mathrm{d}p}{\mathrm{d}x} = T \frac{\mathrm{d}p'}{\mathrm{d}x'} D

    Args:
        internal_values (np.ndarray): Cholesky factors stored in an "internal"
            format.

    Returns:
        deriv: The Jacobian matrix.

    """
    X = chol_params_to_lower_triangular_matrix(internal_values)
    dim = len(X)

    identity = np.eye(dim)
    S = X @ X.T

    #  the wrong formulation in the tinyurl stated: A = np.multiply(I, X)
    A = np.sqrt(np.multiply(identity, S))

    V = np.linalg.inv(A)

    K = _commutation_matrix(dim)
    Y = np.diag(identity.ravel("F"))

    #  with the wrong formulation in the tinyurl we would have had U = Y
    norms = np.sqrt((X**2).sum(axis=1).reshape(-1, 1))
    XX = X / norms
    U = Y @ np.kron(identity, XX) @ K

    N = np.kron(identity, X) @ K + np.kron(X, identity)

    VS = V @ S
    B = np.kron(V, V)
    H = np.kron(VS, identity)
    J = np.kron(identity, VS)

    intermediate = U + B @ N - (H + J) @ B @ U

    T = _transformation_matrix(dim)
    D = _duplication_matrix(dim)

    deriv = T @ intermediate @ D
    return deriv


def probability_to_internal(external_values, constr):
    """Reparametrize probability constrained parameters to internal."""
    return external_values / external_values[-1]


def probability_to_internal_jacobian(external_values, constr):
    r"""Jacobian of ``probability_to_internal``.

    Let :math:`x = \text{external}`. The function ``probability_to_internal``
    has the following structure

    .. math::  f: \mathbb{R}^m \to \mathbb{R}^m, x \mapsto \frac{1}{x_m} x

    where :math:`e_k` denotes the m-dimensional k-th standard basis vector. The
    jacobian can then be computed as

    .. math::
        J(f)(x) =
        \frac{1}{x_m} \sum_{k=1}^{m-1} e_k e_k^\top -
        \frac{1}{x_m^2}  [0, \dots, 0,
            \left ( \begin{matrix} x_{1:m-1} \\ 0 \end{matrix} \right )
        ]

    Args:
        external_values (np.ndarray): Array of probabilities; sums to one.

    Returns:
        deriv: The Jacobian matrix.

    """
    dim = len(external_values)

    deriv = np.eye(dim) / external_values[-1]
    deriv[:, -1] -= external_values / (external_values[-1] ** 2)
    deriv[-1, -1] = 0

    return deriv


def probability_from_internal(internal_values, constr):
    """Reparametrize probability constrained parameters from internal."""
    return internal_values / internal_values.sum()


def probability_from_internal_jacobian(internal_values, constr):
    r"""Jacobian of ``probability_from_internal``.

    Let :math:`x := \text{internal}`. The function ``probability_from_internal``
    has the following structure

    .. math::`f: \mathbb{R}^m \to \mathbb{R}^m, x \mapsto \frac{1}{x^\top 1} x`

    where :math:`1` denotes a vector of all ones and :math:`I_m` the identity
    matrix. The jacobian can be computed as

    .. math::  J(f)(x) = \frac{1}{\sigma} I_m - \frac{1}{\sigma^2} 1 x^\top

    Args:
        internal_values (np.ndarray): Internal (positive) values.

    Returns:
        deriv: The Jacobian matrix.

    """
    dim = len(internal_values)

    sigma = np.sum(internal_values)
    left = np.eye(dim)
    right = (np.ones((dim, dim)) * (internal_values / sigma)).T

    deriv = (left - right) / sigma
    return deriv


def linear_to_internal(external_values, constr):
    """Reparametrize linear constraint to internal."""
    return constr["to_internal"] @ external_values


def linear_to_internal_jacobian(external_values, constr):
    return constr["to_internal"]


def linear_from_internal(internal_values, constr):
    """Reparametrize linear constraint from internal."""
    return constr["from_internal"] @ internal_values


def linear_from_internal_jacobian(internal_values, constr):
    return constr["from_internal"]


def _elimination_matrix(dim):
    r"""Construct (row-wise) elimination matrix.

    Let :math:`A` be a quadratic matrix. Let :math:`\text{vec}(A)` be the
    column-wise vectorization of :math:`A`. Let :math:`\text{vech}(A)` be the
    row-wise half-vectorization of :math:`A`. Then the corresponding elimination
    matrix :math:`L` has the property

    .. math::  L \text{vec}(A) = \text{vech}(A)

    See the wiki entry https://tinyurl.com/yy4sdr43 for further information, but
    note that here we are using :math:`\text{vech}` as the row-wise and not
    column-wise half-vectorization.

    Args:
        dim (int): The dimension.

    Returns:
        eliminator (np.ndarray): The elimination matrix.

    Examples:
    >>> import numpy as np
    >>> from numpy.testing import assert_array_almost_equal
    >>> dim = 10
    >>> A = np.random.default_rng().normal(size=(dim, dim))
    >>> vectorized = A.ravel('F')
    >>> half_vectorized = A[np.tril_indices(dim)]
    >>> L = _elimination_matrix(dim)
    >>> assert_array_almost_equal(L @ vectorized, half_vectorized)

    """
    n = dimension_to_number_of_triangular_elements(dim)

    counter = np.zeros((dim, dim), int) - 1
    counter[np.tril_indices(dim)] = np.arange(n, dtype=int)

    columns = [_unit_vector_or_zeros(i, n) for i in counter.ravel("F")]

    eliminator = np.column_stack(columns)
    return eliminator


def _duplication_matrix(dim):
    r"""Return duplication matrix.

    Let :math:`A` be a lower-triangular quadratic matrix. Let
    :math:`\text{vec}(A)` be the column-wise vectorization of :math:`A`. Let
    :math:`\text{vech}(A)` be the row-wise half-vectorization of :math:`A`.
    Then the corresponding elimination matrix :math:`D` has the property

    .. math::  D \text{vech}(A) = \text{vec}(A)

    In particular note that here :math:`D = L^\top`.

    See the wiki entry https://tinyurl.com/yy4sdr43 for further information, but
    note that here we are using :math:`\text{vech}` as the row-wise and not
    column-wise half-vectorization, and that we are using this operator on a
    lower-triangular matrix and not a symmetric matrix, which allows for the
    identity :math:`D = L^\top`.

    Args:
        dim (int): The dimension.

    Returns:
        duplicator (np.ndarray): The duplication matrix.

    Example:
    >>> import numpy as np
    >>> from numpy.testing import assert_array_almost_equal
    >>> dim = 10
    >>> A = np.tril(np.random.default_rng().normal(size=(dim, dim)))
    >>> vectorized = A.ravel('F')
    >>> half_vectorized = A[np.tril_indices(dim)]
    >>> D = _duplication_matrix(dim)
    >>> assert_array_almost_equal(D @ half_vectorized, vectorized)

    """
    duplicator = _elimination_matrix(dim).T
    return duplicator


def _transformation_matrix(dim):
    r"""Return transformation matrix.

    Let :math:`A` be a quadratic matrix of dimension :math:`m \times m`. Define
    the :math:`m-1 \times m-1` matrix :math:`B` as the lower-triangular matrix
    with entries given by the lower-triangular part of :math:`A` without the
    diagonal. Set :math:`a := \text{diag}(A)`. We define the special
    vectorization operator :math:`\bar{\text{vec}}` as the operator that maps
    the diagonal of a matrix to the first entries of the vector and then
    proceeds to map the remaining lower part of the matrix using a row-wise
    half-vectorization scheme. That is, we would have

    .. math:: \bar{\text{vec}}(A) = (a^\top, \text{vech}(A)^\top)^\top

    Then the transformation matrix :math:`T` is defined by the property that

    .. math:: T \text{vec}(A) = \bar{\text{vec}}(A)

    We use this transformation when we map the vectorization of the modified
    correlation matrix to the externally stored ``sdcorr_params``.

    Args:
        dim (int): The dimension.

    Returns:
        transformer (np.ndarray): The transformation matrix.

    Example:
    >>> import numpy as np
    >>> from numpy.testing import assert_array_almost_equal
    >>> from optimagic.utilities import cov_matrix_to_sdcorr_params
    >>> from optimagic.utilities import cov_to_sds_and_corr
    >>> cov = np.cov(np.random.default_rng().normal(size=(10, 4)))
    >>> sds, corr = cov_to_sds_and_corr(cov)
    >>> corr[np.diag_indices(len(cov))] = sds
    >>> vectorized = corr.ravel('F')
    >>> sdcorr_params = cov_matrix_to_sdcorr_params(cov)
    >>> T = _transformation_matrix(len(cov))
    >>> assert_array_almost_equal(T @ vectorized, sdcorr_params)

    """
    n = dimension_to_number_of_triangular_elements(dim)
    counter = np.zeros((dim, dim)) + np.nan
    counter[np.diag_indices(dim)] = np.arange(dim, dtype=int)
    counter[np.tril_indices(dim, k=-1)] = np.arange(dim, n, dtype=int)

    m = counter.ravel("F")
    num_na = np.count_nonzero(np.isnan(m))
    indices = m.argsort()[:-num_na]

    rows = [_unit_vector_or_zeros(i, dim**2) for i in indices]

    transformer = np.row_stack(rows)
    return transformer


def _commutation_matrix(dim):
    r"""Return commutation matrix.

    Let :math:`A` be a quadratic matrix. Let :math:`\text{vec}(A)` be the
    column-wise vectorization of :math:`A`. Then the corresponding commutation
    matrix :math:`K` has the property

    .. math::  K \text{vec}(A) = \text{vec}(A^\top)

    See the wiki entry https://tinyurl.com/yydgq2z4 for further information.

    Args:
        dim (int): The dimension.

    Returns:
        cummuter (np.ndarrary): The cummutation matrix.

    Example:
    >>> import numpy as np
    >>> from numpy.testing import assert_array_almost_equal
    >>> dim = 10
    >>> A = np.random.default_rng().normal(size=(dim, dim))
    >>> vectorized = A.ravel('F')
    >>> vectorized_transposed = A.T.ravel('F')
    >>> K = _commutation_matrix(dim)
    >>> assert_array_almost_equal(K @ vectorized, vectorized_transposed)

    """
    row = np.arange(dim**2)
    col = row.reshape((dim, dim), order="F").ravel()
    commuter = np.zeros((dim**2, dim**2), dtype=np.int8)
    commuter[row, col] = 1
    return commuter


def _unit_vector_or_zeros(index, size):
    """Return unit vector or vector of all zeroes.

    Args:
        index (int): On which index to set a 1. If it is set to -1 a vector of
            all zeros will be returned.
        size (int): Dimension of the resulting vector.

    Returns:
        u (np.ndarray): The unit or zero vector.

    Example:
    >>> import numpy as np
    >>> _unit_vector_or_zeros(1, 2)
    array([0, 1])
    >>> _unit_vector_or_zeros(-1, 2)
    array([0, 0])

    """
    u = np.zeros(size, int)
    if index != -1:
        u[index] = 1
    return u
