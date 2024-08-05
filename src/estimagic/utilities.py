from optimagic.decorators import deprecated
from optimagic.utilities import (
    calculate_trustregion_initial_radius as _calculate_trustregion_initial_radius,
)
from optimagic.utilities import (
    chol_params_to_lower_triangular_matrix as _chol_params_to_lower_triangular_matrix,
)
from optimagic.utilities import cov_matrix_to_params as _cov_matrix_to_params
from optimagic.utilities import (
    cov_matrix_to_sdcorr_params as _cov_matrix_to_sdcorr_params,
)
from optimagic.utilities import cov_params_to_matrix as _cov_params_to_matrix
from optimagic.utilities import cov_to_sds_and_corr as _cov_to_sds_and_corr
from optimagic.utilities import (
    dimension_to_number_of_triangular_elements as _dimension_to_number_of_triangular_elements,  # noqa: E501
)
from optimagic.utilities import get_rng as _get_rng
from optimagic.utilities import hash_array as _hash_array
from optimagic.utilities import isscalar as _isscalar
from optimagic.utilities import (
    number_of_triangular_elements_to_dimension as _number_of_triangular_elements_to_dimension,  # noqa: E501
)
from optimagic.utilities import propose_alternatives as _propose_alternatives
from optimagic.utilities import read_pickle as _read_pickle
from optimagic.utilities import robust_cholesky as _robust_cholesky
from optimagic.utilities import robust_inverse as _robust_inverse
from optimagic.utilities import sdcorr_params_to_matrix as _sdcorr_params_to_matrix
from optimagic.utilities import (
    sdcorr_params_to_sds_and_corr as _sdcorr_params_to_sds_and_corr,
)
from optimagic.utilities import sds_and_corr_to_cov as _sds_and_corr_to_cov
from optimagic.utilities import to_pickle as _to_pickle

MSG = (
    "estimagic.utilities.{name} has been deprecated in version 0.5.0. Use optimagic."
    "utilities.{name} instead. This function will be removed in version 0.6.0."
)


chol_params_to_lower_triangular_matrix = deprecated(
    _chol_params_to_lower_triangular_matrix,
    MSG.format(name="chol_params_to_lower_triangular_matrix"),
)
cov_params_to_matrix = deprecated(
    _cov_params_to_matrix, MSG.format(name="cov_params_to_matrix")
)
cov_matrix_to_params = deprecated(
    _cov_matrix_to_params, MSG.format(name="cov_matrix_to_params")
)
sdcorr_params_to_sds_and_corr = deprecated(
    _sdcorr_params_to_sds_and_corr, MSG.format(name="sdcorr_params_to_sds_and_corr")
)
sds_and_corr_to_cov = deprecated(
    _sds_and_corr_to_cov, MSG.format(name="sds_and_corr_to_cov")
)
cov_to_sds_and_corr = deprecated(
    _cov_to_sds_and_corr, MSG.format(name="cov_to_sds_and_corr")
)
sdcorr_params_to_matrix = deprecated(
    _sdcorr_params_to_matrix, MSG.format(name="sdcorr_params_to_matrix")
)
cov_matrix_to_sdcorr_params = deprecated(
    _cov_matrix_to_sdcorr_params, MSG.format(name="cov_matrix_to_sdcorr_params")
)
number_of_triangular_elements_to_dimension = deprecated(
    _number_of_triangular_elements_to_dimension,
    MSG.format(name="number_of_triangular_elements_to_dimension"),
)
dimension_to_number_of_triangular_elements = deprecated(
    _dimension_to_number_of_triangular_elements,
    MSG.format(name="dimension_to_number_of_triangular_elements"),
)
propose_alternatives = deprecated(
    _propose_alternatives, MSG.format(name="propose_alternatives")
)
robust_cholesky = deprecated(_robust_cholesky, MSG.format(name="robust_cholesky"))
robust_inverse = deprecated(_robust_inverse, MSG.format(name="robust_inverse"))
hash_array = deprecated(_hash_array, MSG.format(name="hash_array"))
calculate_trustregion_initial_radius = deprecated(
    _calculate_trustregion_initial_radius,
    MSG.format(name="calculate_trustregion_initial_radius"),
)
to_pickle = deprecated(_to_pickle, MSG.format(name="to_pickle"))
read_pickle = deprecated(_read_pickle, MSG.format(name="read_pickle"))
isscalar = deprecated(_isscalar, MSG.format(name="isscalar"))
get_rng = deprecated(_get_rng, MSG.format(name="get_rng"))

__all__ = [
    "chol_params_to_lower_triangular_matrix",
    "cov_params_to_matrix",
    "cov_matrix_to_params",
    "sdcorr_params_to_sds_and_corr",
    "sds_and_corr_to_cov",
    "cov_to_sds_and_corr",
    "sdcorr_params_to_matrix",
    "cov_matrix_to_sdcorr_params",
    "number_of_triangular_elements_to_dimension",
    "dimension_to_number_of_triangular_elements",
    "propose_alternatives",
    "robust_cholesky",
    "robust_inverse",
    "hash_array",
    "calculate_trustregion_initial_radius",
    "to_pickle",
    "read_pickle",
    "isscalar",
    "get_rng",
]
