import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from numpy.testing import assert_array_equal as aae
from optimagic import first_derivative
from optimagic.parameters.conversion import InternalParams
from optimagic.parameters.scale_conversion import get_scale_converter
from optimagic.parameters.scaling import ScalingOptions

TEST_CASES = {
    "start_values": InternalParams(
        values=np.array([0, 1, 1, 1, 1, 1]),
        lower_bounds=np.array([-2, 0, 0.5, 2 / 3, 3 / 4, 4 / 5]),
        upper_bounds=np.array([2, 2, 1.5, 4 / 3, 5 / 4, 6 / 5]),
        names=None,
    ),
    "bounds": InternalParams(
        values=np.full(6, 0.5),
        lower_bounds=np.zeros(6),
        upper_bounds=np.ones(6),
        names=None,
    ),
}

IDS = list(TEST_CASES)
PARAMETRIZATION = list(TEST_CASES.items())


@pytest.mark.parametrize("method, expected", PARAMETRIZATION, ids=IDS)
def test_get_scale_converter_active(method, expected):
    params = InternalParams(
        values=np.arange(6),
        lower_bounds=np.arange(6) - 1,
        upper_bounds=np.arange(6) + 1,
        names=list("abcdef"),
    )

    scaling = ScalingOptions(
        method=method,
        clipping_value=0.5,
    )

    converter, scaled = get_scale_converter(
        internal_params=params,
        scaling=scaling,
    )

    aaae(scaled.values, expected.values)
    aaae(scaled.lower_bounds, expected.lower_bounds)
    aaae(scaled.upper_bounds, expected.upper_bounds)

    aaae(converter.params_to_internal(params.values), expected.values)
    aaae(converter.params_from_internal(expected.values), params.values)

    calculated_jacobian = converter.derivative_to_internal(np.eye(len(params.values)))

    numerical_jacobian = first_derivative(
        converter.params_from_internal, expected.values
    )["derivative"]

    aaae(calculated_jacobian, numerical_jacobian)


def test_scale_conversion_fast_path():
    params = InternalParams(
        values=np.arange(6),
        lower_bounds=np.arange(6) - 1,
        upper_bounds=np.arange(6) + 1,
        names=list("abcdef"),
    )

    converter, scaled = get_scale_converter(
        internal_params=params,
        scaling=None,
    )

    aae(params.values, scaled.values)
    aae(params.lower_bounds, scaled.lower_bounds)
    aae(params.upper_bounds, scaled.upper_bounds)

    aae(converter.params_to_internal(params.values), params.values)
    aae(converter.params_from_internal(params.values), params.values)
    aae(converter.derivative_to_internal(np.ones(3)), np.ones(3))
