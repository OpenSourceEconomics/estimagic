import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic import first_derivative
from optimagic.parameters.space_conversion import (
    InternalParams,
    _multiply_from_left,
    _multiply_from_right,
    get_space_converter,
)
from optimagic.utilities import get_rng


def _get_test_case_no_constraint():
    n_params = 10
    fp = InternalParams(
        values=np.arange(n_params),
        lower_bounds=np.full(n_params, -1),
        upper_bounds=np.full(n_params, 11),
        names=list("abcdefghij"),
    )

    constraints = []
    return constraints, fp, fp


def _get_test_case_fixed(with_value):
    fp = InternalParams(
        values=np.arange(5),
        lower_bounds=np.full(5, -np.inf),
        upper_bounds=np.full(5, np.inf),
        names=list("abcde"),
    )
    if with_value:
        constraints = [{"index": [0, 2, 4], "type": "fixed", "value": [0, 2, 4]}]
    else:
        constraints = [{"index": [0, 2, 4], "type": "fixed"}]

    internal = InternalParams(
        values=np.array([1, 3]),
        lower_bounds=np.full(2, -np.inf),
        upper_bounds=np.full(2, np.inf),
        names=None,
    )

    return constraints, fp, internal


def _get_test_case_increasing(as_one):
    fp = InternalParams(
        values=np.array([0.1, 2.2, 2.3, 10.1, -1]),
        lower_bounds=np.full(5, -np.inf),
        upper_bounds=np.full(5, np.inf),
        names=list("abcde"),
    )

    internal = InternalParams(
        values=np.array([0.1, -2.1, -0.1, -7.8, -1]),
        lower_bounds=np.full(5, -np.inf),
        upper_bounds=np.array([np.inf, 0, 0, 0, np.inf]),
        names=None,
    )

    if as_one:
        constraints = [{"type": "increasing", "index": [0, 1, 2, 3]}]
    else:
        constraints = [
            {"type": "increasing", "index": [0, 1, 2]},
            {"type": "increasing", "index": [2, 3]},
        ]

    return constraints, fp, internal


def _get_test_case_decreasing(as_one):
    fp = InternalParams(
        values=np.array([0.1, 2.2, 2.3, 10.1, -1]),
        lower_bounds=np.full(5, -np.inf),
        upper_bounds=np.full(5, np.inf),
        names=list("abcde"),
    )

    internal = InternalParams(
        values=np.array([0.1, -2.1, -0.1, -7.8, -1]),
        lower_bounds=np.full(5, -np.inf),
        upper_bounds=np.array([np.inf, 0, 0, 0, np.inf]),
        names=None,
    )

    if as_one:
        constraints = [{"type": "decreasing", "index": [3, 2, 1, 0]}]
    else:
        constraints = [
            {"type": "decreasing", "index": [2, 1, 0]},
            {"type": "decreasing", "index": [3, 2]},
        ]

    return constraints, fp, internal


def _get_test_case_equality(as_one):
    fp = InternalParams(
        values=np.array([0, 1.5, 1.5, 0, 1.5, 1]),
        lower_bounds=np.array([-10, 1, 0.9, -np.inf, -np.inf, -10]),
        upper_bounds=np.full(6, np.inf),
        names=list("abcdef"),
    )

    internal = InternalParams(
        values=np.array([0, 1.5, 0, 1]),
        lower_bounds=np.array([-10, 1, -np.inf, -10]),
        upper_bounds=np.full(4, np.inf),
        names=None,
    )

    if as_one:
        constraints = [{"type": "equality", "index": [1, 2, 4]}]
    else:
        constraints = [
            {"type": "equality", "index": [1, 2]},
            {"type": "equality", "index": [1, 4]},
        ]

    return constraints, fp, internal


def _get_test_case_probability():
    fp = InternalParams(
        values=np.array([0.1, 0.2, 0.2, 0.5, 10]),
        lower_bounds=np.full(5, -np.inf),
        upper_bounds=np.full(5, np.inf),
        names=list("abcde"),
    )

    internal = InternalParams(
        values=np.array([0.2, 0.4, 0.4, 10]),
        lower_bounds=np.array([0, 0, 0, -np.inf]),
        upper_bounds=np.full(4, np.inf),
        names=None,
    )

    constraints = [{"type": "probability", "index": [0, 1, 2, 3]}]

    return constraints, fp, internal


def _get_test_case_uncorrelated_covariance():
    fp = InternalParams(
        values=np.array([1, 0, 4, 0, 0, 9, 10]),
        lower_bounds=np.full(7, -np.inf),
        upper_bounds=np.full(7, np.inf),
        names=list("abcdefg"),
    )

    internal = InternalParams(
        values=np.array([1, 4, 9, 10]),
        lower_bounds=np.array([0, 0, 0, -np.inf]),
        upper_bounds=np.full(4, np.inf),
        names=None,
    )

    constraints = [
        {"type": "covariance", "index": [0, 1, 2, 3, 4, 5]},
        {"type": "fixed", "index": [1, 3, 4], "value": 0},
    ]

    return constraints, fp, internal


def _get_test_case_covariance():
    fp = InternalParams(
        values=np.array([1, -0.2, 1.2, -0.2, 0.1, 1.3, 0.1, -0.05, 0.2, 1, 10]),
        lower_bounds=np.full(11, -np.inf),
        upper_bounds=np.full(11, np.inf),
        names=list("abcdefghijk"),
    )

    internal = InternalParams(
        values=np.array(
            [
                1,
                -0.2,
                1.07703296,
                -0.2,
                0.0557086,
                1.12111398,
                0.1,
                -0.0278543,
                0.19761748,
                0.97476739,
                10,
            ]
        ),
        lower_bounds=np.array(
            [0, -np.inf, 0, -np.inf, -np.inf, 0, -np.inf, -np.inf, -np.inf, 0, -np.inf]
        ),
        upper_bounds=np.full(11, np.inf),
        names=None,
    )

    constraints = [{"type": "covariance", "index": np.arange(10)}]

    return constraints, fp, internal


def _get_test_case_normalized_covariance():
    fp = InternalParams(
        values=np.array([4, 0.1, 2, 0.2, 0.3, 3, 10]),
        lower_bounds=np.full(7, -np.inf),
        upper_bounds=np.full(7, np.inf),
        names=list("abcdefg"),
    )

    internal = InternalParams(
        values=np.array([0.05, 1.4133294025, 0.1, 0.2087269956, 1.7165177078, 10]),
        lower_bounds=[-np.inf, 0, -np.inf, -np.inf, 0, -np.inf],
        upper_bounds=np.full(6, np.inf),
        names=None,
    )

    constraints = [
        {"type": "covariance", "index": np.arange(6)},
        {"type": "fixed", "index": [0], "value": 4},
    ]

    return constraints, fp, internal


def _get_test_case_sdcorr():
    fp = InternalParams(
        values=np.array([2, 1.5, 3, 0.2, 0.15, 0.33, 10]),
        lower_bounds=np.full(7, -np.inf),
        upper_bounds=np.full(7, np.inf),
        names=list("abcdefg"),
    )

    internal = InternalParams(
        values=np.array([2, 0.3, 1.46969385, 0.45, 0.91855865, 2.82023935, 10]),
        lower_bounds=np.array([0, -np.inf, 0, -np.inf, -np.inf, 0, -np.inf]),
        upper_bounds=np.full(7, np.inf),
        names=None,
    )

    constraints = [{"type": "sdcorr", "index": np.arange(6)}]

    return constraints, fp, internal


TEST_CASES = {
    "no_constraints": _get_test_case_no_constraint(),
    "fixed_at_start": _get_test_case_fixed(with_value=False),
    "fixed_at_value": _get_test_case_fixed(with_value=True),
    "one_increasing": _get_test_case_increasing(as_one=True),
    "overlapping_increasing": _get_test_case_increasing(as_one=False),
    "one_decreasing": _get_test_case_decreasing(as_one=True),
    "overlapping_decreasing": _get_test_case_decreasing(as_one=False),
    "one_equality": _get_test_case_equality(as_one=True),
    "everlapping_equality": _get_test_case_equality(as_one=False),
    "probability": _get_test_case_probability(),
    "uncorrelated_covariance": _get_test_case_uncorrelated_covariance(),
    "covariance": _get_test_case_covariance(),
    "normalized_covariance": _get_test_case_normalized_covariance(),
    "sdcorr": _get_test_case_sdcorr(),
}


PARAMETRIZATION = list(TEST_CASES.values())
IDS = list(TEST_CASES)


@pytest.mark.parametrize(
    "constraints, params, expected_internal", PARAMETRIZATION, ids=IDS
)
def test_space_converter_with_params(constraints, params, expected_internal):
    converter, internal = get_space_converter(
        internal_params=params,
        internal_constraints=constraints,
    )

    aaae(internal.values, expected_internal.values)
    aaae(internal.lower_bounds, expected_internal.lower_bounds)
    aaae(internal.upper_bounds, expected_internal.upper_bounds)

    aaae(converter.params_to_internal(params.values), expected_internal.values)
    aaae(converter.params_from_internal(expected_internal.values), params.values)

    numerical_jacobian = first_derivative(
        converter.params_from_internal, expected_internal.values
    )["derivative"]

    calculated_jacobian = converter.derivative_to_internal(
        external_derivative=np.eye(len(params.values)),
        internal_values=expected_internal.values,
    )

    aaae(calculated_jacobian, numerical_jacobian)


@pytest.mark.parametrize("seed", range(5))
def test_multiply_from_left_and_right(seed):
    rng = get_rng(seed)
    mat_list = [rng.uniform(size=(10, 10)) for i in range(5)]
    a, b, c, d, e = mat_list

    expected = a @ b @ c @ d @ e

    calc_from_left = _multiply_from_left(mat_list)
    calc_from_right = _multiply_from_right(mat_list)

    aaae(calc_from_left, expected)
    aaae(calc_from_right, expected)
