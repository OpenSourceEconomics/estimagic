import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.differentiation.generate_steps import (
    _calculate_or_validate_base_steps,
    _fillna,
    _rescale_to_accomodate_bounds,
    _set_unused_side_to_nan,
    generate_steps,
)
from optimagic.parameters.bounds import Bounds


def test_scalars_as_base_steps():
    steps_scalar = _calculate_or_validate_base_steps(
        0.1, np.ones(3), "first_derivative", None, scaling_factor=1
    )

    steps_array = _calculate_or_validate_base_steps(
        np.full(3, 0.1), np.ones(3), "first_derivative", None, scaling_factor=1
    )

    aaae(steps_scalar, steps_array)


def test_scalars_as_min_steps():
    steps_scalar = _calculate_or_validate_base_steps(
        0.1, np.ones(3), "first_derivative", 0.12, scaling_factor=1.5
    )

    steps_array = _calculate_or_validate_base_steps(
        np.full(3, 0.1),
        np.ones(3),
        "first_derivative",
        np.full(3, 0.12),
        scaling_factor=1.5,
    )

    aaae(steps_scalar, steps_array)


def test_calculate_or_validate_base_steps_invalid_too_small():
    base_steps = np.array([1e-10, 0.01, 0.01])
    min_steps = np.full(3, 1e-8)
    with pytest.raises(ValueError):
        _calculate_or_validate_base_steps(
            base_steps, np.ones(3), "first_derivative", min_steps, scaling_factor=1
        )


def test_calculate_or_validate_base_steps_wrong_shape():
    base_steps = np.array([0.01, 0.01, 0.01])
    min_steps = np.full(3, 1e-8)
    with pytest.raises(ValueError):
        _calculate_or_validate_base_steps(
            base_steps, np.ones(2), "first_derivative", min_steps, scaling_factor=1
        )


def test_calculate_or_validate_base_steps_jacobian():
    x = np.array([0.05, 1, -5])
    expected = np.array([0.1, 1, 5]) * np.sqrt(np.finfo(float).eps)
    calculated = _calculate_or_validate_base_steps(
        None, x, "first_derivative", 0, scaling_factor=1.0
    )
    aaae(calculated, expected, decimal=12)


def test_calculate_or_validate_base_steps_jacobian_with_scaling_factor():
    x = np.array([0.05, 1, -5])
    expected = np.array([0.1, 1, 5]) * np.sqrt(np.finfo(float).eps) * 2
    calculated = _calculate_or_validate_base_steps(
        None, x, "first_derivative", 0, scaling_factor=2.0
    )
    aaae(calculated, expected, decimal=12)


def test_calculate_or_validate_base_steps_binding_min_step():
    x = np.array([0.05, 1, -5])
    expected = np.array([0.1, 1, 5]) * np.sqrt(np.finfo(float).eps)
    expected[0] = 1e-8
    calculated = _calculate_or_validate_base_steps(
        None, x, "first_derivative", 1e-8, scaling_factor=1.0
    )
    aaae(calculated, expected, decimal=12)


def test_calculate_or_validate_base_steps_hessian():
    x = np.array([0.05, 1, -5])
    expected = np.array([0.1, 1, 5]) * np.finfo(float).eps ** (1 / 3)
    calculated = _calculate_or_validate_base_steps(
        None, x, "second_derivative", 0, scaling_factor=1.0
    )
    aaae(calculated, expected, decimal=12)


def test_set_unused_side_to_nan_forward():
    pos = np.ones((3, 2))
    neg = -np.ones((3, 2))
    method = "forward"
    x = np.zeros(3)
    upper_bounds = np.array([0.5, 2, 3])
    lower_bounds = np.array([-2, -0.1, -0.1])

    expected_pos = np.array([[np.nan, np.nan], [1, 1], [1, 1]])
    expected_neg = np.array([[-1, -1], [np.nan, np.nan], [np.nan, np.nan]])

    calculated_pos, calculated_neg = _set_unused_side_to_nan(
        x, pos, neg, method, lower_bounds, upper_bounds
    )

    assert np.allclose(calculated_pos, expected_pos, equal_nan=True)
    assert np.allclose(calculated_neg, expected_neg, equal_nan=True)


def test_set_unused_side_to_nan_backward():
    pos = np.ones((3, 2))
    neg = -np.ones((3, 2))
    method = "backward"
    x = np.zeros(3)
    upper_bounds = np.array([0.5, 2, 3])
    lower_bounds = np.array([-2, -0.1, -2])

    expected_pos = np.array([[np.nan, np.nan], [1, 1], [np.nan, np.nan]])
    expected_neg = np.array([[-1, -1], [np.nan, np.nan], [-1, -1]])

    calculated_pos, calculated_neg = _set_unused_side_to_nan(
        x, pos, neg, method, lower_bounds, upper_bounds
    )

    assert np.allclose(calculated_pos, expected_pos, equal_nan=True)
    assert np.allclose(calculated_neg, expected_neg, equal_nan=True)


def test_fillna():
    a = np.array([np.nan, 3, 4])
    assert np.allclose(_fillna(a, 0), np.array([0, 3, 4.0]))


def test_rescale_to_accomodate_bounds():
    pos = np.array([[1, 2], [1.5, 3], [1, 2], [3, np.nan]])
    neg = -pos
    base_steps = np.array([1, 1.5, 2, 3])
    min_step = 0.1
    lower_bounds = -4 * np.ones(4)
    upper_bounds = np.ones(4) * 2.5

    expected_pos = np.array([[1, 2], [1.25, 2.5], [1, 2], [2.5, np.nan]])
    expected_neg = -expected_pos

    calculated_pos, calculated_neg = _rescale_to_accomodate_bounds(
        base_steps, pos, neg, lower_bounds, upper_bounds, min_step
    )

    np.allclose(calculated_pos, expected_pos, equal_nan=True)
    np.allclose(calculated_neg, expected_neg, equal_nan=True)


def test_rescale_to_accomodate_bounds_binding_min_step():
    pos = np.array([[1, 2], [1.5, 3], [1, 2]])
    neg = -pos
    base_steps = np.array([1, 1.5, 2])
    min_step = np.array([0, 1.4, 0])
    lower_bounds = -4 * np.ones(3)
    upper_bounds = np.ones(3) * 2.5

    expected_pos = np.array([[1, 2], [1.4, 2.8], [1, 2]])
    expected_neg = -expected_pos

    calculated_pos, calculated_neg = _rescale_to_accomodate_bounds(
        base_steps, pos, neg, lower_bounds, upper_bounds, min_step
    )

    aaae(calculated_pos, expected_pos)
    aaae(calculated_neg, expected_neg)


def test_generate_steps_binding_min_step():
    calculated_steps = generate_steps(
        x=np.arange(3),
        method="central",
        n_steps=2,
        target="first_derivative",
        base_steps=np.array([0.1, 0.2, 0.3]),
        bounds=Bounds(lower=np.full(3, -np.inf), upper=np.full(3, 2.5)),
        step_ratio=2.0,
        min_steps=np.full(3, 1e-8),
        scaling_factor=1.0,
    )

    expected_pos = np.array([[0.1, 0.2], [0.2, 0.4], [0.25, 0.5]]).T
    expected_neg = -expected_pos

    aaae(calculated_steps.pos, expected_pos)
    aaae(calculated_steps.neg, expected_neg)


def test_generate_steps_min_step_equals_base_step():
    calculated_steps = generate_steps(
        x=np.arange(3),
        method="central",
        n_steps=2,
        target="first_derivative",
        base_steps=np.array([0.1, 0.2, 0.3]),
        bounds=Bounds(lower=np.full(3, -np.inf), upper=np.full(3, 2.5)),
        step_ratio=2.0,
        min_steps=None,
        scaling_factor=1.0,
    )

    expected_pos = np.array([[0.1, 0.2], [0.2, 0.4], [0.3, np.nan]]).T
    expected_neg = np.array([[-0.1, -0.2], [-0.2, -0.4], [-0.3, -0.6]]).T
    aaae(calculated_steps.pos, expected_pos)
    aaae(calculated_steps.neg, expected_neg)
