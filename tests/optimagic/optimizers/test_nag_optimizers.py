import pytest
from optimagic.optimizers.nag_optimizers import (
    _build_options_dict,
    _change_evals_per_point_interface,
    _get_fast_start_method,
)


def test_change_evals_per_point_interface_none():
    res = _change_evals_per_point_interface(None)
    assert res is None


def test_change_evals_per_point_interface_func():
    def return_args(
        upper_trustregion_radius, lower_trustregion_radius, n_iterations, n_resets
    ):
        return (
            upper_trustregion_radius,
            lower_trustregion_radius,
            n_iterations,
            n_resets,
        )

    func = _change_evals_per_point_interface(return_args)
    res = func(delta=0, rho=1, iter=2, nrestarts=3)
    expected = (0, 1, 2, 3)
    assert res == expected


def test_get_fast_start_method_auto():
    res = _get_fast_start_method("auto")
    assert res == (None, None)


def test_get_fast_start_method_jacobian():
    res = _get_fast_start_method("jacobian")
    assert res == (True, False)


def test_get_fast_start_method_trust():
    res = _get_fast_start_method("trustregion")
    assert res == (False, True)


def test_get_fast_start_method_error():
    with pytest.raises(ValueError):
        _get_fast_start_method("wrong_input")


def test_build_options_dict_none():
    default = {"a": 1, "b": 2}
    assert default == _build_options_dict(None, default)


def test_build_options_dict_override():
    default = {"a": 1, "b": 2}
    user_input = {"a": 0}
    res = _build_options_dict(user_input, default)
    expected = {"a": 0, "b": 2}
    assert res == expected


def test_build_options_dict_invalid_key():
    default = {"a": 1, "b": 2}
    user_input = {"other_key": 0}
    with pytest.raises(ValueError):
        _build_options_dict(user_input, default)
