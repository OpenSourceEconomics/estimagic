import numpy as np
import pytest
from optimagic.exceptions import InvalidMultistartError
from optimagic.optimization.multistart_options import (
    MultistartOptions,
    _linear_weights,
    _tiktak_weights,
    get_internal_multistart_options_from_public,
    pre_process_multistart,
)


def test_pre_process_multistart_trivial_case():
    multistart = MultistartOptions(n_samples=10, convergence_max_discoveries=55)
    got = pre_process_multistart(multistart)
    assert got == multistart


def test_pre_process_multistart_none_case():
    assert pre_process_multistart(None) is None


def test_pre_process_multistart_false_case():
    assert pre_process_multistart(False) is None


def test_pre_process_multistart_dict_case():
    got = pre_process_multistart(
        multistart={
            "n_samples": 10,
            "convergence_max_discoveries": 55,
        }
    )
    assert got == MultistartOptions(
        n_samples=10,
        convergence_max_discoveries=55,
    )


def test_pre_process_multistart_invalid_type():
    with pytest.raises(InvalidMultistartError, match="Invalid multistart options"):
        pre_process_multistart(multistart="invalid")


def test_pre_process_multistart_invalid_dict_key():
    with pytest.raises(InvalidMultistartError, match="Invalid multistart options"):
        pre_process_multistart(multistart={"invalid": "invalid"})


def test_pre_process_multistart_invalid_dict_value():
    with pytest.raises(InvalidMultistartError, match="Invalid number of samples"):
        pre_process_multistart(multistart={"n_samples": "invalid"})


@pytest.mark.parametrize("value", ["invalid", -1])
def test_multistart_options_invalid_n_samples_value(value):
    with pytest.raises(InvalidMultistartError, match="Invalid number of samples"):
        MultistartOptions(n_samples=value)


@pytest.mark.parametrize("value", ["invalid", -1])
def test_multistart_options_invalid_stopping_maxopt(value):
    with pytest.raises(InvalidMultistartError, match="Invalid number of optimizations"):
        MultistartOptions(stopping_maxopt=value)


def test_multistart_options_stopping_maxopt_less_than_n_samples():
    with pytest.raises(InvalidMultistartError, match="Invalid number of samples"):
        MultistartOptions(n_samples=1, stopping_maxopt=2)


def test_multistart_options_invalid_sampling_distribution():
    with pytest.raises(InvalidMultistartError, match="Invalid sampling distribution"):
        MultistartOptions(sampling_distribution="invalid")


def test_multistart_options_invalid_sampling_method():
    with pytest.raises(InvalidMultistartError, match="Invalid sampling method"):
        MultistartOptions(sampling_method="invalid")


def test_multistart_options_invalid_mixing_weight_method():
    with pytest.raises(InvalidMultistartError, match="Invalid mixing weight method"):
        MultistartOptions(mixing_weight_method="invalid")


@pytest.mark.parametrize("value", [("a", "b"), (1, 2, 3), {"a": 1.0, "b": 3.0}])
def test_multistart_options_invalid_mixing_weight_bounds(value):
    with pytest.raises(InvalidMultistartError, match="Invalid mixing weight bounds"):
        MultistartOptions(mixing_weight_bounds=value)


def test_multistart_options_invalid_convergence_xtol_rel():
    with pytest.raises(InvalidMultistartError, match="Invalid relative params"):
        MultistartOptions(convergence_xtol_rel="invalid")


@pytest.mark.parametrize("value", ["invalid", -1])
def test_multistart_options_invalid_convergence_max_discoveries(value):
    with pytest.raises(InvalidMultistartError, match="Invalid max discoveries"):
        MultistartOptions(convergence_max_discoveries=value)


@pytest.mark.parametrize("value", ["invalid", -1])
def test_multistart_options_invalid_n_cores(value):
    with pytest.raises(InvalidMultistartError, match="Invalid number of cores"):
        MultistartOptions(n_cores=value)


@pytest.mark.parametrize("value", ["invalid", -1])
def test_multistart_options_invalid_batch_size(value):
    with pytest.raises(InvalidMultistartError, match="Invalid batch size"):
        MultistartOptions(batch_size=value)


def test_multistart_options_batch_size_smaller_than_n_cores():
    with pytest.raises(InvalidMultistartError, match="Invalid batch size"):
        MultistartOptions(batch_size=1, n_cores=2)


def test_multistart_options_invalid_batch_evaluator():
    with pytest.raises(InvalidMultistartError, match="Invalid batch evaluator"):
        MultistartOptions(batch_evaluator="invalid")


def test_multistart_options_invalid_seed():
    with pytest.raises(InvalidMultistartError, match="Invalid seed"):
        MultistartOptions(seed="invalid")


def test_multistart_options_invalid_error_handling():
    with pytest.raises(InvalidMultistartError, match="Invalid error handling"):
        MultistartOptions(error_handling="invalid")


def test_linear_weights():
    calculated = _linear_weights(5, 10, 0.4, 0.8)
    expected = 0.6
    assert np.allclose(calculated, expected)


def test_tiktak_weights():
    assert np.allclose(0.3, _tiktak_weights(0, 10, 0.3, 0.8))
    assert np.allclose(0.8, _tiktak_weights(10, 10, 0.3, 0.8))


def test_get_internal_multistart_options_from_public_defaults():
    options = MultistartOptions()

    got = get_internal_multistart_options_from_public(
        options,
        params=np.arange(5),
        params_to_internal=lambda x: x,
    )

    assert got.convergence_xtol_rel == 0.01
    assert got.convergence_max_discoveries == options.convergence_max_discoveries
    assert got.n_cores == options.n_cores
    assert got.error_handling == "continue"
    assert got.n_samples == 500
    assert got.stopping_maxopt == 50
    assert got.batch_size == 1
