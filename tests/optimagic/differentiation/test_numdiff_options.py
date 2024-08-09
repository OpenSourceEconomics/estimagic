import pytest
from optimagic.differentiation.numdiff_options import (
    NumdiffOptions,
    pre_process_numdiff_options,
)
from optimagic.exceptions import InvalidNumdiffOptionsError


def test_pre_process_numdiff_options_trivial_case():
    numdiff_options = NumdiffOptions(
        method="central",
        step_size=0.1,
        scaling_factor=0.5,
        min_steps=None,
        batch_evaluator="joblib",
    )
    got = pre_process_numdiff_options(numdiff_options)
    assert got == numdiff_options


def test_pre_process_numdiff_options_none_case():
    assert pre_process_numdiff_options(None) is None


def test_pre_process_numdiff_options_dict_case():
    got = pre_process_numdiff_options(
        {"method": "central", "step_size": 0.1, "batch_evaluator": "pathos"}
    )
    assert got == NumdiffOptions(
        method="central", step_size=0.1, batch_evaluator="pathos"
    )


def test_pre_process_numdiff_options_invalid_type():
    with pytest.raises(InvalidNumdiffOptionsError):
        pre_process_numdiff_options(numdiff_options="invalid")


def test_pre_process_numdiff_options_invalid_dict_key():
    with pytest.raises(InvalidNumdiffOptionsError, match="Invalid numdiff options"):
        pre_process_numdiff_options(numdiff_options={"wrong_key": "central"})


def test_pre_process_numdiff_options_invalid_dict_value():
    with pytest.raises(InvalidNumdiffOptionsError, match="Invalid numdiff `method`:"):
        pre_process_numdiff_options(numdiff_options={"method": "invalid"})


def test_numdiff_options_invalid_method():
    with pytest.raises(InvalidNumdiffOptionsError, match="Invalid numdiff `method`:"):
        NumdiffOptions(method="invalid")


def test_numdiff_options_invalid_step_size():
    with pytest.raises(
        InvalidNumdiffOptionsError, match="Invalid numdiff `step_size`:"
    ):
        NumdiffOptions(step_size=0)


def test_numdiff_options_invalid_scaling_factor():
    with pytest.raises(
        InvalidNumdiffOptionsError, match="Invalid numdiff `scaling_factor`:"
    ):
        NumdiffOptions(scaling_factor=-1)


def test_numdiff_options_invalid_min_steps():
    with pytest.raises(
        InvalidNumdiffOptionsError, match="Invalid numdiff `min_steps`:"
    ):
        NumdiffOptions(min_steps=-1)


def test_numdiff_options_invalid_n_cores():
    with pytest.raises(InvalidNumdiffOptionsError, match="Invalid numdiff `n_cores`:"):
        NumdiffOptions(n_cores=-1)


def test_numdiff_options_invalid_batch_evaluator():
    with pytest.raises(
        InvalidNumdiffOptionsError, match="Invalid numdiff `batch_evaluator`:"
    ):
        NumdiffOptions(batch_evaluator="invalid")
