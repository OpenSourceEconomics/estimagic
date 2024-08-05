import numpy as np
import pytest
from optimagic.benchmarking.more_wild import (
    MORE_WILD_PROBLEMS,
    get_start_points_mancino,
)


@pytest.mark.parametrize("name, specification", list(MORE_WILD_PROBLEMS.items()))
def test_more_wild_function_at_start_x(name, specification):  # noqa: ARG001
    _criterion = specification["fun"]
    assert isinstance(specification["start_x"], list)
    _x = np.array(specification["start_x"])
    _contributions = _criterion(_x)
    calculated = _contributions @ _contributions
    expected = specification["start_criterion"]
    assert np.allclose(calculated, expected)

    if specification.get("solution_x") is not None:
        assert isinstance(specification["solution_x"], list)
        _x = np.array(specification["solution_x"])
        _contributions = _criterion(_x)
        calculated = _contributions @ _contributions
        expected = specification["solution_criterion"]
        assert np.allclose(calculated, expected, rtol=1e-8, atol=1e-8)


def test_get_start_points_mancino():
    expected = (np.array([102.4824, 96.3335, 90.4363, 84.7852, 79.3747]),)
    result = get_start_points_mancino(5)
    assert np.allclose(expected, result)
