import numpy as np
import pytest
from estimagic.benchmarking.scalar_functions import SCALAR_FUNCTIONS_EXTRA_PROBLEMS
from estimagic.benchmarking.scalar_functions import SCALAR_FUNCTIONS_PROBLEMS


@pytest.mark.parametrize(
    "name, specification",
    [
        *list(SCALAR_FUNCTIONS_PROBLEMS.items()),
        *list(SCALAR_FUNCTIONS_EXTRA_PROBLEMS.items()),
    ],
)
def test_scalar_function_at_start_x(name, specification):
    _criterion = specification["criterion"]
    _x = specification["start_x"]
    calculated = _criterion(_x)
    expected = specification["start_criterion"]
    assert np.allclose(calculated, expected)
    if specification.get("solution_x") is not None:
        _x = specification["solution_x"]
        calculated = _criterion(_x)
        expected = specification["solution_criterion"]
        assert np.allclose(calculated, expected, rtol=1e-8, atol=1e-8)
        assert len(specification["start_x"]) == len(specification["solution_x"])
