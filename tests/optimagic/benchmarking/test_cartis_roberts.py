import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from optimagic.benchmarking.cartis_roberts import (
    CARTIS_ROBERTS_PROBLEMS,
    get_start_points_bdvalues,
    get_start_points_msqrta,
)


@pytest.mark.parametrize("name, specification", list(CARTIS_ROBERTS_PROBLEMS.items()))
def test_cartis_roberts_function_at_start_x(name, specification):  # noqa: ARG001
    _criterion = specification["fun"]
    _x = np.array(specification["start_x"])
    assert isinstance(specification["start_x"], list)
    _contributions = _criterion(_x)
    calculated = _contributions @ _contributions
    expected = specification["start_criterion"]
    assert np.allclose(calculated, expected)
    assert isinstance(specification["start_x"], list)


@pytest.mark.parametrize("name, specification", list(CARTIS_ROBERTS_PROBLEMS.items()))
def test_cartis_roberts_function_at_solution_x(name, specification):  # noqa: ARG001
    _criterion = specification["fun"]
    _x = specification["solution_x"]
    if _x is not None:
        assert isinstance(_x, list)
        _x = np.array(_x)
        _contributions = _criterion(_x)
        calculated = _contributions @ _contributions
        expected = specification["solution_criterion"]
        assert np.allclose(calculated, expected, atol=1e-7)


def test_get_start_points_bdvalues():
    expected = np.array([-0.1389, -0.2222, -0.2500, -0.2222, -0.1389])
    result = get_start_points_bdvalues(5)
    assert_array_almost_equal(expected, result, decimal=4)


def test_get_start_points_msqrta():
    matlab_mat = np.array(
        [
            [0.8415, -0.7568, 0.4121, -0.2879, -0.1324],
            [-0.9918, -0.9538, 0.9200, -0.6299, -0.5064],
            [0.9988, -0.4910, -0.6020, 0.9395, -0.9301],
            [-0.9992, -0.0265, -0.4041, 0.2794, -0.8509],
            [0.9235, 0.1935, 0.9365, -0.8860, 0.1760],
        ]
    )
    expected = 0.2 * matlab_mat.flatten()
    result = get_start_points_msqrta(5)
    assert_array_almost_equal(result, expected, decimal=4)
