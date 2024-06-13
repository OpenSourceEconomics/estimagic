import numpy as np
from estimagic.parameters.check_constraints import _iloc


def test_iloc():
    dictionary = {
        "index": ["a", "b", "c"],
        "lower_bounds": [0, 0, 0],
        "upper_bounds": [1, 1, 1],
        "is_fixed_to_value": [False, False, True],
    }
    position = [0, 2]
    expected_result = {
        "lower_bounds": np.array([0, 0]),
        "upper_bounds": np.array([1, 1]),
        "is_fixed_to_value": np.array([False, True]),
    }
    result = _iloc(dictionary, position)
    assert len(result) == len(expected_result)
    for k, v in expected_result.items():
        assert k in result
        assert np.array_equal(result[k], v)
