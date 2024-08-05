import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.optimization.history_tools import get_history_arrays


@pytest.fixture()
def history():
    hist = {
        "criterion": [5, 4, 5.5, 4.2],
        "params": [{"a": 0}, {"a": 1}, {"a": 2}, {"a": 3}],
        "runtime": [0, 1, 2, 3],
    }
    return hist


def test_get_history_arrays_minimize(history):
    calculated = get_history_arrays(history, "minimize")
    for val in calculated.values():
        assert isinstance(val, np.ndarray)

    aaae(calculated["is_accepted"], np.array([True, True, False, False]))


def test_get_history_arrays_maximize(history):
    calculated = get_history_arrays(history, "maximize")
    for val in calculated.values():
        assert isinstance(val, np.ndarray)

    aaae(calculated["is_accepted"], np.array([True, False, True, False]))
