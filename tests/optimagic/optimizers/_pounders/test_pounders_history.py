"""Test the history class for least-squares optimizers."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.optimizers._pounders.pounders_history import LeastSquaresHistory

ENTRIES = [
    (np.arange(3), [np.arange(5)]),
    ([np.arange(3)], list(range(5))),
    (np.arange(3).reshape(1, 3), np.arange(5).reshape(1, 5)),
]

TEST_CASES = []
for entries in ENTRIES:
    for is_center in True, False:
        TEST_CASES.append((entries, is_center))


@pytest.mark.parametrize("entries, is_center", TEST_CASES)
def test_add_entries_not_initialized(entries, is_center):
    history = LeastSquaresHistory()

    if is_center:
        c_info = {"x": np.zeros(3), "residuals": np.zeros(5), "radius": 1}
        history.add_centered_entries(*entries, c_info)
    else:
        history.add_entries(*entries)

    xs, residuals, critvals = history.get_entries()
    xs_sinlge = history.get_xs()
    residuals_sinlge = history.get_residuals()
    critvals_sinlge = history.get_critvals()

    for entry in xs, residuals, critvals:
        assert isinstance(entry, np.ndarray)

    aaae(xs, np.arange(3).reshape(1, 3))
    aaae(xs_sinlge, np.arange(3).reshape(1, 3))
    aaae(residuals, np.arange(5).reshape(1, 5))
    aaae(residuals_sinlge, np.arange(5).reshape(1, 5))
    aaae(critvals, np.array([30.0]))
    aaae(critvals_sinlge, np.array([30.0]))


@pytest.mark.parametrize("entries, is_center", TEST_CASES)
def test_add_entries_initialized_with_space(entries, is_center):
    history = LeastSquaresHistory()
    history.add_entries(np.ones((4, 3)), np.zeros((4, 5)))

    if is_center:
        c_info = {"x": np.zeros(3), "residuals": np.zeros(5), "radius": 1}
        history.add_centered_entries(*entries, c_info)
    else:
        history.add_entries(*entries)

    xs, residuals, critvals = history.get_entries(index=-1)
    xs_sinlge = history.get_xs(index=-1)
    residuals_sinlge = history.get_residuals(index=-1)
    critvals_sinlge = history.get_critvals(index=-1)

    for entry in xs, residuals:
        assert isinstance(entry, np.ndarray)

    aaae(xs, np.arange(3))
    aaae(xs_sinlge, np.arange(3))
    aaae(residuals, np.arange(5))
    aaae(residuals_sinlge, np.arange(5))
    assert critvals == 30
    assert critvals_sinlge == 30


def test_add_entries_initialized_extension_needed():
    history = LeastSquaresHistory()
    history.add_entries(np.ones((4, 3)), np.zeros((4, 5)))
    history.xs = history.xs[:5]
    history.residuals = history.residuals[:5]
    history.critvals = history.critvals[:5]

    history.add_entries(np.arange(12).reshape(4, 3), np.arange(20).reshape(4, 5))

    assert len(history.xs) == 10
    assert len(history.residuals) == 10
    assert len(history.critvals) == 10

    xs, residuals, _ = history.get_entries(index=-1)
    xs_sinlge = history.get_xs(index=-1)
    residuals_sinlge = history.get_residuals(index=-1)

    for entry in xs, xs_sinlge, residuals, residuals_sinlge:
        assert isinstance(entry, np.ndarray)

    assert history.get_n_fun() == 8


def test_add_centered_entries():
    history = LeastSquaresHistory()
    history.add_entries(np.ones((2, 2)), np.ones((2, 4)))
    center_info = {
        "x": history.get_xs(index=-1),
        "residuals": history.get_residuals(index=-1),
        "radius": 0.5,
    }
    history.add_centered_entries(
        xs=np.ones(2), residuals=np.ones(4) * 2, center_info=center_info
    )

    xs, residuals, critvals = history.get_entries(index=-1)

    aaae(xs, np.array([1.5, 1.5]))
    aaae(residuals, np.array([3, 3, 3, 3]))
    assert critvals == 36
    assert history.get_n_fun() == 3


def test_get_centered_entries():
    history = LeastSquaresHistory()
    history.add_entries(np.ones((4, 3)), np.ones((4, 5)))
    center_info = {
        "x": np.arange(3),
        "residuals": np.arange(5),
        "radius": 0.25,
    }

    xs, residuals, critvals = history.get_centered_entries(
        center_info=center_info, index=-1
    )

    aaae(xs, np.array([4, 0, -4]))
    aaae(residuals, np.arange(1, -4, -1))
    assert critvals == 15
    assert history.get_n_fun() == 4
