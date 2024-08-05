import numpy as np
import pandas as pd
import pytest
from estimagic.bootstrap_samples import (
    _convert_cluster_ids_to_indices,
    _get_bootstrap_samples_from_indices,
    get_bootstrap_indices,
    get_bootstrap_samples,
)
from numpy.testing import assert_array_equal as aae
from optimagic.utilities import get_rng
from pandas.testing import assert_frame_equal as afe


@pytest.fixture()
def data():
    df = pd.DataFrame()
    df["id"] = np.arange(900)
    df["hh"] = [3, 1, 2, 0, 0, 2, 5, 4, 5] * 100
    return df


def test_get_bootstrap_indices_randomization_works_without_clustering(data):
    rng = get_rng(seed=12345)
    res = get_bootstrap_indices(data, n_draws=2, rng=rng)
    assert set(res[0]) != set(res[1])


def test_get_bootstrap_indices_radomization_works_with_clustering(data):
    rng = get_rng(seed=12345)
    res = get_bootstrap_indices(data, cluster_by="hh", n_draws=2, rng=rng)
    assert set(res[0]) != set(res[1])


def test_clustering_leaves_households_intact(data):
    rng = get_rng(seed=12345)
    indices = get_bootstrap_indices(data, cluster_by="hh", n_draws=1, rng=rng)[0]
    sampled = data.iloc[indices]
    sampled_households = sampled["hh"].unique()
    for household in sampled_households:
        expected_ids = set(data[data["hh"] == household]["id"].unique())
        actual_ids = set(sampled[sampled["hh"] == household]["id"].unique())
        assert expected_ids == actual_ids


def test_convert_cluster_ids_to_indices():
    cluster_col = pd.Series([2, 2, 0, 1, 0, 1])
    drawn_clusters = np.array([[1, 0]])
    expected = np.array([3, 5, 2, 4])
    calculated = _convert_cluster_ids_to_indices(cluster_col, drawn_clusters)[0]
    aae(calculated, expected)


def test_get_bootstrap_samples_from_indices():
    indices = [np.array([0, 1])]
    data = pd.DataFrame(np.arange(6).reshape(3, 2))
    expected = pd.DataFrame(np.arange(4).reshape(2, 2))
    calculated = _get_bootstrap_samples_from_indices(data, indices)[0]
    afe(calculated, expected)


def test_get_bootstrap_samples_runs(data):
    rng = get_rng(seed=12345)
    get_bootstrap_samples(data, n_draws=2, rng=rng)
