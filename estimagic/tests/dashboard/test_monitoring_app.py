"""Test the functions of the monitoring app."""
import webbrowser
from pathlib import Path

import pandas as pd
import pytest
from bokeh.document import Document
from bokeh.io import output_file
from bokeh.io import save
from bokeh.models import ColumnDataSource

import estimagic.dashboard.monitoring_app as monitoring
from estimagic.logging.create_database import load_database


@pytest.fixture()
def database():
    database_name = "db1.db"
    current_dir_path = Path(__file__).resolve().parent
    database_path = current_dir_path / database_name
    database = load_database(database_path)
    return database


def test_monitoring_app():
    """Integration test that no Error is raised when calling the monitoring app."""
    doc = Document()
    database_name = "test_db"
    current_dir_path = Path(__file__).resolve().parent
    session_data = {"last_retrieved": 0, "database_path": current_dir_path / "db1.db"}

    monitoring.monitoring_app(
        doc=doc, database_name=database_name, session_data=session_data
    )


def test_create_bokeh_data_sources(database):
    tables = ["criterion_history", "params_history"]
    criterion_history, params_history = monitoring._create_bokeh_data_sources(
        database=database, tables=tables
    )
    assert criterion_history.data == {"iteration": [1], "value": [426.5586492569206]}
    assert params_history.data == {
        "iteration": [1],
        "beta_pared": [0.47738201898674737],
        "beta_public": [0.22650218067445926],
        "beta_gpa": [-0.46745804687921866],
        "cutoff_0": [0.0],
        "cutoff_1": [2.0],
    }


# skip test create_initial_convergence_plots


def test_plot_time_series_with_large_initial_values():
    cds = ColumnDataSource({"y": [2e17, 1e16, 1e5], "x": [1, 2, 3]})
    title = "Are large initial values shown?"
    fig = monitoring._plot_time_series(data=cds, y_keys=["y"], x_name="x", title=title)
    title = "Test _plot_time_series can handle large initial values."
    output_file("time_series_initial_value.html", title=title)
    path = save(obj=fig)
    webbrowser.open_new_tab("file://" + path)


def test_map_groups_to_params_group_none():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = None
    params["name"] = ["a", "b", "c", "d"]
    params.index = ["a", "b", "c", "d"]
    expected = {}
    res = monitoring._map_groups_to_params(params)
    assert expected == res


def test_map_groups_to_params_group_not_none():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = [None, "A", "B", "B"]
    params.index = ["a", "b", "c", "d"]
    params["name"] = ["a", "b", "c", "d"]
    expected = {"A": ["b"], "B": ["c", "d"]}
    res = monitoring._map_groups_to_params(params)
    assert expected == res


def test_map_groups_to_params_group_int_index():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params.index = ["0", "1", "2", "3"]
    params["name"] = ["0", "1", "2", "3"]
    params["group"] = [None, "A", "B", "B"]
    expected = {"A": ["1"], "B": ["2", "3"]}
    res = monitoring._map_groups_to_params(params)
    assert expected == res


def test_map_groups_to_params_group_multi_index():
    params = pd.DataFrame()
    params["value"] = [0, 1, 2, 3]
    params["group"] = [None, "A", "B", "B"]
    params["ind1"] = ["beta", "beta", "cutoff", "cutoff"]
    params["ind2"] = ["edu", "exp", 1, 2]
    params.set_index(["ind1", "ind2"], inplace=True)
    params["name"] = ["beta_edu", "beta_exp", "cutoff_1", "cutoff_2"]
    expected = {"A": ["beta_exp"], "B": ["cutoff_1", "cutoff_2"]}
    res = monitoring._map_groups_to_params(params)
    assert expected == res
