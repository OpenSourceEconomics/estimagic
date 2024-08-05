import pickle

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from optimagic.logging.create_tables import (
    make_optimization_iteration_table,
    make_optimization_problem_table,
    make_steps_table,
)
from optimagic.logging.load_database import DataBase, load_database
from optimagic.logging.read_from_database import (
    read_last_rows,
    read_new_rows,
    read_table,
)
from optimagic.logging.write_to_database import append_row, update_row


@pytest.fixture()
def iteration_data():
    data = {
        "params": np.ones(1),
        "timestamp": 0.5,
        "value": 5.0,
    }
    return data


@pytest.fixture()
def problem_data():
    data = {
        "direction": "maximize",
        "params": np.arange(3),
        "algorithm": "bla",
        "constraints": [{"type": "bla"}],
        "algo_options": None,
        "numdiff_options": {},
        "log_options": {"fast_logging": False},
    }
    return data


def test_load_database_from_path(tmp_path):
    """Test that database is generated because it does not exist."""
    path = tmp_path / "test.db"
    database = load_database(path_or_database=path, fast_logging=False)
    assert isinstance(database, DataBase)
    assert database.path is not None
    assert database.fast_logging is False


def test_load_database_after_pickling(tmp_path):
    """Pickling unsets database.bind.

    Test that load_database sets it again.

    """
    path = tmp_path / "test.db"
    database = load_database(path_or_database=path, fast_logging=False)
    database = pickle.loads(pickle.dumps(database))
    assert hasattr(database.engine, "connect")


def test_optimization_iteration_table_scalar(tmp_path, iteration_data):
    path = tmp_path / "test.db"
    database = load_database(path_or_database=path)
    make_optimization_iteration_table(database)
    append_row(iteration_data, "optimization_iterations", database)
    res = read_last_rows(database, "optimization_iterations", 1, "list_of_dicts")
    assert isinstance(res, list)
    assert isinstance(res[0], dict)
    res = res[0]
    assert res["rowid"] == 1
    assert_array_equal(res["params"], iteration_data["params"])

    for key in ["value", "timestamp"]:
        assert res[key] == iteration_data[key]


def test_steps_table(tmp_path):
    path = tmp_path / "test.db"
    database = load_database(path_or_database=path)
    make_steps_table(database)
    for status in ["scheduled", "running", "completed"]:
        append_row(
            {
                "status": status,
                "n_iterations": 0,
                "type": "optimization",
                "name": "bla",
            },
            "steps",
            database,
        )

    res, _ = read_new_rows(database, "steps", 1, "dict_of_lists")

    expected = {
        "rowid": [2, 3],
        "status": ["running", "completed"],
        "type": ["optimization", "optimization"],
        "name": ["bla", "bla"],
        "n_iterations": [0, 0],
    }
    assert res == expected


def test_optimization_problem_table(tmp_path, problem_data):
    path = tmp_path / "test.db"
    database = load_database(path_or_database=path)
    make_optimization_problem_table(database)
    append_row(problem_data, "optimization_problem", database)
    res = read_last_rows(database, "optimization_problem", 1, "list_of_dicts")[0]
    assert res["rowid"] == 1
    for key, expected in problem_data.items():
        if key == "criterion":
            assert res[key](np.ones(3)) == 3
        elif isinstance(expected, np.ndarray):
            assert_array_equal(res[key], expected)
        else:
            assert res[key] == expected


def test_read_new_rows_stride(tmp_path, iteration_data):
    path = tmp_path / "test.db"
    database = load_database(path_or_database=path)
    make_optimization_iteration_table(database)
    for i in range(1, 11):  # sqlalchemy starts counting at 1
        iteration_data["value"] = i
        append_row(iteration_data, "optimization_iterations", database)

    res = read_new_rows(
        database=database,
        table_name="optimization_iterations",
        last_retrieved=1,
        return_type="dict_of_lists",
        stride=2,
    )[0]["value"]

    expected = [2.0, 4.0, 6.0, 8.0, 10.0]
    assert res == expected


def test_update_row(tmp_path, iteration_data):
    path = tmp_path / "test.db"
    database = load_database(path_or_database=path)
    make_optimization_iteration_table(database)
    for i in range(1, 11):  # sqlalchemy starts counting at 1
        iteration_data["value"] = i
        append_row(iteration_data, "optimization_iterations", database)

    update_row({"value": 20}, 8, "optimization_iterations", database)

    res = read_new_rows(
        database=database,
        table_name="optimization_iterations",
        last_retrieved=3,
        return_type="dict_of_lists",
    )[0]["value"]

    expected = [4, 5, 6, 7, 20, 9, 10]
    assert res == expected


def test_read_last_rows_stride(tmp_path, iteration_data):
    path = tmp_path / "test.db"
    database = load_database(path_or_database=path)
    make_optimization_iteration_table(database)
    for i in range(1, 11):  # sqlalchemy starts counting at 1
        iteration_data["value"] = i
        append_row(iteration_data, "optimization_iterations", database)

    res = read_last_rows(
        database=database,
        table_name="optimization_iterations",
        n_rows=3,
        return_type="dict_of_lists",
        stride=2,
    )["value"]

    expected = [6.0, 8.0, 10.0]
    assert res == expected


def test_read_new_rows_with_step(tmp_path, iteration_data):
    path = tmp_path / "test.db"
    database = load_database(path_or_database=path)
    make_optimization_iteration_table(database)
    for i in range(1, 11):  # sqlalchemy starts counting at 1
        iteration_data["value"] = i
        iteration_data["step"] = i % 2
        append_row(iteration_data, "optimization_iterations", database)

    res, _ = read_new_rows(
        database=database,
        table_name="optimization_iterations",
        last_retrieved=0,
        return_type="dict_of_lists",
        step=0,
    )

    expected = [2, 4, 6, 8, 10]
    assert res["rowid"] == expected


def test_read_last_rows_with_step(tmp_path, iteration_data):
    path = tmp_path / "test.db"
    database = load_database(path_or_database=path)
    make_optimization_iteration_table(database)
    for i in range(1, 11):  # sqlalchemy starts counting at 1
        iteration_data["value"] = i
        iteration_data["step"] = i % 2
        append_row(iteration_data, "optimization_iterations", database)

    res = read_last_rows(
        database=database,
        table_name="optimization_iterations",
        n_rows=20,
        return_type="dict_of_lists",
        step=0,
    )

    expected = [2, 4, 6, 8, 10]
    assert res["rowid"] == expected


def test_read_table(tmp_path, iteration_data):
    path = tmp_path / "test.db"
    database = load_database(path_or_database=path)
    make_optimization_iteration_table(database)
    for i in range(1, 11):  # sqlalchemy starts counting at 1
        iteration_data["value"] = i
        iteration_data["step"] = i % 2
        append_row(iteration_data, "optimization_iterations", database)

    table = read_table(
        database=database,
        table_name="optimization_iterations",
        return_type="dict_of_lists",
    )

    assert table["rowid"] == list(range(1, 11))
    assert table["step"] == [1, 0] * 5
