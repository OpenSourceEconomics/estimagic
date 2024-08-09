from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
from optimagic.logging.logger import SQLiteLogger
from optimagic.optimization.optimize import minimize
from optimagic.parameters.tree_registry import get_registry
from pybaum import tree_equal, tree_just_flatten


@pytest.fixture()
def example_db(tmp_path):
    path = tmp_path / "test.db"

    def _crit(params):
        x = np.array(list(params.values()))
        return x @ x

    minimize(
        fun=_crit,
        params={"a": 1, "b": 2, "c": 3},
        algorithm="scipy_lbfgsb",
        logging=path,
    )
    return path


def test_read_start_params(example_db):
    res = SQLiteLogger(example_db).read_start_params()
    assert res == {"a": 1, "b": 2, "c": 3}


def test_log_reader_read_start_params(example_db):
    reader = SQLiteLogger(example_db)
    res = reader.read_start_params()
    assert res == {"a": 1, "b": 2, "c": 3}


def test_log_reader_read_iteration(example_db):
    reader = SQLiteLogger(example_db)
    first_row = reader.read_iteration(0)
    assert first_row["params"] == {"a": 1, "b": 2, "c": 3}
    assert first_row["rowid"] == 1
    assert first_row["value"] == 14

    last_row = reader.read_iteration(-1)
    assert list(last_row["params"]) == ["a", "b", "c"]
    assert np.allclose(last_row["value"], 0)


def test_log_reader_read_history(example_db):
    reader = SQLiteLogger(example_db)
    res = reader.read_history()
    assert res["runtime"][0] == 0
    assert res["criterion"][0] == 14
    assert res["params"][0] == {"a": 1, "b": 2, "c": 3}


def test_log_reader_read_multistart_history(example_db):
    reader = SQLiteLogger(example_db)
    history, local_history, exploration = reader.read_multistart_history(
        direction="minimize"
    )
    assert local_history is None
    assert exploration is None

    registry = get_registry(extended=True)
    assert tree_equal(
        tree_just_flatten(asdict(history), registry=registry),
        tree_just_flatten(asdict(reader.read_history()), registry=registry),
    )


def test_read_steps_table(example_db):
    res = SQLiteLogger(example_db).step_store.to_df()
    assert isinstance(res, pd.DataFrame)
    assert res.loc[0, "rowid"] == 1
    assert res.loc[0, "type"] == "optimization"
    assert res.loc[0, "status"] == "complete"


def test_read_optimization_problem_table(example_db):
    res = SQLiteLogger(example_db).problem_store.to_df()
    assert isinstance(res, pd.DataFrame)


# TODO: db file is created at instantiation of the logger, decide how to handle
#  empty tables. By now, the logger methods may raise unspecific errors
#  (like IndexError)
@pytest.mark.skip
def test_non_existing_database_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        SQLiteLogger(tmp_path / "i_do_not_exist.db").read_start_params()
