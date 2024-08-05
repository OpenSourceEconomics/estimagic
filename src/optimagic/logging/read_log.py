"""Functions to read data from the database used for logging.

The functions in the module are meant for end users of optimagic. They do not require
any knowledge of databases.

When using them internally, make sure to supply a database to path_or_database.
Otherwise, the functions may be very slow.

"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from pybaum import tree_flatten, tree_unflatten

from optimagic.logging.load_database import load_database
from optimagic.logging.read_from_database import (
    read_last_rows,
    read_new_rows,
    read_specific_row,
)
from optimagic.parameters.tree_registry import get_registry


def load_existing_database(path_or_database):
    if isinstance(path_or_database, (Path, str)):
        path = Path(path_or_database)
        if not path.exists():
            raise FileNotFoundError(f"Database {path} does not exist.")
    return load_database(path_or_database)


def read_start_params(path_or_database):
    """Load the start parameters DataFrame.

    Args:
        path_or_database (pathlib.Path, str or sqlalchemy.MetaData)

    Returns:
        params (pd.DataFrame): see :ref:`params`.

    """
    database = load_existing_database(path_or_database)
    optimization_problem = read_last_rows(
        database=database,
        table_name="optimization_problem",
        n_rows=1,
        return_type="dict_of_lists",
    )
    start_params = optimization_problem["params"][0]
    return start_params


def read_steps_table(path_or_database):
    """Load the steps table.

    Args:
        path_or_database (pathlib.Path, str or sqlalchemy.MetaData)

    Returns:
        steps_df (pandas.DataFrame)

    """
    database = load_existing_database(path_or_database)
    steps_table, _ = read_new_rows(
        database=database,
        table_name="steps",
        last_retrieved=0,
        return_type="list_of_dicts",
    )
    steps_df = pd.DataFrame(steps_table)

    return steps_df


def read_optimization_problem_table(path_or_database):
    """Load the start parameters DataFrame.

    Args:
        path_or_database (pathlib.Path, str or sqlalchemy.MetaData)

    Returns:
        params (pd.DataFrame): see :ref:`params`.

    """
    database = load_existing_database(path_or_database)
    steps_table, _ = read_new_rows(
        database=database,
        table_name="optimization_problem",
        last_retrieved=0,
        return_type="list_of_dicts",
    )
    steps_df = pd.DataFrame(steps_table)

    return steps_df


@dataclass
class OptimizeLogReader:
    """Read information about an optimization from a sqlite database."""

    path: Union[str, Path]

    def __post_init__(self):
        _database = load_existing_database(self.path)
        _start_params = read_start_params(_database)
        _registry = get_registry(extended=True)
        _, _treedef = tree_flatten(_start_params, registry=_registry)
        self._database = _database
        self._registry = _registry
        self._treedef = _treedef
        self._start_params = _start_params

    def read_iteration(self, iteration):
        out = _read_optimization_iteration(
            database=self._database,
            iteration=iteration,
            params_treedef=self._treedef,
            registry=self._registry,
        )
        return out

    def read_history(self):
        out = _read_optimization_history(
            database=self._database,
            params_treedef=self._treedef,
            registry=self._registry,
        )
        return out

    def read_multistart_history(self, direction):
        out = _read_multistart_optimization_history(
            database=self._database,
            params_treedef=self._treedef,
            registry=self._registry,
            direction=direction,
        )
        return out

    def read_start_params(self):
        return self._start_params


def _read_optimization_iteration(database, iteration, params_treedef, registry):
    """Get information about an optimization iteration."""
    if iteration >= 0:
        rowid = iteration + 1
    else:
        last_iteration = read_last_rows(
            database=database,
            table_name="optimization_iterations",
            n_rows=1,
            return_type="list_of_dicts",
        )
        highest_rowid = last_iteration[0]["rowid"]

        # iteration is negative here!
        rowid = highest_rowid + iteration + 1

    data = read_specific_row(
        database,
        table_name="optimization_iterations",
        rowid=rowid,
        return_type="list_of_dicts",
    )

    if len(data) == 0:
        raise IndexError(f"Invalid iteration requested: {iteration}")
    else:
        data = data[0]

    params = tree_unflatten(params_treedef, data["params"], registry=registry)
    data["params"] = params

    return data


def _read_optimization_history(database, params_treedef, registry):
    """Read a histories out values, parameters and other information."""
    raw_res, _ = read_new_rows(
        database=database,
        table_name="optimization_iterations",
        last_retrieved=0,
        return_type="list_of_dicts",
    )

    history = {"params": [], "criterion": [], "runtime": []}
    for data in raw_res:
        if data["value"] is not None:
            params = tree_unflatten(params_treedef, data["params"], registry=registry)
            history["params"].append(params)
            history["criterion"].append(data["value"])
            history["runtime"].append(data["timestamp"])

    times = np.array(history["runtime"])
    times -= times[0]
    history["runtime"] = times

    return history


def _read_multistart_optimization_history(
    database, params_treedef, registry, direction
):
    """Read multistart histories out values, parameters and other information.

    Returns:
        tuple:
        - dict: history that led to lowest criterion
        - dict: all other histories
        - dict: exploration phase

    """
    # ==================================================================================
    # Process raw data
    # ==================================================================================
    steps = read_steps_table(database)

    raw_res, _ = read_new_rows(
        database=database,
        table_name="optimization_iterations",
        last_retrieved=0,
        return_type="list_of_dicts",
    )

    history = {"params": [], "criterion": [], "runtime": [], "step": []}
    for data in raw_res:
        if data["value"] is not None:
            params = tree_unflatten(params_treedef, data["params"], registry=registry)
            history["params"].append(params)
            history["criterion"].append(data["value"])
            history["runtime"].append(data["timestamp"])
            history["step"].append(data["step"])

    times = np.array(history["runtime"])
    times -= times[0]
    history["runtime"] = times

    # ==================================================================================
    # Format data as data frames
    # ==================================================================================
    df = pd.DataFrame(history)
    df = df.merge(steps[["rowid", "type"]], left_on="step", right_on="rowid")
    df = df.drop(columns="rowid")

    # ==================================================================================
    # Extract data from df
    # ==================================================================================
    exploration = df.query("type == 'exploration'").drop(columns=["step", "type"])

    histories = df.query("type == 'optimization'")
    histories = histories.drop(columns="type")
    histories = histories.set_index("step", append=True)

    # ==================================================================================
    # The best history is given by the history that attains the global minimum or
    # maximum. All other histories are defined as local histories.

    if direction == "minimize":
        best_idx = histories["criterion"].groupby(level="step").min().idxmin()
        exploration = exploration.sort_values(by="criterion", ascending=True)
    elif direction == "maximize":
        best_idx = histories["criterion"].groupby(level="step").max().idxmax()
        exploration = exploration.sort_values(by="criterion", ascending=False)
    else:
        raise ValueError()

    history = histories.xs(best_idx, level="step").to_dict(orient="list")

    exploration = None if len(exploration) == 0 else exploration
    if exploration is not None:
        exploration = exploration.to_dict(orient="list")

    local_histories = []
    for idx in histories.index.get_level_values("step").unique().difference([best_idx]):
        _local_history = histories.xs(idx, level="step").to_dict(orient="list")
        local_histories.append(_local_history)

    local_histories = None if len(local_histories) == 0 else local_histories

    return history, local_histories, exploration
