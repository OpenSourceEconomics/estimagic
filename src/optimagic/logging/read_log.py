"""Functions to read data from the database used for logging.

The functions in the module are meant for end users of optimagic. They do not require
any knowledge of databases.

When using them internally, make sure to supply a database to path_or_database.
Otherwise, the functions may be very slow.

"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from optimagic.logging.load_database import load_database
from optimagic.logging.logger import SQLiteLogger
from optimagic.logging.read_from_database import (
    read_last_rows,
    read_new_rows,
)


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
    def __new__(cls, *args, **kwargs):
        warnings.warn(
            "OptimizeLogReader is deprecated and will be removed in a future "
            "version. Please use optimagic.logging.SQLiteLogger instead.",
            FutureWarning,
        )
        return SQLiteLogger(*args, **kwargs)
