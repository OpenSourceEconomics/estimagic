import sqlalchemy as sql

from optimagic.exceptions import TableExistsError
from optimagic.logging.load_database import RobustPickler


def make_optimization_iteration_table(database, if_exists="extend"):
    """Generate a table for information that is generated with each function evaluation.

    Args:
        database (DataBase): DataBase object containing the engine and metadata.
        if_exists (str): What to do if the table already exists. Can be "extend",
            "replace" or "raise".

    Returns:
        database (sqlalchemy.MetaData):Bound metadata object with added table.

    """
    table_name = "optimization_iterations"
    _handle_existing_table(database, "optimization_iterations", if_exists)

    columns = [
        sql.Column("rowid", sql.Integer, primary_key=True),
        sql.Column("params", sql.PickleType(pickler=RobustPickler)),
        sql.Column("internal_derivative", sql.PickleType(pickler=RobustPickler)),
        sql.Column("timestamp", sql.Float),
        sql.Column("exceptions", sql.String),
        sql.Column("valid", sql.Boolean),
        sql.Column("hash", sql.String),
        sql.Column("value", sql.Float),
        sql.Column("step", sql.Integer),
        sql.Column("criterion_eval", sql.PickleType(pickler=RobustPickler)),
    ]

    sql.Table(
        table_name,
        database.metadata,
        *columns,
        sqlite_autoincrement=True,
        extend_existing=True,
    )

    database.metadata.create_all(database.engine)


def make_steps_table(database, if_exists="extend"):
    table_name = "steps"
    _handle_existing_table(database, table_name, if_exists)
    columns = [
        sql.Column("rowid", sql.Integer, primary_key=True),
        sql.Column("type", sql.String),  # e.g. optimization
        sql.Column("status", sql.String),  # e.g. running
        sql.Column("n_iterations", sql.Integer),  # optional
        sql.Column(
            "name", sql.String
        ),  # e.g. "optimization-1", "exploration", not unique
    ]
    sql.Table(
        table_name,
        database.metadata,
        *columns,
        extend_existing=True,
        sqlite_autoincrement=True,
    )
    database.metadata.create_all(database.engine)


def make_optimization_problem_table(database, if_exists="extend"):
    table_name = "optimization_problem"
    _handle_existing_table(database, table_name, if_exists)

    columns = [
        sql.Column("rowid", sql.Integer, primary_key=True),
        sql.Column("direction", sql.String),
        sql.Column("params", sql.PickleType(pickler=RobustPickler)),
        sql.Column("algorithm", sql.PickleType(pickler=RobustPickler)),
        sql.Column("algo_options", sql.PickleType(pickler=RobustPickler)),
        sql.Column("numdiff_options", sql.PickleType(pickler=RobustPickler)),
        sql.Column("log_options", sql.PickleType(pickler=RobustPickler)),
        sql.Column("error_handling", sql.String),
        sql.Column("error_penalty", sql.PickleType(pickler=RobustPickler)),
        sql.Column("constraints", sql.PickleType(pickler=RobustPickler)),
        sql.Column("free_mask", sql.PickleType(pickler=RobustPickler)),
    ]

    sql.Table(
        table_name,
        database.metadata,
        *columns,
        extend_existing=True,
        sqlite_autoincrement=True,
    )

    database.metadata.create_all(database.engine)


def _handle_existing_table(database, table_name, if_exists):
    assert if_exists in ["replace", "extend", "raise"]

    if table_name in database.metadata.tables:
        if if_exists == "replace":
            database.metadata.tables[table_name].drop(database.engine)
        elif if_exists == "raise":
            raise TableExistsError(f"The table {table_name} already exists.")
