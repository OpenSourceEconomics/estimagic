import traceback
import warnings

import sqlalchemy as sql


def update_row(data, rowid, table_name, database):
    table = database.metadata.tables[table_name]
    stmt = sql.update(table).where(table.c.rowid == rowid).values(**data)

    _execute_write_statement(stmt, database)


def append_row(data, table_name, database):
    """

    Args:
        data (dict): The keys correspond to columns in the database table.
        table_name (str): Name of the database table to which the row is added.
        database (DataBase): The database to which the row is added.

    """
    stmt = database.metadata.tables[table_name].insert().values(**data)

    _execute_write_statement(stmt, database)


def _execute_write_statement(statement, database):
    try:
        # this will automatically roll back the transaction if any exception is raised
        # and then raise the exception
        with database.engine.begin() as connection:
            connection.execute(statement)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        exception_info = traceback.format_exc()
        warnings.warn(
            f"Unable to write to database. The traceback was:\n\n{exception_info}"
        )
