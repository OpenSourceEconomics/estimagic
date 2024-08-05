"""Functions to generate, load, write to and read from databases.

The functions here are meant for internal use in optimagic, e.g. for logging during
the optimization and reading from the database. They do not require
detailed knowledge of databases in general but some knowledge of the schema
(e.g. table names) of the database we use for logging.

Therefore, users who simply want to read the database should use the functions in
``read_log.py`` instead.

"""

import traceback
import warnings

import sqlalchemy as sql


def read_new_rows(
    database,
    table_name,
    last_retrieved,
    return_type,
    limit=None,
    stride=1,
    step=None,
):
    """Read all iterations after last_retrieved up to a limit.

    Args:
        database (DataBase): Object containing everything to work with the database.
        table_name (str): name of the table to retrieve.
        last_retrieved (int): The last iteration that was retrieved.
        return_type (str): either "list_of_dicts" or "dict_of_lists".
        limit (int): maximum number of rows to extract from the table.
        stride (int): Only return every n-th row. Default is every row (stride=1).
        step (int): Only return iterations that belong to step.

    Returns:
        result (return_type): up to limit rows after last_retrieved of the
            `table_name` table as `return_type`.
        int: The new last_retrieved value.

    """
    last_retrieved = int(last_retrieved)
    limit = int(limit) if limit is not None else limit

    table = database.metadata.tables[table_name]
    stmt = table.select().where(table.c.rowid > last_retrieved).limit(limit)
    conditions = [table.c.rowid > last_retrieved]

    if stride != 1:
        conditions.append(table.c.rowid % stride == 0)

    if step is not None:
        conditions.append(table.c.step == int(step))

    stmt = table.select().where(sql.and_(*conditions)).limit(limit)

    data = _execute_read_statement(database, table_name, stmt, return_type)

    if return_type == "list_of_dicts":
        new_last = data[-1]["rowid"] if data else last_retrieved
    else:
        new_last = data["rowid"][-1] if data["rowid"] else last_retrieved

    return data, new_last


def read_last_rows(
    database,
    table_name,
    n_rows,
    return_type,
    stride=1,
    step=None,
):
    """Read the last n_rows rows from a table.

    If a table has less than n_rows rows, the whole table is returned.

    Args:
        database (DataBase)
        table_name (str): name of the table to retrieve.
        n_rows (int): number of rows to retrieve.
        return_type (str): either "list_of_dicts" or "dict_of_lists".
        stride (int): Only return every n-th row. Default is every row (stride=1).
        step (int): Only return rows that belong to step.

    Returns:
        result (return_type): the last rows of the `table_name` table as `return_type`.

    """
    n_rows = int(n_rows)

    table = database.metadata.tables[table_name]

    conditions = []

    if stride != 1:
        conditions.append(table.c.rowid % stride == 0)

    if step is not None:
        conditions.append(table.c.step == int(step))

    if conditions:
        stmt = (
            table.select()
            .order_by(table.c.rowid.desc())
            .where(sql.and_(*conditions))
            .limit(n_rows)
        )
    else:
        stmt = table.select().order_by(table.c.rowid.desc()).limit(n_rows)

    reversed_ = _execute_read_statement(database, table_name, stmt, return_type)
    if return_type == "list_of_dicts":
        out = reversed_[::-1]
    else:
        out = {key: val[::-1] for key, val in reversed_.items()}

    return out


def read_specific_row(database, table_name, rowid, return_type):
    """Read a specific row from a table.

    Args:
        database (sqlalchemy.MetaData)
        table_name (str): name of the table to retrieve.
        n_rows (int): number of rows to retrieve.
        return_type (str): either "list_of_dicts" or "dict_of_lists".

    Returns:
        dict or list: The requested row from the database.

    """
    rowid = int(rowid)
    table = database.metadata.tables[table_name]
    stmt = table.select().where(table.c.rowid == rowid)
    data = _execute_read_statement(database, table_name, stmt, return_type)
    return data


def read_table(database, table_name, return_type):
    table = database.metadata.tables[table_name]
    stmt = table.select()
    data = _execute_read_statement(database, table_name, stmt, return_type)
    return data


def _execute_read_statement(database, table_name, statement, return_type):
    try:
        with database.engine.begin() as connection:
            raw_result = list(connection.execute(statement))
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        exception_info = traceback.format_exc()
        warnings.warn(
            "Unable to read {table_name} from database. Try again later. The traceback "
            f"was: \n\n{exception_info}"
        )
        # if we only want to warn we must provide a raw_result to be processed below.
        raw_result = []

    columns = database.metadata.tables[table_name].columns.keys()

    if return_type == "list_of_dicts":
        result = [dict(zip(columns, row, strict=False)) for row in raw_result]

    elif return_type == "dict_of_lists":
        raw_result = transpose_nested_list(raw_result)
        result = dict(zip(columns, raw_result, strict=False))
        if result == {}:
            result = {col: [] for col in columns}
    else:
        raise NotImplementedError(
            "The return_type must be 'list_of_dicts' or 'dict_of_lists', "
            f"not {return_type}."
        )

    return result


def transpose_nested_list(nested_list):
    """Transpose a list of lists.

    Args:
        nested_list (list): Nested list where all sublists have the same length.

    Returns:
        list

    Examples:
        >>> transpose_nested_list([[1, 2], [3, 4]])
        [[1, 3], [2, 4]]

    """
    return list(map(list, zip(*nested_list, strict=False)))


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    """Convert a list of dicts to a dict of lists.

    Args:
        list_of_dicts (list): List of dictionaries. All dictionaries have the same keys.

    Returns:
        dict

    Examples:
        >>> list_of_dicts_to_dict_of_lists([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        {'a': [1, 3], 'b': [2, 4]}

    """
    return {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}


def dict_of_lists_to_list_of_dicts(dict_of_lists):
    """Convert a dict of lists to a list of dicts.

    Args:
        dict_of_lists (dict): Dictionary of lists where all lists have the same length.

    Returns:
        list

    Examples:

        >>> dict_of_lists_to_list_of_dicts({'a': [1, 3], 'b': [2, 4]})
        [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]

    """
    return [
        dict(zip(dict_of_lists, t, strict=False))
        for t in zip(*dict_of_lists.values(), strict=False)
    ]
