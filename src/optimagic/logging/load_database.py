import io
import warnings

import cloudpickle
import pandas as pd
import sqlalchemy as sql

from optimagic.exceptions import get_traceback


class DataBase:
    """Class containing everything to work with a logging database.

    Importantly, the class is pickle-serializable which is important to share it across
    multiple processes. Upon unpickling, it will automatically re-create an engine to
    connect to the database.

    """

    def __init__(self, metadata, path, fast_logging, engine=None):
        self.metadata = metadata
        self.path = path
        self.fast_logging = fast_logging
        if engine is None:
            self.engine = _create_engine(path, fast_logging)
        else:
            self.engine = engine

    def __reduce__(self):
        return (DataBase, (self.metadata, self.path, self.fast_logging))


def load_database(path_or_database, fast_logging=False):
    """Load or create a database from a path and configure it for our needs.

    This is the only acceptable way of loading or creating a database in optimagic!

    Args:
        path_or_database (str or pathlib.Path): Path to the database or DataBase.
        fast_logging (bool): If True, use unsafe optimizations to speed up the logging.
            If False, only use ultra safe optimizations.

    Returns:
        database (Database): Object containing everything to work with the
            database.

    """
    if isinstance(path_or_database, DataBase):
        out = path_or_database
    else:
        engine = _create_engine(path_or_database, fast_logging)
        metadata = sql.MetaData()
        _configure_reflect()
        metadata.reflect(engine)

        out = DataBase(
            metadata=metadata,
            path=path_or_database,
            fast_logging=fast_logging,
            engine=engine,
        )
    return out


def _create_engine(path, fast_logging):
    engine = sql.create_engine(f"sqlite:///{path}")
    _configure_engine(engine, fast_logging)
    return engine


def _configure_engine(engine, fast_logging):
    """Configure the sqlite engine.

    The two functions that configure the emission of the begin statement are taken from
    the sqlalchemy documentation the documentation:
    https://tinyurl.com/u9xea5z
    and are
    the recommended way of working around a bug in the pysqlite driver.

    The other function speeds up the write process. If fast_logging is False, it does so
    using only completely safe optimizations. Of fast_logging is True, it also uses
    unsafe optimizations.

    """

    @sql.event.listens_for(engine, "connect")
    def do_connect(dbapi_connection, connection_record):  # noqa: ARG001
        # disable pysqlite's emitting of the BEGIN statement entirely.
        # also stops it from emitting COMMIT before absolutely necessary.
        dbapi_connection.isolation_level = None

    @sql.event.listens_for(engine, "begin")
    def do_begin(conn):
        # emit our own BEGIN
        conn.exec_driver_sql("BEGIN DEFERRED")

    @sql.event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):  # noqa: ARG001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode = WAL")
        if fast_logging:
            cursor.execute("PRAGMA synchronous = OFF")
        else:
            cursor.execute("PRAGMA synchronous = NORMAL")
        cursor.close()


def _configure_reflect():
    """Mark all BLOB dtypes as PickleType with our custom pickle reader.

    Code ist taken from the documentation: https://tinyurl.com/y7q287jr

    """

    @sql.event.listens_for(sql.Table, "column_reflect")
    def _setup_pickletype(inspector, table, column_info):  # noqa: ARG001
        if isinstance(column_info["type"], sql.BLOB):
            column_info["type"] = sql.PickleType(pickler=RobustPickler)


class RobustPickler:
    @staticmethod
    def loads(
        data,
        fix_imports=True,  # noqa: ARG004
        encoding="ASCII",  # noqa: ARG004
        errors="strict",  # noqa: ARG004
        buffers=None,  # noqa: ARG004
    ):
        """Robust pickle loading.

        We first try to unpickle the object with pd.read_pickle. This makes no
        difference for non-pandas objects but makes the de-serialization
        of pandas objects more robust across pandas versions. If that fails, we use
        cloudpickle. If that fails, we return None but do not raise an error.

        See: https://github.com/pandas-dev/pandas/issues/16474

        """
        try:
            res = pd.read_pickle(io.BytesIO(data), compression=None)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            try:
                res = cloudpickle.loads(data)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                res = None
                tb = get_traceback()
                warnings.warn(
                    f"Unable to read PickleType column from database:\n{tb}\n "
                    "The entry was replaced by None."
                )

        return res

    @staticmethod
    def dumps(
        obj,
        protocol=None,
        *,
        fix_imports=True,  # noqa: ARG001
        buffer_callback=None,  # noqa: ARG004
    ):
        return cloudpickle.dumps(obj, protocol=protocol)
