from pathlib import Path
from typing import Any, cast

import sqlalchemy as sql
from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Integer,
    PickleType,
    String,
)
from sqlalchemy.engine.base import Engine

from optimagic.logging.load_database import (
    RobustPickler,
)
from optimagic.logging.sqlalchemy import (
    SQLAlchemyConfig,
    SQLAlchemyTableStore,
    TableConfig,
)
from optimagic.logging.types import (
    CriterionEvaluationResult,
    CriterionEvaluationWithId,
    ExistenceStrategy,
    ProblemInitialization,
    ProblemInitializationWithId,
    StepResult,
    StepResultWithId,
)


class SQLiteConfig(SQLAlchemyConfig):
    def __init__(self, path: str | Path, fast_logging: bool = True):
        url = f"sqlite:///{path}"
        self._fast_logging = fast_logging
        super().__init__(url)

    def _create_engine(self) -> Engine:
        engine = sql.create_engine(self.url)
        self._configure_engine(engine)
        return engine

    def _configure_engine(self, engine: Engine) -> None:
        """Configure the sqlite engine.

        The two functions that configure the emission of the `begin` statement are taken
        from the sqlalchemy documentation the documentation:
        https://tinyurl.com/u9xea5z
        and are
        the recommended way of working around a bug in the pysqlite driver.

        The other function speeds up the write process. If fast_logging is False, it
        does so using only completely safe optimizations. Of fast_logging is True,
        it also uses unsafe optimizations.

        """

        @sql.event.listens_for(engine, "connect")
        def do_connect(dbapi_connection: Any, connection_record: Any) -> None:  # noqa: ARG001
            # disable pysqlite's emitting of the BEGIN statement entirely.
            # also stops it from emitting COMMIT before absolutely necessary.
            dbapi_connection.isolation_level = None

        @sql.event.listens_for(engine, "begin")
        def do_begin(conn: Any) -> None:
            # emit our own BEGIN
            conn.exec_driver_sql("BEGIN DEFERRED")

        @sql.event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:  # noqa: ARG001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode = WAL")
            if self._fast_logging:
                cursor.execute("PRAGMA synchronous = OFF")
            else:
                cursor.execute("PRAGMA synchronous = NORMAL")
            cursor.close()


class IterationStore(
    SQLAlchemyTableStore[CriterionEvaluationResult, CriterionEvaluationWithId]
):
    table_name = "optimization_iterations"

    def __init__(
        self,
        db_config: SQLiteConfig,
        existence_strategy: ExistenceStrategy = ExistenceStrategy.EXTEND,
    ):
        columns = [
            Column("rowid", Integer, primary_key=True, autoincrement=True),
            Column("params", PickleType(pickler=RobustPickler)),  # type:ignore
            Column("internal_derivative", PickleType(pickler=RobustPickler)),  # type:ignore
            Column("timestamp", Float),
            Column("exceptions", String),
            Column("valid", Boolean),
            Column("hash", String),
            Column("value", Float),
            Column("step", Integer),
            Column("criterion_eval", PickleType(pickler=RobustPickler)),  # type:ignore
        ]

        table_config = TableConfig(
            self.table_name,
            columns,
            "rowid",
            existence_strategy,
        )

        super().__init__(
            table_config,
            db_config,
            CriterionEvaluationResult,
            CriterionEvaluationWithId,
        )


class StepStore(SQLAlchemyTableStore[StepResult, StepResultWithId]):
    table_name = "steps"

    def __init__(
        self,
        db_config: SQLiteConfig,
        existence_strategy: ExistenceStrategy = ExistenceStrategy.EXTEND,
    ):
        columns = [
            Column("rowid", Integer, primary_key=True, autoincrement=True),
            Column("type", String),  # e.g. optimization
            Column("status", String),  # e.g. running
            Column("n_iterations", Integer),  # optional
            Column("name", String),  # e.g. "optimization-1", "exploration", not unique
        ]

        table_config = TableConfig(
            self.table_name,
            cast(list[Column[Any]], columns),
            "rowid",
            existence_strategy,
        )

        super().__init__(
            table_config,
            db_config,
            StepResult,
            StepResultWithId,
        )


class ProblemStore(
    SQLAlchemyTableStore[ProblemInitialization, ProblemInitializationWithId]
):
    table_name = "optimization_problem"

    def __init__(
        self,
        db_config: SQLiteConfig,
        existence_strategy: ExistenceStrategy = ExistenceStrategy.EXTEND,
    ):
        columns = [
            Column("rowid", Integer, primary_key=True, autoincrement=True),
            Column("direction", String),
            Column("params", PickleType(pickler=RobustPickler)),  # type:ignore
        ]

        table_config = TableConfig(
            self.table_name,
            cast(list[Column[Any]], columns),
            "rowid",
            existence_strategy,
        )

        super().__init__(
            table_config,
            db_config,
            ProblemInitialization,
            ProblemInitializationWithId,
        )
