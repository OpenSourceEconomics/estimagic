from __future__ import annotations

import traceback
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Sequence, Type

import sqlalchemy as sql
from sqlalchemy.engine import Engine
from sqlalchemy.sql.base import Executable
from sqlalchemy.sql.schema import MetaData

from optimagic.exceptions import TableExistsError
from optimagic.logging.base import AbstractKeyValueStore, InputType, OutputType
from optimagic.logging.load_database import RobustPickler
from optimagic.logging.types import ExistenceStrategy


class SQLAlchemyConfig:
    def __init__(self, url: str):
        self.url = url
        engine = self.create_engine()
        metadata = MetaData()
        self._configure_reflect()
        metadata.reflect(engine)
        self._metadata = metadata

    @property
    def metadata(self) -> MetaData:
        return self._metadata

    def create_engine(self) -> Engine:
        return sql.create_engine(self.url)

    @staticmethod
    def _configure_reflect() -> None:
        """Mark all BLOB dtypes as PickleType with our custom pickle reader.

        Code ist taken from the documentation: https://tinyurl.com/y7q287jr

        """

        @sql.event.listens_for(sql.Table, "column_reflect")
        def _setup_pickletype(
            inspector: Any, table: sql.Table, column_info: dict[str, Any]
        ) -> None:  # noqa: ARG001
            if isinstance(column_info["type"], sql.BLOB):
                column_info["type"] = sql.PickleType(pickler=RobustPickler)  # type:ignore


@dataclass
class TableConfig:
    table_name: str
    columns: list[sql.Column[Any]]
    primary_key: str
    existence_strategy: ExistenceStrategy = ExistenceStrategy.EXTEND

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def _handle_existing_table(self, metadata: MetaData, engine: Engine) -> None:
        if self.table_name in metadata.tables:
            if self.existence_strategy is ExistenceStrategy.REPLACE:
                metadata.tables[self.table_name].drop(engine)
            elif self.existence_strategy is ExistenceStrategy.RAISE:
                raise TableExistsError(f"The table {self.table_name} already exists.")

    def create_table(self, metadata: MetaData, engine: Engine) -> sql.Table:
        metadata.reflect(engine)
        self._handle_existing_table(metadata, engine)
        table = sql.Table(
            self.table_name, metadata, *self.columns, extend_existing=True
        )
        metadata.create_all(engine)
        return table


class SQLAlchemyTableStore(AbstractKeyValueStore[InputType, OutputType]):
    def __init__(
        self,
        table_config: TableConfig,
        db_config: SQLAlchemyConfig,
        input_type: Type[InputType],
        output_type: Type[OutputType],
    ):
        self._db_config = db_config
        self._engine = db_config.create_engine()
        self._table_config = table_config
        self._table = table_config.create_table(db_config.metadata, self._engine)
        super().__init__(input_type, output_type, self._table_config.primary_key)

    @property
    def column_names(self) -> list[str]:
        return self._table_config.column_names

    @property
    def primary_key(self) -> str:
        return self._table_config.primary_key

    @property
    def table_name(self) -> str:
        return self._table_config.table_name

    @property
    def table(self) -> sql.Table:
        return self._table

    @property
    def engine(self) -> Engine:
        return self._engine

    def __reduce__(
        self,
    ) -> tuple[
        Type[SQLAlchemyTableStore[Any, Any]],
        tuple[TableConfig, SQLAlchemyConfig, Type[Any], Type[Any]],
    ]:
        return SQLAlchemyTableStore, (
            self._table_config,
            self._db_config,
            self._input_type,
            self._output_type,
        )

    def _insert(self, value: InputType | dict[str, Any]) -> None:
        insert_values = self._pre_process(value)
        stmt = self._table.insert().values(**insert_values)
        self._execute_write_statement(stmt)

    def _pre_process(self, value: InputType | dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            insert_values = asdict(value)
        else:
            insert_values = value
        return insert_values

    def _update(self, key: int, value: InputType | dict[str, Any]) -> None:
        update_values = self._pre_process(value)
        stmt = (
            self._table.update()
            .where(getattr(self._table.c, self.primary_key) == key)
            .values(**update_values)
        )
        self._execute_write_statement(stmt)

    def _select_by_key(self, key: int) -> list[OutputType]:
        stmt = self._table.select().where(
            getattr(self._table.c, self.primary_key) == key
        )
        return self._execute_read_statement(stmt)

    def _select_all(self) -> list[OutputType]:
        stmt = self._table.select()
        return self._execute_read_statement(stmt)

    def select_last_rows(self, n_rows: int) -> list[OutputType]:
        stmt = (
            self._table.select()
            .order_by(getattr(self._table.c, self.primary_key).desc())
            .limit(n_rows)
        )
        return self._execute_read_statement(stmt)

    def _execute_read_statement(self, statement: Executable) -> list[OutputType]:
        with self._engine.connect() as connection:
            results = connection.execute(statement).fetchall()
            return self._post_process(results)

    def _post_process(self, results: Sequence[sql.Row]) -> list[OutputType]:  # type:ignore
        return [
            self._output_type(**dict(zip(self.column_names, row, strict=False)))
            for row in results
        ]

    def _execute_write_statement(self, statement: Executable) -> None:
        try:
            with self._engine.begin() as connection:
                connection.execute(statement)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            exception_info = traceback.format_exc()
            warnings.warn(
                f"Unable to write to database. The traceback was:\n\n{exception_info}"
            )
