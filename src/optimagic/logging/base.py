from abc import ABC, abstractmethod
from dataclasses import asdict, fields, is_dataclass
from typing import Any, Generic, Type, TypeVar

import pandas as pd

from optimagic.typing import DictLikeAccess

InputType = TypeVar("InputType", bound=DictLikeAccess)
OutputType = TypeVar("OutputType", bound=DictLikeAccess)


class AbstractKeyValueStore(Generic[InputType, OutputType], ABC):
    def __init__(
        self,
        input_type: Type[InputType],
        output_type: Type[OutputType],
        primary_key: str,
    ):
        if not (is_dataclass(input_type) and is_dataclass(output_type)):
            raise ValueError("Arguments input_type and output_type must by dataclasses")

        output_fields = {f.name for f in fields(output_type)}
        if primary_key not in output_fields:
            raise ValueError(
                f"Primary key {primary_key} not found in output_type fields "
                f"{fields(output_type)}"
            )

        self._output_type = output_type
        self._input_type = input_type
        self._primary_key = primary_key
        self._supported_fields = {f.name for f in fields(input_type)}

    @property
    def primary_key(self) -> str:
        return self._primary_key

    def insert(self, value: InputType | dict[str, Any]) -> None:
        self._check_fields(value)
        self._insert(value)

    @abstractmethod
    def _insert(self, value: InputType | dict[str, Any]) -> None:
        pass

    def update(self, key: int, value: InputType | dict[str, Any]) -> None:
        self._check_fields(value)
        self._update(key, value)

    def _check_fields(self, value: InputType | dict[str, Any]) -> None:
        if isinstance(value, dict):
            not_supported_fields = set(value.keys()).difference(self._supported_fields)
            if not_supported_fields:
                raise ValueError(
                    f"Not supported fields {not_supported_fields}. "
                    f"Only supports fields {self._supported_fields}"
                )

    @abstractmethod
    def _update(self, key: int, value: InputType | dict[str, Any]) -> None:
        pass

    @abstractmethod
    def _select_by_key(self, key: int) -> list[OutputType]:
        pass

    @abstractmethod
    def _select_all(self) -> list[OutputType]:
        pass

    def select(self, key: int | None = None) -> list[OutputType]:
        if key is None:
            return self._select_all()

        return self._select_by_key(key)

    @abstractmethod
    def select_last_rows(self, n_rows: int) -> list[OutputType]:
        pass

    def to_df(self) -> pd.DataFrame:
        items = self._select_all()
        return pd.DataFrame([asdict(item) for item in items])
