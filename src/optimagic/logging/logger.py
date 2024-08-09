from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pybaum import tree_flatten, tree_unflatten

from optimagic.logging.base import AbstractKeyValueStore
from optimagic.logging.sqlite import (
    IterationStore,
    ProblemStore,
    SQLiteConfig,
    StepStore,
)
from optimagic.logging.types import (
    CriterionEvaluationResult,
    CriterionEvaluationWithId,
    ExistenceStrategy,
    ExistenceStrategyLiteral,
    IterationHistory,
    MultiStartIterationHistory,
    OptimizationType,
    OptimizationTypeLiteral,
    ProblemInitialization,
    ProblemInitializationWithId,
    StepResult,
    StepResultWithId,
    StepType,
)
from optimagic.parameters.tree_registry import get_registry
from optimagic.typing import PyTree


class Logger:
    def __init__(
        self,
        iteration_store: AbstractKeyValueStore[
            CriterionEvaluationResult, CriterionEvaluationWithId
        ],
        step_store: AbstractKeyValueStore[StepResult, StepResultWithId],
        problem_store: AbstractKeyValueStore[
            ProblemInitialization, ProblemInitializationWithId
        ],
    ):
        self.step_store = step_store
        self.iteration_store = iteration_store
        self.problem_store = problem_store
        self._pytree_registry = get_registry(extended=True)
        self._treedef = tree_flatten(
            self.read_start_params(), registry=self._pytree_registry
        )[-1]

    def read_iteration(self, iteration: int) -> CriterionEvaluationWithId:
        if iteration >= 0:
            rowid = iteration + 1
        else:
            try:
                last_row = self.iteration_store.select_last_rows(1)
                highest_rowid = last_row[0].rowid
            except IndexError as e:
                raise IndexError(
                    "Invalid iteration request, iteration store is empty"
                ) from e

            # iteration is negative here!
            assert highest_rowid is not None
            rowid = highest_rowid + iteration + 1

        row_list = self.iteration_store.select(rowid)

        if len(row_list) == 0:
            raise IndexError(f"Invalid iteration requested: {iteration}")
        else:
            data = row_list[0]

        params = tree_unflatten(
            self._treedef, data.params, registry=self._pytree_registry
        )

        return replace(data, params=params)

    def read_history(self) -> IterationHistory:
        raw_res = self.iteration_store.select()
        params_list = []
        criterion_list = []
        runtime_list = []
        for data in raw_res:
            if data.value is not None:
                params = tree_unflatten(
                    self._treedef, data.params, registry=self._pytree_registry
                )
                params_list.append(params)
                criterion_list.append(data.value)
                runtime_list.append(data.timestamp)

        times = np.array(runtime_list)
        times -= times[0]

        return IterationHistory(params_list, criterion_list, times)

    @staticmethod
    def _normalize_direction(
        direction: OptimizationType | OptimizationTypeLiteral,
    ) -> OptimizationType:
        if isinstance(direction, str):
            direction = OptimizationType(direction)
        return direction

    def _build_history_dataframe(self) -> pd.DataFrame:
        steps = self.step_store.to_df()
        raw_res = self.iteration_store.select()

        history: dict[str, list[Any]] = {
            "params": [],
            "criterion": [],
            "runtime": [],
            "step": [],
        }

        for data in raw_res:
            if data.value is not None:
                params = tree_unflatten(
                    self._treedef, data.params, registry=self._pytree_registry
                )
                history["params"].append(params)
                history["criterion"].append(data.value)
                history["runtime"].append(data.timestamp)
                history["step"].append(data.step)

        times = np.array(history["runtime"])
        times -= times[0]
        history["runtime"] = times.tolist()

        df = pd.DataFrame(history)
        df = df.merge(
            steps[[f"{self.step_store.primary_key}", "type"]],
            left_on="step",
            right_on=f"{self.step_store.primary_key}",
        )
        return df.drop(columns=f"{self.step_store.primary_key}")

    @staticmethod
    def _split_exploration_and_optimization(
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame]:
        exploration = df.query(f"type == '{StepType.EXPLORATION.value}'").drop(
            columns=["step", "type"]
        )
        histories = df.query(f"type == '{StepType.OPTIMIZATION.value}'")
        histories = histories.drop(columns="type").set_index("step", append=True)

        return None if exploration.empty else exploration, histories

    @staticmethod
    def _sort_exploration(
        exploration: pd.DataFrame | None, optimization_type: OptimizationType
    ) -> IterationHistory | None:
        if exploration is not None:
            is_minimization = optimization_type is OptimizationType.MINIMIZE
            exploration = exploration.sort_values(
                by="criterion", ascending=is_minimization
            )
            exploration_dict = cast(dict[str, Any], exploration.to_dict(orient="list"))
            return IterationHistory(**exploration_dict)
        return exploration

    @staticmethod
    def _extract_best_history(
        histories: pd.DataFrame, optimization_type: OptimizationType
    ) -> tuple[IterationHistory, list[IterationHistory] | None]:
        groupby_step_criterion = histories["criterion"].groupby(level="step")

        if optimization_type is OptimizationType.MINIMIZE:
            best_idx = groupby_step_criterion.min().idxmin()
        else:
            best_idx = groupby_step_criterion.max().idxmax()

        remaining_indices = (
            histories.index.get_level_values("step").unique().difference([best_idx])
        )

        best_history: pd.DataFrame | pd.Series[Any] = histories.xs(
            best_idx, level="step"
        )

        def _to_dict(pandas_obj: pd.DataFrame | pd.Series) -> dict[str, Any]:  # type:ignore
            if isinstance(pandas_obj, pd.DataFrame):
                result = pandas_obj.to_dict(orient="list")
            else:
                result = best_history.to_dict()
            return cast(dict[str, Any], result)

        best_history_dict = _to_dict(best_history)
        local_histories = [
            _to_dict(histories.xs(idx, level="step")) for idx in remaining_indices
        ]
        if len(local_histories) == 0:
            remaining_histories = None
        else:
            remaining_histories = [
                IterationHistory(**history_dict) for history_dict in local_histories
            ]

        return IterationHistory(**best_history_dict), remaining_histories

    def read_multistart_history(
        self, direction: OptimizationType | OptimizationTypeLiteral
    ) -> MultiStartIterationHistory:
        optimization_type = self._normalize_direction(direction)
        history_df = self._build_history_dataframe()
        exploration, optimization_history = self._split_exploration_and_optimization(
            history_df
        )
        exploration_history = self._sort_exploration(exploration, optimization_type)
        best_history, remaining_histories = self._extract_best_history(
            optimization_history, optimization_type
        )

        return MultiStartIterationHistory(
            best_history,
            local_histories=remaining_histories,
            exploration=exploration_history,
        )

    def read_start_params(self) -> PyTree:
        return self.problem_store.select_last_rows(1)[0].params


class SQLiteLogger(Logger):
    def __init__(
        self,
        path: str | Path,
        fast_logging: bool = True,
        existence_strategy: ExistenceStrategy
        | ExistenceStrategyLiteral = ExistenceStrategy.EXTEND,
    ):
        if isinstance(existence_strategy, str):
            existence_strategy = ExistenceStrategy(existence_strategy)

        db_config = SQLiteConfig(path, fast_logging=fast_logging)
        iteration_store = IterationStore(
            db_config, existence_strategy=existence_strategy
        )
        step_store = StepStore(db_config, existence_strategy=existence_strategy)
        problem_store = ProblemStore(db_config, existence_strategy=existence_strategy)
        super().__init__(iteration_store, step_store, problem_store)
