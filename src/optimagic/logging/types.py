from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from optimagic.typing import DictLikeAccess, PyTree, TupleLikeAccess


class StepStatus(str, Enum):
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETE = "complete"
    SKIPPED = "skipped"


class StepType(str, Enum):
    OPTIMIZATION = "optimization"
    EXPLORATION = "exploration"


class ExistenceStrategy(str, Enum):
    RAISE = "raise"
    EXTEND = "extend"
    REPLACE = "replace"


ExistenceStrategyLiteral = Literal["raise", "extend", "replace"]


class OptimizationType(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


OptimizationTypeLiteral = Literal["minimize", "maximize"]


@dataclass(frozen=True)
class CriterionEvaluationResult(DictLikeAccess):
    params: PyTree
    timestamp: float
    value: float
    valid: bool
    criterion_eval: PyTree | None = None
    internal_derivative: PyTree | None = None
    step: int | None = None
    exceptions: str | None = None
    hash: str | None = None


@dataclass(frozen=True)
class CriterionEvaluationWithId(CriterionEvaluationResult):
    rowid: int | None = None

    def __post_init__(self) -> None:
        if self.rowid is None:
            raise ValueError()


@dataclass(frozen=True)
class StepResult(DictLikeAccess):
    name: str
    type: StepType | str
    status: StepStatus | str
    n_iterations: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.type, str):
            object.__setattr__(self, "type", StepType(self.type))
        if isinstance(self.status, str):
            object.__setattr__(self, "status", StepStatus(self.status))


@dataclass(frozen=True)
class StepResultWithId(StepResult):
    rowid: int | None = None

    def __post_init__(self) -> None:
        if self.rowid is None:
            raise ValueError()
        super().__post_init__()


@dataclass(frozen=True)
class ProblemInitialization(DictLikeAccess):
    direction: str
    params: PyTree


@dataclass(frozen=True)
class ProblemInitializationWithId(ProblemInitialization):
    rowid: int | None = None

    def __post_init__(self) -> None:
        if self.rowid is None:
            raise ValueError()


@dataclass(frozen=True)
class IterationHistory(DictLikeAccess):
    params: list[PyTree]
    criterion: list[float]
    runtime: list[float] | NDArray[np.float64]


@dataclass(frozen=True)
class MultiStartIterationHistory(TupleLikeAccess):
    history: IterationHistory
    local_histories: list[IterationHistory] | None = None
    exploration: IterationHistory | None = None
