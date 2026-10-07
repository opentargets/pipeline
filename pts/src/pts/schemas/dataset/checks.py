"""Procedural checks: rules a dataset model cannot state declaratively.

Anything a single row cannot decide (a rule across rows, or across columns in ways no constraint
expresses) is written as a plain function over pandera's `PolarsData` and registered against its
model, in the same module as the model::

    @column_check(EncoreEvidence, 'score')
    def score_matches_interaction_strength(data: PolarsData) -> pl.LazyFrame:
        \"\"\"`score` must be `abs(geneticInteractionScore) / 16`.\"\"\"
        ...

`to_pandera` appends registered checks to the ones it derives from the model -- it never replaces
those -- and the documentation lists them by name and docstring. Checks registered on a shared base
(e.g. `EvidenceBase`) apply to every dataset built on it.
"""

import inspect
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import polars as pl
from pandera.polars import PolarsData

from pts.schemas.dataset.base import DatasetModel

CheckFunction = Callable[[PolarsData], pl.LazyFrame]


@dataclass(frozen=True)
class RegisteredCheck:
    """A procedural check registered against a dataset model.

    Attributes:
        function: the check, returning one boolean per row (true meaning the row passes).
        column: the column it validates, or `None` for a check over the whole frame.
    """

    function: CheckFunction
    column: str | None

    @property
    def name(self) -> str:
        """The check's name: its function name."""
        return getattr(self.function, '__name__', repr(self.function))

    @property
    def description(self) -> str:
        """The check's docstring."""
        return inspect.getdoc(self.function) or ''


_REGISTRY: defaultdict[type[DatasetModel], list[RegisteredCheck]] = defaultdict(list)


def column_check(model: type[DatasetModel], column: str) -> Callable[[CheckFunction], CheckFunction]:
    """Register a check on one column of `model` (and of every dataset built on it)."""

    def register(function: CheckFunction) -> CheckFunction:
        _REGISTRY[model].append(RegisteredCheck(function, column))
        return function

    return register


def frame_check(model: type[DatasetModel]) -> Callable[[CheckFunction], CheckFunction]:
    """Register a check over the whole frame of `model` (and of every dataset built on it)."""

    def register(function: CheckFunction) -> CheckFunction:
        _REGISTRY[model].append(RegisteredCheck(function, None))
        return function

    return register


def registered_checks(model: type[DatasetModel]) -> tuple[RegisteredCheck, ...]:
    """Every check registered on `model` or one of its bases, bases first."""
    return tuple(check for cls in reversed(model.__mro__) for check in _REGISTRY.get(cls, ()))
