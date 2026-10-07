"""Tests for pts.schemas.dataset.converter and the procedural-check registry.

How each declarative rule is enforced is covered exhaustively by `test_schema_agreement.py`,
against Pydantic; this file covers what that comparison cannot see: keys, registered checks, and
the error messages a failing write produces.
"""

import math
from typing import Annotated, ClassVar

import pandera.errors as pe
import polars as pl
import pytest
from pandera.polars import PolarsData
from pydantic import Field

from pts.schemas.dataset import (
    DatasetModel,
    PrimaryKey,
    SchemaDefinitionError,
    StructModel,
    column_check,
    frame_check,
    to_pandera,
    to_polars_schema,
)
from pts.schemas.dataset.converter import to_base_pandera


class _Sample(StructModel):
    code: str = Field(pattern=r'^[A-Z]\d$')


class _Base(DatasetModel):
    id: Annotated[str, PrimaryKey()]
    value: float = Field(ge=0.0)


class _Measurement(_Base):
    dataset_name: ClassVar[str] = '_measurement'
    dataset_description: ClassVar[str] = 'test dataset'
    example: ClassVar[dict] = {'id': 'a', 'value': 1.0, 'samples': [{'code': 'A1'}]}

    samples: list[_Sample]


@column_check(_Base, 'value')
def value_is_finite(data: PolarsData) -> pl.LazyFrame:
    """`value` must be finite."""
    return data.lazyframe.select(pl.col(data.key).is_finite())


@frame_check(_Measurement)
def sorted_by_id(data: PolarsData) -> pl.LazyFrame:
    """Rows must be sorted by `id`."""
    return data.lazyframe.select(pl.col('id') >= pl.col('id').shift(1).fill_null(pl.col('id')))


class _Pair(DatasetModel):
    dataset_name: ClassVar[str] = '_pair'
    dataset_description: ClassVar[str] = 'composite key'
    example: ClassVar[dict] = {'left': 'a', 'right': 1}

    left: Annotated[str, PrimaryKey()]
    right: Annotated[int, PrimaryKey()]


def _frame(model: type[DatasetModel], rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=to_polars_schema(model))


def _failed_checks(model: type[DatasetModel], frame: pl.DataFrame) -> set[str]:
    with pytest.raises(pe.SchemaErrors) as caught:
        to_pandera(model).validate(frame, lazy=True)
    return set(caught.value.failure_cases['check'].to_list())


def test_registered_checks_are_appended_to_derived_ones_and_inherited_from_bases() -> None:
    schema = to_pandera(_Measurement)
    base = to_base_pandera(_Measurement)

    assert [check.name for check in schema.columns['value'].checks] == [
        *(check.name for check in base.columns['value'].checks),
        'value_is_finite',
    ]
    assert [check.name for check in schema.checks or []] == ['sorted_by_id']
    assert base.checks == []


def test_registered_checks_run() -> None:
    frame = _frame(
        _Measurement, [{'id': 'b', 'value': math.inf, 'samples': []}, {'id': 'a', 'value': 1.0, 'samples': []}]
    )

    assert _failed_checks(_Measurement, frame) == {
        'value_is_finite: `value` must be finite.',
        'sorted_by_id: Rows must be sorted by `id`.',
    }


def test_a_check_on_an_undeclared_column_is_rejected() -> None:
    class Orphan(DatasetModel):
        dataset_name: ClassVar[str] = '_orphan'
        dataset_description: ClassVar[str] = 'orphan'
        example: ClassVar[dict] = {}
        x: str

    column_check(Orphan, 'missing')(value_is_finite)

    with pytest.raises(SchemaDefinitionError, match=r"undeclared column\(s\) \['missing'\]"):
        to_pandera(Orphan)


def test_single_primary_key_is_unique() -> None:
    frame = _frame(_Measurement, [{'id': 'a', 'value': 1.0, 'samples': []}] * 2)

    assert 'field_uniqueness' in _failed_checks(_Measurement, frame)


def test_composite_primary_key_is_unique_together_only() -> None:
    to_pandera(_Pair).validate(_frame(_Pair, [{'left': 'a', 'right': 1}, {'left': 'a', 'right': 2}]))

    assert 'multiple_fields_uniqueness' in _failed_checks(_Pair, _frame(_Pair, [{'left': 'a', 'right': 1}] * 2))


def test_undeclared_columns_are_rejected() -> None:
    frame = _frame(_Pair, [{'left': 'a', 'right': 1}]).with_columns(extra=pl.lit(0))

    assert 'column_in_schema' in _failed_checks(_Pair, frame)


def test_pattern_matches_anywhere_like_pydantic_does() -> None:
    class Loose(DatasetModel):
        dataset_name: ClassVar[str] = '_loose'
        dataset_description: ClassVar[str] = 'unanchored pattern'
        example: ClassVar[dict] = {}
        x: str = Field(pattern=r'\d{3}')

    to_pandera(Loose).validate(pl.DataFrame({'x': ['123', 'ab123', '123ab']}))
    Loose.model_validate({'x': 'ab123'})


def test_failures_name_the_nested_path_and_rule() -> None:
    frame = _frame(_Measurement, [{'id': 'a', 'value': -1.0, 'samples': [{'code': 'A1'}, {'code': 'bad'}, None]}])

    assert _failed_checks(_Measurement, frame) == {
        'value: >= 0.0',
        'samples[]: not null',
        'samples[]/code: matches ^[A-Z]\\d$',
    }


def test_nan_counts_as_null() -> None:
    class Floats(DatasetModel):
        dataset_name: ClassVar[str] = '_floats'
        dataset_description: ClassVar[str] = 'floats'
        example: ClassVar[dict] = {}
        required: float = Field(le=1.0)
        optional: float | None = Field(le=1.0)
        elements: list[float]

    frame = pl.DataFrame(
        {'required': [math.nan], 'optional': [math.nan], 'elements': [[math.nan]]}, schema=to_polars_schema(Floats)
    )

    assert _failed_checks(Floats, frame) == {'not_nullable', 'elements[]: not null'}
