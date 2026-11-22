"""Tests for ot_datamodels.derive: view columns copied from the outputs they come from."""

import dataclasses
from typing import Annotated, ClassVar, Literal

import pytest
from pydantic import Field, create_model

from ot_datamodels import (
    Bioregistry,
    ClickHouseType,
    DatasetModel,
    Note,
    PrimaryKey,
    Reference,
    SchemaDefinitionError,
    StructModel,
    from_output,
    row_of,
)
from ot_datamodels.discovery import discover_datasets
from ot_datamodels.spec import FieldSpec, Node, dataset_spec, walk
from ot_datamodels.vocabularies import Datasource, enum_of


class _Pair(StructModel):
    target_id: Annotated[str, Reference('target', 'id')] = Field(description='a target')
    rank: int


class _Out(DatasetModel):
    dataset_name: ClassVar[str] = 'out'
    dataset_title: ClassVar[str] = 'an output'
    dataset_description: ClassVar[str] = 'an output'
    example: ClassVar[dict] = {}

    id: Annotated[
        str,
        PrimaryKey(),
        Reference('target', 'id'),
        Bioregistry('ensembl'),
        ClickHouseType('LowCardinality(String)'),
    ] = Field(pattern=r'^ENSG\d{11}$', description='a target id')
    datasource_id: Annotated[Literal['eva'], enum_of(Datasource)] = Field(description='a datasource')
    score: float = Field(ge=0, le=1, description='a score')
    count: Annotated[int, ClickHouseType('UInt32')] = Field(ge=0, le=2**32 - 1, description='a count')
    label: Annotated[str | None, Note('free text')] = Field(min_length=1, description='a label')
    tags: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1, description='some tags')
    pair: _Pair | None = Field(description='a pair')


_COLUMNS = ('id', 'datasource_id', 'score', 'count', 'label', 'tags', 'pair')

Id = from_output(_Out, 'id')
DatasourceColumn = from_output(_Out, 'datasource_id')
Score = from_output(_Out, 'score')
Count = from_output(_Out, 'count')
Label = from_output(_Out, 'label')
Tags = from_output(_Out, 'tags')
Pair = from_output(_Out, 'pair')
RowScore = from_output(_Out, 'score')
NullableCount = from_output(_Out, 'count', nullable=True)
RequiredLabel = from_output(_Out, 'label', nullable=False)
KeyId = from_output(_Out, 'id', primary_key=True)
OutRow = row_of(_Out, ['score', 'pair'], name='OutRow')


class _View(DatasetModel):
    dataset_name: ClassVar[str] = 'view'
    dataset_title: ClassVar[str] = 'a view'
    dataset_description: ClassVar[str] = 'a view'
    example: ClassVar[dict] = {}

    id: Id
    datasource_id: DatasourceColumn
    score: Score
    count: Count
    label: Label
    tags: Tags
    pair: Pair
    row_score: RowScore
    nullable_count: NullableCount
    required_label: RequiredLabel
    rows: list[OutRow]


class _KeyedView(DatasetModel):
    dataset_name: ClassVar[str] = 'keyed_view'
    dataset_title: ClassVar[str] = 'a keyed view'
    dataset_description: ClassVar[str] = 'a keyed view'
    example: ClassVar[dict] = {}

    id: KeyId


def _field(model: type[DatasetModel], name: str) -> FieldSpec:
    return next(f for f in dataset_spec(model).fields if f.name == name)


def _subtree(model: type[DatasetModel], column: str) -> dict[str, Node]:
    """Every value under a column, with paths relative to it."""
    return {
        path.removeprefix(column): node
        for path, node in walk(dataset_spec(model))
        if path == column or path.startswith((f'{column}[]', f'{column}/'))
    }


@pytest.mark.parametrize(('view_column', 'out_column'), [*((c, c) for c in _COLUMNS), ('row_score', 'score')])
def test_from_output_keeps_spec_metadata(view_column: str, out_column: str) -> None:
    assert _subtree(_View, view_column) == _subtree(_Out, out_column)
    derived, output = _field(_View, view_column), _field(_Out, out_column)
    assert (derived.description, derived.note, derived.required) == (output.description, output.note, output.required)


def test_from_output_keeps_the_markers_it_copies() -> None:
    node = _field(_View, 'id').node
    assert node.clickhouse == 'LowCardinality(String)'
    assert node.reference == Reference('target', 'id')
    assert node.bioregistry == 'ensembl'
    assert node.constraints.pattern == r'^ENSG\d{11}$'
    assert _field(_View, 'datasource_id').node.dtype == _field(_Out, 'datasource_id').node.dtype


def test_from_output_nullable_override() -> None:
    derived, output = _field(_View, 'nullable_count').node, _field(_Out, 'count').node
    assert derived.nullable
    assert not output.nullable
    assert derived == dataclasses.replace(output, nullable=True)

    required, nullable = _field(_View, 'required_label').node, _field(_Out, 'label').node
    assert nullable.nullable
    assert required == dataclasses.replace(nullable, nullable=False)


def test_from_output_drops_primary_key_unless_asked() -> None:
    assert _field(_Out, 'id').primary_key
    assert not _field(_View, 'id').primary_key
    assert dataset_spec(_View).primary_key == ()
    assert dataset_spec(_KeyedView).primary_key == ('id',)


def test_from_output_rejects_an_unknown_field() -> None:
    with pytest.raises(SchemaDefinitionError, match="_Out has no field 'nope'"):
        from_output(_Out, 'nope')


def test_row_of_projects_fields() -> None:
    assert issubclass(OutRow, StructModel)
    assert not issubclass(OutRow, DatasetModel)
    assert OutRow.__name__ == 'OutRow'
    element = _field(_View, 'rows').node.element
    assert element is not None
    assert [(f.name, f.node, f.description) for f in element.fields] == [
        (name, _field(_Out, name).node, _field(_Out, name).description) for name in ('score', 'pair')
    ]


IdRow = row_of(_Out, ['id'], name='IdRow')


class _RowView(DatasetModel):
    dataset_name: ClassVar[str] = 'row_view'
    dataset_title: ClassVar[str] = 'a row view'
    dataset_description: ClassVar[str] = 'a row view'
    example: ClassVar[dict] = {}

    rows: list[IdRow]


def test_row_of_drops_primary_key() -> None:
    element = _field(_RowView, 'rows').node.element
    assert element is not None
    assert element.fields[0].node == _field(_Out, 'id').node


def test_row_of_nullable_override() -> None:
    row = row_of(_Out, ['score', 'label'], name='NullableRow', nullable={'score': True, 'label': False})
    assert row.model_fields['score'].annotation == float | None
    assert row.model_fields['label'].annotation is str


def test_row_of_rejects_unknown_fields() -> None:
    with pytest.raises(SchemaDefinitionError, match="_Out has no field 'nope'"):
        row_of(_Out, ['score', 'nope'], name='BadRow')
    with pytest.raises(SchemaDefinitionError, match="nullable names 'gone'"):
        row_of(_Out, ['score'], name='BadRow', nullable={'gone': True})
    with pytest.raises(SchemaDefinitionError, match="name 'Bad Row' is not a valid identifier"):
        row_of(_Out, ['score'], name='Bad Row')


class _Extended(_Out):
    dataset_name: ClassVar[str] = 'extended'
    dataset_title: ClassVar[str] = 'an extended output'
    dataset_description: ClassVar[str] = 'an extended output'
    example: ClassVar[dict] = {}

    extra: bool = Field(description='an extra column')


def test_subclass_inherits_columns() -> None:
    fields = dataset_spec(_Extended).fields
    assert [f.name for f in fields] == [*_COLUMNS, 'extra']
    assert fields[: len(_COLUMNS)] == dataset_spec(_Out).fields


@pytest.mark.parametrize(('name', 'output'), sorted(discover_datasets().items()))
def test_every_output_column_derives_unchanged(name: str, output: type[DatasetModel]) -> None:
    keyed = {f.name for f in dataset_spec(output).fields if f.primary_key}
    view = create_model(
        f'{output.__name__}Copy',
        __base__=DatasetModel,
        **{f: from_output(output, f, primary_key=f in keyed) for f in output.model_fields},
    )
    view.dataset_name, view.dataset_title, view.dataset_description, view.example = name, '', '', {}
    assert dataset_spec(view).fields == dataset_spec(output).fields
