"""Tests for pts.schemas.dataset.spec: how model annotations become the field tree."""

import datetime
from typing import Annotated, Any, ClassVar, Literal

import polars as pl
import pytest
from pydantic import Field, computed_field, create_model, field_validator, model_validator
from pydantic.fields import FieldInfo

from pts.schemas.dataset import (
    DatasetModel,
    Note,
    PolarsDtype,
    PrimaryKey,
    Reference,
    SchemaDefinitionError,
    StructModel,
)
from pts.schemas.dataset.spec import Constraints, DatasetSpec, Node, dataset_spec, walk


def _dataset(**fields: Any) -> type[DatasetModel]:
    """A concrete dataset model with the given `name=annotation` or `name=(annotation, Field(...))` fields."""
    definitions: dict[str, Any] = {
        name: spec if isinstance(spec, tuple) else (spec, ...) for name, spec in fields.items()
    }
    model = create_model('_Model', __base__=DatasetModel, **definitions)
    model.dataset_name = 'model'
    model.dataset_description = 'a model'
    model.example = {}
    return model


def _spec(**fields: Any) -> DatasetSpec:
    return dataset_spec(_dataset(**fields))


def _node(annotation: Any, field: FieldInfo | None = None) -> Node:
    return _spec(x=(annotation, field if field is not None else ...)).fields[0].node


class _Pair(StructModel):
    left: str
    right: int


class _Inner(StructModel):
    code: str


class _Middle(StructModel):
    inners: list[_Inner]


class _Outer(StructModel):
    middle: _Middle
    label: str | None


@pytest.mark.parametrize(
    ('annotation', 'field', 'dtype', 'nullable', 'constraints'),
    [
        (str, None, pl.String(), False, Constraints()),
        (str | None, None, pl.String(), True, Constraints()),
        (bool, None, pl.Boolean(), False, Constraints()),
        (int, None, pl.Int64(), False, Constraints()),
        (Annotated[int, PolarsDtype(pl.UInt32)], None, pl.UInt32(), False, Constraints()),
        (Annotated[float, PolarsDtype(pl.Float32())], None, pl.Float32(), False, Constraints()),
        (float, Field(ge=0.0, lt=1.0), pl.Float64(), False, Constraints(ge=0.0, lt=1.0)),
        (int, Field(gt=0, le=9), pl.Int64(), False, Constraints(gt=0, le=9)),
        (str | None, Field(pattern=r'^\d+$'), pl.String(), True, Constraints(pattern=r'^\d+$')),
        (str, Field(min_length=1, max_length=3), pl.String(), False, Constraints(min_length=1, max_length=3)),
        (Literal['a', 'b'], None, pl.String(), False, Constraints(allowed=('a', 'b'))),
        (Literal[1, 2] | None, None, pl.Int64(), True, Constraints(allowed=(1, 2))),
        (list[str], Field(min_length=1), pl.List(pl.String), False, Constraints(min_length=1)),
        (_Pair, None, pl.Struct({'left': pl.String, 'right': pl.Int64}), False, Constraints()),
    ],
)
def test_maps_annotations_to_dtype_nullability_and_constraints(
    annotation: Any, field: FieldInfo | None, dtype: pl.DataType, nullable: bool, constraints: Constraints
) -> None:
    node = _node(annotation, field)

    assert node.dtype == dtype
    assert node.nullable is nullable
    assert node.constraints == constraints


def test_list_element_carries_its_own_nullability_constraints_and_markers() -> None:
    node = _node(list[Annotated[str, Field(pattern=r'^\d+$'), Reference('target', 'id')] | None] | None)

    assert node.nullable is True
    assert node.element is not None
    assert node.element.nullable is True
    assert node.element.constraints == Constraints(pattern=r'^\d+$')
    assert node.element.reference == Reference('target', 'id')


def test_struct_dtype_nests_to_any_depth_in_declaration_order() -> None:
    node = _node(list[_Outer])

    assert node.dtype == pl.List(
        pl.Struct({'middle': pl.Struct({'inners': pl.List(pl.Struct({'code': pl.String}))}), 'label': pl.String})
    )
    assert [path for path, _ in walk(_spec(x=list[_Outer]))] == [
        'x',
        'x[]',
        'x[]/middle',
        'x[]/middle/inners',
        'x[]/middle/inners[]',
        'x[]/middle/inners[]/code',
        'x[]/label',
    ]


def test_field_level_facts() -> None:
    spec = _spec(
        key=(Annotated[str, PrimaryKey()], Field(description='the key')),
        optional=(Annotated[str | None, Note('may be absent')], None),
    )
    key, optional = spec.fields

    assert spec.primary_key == ('key',)
    assert (key.description, key.required, key.primary_key) == ('the key', True, True)
    assert (optional.required, optional.note) == (False, 'may be absent')


def test_concrete_dataset_detection() -> None:
    class Base(DatasetModel):
        shared: str

    class Concrete(Base):
        dataset_name: ClassVar[str] = 'concrete'
        dataset_description: ClassVar[str] = 'concrete'
        example: ClassVar[dict] = {}

    assert not Base.is_concrete()
    assert Concrete.is_concrete()
    assert [f.name for f in dataset_spec(Concrete).fields] == ['shared']
    with pytest.raises(SchemaDefinitionError, match='shared base'):
        dataset_spec(Base)


class _Plain(StructModel):
    x: str


class _WithDefault(StructModel):
    x: str = 'a'


class _KeyedStruct(StructModel):
    x: Annotated[str, PrimaryKey()]


class _Recursive(StructModel):
    children: list['_Recursive']


_Recursive.model_rebuild()


@pytest.mark.parametrize(
    ('fields', 'message'),
    [
        ({'x': int | str}, 'only `T | None`'),
        ({'x': dict[str, str]}, 'unsupported type'),
        ({'x': datetime.date}, 'unsupported type'),
        ({'x': list}, 'needs its element type'),
        ({'x': Literal['a', 1]}, 'all str or all int'),
        ({'x': (int, Field(multiple_of=2))}, 'unsupported annotation'),
        ({'x': (str, Field(alias='y'))}, 'alias'),
        ({'x': (str, Field(json_schema_extra={'k': 'v'}))}, 'json_schema_extra'),
        ({'x': Annotated[list[str], Reference('a', 'b')]}, 'on the list element type'),
        ({'x': Annotated[int, PolarsDtype(pl.Float32)]}, 'does not fit int'),
        ({'x': Annotated[str, PolarsDtype(pl.Categorical)]}, 'does not fit str'),
        ({'x': list[Annotated[str, Note('n')]]}, 'must annotate the field itself'),
        ({'x': list[Annotated[str, Field(description='d')]]}, 'may only carry constraints'),
        ({'x': Annotated[str, Reference('a', 'b'), Reference('c', 'd')]}, 'more than once'),
        ({'x': (Annotated[str | None, PrimaryKey()], ...)}, 'must be required and non-null'),
        ({'x': (Annotated[str, PrimaryKey()], 'a')}, 'must be required and non-null'),
        ({'x': _WithDefault}, 'cannot have a default'),
        ({'x': _KeyedStruct}, 'only applies to top-level columns'),
        ({'x': _Recursive}, 'contains itself'),
        ({'x': create_model('_Bare', y=(str, ...))}, 'must subclass StructModel'),
    ],
)
def test_rejects_what_it_cannot_translate(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(SchemaDefinitionError, match=message):
        _spec(**fields)


def test_rejects_validators_serializers_and_computed_fields() -> None:
    class WithFieldValidator(DatasetModel):
        x: str

        @field_validator('x')
        @classmethod
        def strip(cls, value: str) -> str:
            return value.strip()

    class WithModelValidator(DatasetModel):
        x: str

        @model_validator(mode='after')
        def check(self) -> 'WithModelValidator':
            return self

    class WithComputedField(DatasetModel):
        x: str

        @computed_field
        @property
        def y(self) -> str:
            return self.x

    for model, kind in [
        (WithFieldValidator, 'field_validators'),
        (WithModelValidator, 'model_validators'),
        (WithComputedField, 'computed_fields'),
    ]:
        model.dataset_name = 'm'
        model.dataset_description = 'm'
        model.example = {}
        with pytest.raises(SchemaDefinitionError, match=kind):
            dataset_spec(model)


def test_rejects_a_dataset_without_description_or_example() -> None:
    class Incomplete(DatasetModel):
        dataset_name: ClassVar[str] = 'incomplete'
        x: str

    with pytest.raises(SchemaDefinitionError, match='dataset_description, example'):
        dataset_spec(Incomplete)


def test_constraints_as_dict_lists_only_what_is_set() -> None:
    assert Constraints(ge=0, allowed=('a', 'b')).as_dict() == {'ge': 0, 'allowed': ['a', 'b']}
    assert _node(_Plain).constraints.as_dict() == {}
