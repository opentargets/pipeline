"""Turn a `DatasetModel` into a neutral field tree.

The one place that interprets a model's annotations. The pandera converter and the documentation
generator both read the tree built here, so they cannot disagree about what a model means, and a
further target (a Spark `StructType`, say) only needs another reader of the same tree.

Only a strict subset of Pydantic is supported, and anything outside it raises
`SchemaDefinitionError` instead of being silently left unenforced:

* types: `str`, `bool`, `int`, `float` (see `PolarsDtype` to change their Polars precision),
  `Literal[...]` of all-`str` or all-`int` values, `list[T]`, `StructModel` subclasses, and
  `T | None` for any of those;
* constraints: `ge`/`gt`/`le`/`lt` on numbers, `pattern`/`min_length`/`max_length` on strings,
  `min_length`/`max_length` on lists -- on a column, a struct field or a list element alike
  (`list[Annotated[str, Field(pattern=...)]]`);
* markers: `PrimaryKey` and `Note` on a top-level column (`Note` on a struct field too),
  `Reference`/`Bioregistry`/`PolarsDtype` on any primitive value;
* a default on a top-level column, making it optional (the column may be absent).

Everything Pydantic validates in code rather than declaratively -- validators, serializers,
computed fields -- is rejected, as are aliases and discriminators.
"""

import dataclasses
import functools
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import annotated_types as at
import polars as pl
from pydantic import BaseModel
from pydantic._internal._fields import _general_metadata_cls
from pydantic.fields import FieldInfo

from pts.schemas.dataset.base import (
    Bioregistry,
    DatasetModel,
    Note,
    PolarsDtype,
    PrimaryKey,
    Reference,
    StructModel,
)


class SchemaDefinitionError(ValueError):
    """A dataset model uses something the schema tooling cannot translate faithfully."""


#: The class Pydantic wraps `Field(pattern=...)` (and other non-`annotated_types` constraints) in.
#: Private, and only built on first use, hence fetched rather than imported; the test suite covers
#: `pattern` at every depth, so a change on Pydantic's side surfaces there.
_GeneralMetadata = _general_metadata_cls()

#: Default Polars dtype of each supported primitive.
_PRIMITIVES: dict[type, pl.DataType] = {
    str: pl.String(),
    bool: pl.Boolean(),
    int: pl.Int64(),
    float: pl.Float64(),
}

#: Constraints each kind of value accepts.
_STRING_CONSTRAINTS = frozenset({'pattern', 'min_length', 'max_length'})
_NUMBER_CONSTRAINTS = frozenset({'ge', 'gt', 'le', 'lt'})
_LIST_CONSTRAINTS = frozenset({'min_length', 'max_length'})

#: `FieldInfo` attributes that change what a field means in ways the converter cannot follow.
_REJECTED_FIELD_ATTRIBUTES = ('alias', 'validation_alias', 'serialization_alias', 'discriminator', 'json_schema_extra')

#: Pydantic decorator kinds that validate or reshape data in code.
_REJECTED_DECORATORS = (
    'validators',
    'field_validators',
    'root_validators',
    'model_validators',
    'field_serializers',
    'model_serializers',
    'computed_fields',
)


@dataclass(frozen=True)
class Constraints:
    """Declarative rules on one value. `None` means unconstrained."""

    ge: float | None = None
    gt: float | None = None
    le: float | None = None
    lt: float | None = None
    pattern: str | None = None
    min_length: int | None = None
    max_length: int | None = None
    allowed: tuple[str | int, ...] | None = None

    def as_dict(self) -> dict[str, Any]:
        """The constraints that are set, JSON-ready."""
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in dataclasses.asdict(self).items()
            if value is not None
        }


@dataclass(frozen=True)
class Node:
    """One value -- a column, a struct field or a list element: its type and the rules on it.

    Attributes:
        kind: `primitive`, `list` or `struct`.
        dtype: the full Polars dtype of the value.
        nullable: whether the value may be null.
        constraints: declarative rules on the value itself.
        element: the element of a list.
        fields: the fields of a struct.
        reference: the dataset/column the value resolves against, if any.
        bioregistry: the identifier registry the value belongs to, if any.
    """

    kind: Literal['primitive', 'list', 'struct']
    dtype: pl.DataType
    nullable: bool
    constraints: Constraints = field(default_factory=Constraints)
    element: 'Node | None' = None
    fields: tuple['FieldSpec', ...] = ()
    reference: Reference | None = None
    bioregistry: str | None = None


@dataclass(frozen=True)
class FieldSpec:
    """A named field: a top-level column or a struct field.

    Attributes:
        name: the column or struct field name.
        node: its value.
        description: its description, if any.
        required: whether a top-level column must be present (always true for struct fields).
        primary_key: whether a top-level column is (part of) the primary key.
        note: a free-text remark, if any.
    """

    name: str
    node: Node
    description: str | None
    required: bool
    primary_key: bool
    note: str | None


@dataclass(frozen=True, eq=False)
class DatasetSpec:
    """A whole dataset.

    Attributes:
        name: the dataset name.
        description: one-line description of the dataset.
        fields: its top-level columns, in declaration order.
        example: one valid row.
    """

    name: str
    description: str
    fields: tuple[FieldSpec, ...]
    example: dict[str, Any]

    @property
    def primary_key(self) -> tuple[str, ...]:
        """The primary-key columns, in declaration order."""
        return tuple(f.name for f in self.fields if f.primary_key)


@functools.cache
def dataset_spec(model: type[DatasetModel]) -> DatasetSpec:
    """Build the field tree of a concrete dataset model.

    Args:
        model: a concrete `DatasetModel` subclass.

    Returns:
        The model's `DatasetSpec`.

    Raises:
        SchemaDefinitionError: if `model` is not a concrete dataset, or uses anything outside the
            supported subset (see the module docstring).
    """
    if not (isinstance(model, type) and issubclass(model, DatasetModel)):
        raise SchemaDefinitionError(f'{model!r} is not a DatasetModel subclass')
    if not model.is_concrete():
        msg = f'{model.__name__} sets no dataset_name of its own: it is a shared base, not a dataset'
        raise SchemaDefinitionError(msg)
    missing = [name for name in ('dataset_description', 'example') if not hasattr(model, name)]
    if missing:
        raise SchemaDefinitionError(f'{model.__name__} does not set {", ".join(missing)}')

    fields = _fields(model, prefix='', top_level=True, seen=(model,))
    primary_key = [f for f in fields if f.primary_key]
    nullable_keys = [f.name for f in primary_key if f.node.nullable or not f.required]
    if nullable_keys:
        raise SchemaDefinitionError(
            f'{model.__name__}: primary-key column(s) {nullable_keys} must be required and non-null'
        )
    return DatasetSpec(model.dataset_name, model.dataset_description, fields, model.example)


def walk(spec: DatasetSpec) -> Iterator[tuple[str, Node]]:
    """Every value in a dataset with its path, depth first.

    Paths read `column`, `column[]` (a list element) and `column/field` (a struct field), so a field
    of a struct inside a list is `column[]/field`.
    """
    for f in spec.fields:
        yield from _walk(f.node, f.name)


def _walk(node: Node, path: str) -> Iterator[tuple[str, Node]]:
    yield path, node
    if node.element is not None:
        yield from _walk(node.element, f'{path}[]')
    for f in node.fields:
        yield from _walk(f.node, f'{path}/{f.name}')


def _fields(model: type[BaseModel], *, prefix: str, top_level: bool, seen: tuple[type, ...]) -> tuple[FieldSpec, ...]:
    used = [kind for kind in _REJECTED_DECORATORS if getattr(model.__pydantic_decorators__, kind)]
    if used:
        msg = f'{model.__name__} uses {", ".join(used)}, which cannot be translated to column checks'
        raise SchemaDefinitionError(msg)
    return tuple(
        _field(name, info, path=f'{prefix}{name}', top_level=top_level, seen=seen)
        for name, info in model.model_fields.items()
    )


def _field(name: str, info: FieldInfo, *, path: str, top_level: bool, seen: tuple[type, ...]) -> FieldSpec:
    rejected = [attribute for attribute in _REJECTED_FIELD_ATTRIBUTES if getattr(info, attribute) is not None]
    if rejected:
        raise SchemaDefinitionError(f'{path}: {", ".join(rejected)} not supported')
    required = info.is_required()
    if not required and not top_level:
        msg = f'{path}: a struct field cannot have a default, since every struct value carries every field'
        raise SchemaDefinitionError(msg)

    primary_key = False
    notes: list[str] = []
    metadata: list[Any] = []
    for item in info.metadata:
        if isinstance(item, PrimaryKey):
            if not top_level:
                raise SchemaDefinitionError(f'{path}: PrimaryKey only applies to top-level columns')
            primary_key = True
        elif isinstance(item, Note):
            notes.append(item.text)
        else:
            metadata.append(item)
    if len(notes) > 1:
        raise SchemaDefinitionError(f'{path}: more than one Note')

    node = _node(info.annotation, metadata, path=path, seen=seen)
    return FieldSpec(name, node, info.description, required, primary_key, notes[0] if notes else None)


def _node(annotation: Any, metadata: list[Any], *, path: str, seen: tuple[type, ...]) -> Node:
    annotation, metadata = _unwrap_annotated(annotation, metadata, path)
    nullable = False
    if get_origin(annotation) in {Union, types.UnionType}:
        members = get_args(annotation)
        non_null = [member for member in members if member is not type(None)]
        if len(members) != 2 or len(non_null) != 1:
            raise SchemaDefinitionError(f'{path}: unsupported union {annotation!r}, only `T | None` is supported')
        nullable = True
        annotation, metadata = _unwrap_annotated(non_null[0], metadata, path)

    constraints, reference, bioregistry, dtype_override = _classify(metadata, path)

    if annotation is list or get_origin(annotation) is list:
        if dtype_override is not None or reference is not None or bioregistry is not None:
            raise SchemaDefinitionError(f'{path}: put PolarsDtype/Reference/Bioregistry on the list element type')
        _accept_only(constraints, _LIST_CONSTRAINTS, path, 'a list')
        element = _node(_single_arg(annotation, path), [], path=f'{path}[]', seen=seen)
        return Node('list', pl.List(element.dtype), nullable, constraints, element=element)

    if get_origin(annotation) is Literal:
        values = get_args(annotation)
        kinds = {type(value) for value in values}
        if kinds == {str}:
            if dtype_override is not None:
                raise SchemaDefinitionError(f'{path}: PolarsDtype does not apply to a str Literal')
            dtype = pl.String()
        elif kinds == {int}:
            dtype = _override(pl.Int64(), dtype_override, int, path)
        else:
            raise SchemaDefinitionError(f'{path}: Literal values must be all str or all int, got {values!r}')
        _accept_only(constraints, frozenset(), path, 'a Literal')
        return Node(
            'primitive', dtype, nullable, Constraints(allowed=values), reference=reference, bioregistry=bioregistry
        )

    if isinstance(annotation, type) and issubclass(annotation, StructModel):
        if annotation in seen:
            raise SchemaDefinitionError(f'{path}: {annotation.__name__} contains itself')
        if dtype_override is not None or reference is not None or bioregistry is not None:
            raise SchemaDefinitionError(f'{path}: PolarsDtype/Reference/Bioregistry do not apply to a struct')
        _accept_only(constraints, frozenset(), path, 'a struct')
        fields = _fields(annotation, prefix=f'{path}/', top_level=False, seen=(*seen, annotation))
        dtype = pl.Struct([pl.Field(f.name, f.node.dtype) for f in fields])
        return Node('struct', dtype, nullable, fields=fields)

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        raise SchemaDefinitionError(f'{path}: nested model {annotation.__name__} must subclass StructModel')

    if isinstance(annotation, type) and annotation in _PRIMITIVES:
        dtype = _override(_PRIMITIVES[annotation], dtype_override, annotation, path)
        accepted = (
            _STRING_CONSTRAINTS
            if annotation is str
            else _NUMBER_CONSTRAINTS
            if annotation in {int, float}
            else frozenset()
        )
        _accept_only(constraints, accepted, path, annotation.__name__)
        return Node('primitive', dtype, nullable, constraints, reference=reference, bioregistry=bioregistry)

    raise SchemaDefinitionError(f'{path}: unsupported type {annotation!r}')


def _unwrap_annotated(annotation: Any, metadata: list[Any], path: str) -> tuple[Any, list[Any]]:
    """Strip `Annotated[...]` layers, collecting their metadata (a nested `Field` contributes its own)."""
    metadata = list(metadata)
    while get_origin(annotation) is Annotated:
        annotation, *extras = get_args(annotation)
        for extra in extras:
            if isinstance(extra, FieldInfo):
                if extra.description is not None or not extra.is_required():
                    raise SchemaDefinitionError(f'{path}: a Field inside a type may only carry constraints')
                metadata.extend(extra.metadata)
            else:
                metadata.append(extra)
    return annotation, metadata


def _classify(
    metadata: list[Any], path: str
) -> tuple[Constraints, Reference | None, str | None, pl.DataType | type[pl.DataType] | None]:
    """Sort one value's metadata into constraints and markers, rejecting anything unrecognised."""
    constraints: dict[str, Any] = {}
    reference: Reference | None = None
    bioregistry: str | None = None
    dtype: pl.DataType | type[pl.DataType] | None = None

    def put(name: str, value: Any) -> None:
        if name in constraints:
            raise SchemaDefinitionError(f'{path}: {name} given more than once')
        constraints[name] = value

    for item in metadata:
        match item:
            case at.Ge(ge=value):
                put('ge', value)
            case at.Gt(gt=value):
                put('gt', value)
            case at.Le(le=value):
                put('le', value)
            case at.Lt(lt=value):
                put('lt', value)
            case at.MinLen(min_length=value):
                put('min_length', value)
            case at.MaxLen(max_length=value):
                put('max_length', value)
            case _GeneralMetadata():
                unsupported = sorted(set(vars(item)) - {'pattern'})
                if unsupported:
                    raise SchemaDefinitionError(f'{path}: {unsupported} not supported')
                put('pattern', item.pattern)
            case Reference() if reference is None:
                reference = item
            case Bioregistry(prefix=prefix) if bioregistry is None:
                bioregistry = prefix
            case PolarsDtype(dtype=override) if dtype is None:
                dtype = override
            case Reference() | Bioregistry() | PolarsDtype():
                raise SchemaDefinitionError(f'{path}: {type(item).__name__} given more than once')
            case PrimaryKey() | Note():
                raise SchemaDefinitionError(f'{path}: {type(item).__name__} must annotate the field itself')
            case _:
                raise SchemaDefinitionError(f'{path}: unsupported annotation {item!r}')
    return Constraints(**constraints), reference, bioregistry, dtype


def _accept_only(constraints: Constraints, accepted: frozenset[str], path: str, what: str) -> None:
    unsupported = sorted(set(constraints.as_dict()) - accepted)
    if unsupported:
        raise SchemaDefinitionError(f'{path}: {", ".join(unsupported)} not supported on {what}')


def _override(
    default: pl.DataType, override: pl.DataType | type[pl.DataType] | None, python_type: type, path: str
) -> pl.DataType:
    if override is None:
        return default
    dtype = override() if isinstance(override, type) else override
    fits = dtype.is_integer() if python_type is int else dtype.is_float() if python_type is float else False
    if not fits:
        raise SchemaDefinitionError(f'{path}: PolarsDtype({dtype}) does not fit {python_type.__name__}')
    return dtype


def _single_arg(annotation: Any, path: str) -> Any:
    args = get_args(annotation)
    if len(args) != 1:
        raise SchemaDefinitionError(f'{path}: a list needs its element type, e.g. list[str]')
    return args[0]
