"""Derive a view's columns from the output columns they copy, so the two cannot drift apart.

A whole output is copied by subclassing it (`class TargetView(Target)`). A single column is copied
with `from_output`, and a nested row of chosen columns with `row_of`. Both return a type to bind to
a module-level name and use as an annotation (type checkers reject a call inside one)::

    Score = from_output(Evidence, 'score')
    HomologueRow = row_of(Homology, ['targetId', 'type'], name='HomologueRow')


    class AssociationView(DatasetModel):
        rowScore: Score
        homologues: list[HomologueRow]

The copy keeps the output field's type, `Annotated` markers, constraints and description; the
column may be renamed. `PrimaryKey` is dropped unless asked for, and nullability is the one
override.
"""

import copy
import types
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from ot_datamodels.base import PrimaryKey, StructModel
from ot_datamodels.spec import SchemaDefinitionError


def from_output(model: type[BaseModel], field: str, *, nullable: bool | None = None, primary_key: bool = False) -> Any:
    """The annotation of an output field, to declare a view column (or struct field) as a copy of it.

    Args:
        model: the `DatasetModel` or `StructModel` holding the field.
        field: the field's name.
        nullable: `True` or `False` to override the field's nullability; `None` keeps it.
        primary_key: whether the copy is (part of) the view's primary key.

    Returns:
        `Annotated[<the field's type>, <a copy of its FieldInfo>]`.

    Raises:
        SchemaDefinitionError: if `model` has no such field.
    """
    info = _info(model, field)
    copied = copy.copy(info)
    copied.metadata = [item for item in info.metadata if not isinstance(item, PrimaryKey)]
    if primary_key:
        copied.metadata.append(PrimaryKey())
    annotation = _with_nullability(info.annotation, nullable)
    copied.annotation = annotation
    return Annotated[annotation, copied]


def row_of(
    model: type[BaseModel], fields: Sequence[str], *, name: str, nullable: Mapping[str, bool] | None = None
) -> Any:
    """A `StructModel` holding copies of chosen fields of an output, in the order given.

    Args:
        model: the `DatasetModel` or `StructModel` holding the fields.
        fields: the fields to copy.
        name: the struct's class name.
        nullable: per field, `True` or `False` to override its nullability.

    Returns:
        A `StructModel` subclass named `name`.

    Raises:
        SchemaDefinitionError: if `model` lacks a field, or `nullable` names one not copied.
    """
    nullable = nullable or {}
    unknown = sorted(set(nullable) - set(fields))
    if unknown:
        raise SchemaDefinitionError(f'row_of {name}: nullable names {", ".join(map(repr, unknown))}, not copied')
    definitions: dict[str, Any] = {f: from_output(model, f, nullable=nullable.get(f)) for f in fields}
    return create_model(name, __base__=StructModel, **definitions)


def _info(model: type[BaseModel], field: str) -> FieldInfo:
    try:
        return model.model_fields[field]
    except KeyError:
        raise SchemaDefinitionError(f'{model.__name__} has no field {field!r}') from None


def _with_nullability(annotation: Any, nullable: bool | None) -> Any:
    members = get_args(annotation) if get_origin(annotation) in {Union, types.UnionType} else (annotation,)
    is_nullable = type(None) in members
    if nullable is None or nullable == is_nullable:
        return annotation
    if nullable:
        return annotation | None
    non_null = [member for member in members if member is not type(None)]
    if len(non_null) != 1:
        raise SchemaDefinitionError(f'cannot make {annotation!r} non-nullable')
    return non_null[0]
