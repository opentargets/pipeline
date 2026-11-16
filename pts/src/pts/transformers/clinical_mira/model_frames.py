"""Polars frames of pydantic response models: the schema a model's ``model_dump`` has, and rows of models."""

from __future__ import annotations

from types import NoneType, UnionType
from typing import TYPE_CHECKING, Any, Literal, Union, get_args, get_origin

import polars as pl
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Sequence

POLARS_SCALARS: dict[Any, type[pl.DataType]] = {str: pl.String, int: pl.Int64, float: pl.Float64, bool: pl.Boolean}
"""Polars dtype of each scalar type a response model field can hold."""


def polars_schema(model_class: type[BaseModel]) -> pl.Schema:
    """The polars schema of ``model_class``'s ``model_dump``: one column per field, in declared order.

    Strings, numbers, booleans, string ``Literal``s, nested models, lists and optionals are supported. Enums,
    ``Annotated`` types inside a container, and fields excluded from ``model_dump`` are not.

    Raises:
        TypeError: If a field is excluded from ``model_dump`` or its type has no polars dtype here, naming the field.
    """
    schema = {}
    for name, field in model_class.model_fields.items():
        if field.exclude:
            msg = f'{model_class.__name__}.{name} is excluded from model_dump'
            raise TypeError(msg)
        try:
            schema[name] = _polars_dtype(field.annotation)
        except TypeError as exc:
            msg = f'{model_class.__name__}.{name}: {exc}'
            raise TypeError(msg) from exc
    return pl.Schema(schema)


def models_to_frame(models: Sequence[BaseModel], model_class: type[BaseModel]) -> pl.DataFrame:
    """One row per model, with the schema :py:func:`polars_schema` derives from ``model_class``."""
    return pl.DataFrame([model.model_dump() for model in models], schema=polars_schema(model_class))


def _polars_dtype(annotation: Any) -> pl.DataType:
    """The polars dtype of a response model field's type."""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, UnionType):
        present = [arg for arg in args if arg is not NoneType]
        if len(present) == 1:
            return _polars_dtype(present[0])
    elif origin is Literal and all(isinstance(arg, str) for arg in args):
        return pl.String()
    elif origin is list:
        return pl.List(_polars_dtype(args[0]))
    elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return pl.Struct(polars_schema(annotation))
    elif annotation in POLARS_SCALARS:
        return POLARS_SCALARS[annotation]()
    msg = f'no polars dtype for {annotation!r}'
    raise TypeError(msg)
