"""Cast a dataset to its model's dtypes before it is written, so its closed vocabularies are stored as Enums.

Only what differs from the model is cast, nested values included; a column or struct field the model does not
declare is kept as it is, for validation to report. The cast is strict: a value outside an Enum raises.
"""

from collections.abc import Callable, Iterator
from typing import overload

import polars as pl
from ot_datamodels.base import DatasetModel
from ot_datamodels.polars import to_polars_schema

#: Values shown per path in an error.
EXAMPLES = 5

#: A dtype, as polars' stubs type a nested one: an instance or a bare class.
Dtype = pl.DataType | pl.datatypes.DataTypeClass

#: Whether a leaf of the dtype read is cast to the model's.
LeafRule = Callable[[Dtype, Dtype], bool]


@overload
def conform(frame: pl.DataFrame, model: type[DatasetModel]) -> pl.DataFrame: ...
@overload
def conform(frame: pl.LazyFrame, model: type[DatasetModel]) -> pl.LazyFrame: ...
def conform(frame: pl.DataFrame | pl.LazyFrame, model: type[DatasetModel]) -> pl.DataFrame | pl.LazyFrame:
    """`frame` with every column cast to its dtype in `model`.

    A `LazyFrame` is cast when it is collected or sunk; polars' error then names the column and the value.

    Raises:
        ValueError: for a `DataFrame` with a value outside an Enum, naming the dataset, the value's path and
            the values.
    """
    casts = targets(frame.collect_schema(), model)
    if not casts:
        return frame
    if isinstance(frame, pl.LazyFrame):
        return frame.with_columns(pl.col(name).cast(dtype, strict=True) for name, dtype in casts.items())
    return _cast_eager(frame, casts, model)


def _cast_eager(frame: pl.DataFrame, casts: dict[str, Dtype], model: type[DatasetModel]) -> pl.DataFrame:
    try:
        return frame.with_columns(pl.col(name).cast(dtype, strict=True) for name, dtype in casts.items())
    except pl.exceptions.InvalidOperationError as error:
        unknown = [
            f'{path}: {values}'
            for name, dtype in casts.items()
            for path, values, _ in unknown_values(frame[name], dtype, name)
        ]
        detail = '; '.join(unknown) if unknown else str(error)
        raise ValueError(f'{model.dataset_name} does not conform to {model.__name__}: {detail}') from error


def targets(schema: pl.Schema, model: type[DatasetModel], leaf: LeafRule | None = None) -> dict[str, Dtype]:
    """The dtype each column of `schema` is cast to, for the columns that differ from `model`.

    Args:
        schema: the dtypes read.
        model: the dataset model.
        leaf: which differing leaves are cast; every one by default.
    """
    wanted = to_polars_schema(model)
    casts: dict[str, Dtype] = {}
    for name, have in schema.items():
        if name in wanted and (dtype := _target(have, wanted[name], leaf or _any)) != have:
            casts[name] = dtype
    return casts


def conformed_schema(schema: pl.Schema, model: type[DatasetModel]) -> pl.Schema:
    """The schema a frame of `schema` has once conformed to `model`."""
    return pl.Schema({**schema, **targets(schema, model)})


def strings_to_enums(have: Dtype, want: Dtype) -> bool:
    """A `LeafRule` casting only a string the model stores as an Enum."""
    return isinstance(want, pl.Enum) and have == pl.String


def unknown_values(series: pl.Series, dtype: Dtype, path: str) -> Iterator[tuple[str, list[str], int]]:
    """The values of `series` outside the Enums of `dtype`: per path, a few distinct values and their count."""
    if isinstance(dtype, pl.Enum):
        if series.dtype == pl.String or isinstance(series.dtype, (pl.Enum, pl.Categorical)):
            values = series.cast(pl.String).drop_nulls()
            outside = values.filter(~values.is_in(dtype.categories.implode()))
            if outside.len():
                yield path, outside.unique(maintain_order=True).head(EXAMPLES).to_list(), outside.len()
    elif isinstance(dtype, pl.List | pl.Array) and isinstance(series.dtype, pl.List | pl.Array):
        yield from unknown_values(series.explode(), dtype.inner, f'{path}[]')
    elif isinstance(dtype, pl.Struct) and isinstance(series.dtype, pl.Struct):
        names = {field.name for field in series.dtype.fields}
        for field in dtype.fields:
            if field.name in names:
                yield from unknown_values(series.struct.field(field.name), field.dtype, f'{path}/{field.name}')


@overload
def without_enums(frame: pl.DataFrame) -> pl.DataFrame: ...
@overload
def without_enums(frame: pl.LazyFrame) -> pl.LazyFrame: ...
def without_enums(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """`frame` with every Enum in it, at any depth, cast to a string, for a reader mixing it with strings."""
    schema = frame.collect_schema()
    casts = {name: enums_as_strings(dtype) for name, dtype in schema.items()}
    return frame.cast({name: dtype for name, dtype in casts.items() if dtype != schema[name]})


def enums_as_strings(dtype: Dtype) -> Dtype:
    """`dtype` with every Enum in it, at any depth, a string."""
    if isinstance(dtype, pl.Enum):
        return pl.String()
    if isinstance(dtype, pl.List):
        return pl.List(enums_as_strings(dtype.inner))
    if isinstance(dtype, pl.Struct):
        return pl.Struct([pl.Field(f.name, enums_as_strings(f.dtype)) for f in dtype.fields])
    return dtype


def _any(have: Dtype, want: Dtype) -> bool:
    return True


def _target(have: Dtype, want: Dtype, leaf: LeafRule) -> Dtype:
    """`have` with each leaf the model declares replaced by the model's dtype, as `leaf` allows."""
    if have == want:
        return have
    if isinstance(have, pl.Struct) and isinstance(want, pl.Struct):
        wanted = {field.name: field.dtype for field in want.fields}
        return pl.Struct([
            pl.Field(f.name, _target(f.dtype, wanted[f.name], leaf) if f.name in wanted else f.dtype)
            for f in have.fields
        ])
    if isinstance(have, pl.List) and isinstance(want, pl.List):
        return pl.List(_target(have.inner, want.inner, leaf))
    if have == pl.Null:
        return want if leaf(have, want) else have
    if _nested(have) or _nested(want):
        return have
    return want if leaf(have, want) else have


def _nested(dtype: Dtype) -> bool:
    return isinstance(dtype, pl.List | pl.Array | pl.Struct)
