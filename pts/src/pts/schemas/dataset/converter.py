"""Derive the pandera schema a dataset is validated against from its model.

Pandera validates a column's dtype, nullability, presence and uniqueness natively. Everything else
-- ranges, patterns, allowed values, lengths, and the nullability of list elements and struct
fields -- becomes one pandera check per rule, generated as a Polars expression. Using one
generator for top-level and nested values alike means a rule is enforced the same way at every
depth: `list[Annotated[str, Field(pattern=...)]]` checks each element exactly as a `str` column
with that pattern would be checked.

Semantics, chosen to agree with Pydantic validating the same row in strict mode (the test suite
checks this agreement for every dataset):

* `pattern` matches anywhere in the value (`str.contains`), as Pydantic's does -- anchor it with
  `^...$` to match the whole value. (pandera's own `str_matches` silently anchors the start, which
  Pydantic does not.)
* NaN counts as null, as pandera already treats it for top-level float columns: a non-nullable
  float rejects NaN, and constraints never fail on a NaN or null value -- nullability alone decides
  those.
* A null list, or a null struct, passes every rule on its elements or fields; whether the list or
  struct itself may be null is its own nullability.

A model's `row_checks` run after the derived checks, over the whole frame.
"""

import functools

import pandera.polars as pa
import polars as pl

from pts.schemas.dataset.base import DatasetModel, RowCheck
from pts.schemas.dataset.spec import Node, SchemaDefinitionError, dataset_spec


@functools.cache
def to_pandera(model: type[DatasetModel], *, unique: bool = True) -> pa.DataFrameSchema:
    """The pandera schema of a dataset: derived checks plus its `row_checks`.

    Args:
        model: the dataset model.
        unique: whether the schema checks that the primary key is unique. Without it, every check
            is decided within the rows it is given, so a dataset can be validated one part at a
            time and its key checked separately (`pts.schemas.dataset.validate`).

    Raises:
        SchemaDefinitionError: if the model is not convertible, or a row check reads a column the
            model does not declare.
    """
    return _build(model, model.row_checks, unique=unique)


@functools.cache
def to_base_pandera(model: type[DatasetModel], *, unique: bool = True) -> pa.DataFrameSchema:
    """The pandera schema of a dataset derived from its fields alone, without its `row_checks`.

    `unique` is as for `to_pandera`: without it, each row is judged on its own.
    """
    return _build(model, (), unique=unique)


def to_polars_schema(model: type[DatasetModel]) -> pl.Schema:
    """The Polars schema of a dataset: its columns and their dtypes, in declaration order."""
    return pl.Schema({f.name: f.node.dtype for f in dataset_spec(model).fields})


def node_conditions(node: Node, value: pl.Expr, path: str) -> list[tuple[str, pl.Expr]]:
    """Every rule on `value` and the values inside it, as `(description, passes)` pairs.

    `passes` is a boolean expression evaluated where `value` is: one result per row for a column,
    or per element inside `list.eval`. The nullability of `value` itself is not included -- for a
    column pandera checks that natively, and for a nested value its parent adds it.

    Args:
        node: the value's node in the field tree.
        value: an expression selecting the value.
        path: the value's path, used in each description.

    Returns:
        One pair per rule, in tree order.
    """
    conditions = [(f'{path}: {rule}', _ignoring_missing(passes, value, node)) for rule, passes in _rules(node, value)]

    if node.element is not None:
        element = pl.element()
        nested = _nullability(node.element, element, f'{path}[]') + node_conditions(node.element, element, f'{path}[]')
        conditions += [(rule, value.list.eval(passes).list.all().fill_null(True)) for rule, passes in nested]

    for child in node.fields:
        child_value = value.struct.field(child.name)
        child_path = f'{path}/{child.name}'
        nested = _nullability(child.node, child_value, child_path) + node_conditions(
            child.node, child_value, child_path
        )
        # not `when/then`: it panics on a multi-chunk frame once `passes` holds a `list.eval`
        conditions += [(rule, value.is_null() | passes) for rule, passes in nested]

    return conditions


def _build(model: type[DatasetModel], row_checks: tuple[RowCheck, ...], *, unique: bool = True) -> pa.DataFrameSchema:
    spec = dataset_spec(model)
    columns = {f.name for f in spec.fields}
    unknown = sorted({name for check in row_checks for name in check.passes.meta.root_names()} - columns)
    if unknown:
        raise SchemaDefinitionError(f'{model.__name__}: row checks read undeclared column(s) {unknown}')

    single_key = spec.primary_key[0] if unique and len(spec.primary_key) == 1 else None
    schema_columns = {
        f.name: pa.Column(
            f.node.dtype,
            checks=[_check(rule, passes) for rule, passes in node_conditions(f.node, pl.col(f.name), f.name)],
            nullable=f.node.nullable,
            required=f.required,
            unique=f.name == single_key,
            description=f.description,
        )
        for f in spec.fields
    }
    return pa.DataFrameSchema(
        schema_columns,
        checks=[_row_check(check) for check in row_checks],
        strict=True,
        unique=list(spec.primary_key) if unique and len(spec.primary_key) > 1 else None,
        name=spec.name,
        description=spec.description,
    )


def _rules(node: Node, value: pl.Expr) -> list[tuple[str, pl.Expr]]:
    """The node's own constraints, as `(rule, passes)` pairs on a non-missing value."""
    c = node.constraints
    if node.dtype.is_float():
        # compared at full precision, as Pydantic compares the stored value: a Float32 0.1 is above 0.1
        value = value.cast(pl.Float64)
    length = value.list.len() if node.kind == 'list' else value.str.len_chars()
    rules: list[tuple[str, pl.Expr | None]] = [
        (f'>= {c.ge}', value >= c.ge if c.ge is not None else None),
        (f'> {c.gt}', value > c.gt if c.gt is not None else None),
        (f'<= {c.le}', value <= c.le if c.le is not None else None),
        (f'< {c.lt}', value < c.lt if c.lt is not None else None),
        (f'matches {c.pattern}', value.str.contains(c.pattern) if c.pattern is not None else None),
        (f'length >= {c.min_length}', length >= c.min_length if c.min_length is not None else None),
        (f'length <= {c.max_length}', length <= c.max_length if c.max_length is not None else None),
        (f'one of {list(c.allowed or ())}', value.is_in(list(c.allowed)) if c.allowed is not None else None),
    ]
    return [(rule, passes) for rule, passes in rules if passes is not None]


def _ignoring_missing(passes: pl.Expr, value: pl.Expr, node: Node) -> pl.Expr:
    """A rule that passes on a null (or, for floats, NaN) value, leaving those to nullability."""
    if node.dtype.is_float():
        passes = passes | value.is_nan()
    return passes.fill_null(True)


def _nullability(node: Node, value: pl.Expr, path: str) -> list[tuple[str, pl.Expr]]:
    """The not-null rule of a nested value, if it is not nullable. NaN counts as null."""
    if node.nullable:
        return []
    present = value.is_not_null()
    if node.dtype.is_float():
        present &= ~value.is_nan()
    return [(f'{path}: not null', present)]


def _check(rule: str, passes: pl.Expr) -> pa.Check:
    # `ignore_na=False`: `passes` already decides missing values itself, so a null result is a failure
    return pa.Check(lambda data: data.lazyframe.select(passes), name=rule, error=rule, ignore_na=False)


def _row_check(check: RowCheck) -> pa.Check:
    return pa.Check(
        lambda data: data.lazyframe.select(check.passes), name=check.name, error=f'{check.name}: {check.description}'
    )
