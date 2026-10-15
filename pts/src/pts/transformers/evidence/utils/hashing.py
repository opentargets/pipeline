"""Rendering of evidence columns as strings, for the evidence `id` and the de-duplication hash."""

from __future__ import annotations

from typing import Any

import polars as pl


class UnsupportedIdentifierField(ValueError):  # noqa: N818 -- reads as a condition, not an error type
    """A dtype `render_for_hash` has no rendering for.

    Raising beats guessing: an unrenderable dtype means the frame carries something none of the
    measured shapes cover, and inventing a rendering for it would silently feed the de-duplication
    hash -- deciding which of a set of duplicate rows gets published.

    Dtype-level only. The VALUE-level guards this class also used to serve (doubles too large to
    render exactly like java, and low-popcount subnormals) were removed once matching the release's
    evidence `id` stopped being a requirement: they killed whole datasources over a digit in an
    identifier nobody diffs.
    """


def _cast_scalar_to_string(expr: pl.Expr, dtype: Any, name: str) -> pl.Expr:
    """Render a String/Int64/Boolean/Float64 leaf as a string, for hashing.

    Nulls are preserved (not substituted) here -- the caller decides whether a null means "the
    whole column is null" (real `None`, `render_for_hash`'s job) or "a nested element/field
    is null" (the literal token `'null'`, `_render_nested`'s job).

    String, Int64 and Boolean render identically to spark (measured, task-9-report.md): spark's
    long-to-string and boolean-to-string agree digit-for-digit and `true`/`false`-for-`true`/
    `false` with `.cast(pl.String)`.

    FLOAT64 DOES NOT MATCH SPARK, deliberately. Spark renders a double with java's
    `Double.toString`, which switches to scientific notation outside `0.001 <= |x| < 1e7` --
    polars' cast does not, so e.g. `1e-5` is `'1.0E-5'` in spark and `'0.00001'` here. That was
    reproduced by a per-element python UDF until matching the release's evidence `id` stopped
    being a requirement; the UDF measured 56x slower than the native cast (8.64s against 0.16s
    per 4M doubles) and 11% of real float values in the staged outputs fall outside the range
    where the two agree, so it was doing real work on real data for a guarantee nothing consumes.

    What the rendering must still be is FAITHFUL and DETERMINISTIC: it round-trips to the same
    double (`float(rendered) == value` for every finite value, including subnormals), so distinct
    values keep distinct renderings and the de-duplication hash still ranks rows meaningfully.

    Args:
        expr: the leaf expression to render.
        dtype: its polars dtype. Typed `Any` for the same reason as `_harmonise_expr`'s
            `source_dtype`/`target_dtype` -- see that function's docstring.
        name: the originating column name, threaded through only for the raise message.

    Returns:
        A string expression, null-preserving.

    Raises:
        UnsupportedIdentifierField: for a dtype this function has not been measured against.
    """
    if dtype in (pl.String, pl.Int64, pl.Boolean, pl.Float64):
        return expr.cast(pl.String)
    msg = f'render_for_hash: unsupported scalar dtype {dtype!r} for column {name!r}'
    raise UnsupportedIdentifierField(msg)


def _render_nested(expr: pl.Expr, dtype: Any, name: str) -> pl.Expr:
    """Render a struct field or list element the way spark's nested `CAST(... AS STRING)` does.

    Recurses through `Struct` and `List` the same way `_harmonise_expr` recurses through the
    target schema -- `evidence_schema`'s `biomarkers` is a `Struct` of `List(Struct)`, so a struct
    renderer that only handles one level deep cannot cover it. Measured against real spark
    (task-9-report.md), including that exact struct-of-list-of-struct shape: a null value at ANY
    nesting level -- a null leaf, a null nested list, or a null nested struct -- renders as the
    bare token `'null'`, not an empty/absent value (`{geneExpression: null} -> '{null}'`, not
    `'{[]}'`). So every branch below folds a null into the literal token `'null'` rather than
    propagating a real `None` up through the surrounding `{...}`/`[...]`.

    The measurements are per-shape (a struct's `{f1, f2, ...}` braces, a list's `[e1, e2, ...]`
    brackets, each leaf type's own rendering, and the null-anywhere-becomes-`'null'` rule); an
    arbitrary composition of them (e.g. a `List(Struct)` nested inside a `Struct` nested inside a
    `List`) is not separately re-measured for every possible shape, only verified for the one
    real schema needs (`biomarkers`) -- see the module's `_render_nested` spark-parity
    test.

    Args:
        expr: the field/element expression to render.
        dtype: its polars dtype (typed `Any`, same reason as `_cast_scalar_to_string`'s `dtype`).
        name: the originating column name (or a dotted field path), for the raise message.

    Returns:
        A string expression that is never null -- a null value renders as the token `'null'`.

    Raises:
        UnsupportedIdentifierField: for a dtype this function has not been measured against.
    """
    if isinstance(dtype, pl.Struct):
        fields = [_render_nested(expr.struct.field(f.name), f.dtype, f'{name}.{f.name}') for f in dtype.fields]
        rendered = pl.lit('{') + pl.concat_str(fields, separator=', ') + pl.lit('}')
        return pl.when(expr.is_null()).then(pl.lit('null')).otherwise(rendered)
    if isinstance(dtype, pl.List):
        element = expr.list.eval(_render_nested(pl.element(), dtype.inner, name))
        rendered = pl.lit('[') + element.list.join(', ') + pl.lit(']')
        return pl.when(expr.is_null()).then(pl.lit('null')).otherwise(rendered)
    return _cast_scalar_to_string(expr, dtype, name).fill_null('null')


def render_for_hash(name: str, dtype: pl.DataType) -> pl.Expr:
    """Render a column as a string, for the sha1 evidence `id` and the de-duplication hash.

    Not `.cast(pl.String)`: polars refuses to cast a `List` at all
    (`InvalidOperationError: cannot cast List type`), and evidence is full of them
    (`literature`, `urls`, `qualityControls`, ...). `validate_uniqueness` hashes EVERY column, so
    some hand-written rendering of the nested shapes is unavoidable.

    The contract is FAITHFUL and DETERMINISTIC, not spark-identical -- matching the release's
    evidence ids is not a requirement. Distinct values must keep distinct renderings, or the
    de-duplication hash stops ranking rows meaningfully and the surviving row among a set of
    duplicates becomes arbitrary.

    Most shapes DO still render exactly as spark's `CAST(x AS STRING)` does, measured against real
    spark (task-8-report.md / task-9-report.md) rather than assumed -- the renderings were written
    for parity and kept because they are as good as any other stable choice:

    * String: identity cast.
    * `Int64`, `Boolean`: identical rendering in both engines -- see `_cast_scalar_to_string`.
    * `Float64`: deliberately NOT spark's rendering -- see `_cast_scalar_to_string`.
    * `List(...)`: spark renders `['1','2']` as `'[1, 2]'`, `[]` as `'[]'`, a null list as null,
      and a null *element* -- of any supported element type, not just String -- as the literal
      token `null`.
    * `Struct(...)`: `{f1, f2, ...}` -- braces, no field names, comma-space separated, a null
      field as the literal token `null`, a null struct column as null.
    * `List(Struct(...))`: the two shapes above composed -- a struct element renders `{...}`
      nested inside the list's `[...]`.
    * A `Struct`/`List` field can itself be a `List`/`Struct` (`evidence_schema`'s `biomarkers` is a
      `Struct` of `List(Struct)`); `_render_nested` recurses to cover that.

    Anything else -- an unmeasured leaf dtype anywhere in the (possibly nested) shape -- raises
    rather than guess at a rendering, since an invented one would feed the de-duplication hash and
    silently decide which of a set of duplicate rows gets published.

    Args:
        name: column to render.
        dtype: the column's polars dtype.

    Returns:
        An expression yielding the column's string rendering.

    Raises:
        UnsupportedIdentifierField: for a dtype this function has not been measured against.
    """
    column = pl.col(name)
    if isinstance(dtype, (pl.Struct, pl.List)):
        return pl.when(column.is_null()).then(None).otherwise(_render_nested(column, dtype, name))
    return _cast_scalar_to_string(column, dtype, name)
