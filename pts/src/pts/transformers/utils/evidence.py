"""Evidence post-processing in polars: harmonisation, entity validation, identifier assignment.

Polars port of `pts.pyspark.evidence_utils.evidence.Evidence`'s first half. Scoring, directions,
uniqueness and dating (the pyspark class's other half) land in a follow-up task on top of this
module -- `Evidence`'s methods each return a new `Evidence`, so that half can chain onto this one
without changing this file.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any

import polars as pl
import polars_hash as plh

from pts.transformers.utils.schemas import load_spark_schema_as_polars

QC_COLUMN = 'qualityControls'


class EvidenceFlags(StrEnum):
    INVALID_DISEASE = 'No valid disease'
    INVALID_TARGET = 'No valid target'
    DUPLICATED = 'Duplicated'
    NO_VALID_SCORE = 'No valid score'
    INVALID_BIOTYPE = 'Invalid biotype'


class UnsupportedIdentifierField(ValueError):  # noqa: N818 -- reads as a condition, not an error type
    """A `unique_fields` dtype (or value) without a verified spark string-cast rendering.

    `spark_cast_to_string` feeds the sha1 that becomes the evidence `id`, a user-visible
    identifier, so a wrong rendering silently changes ids. Raising here is deliberate: a loud
    failure at config-load/run time is an acceptable outcome for an unmeasured type or value; a
    wrong id is not. This is a VALUE-level check where it matters (`_java_double_to_string`'s
    magnitude/subnormal guard) rather than a dtype-level one, so it costs the real datasources
    nothing -- see that function's docstring for the measurements behind it.
    """


def _java_double_to_string(x: float | None) -> str | None:
    """Render a double exactly as spark's `CAST(DOUBLE AS STRING)` does.

    Measured against real spark (task-8-report.md): the rendering matches Java's
    `Double.toString` -- plain decimal for `0.001 <= |x| < 1e7`, scientific `d.dddEn` (no `+`
    sign, always at least one fractional digit) outside that range. Reproduced here by taking
    python's `repr`, which is guaranteed to be the shortest decimal that round-trips to the same
    double -- the same guarantee `Double.toString` makes for ordinary doubles -- and reformatting
    those digits under spark's plain/scientific threshold instead of python's own.

    That guarantee has two measured exceptions, both refused rather than guessed at -- a wrong
    rendering here silently changes a published evidence id, and that is worse than an
    unrunnable datasource. In practice it costs nothing: this is a VALUE-level check, not a
    dtype-level rejection, and no real `resourceScore` (evidence.json's only `Float64`
    `unique_fields` type) ever reaches either range.

    * Subnormal doubles (`0 < |x| < sys.float_info.min`): java's legacy `Double.toString`
      algorithm is not actually shortest-round-trip here (`Double.MIN_VALUE` renders
      `'4.9E-324'` in java but python's shortest round-trip repr is `'5e-324'`).
    * Magnitudes at or above `1.7e16`: a dense fuzz run of 200,000+ random doubles against real
      spark in `[1e15, 2e16)` found the same algorithm emitting extra, non-shortest digits
      starting at `1.801526131658083e16` (e.g. spark's `1.802953665778383e16` renders
      `'1.8029536657783832E16'`, one digit longer than the shortest round-trip
      `'1.802953665778383E16'`). The onset is NOT a clean step function -- only a fraction of
      values above it mismatch, most below it match -- so `1.7e16` is not "the boundary", it is
      a cutoff with a full `1e15` of measured headroom below the smallest mismatch found: 0
      mismatches in 85,297 actual samples `< 1.7e16` from that run, plus a dedicated 150,000-
      sample dense re-check strictly inside `[1.6e16, 1.7e16)` (0 mismatches). An earlier version
      of this guard used `1e15` -- conservative enough to also reject values like `1e15` and
      `1.5e16` that render exactly; `1.7e16` keeps those exact while still refusing the range
      that measurably diverges.

    Args:
        x: the double value to render, or None.

    Returns:
        The spark-equivalent string rendering, or None for a null input.

    Raises:
        UnsupportedIdentifierField: if `x` is subnormal or `|x| >= 1.7e16`.
    """
    if x is None:
        return None
    if math.isnan(x):
        return 'NaN'
    if math.isinf(x):
        return 'Infinity' if x > 0 else '-Infinity'
    if x == 0.0:
        return '-0.0' if math.copysign(1.0, x) < 0 else '0.0'

    negative = x < 0
    ax = -x if negative else x
    if ax < sys.float_info.min or ax >= 1.7e16:
        msg = (
            f'spark_cast_to_string: double {x!r} is outside the range verified against spark '
            '(subnormal, or |x| >= 1.7e16) -- see _java_double_to_string docstring'
        )
        raise UnsupportedIdentifierField(msg)

    digits, point_pos = _shortest_round_trip_digits(ax)
    exponent = point_pos - 1
    body = _plain_decimal(digits, point_pos) if 1e-3 <= ax < 1e7 else _scientific_notation(digits, exponent)
    return ('-' if negative else '') + body


def _shortest_round_trip_digits(ax: float) -> tuple[str, int]:
    """The shortest round-tripping decimal digits of a positive float, and where its point sits.

    `repr` already picked the shortest digit sequence; `Decimal` just recovers it as digits plus
    a decimal-point position instead of a formatted string.
    """
    _, digit_tuple, exp = Decimal(repr(ax)).as_tuple()
    # `exp` is typed as `int | Literal['n', 'N', 'F']` because Decimal.as_tuple() also covers
    # NaN/Infinity, which never reach here -- `ax` is a finite, non-zero, non-nan float already
    # filtered by `_java_double_to_string` before this is called.
    assert isinstance(exp, int)
    digit_str = ''.join(map(str, digit_tuple))
    return digit_str, len(digit_str) + exp


def _plain_decimal(digits: str, point_pos: int) -> str:
    """Format digits as plain decimal notation with the point at `point_pos`."""
    if point_pos <= 0:
        return '0.' + '0' * (-point_pos) + digits
    if point_pos >= len(digits):
        return digits + '0' * (point_pos - len(digits)) + '.0'
    return digits[:point_pos] + '.' + digits[point_pos:]


def _scientific_notation(digits: str, exponent: int) -> str:
    """Format digits as `d.dddEn`. Trailing zeros are `_plain_decimal` padding, not significant."""
    fraction = digits[1:].rstrip('0') or '0'
    return f'{digits[0]}.{fraction}E{exponent}'


def spark_cast_to_string(name: str, dtype: pl.DataType) -> pl.Expr:
    """Reproduce spark's `cast(x AS STRING)` for the `unique_fields` types evidence.json carries.

    Every case is measured against real spark (task-8-report.md), not assumed:

    * String: identity cast.
    * `List(String)`: spark renders `['1','2']` as `'[1, 2]'`, `[]` as `'[]'`, a null list as
      null, and a null *element* as the literal token `null` -- `list.join` alone silently drops
      a null element instead, so it is filled first.
    * `Float64`: see `_java_double_to_string`.
    * `List(Struct(...))` with all-String fields (evidence.json's only such `unique_fields`
      type, `diseaseCellLines`): a struct element renders as `'{f1, f2, f3, f4}'` (braces, no
      field names, comma-space separated, a null field as the literal token `null`), nested
      inside the same `'[...]'` array rendering as `List(String)`.

    Anything else -- an unmeasured dtype, or a `List(Struct(...))` with a non-String field --
    raises rather than guess at a rendering that could silently change an evidence id. This
    module never trades exactness for keeping a datasource runnable; see `_java_double_to_string`
    for why that trade is unnecessary in practice.

    Args:
        name: column to render.
        dtype: the column's polars dtype.

    Returns:
        An expression yielding spark's string rendering of the column.

    Raises:
        UnsupportedIdentifierField: for a dtype this function has not been measured against.
    """
    column = pl.col(name)

    if dtype == pl.String:
        return column.cast(pl.String)

    if dtype == pl.Float64:
        return column.map_elements(_java_double_to_string, return_dtype=pl.String)

    if isinstance(dtype, pl.List):
        inner = dtype.inner
        if inner == pl.String:
            rendered = column.list.eval(pl.element().fill_null('null'))
        elif isinstance(inner, pl.Struct):
            non_string = [f.name for f in inner.fields if f.dtype != pl.String]
            if non_string:
                msg = (
                    f'spark_cast_to_string: List(Struct) field(s) {non_string} of column {name!r} '
                    f'are not String ({inner}) -- only all-String struct fields have been measured'
                )
                raise UnsupportedIdentifierField(msg)
            element = pl.concat_str(
                [pl.element().struct.field(f.name).fill_null('null') for f in inner.fields], separator=', '
            )
            rendered = column.list.eval(pl.lit('{') + element + pl.lit('}'))
        else:
            msg = f'spark_cast_to_string: unsupported list element dtype {inner!r} for column {name!r}'
            raise UnsupportedIdentifierField(msg)
        joined = pl.lit('[') + rendered.list.join(', ') + pl.lit(']')
        return pl.when(column.is_null()).then(None).otherwise(joined)

    msg = f'spark_cast_to_string: unsupported dtype {dtype!r} for column {name!r}'
    raise UnsupportedIdentifierField(msg)


def _flag(condition: pl.Expr, flag: EvidenceFlags) -> pl.Expr:
    """Append a flag to the QC list where the condition holds, deduped and sorted.

    A second, expression-based QC helper exists alongside
    `pts.transformers.utils.quality_flags.update_quality_flag` deliberately: that one is
    DataFrame-based (this pipeline is lazy end to end) and, measured in precheck-parity.md Q2,
    shares the same null-qc bug this one fixes -- see the `.fill_null([])` below. Do not
    "converge" the two; if `update_quality_flag` is ever fixed the same way, this one becomes
    redundant, not the other way around.

    Spark's `update_quality_flag` (pts/src/pts/pyspark/common/utils.py) normalises a null `qc`
    to `[]` *before* branching, so the normalised value comes back out of the `otherwise` branch
    too. A candidate that only normalises inside the `then` branch leaves a null-qc/condition-
    false row as null -- and `pl.col('qc').list.len()` on a null list is itself null, satisfying
    neither `== 0` (valid) nor `!= 0` (failed), so that row would vanish from both outputs.
    """
    normalised = pl.col(QC_COLUMN).fill_null([])
    # `nulls_last=True`: polars' `list.sort()` puts a null element first by default, spark's
    # `array_sort` puts it last. The flag text itself (an `EvidenceFlags` value) is never null --
    # this guards an incoming `qualityControls` that already carries a null element, not the
    # flag being appended here.
    appended = normalised.list.set_union([flag.value]).list.unique().list.sort(nulls_last=True)
    return pl.when(condition).then(appended).otherwise(normalised)


def _harmonise_expr(expr: pl.Expr, source_dtype: Any, target_dtype: Any) -> pl.Expr:
    """Cast one column/field expression from `source_dtype` to `target_dtype`.

    Mirrors `pts.pyspark.common.cast_to_schema.cast_column_to_target_type`: a `List(Struct)` is
    recast element-by-element, a `Struct` is rebuilt in the target's field order (a field the
    target has but the source lacks becomes null, a field the source has but the target lacks is
    dropped), and anything else is a plain non-strict cast -- spark's cast yields null on an
    unconvertible value rather than raising, so `strict=False` mirrors that instead of erroring
    the whole pipeline over one bad value.

    `source_dtype`/`target_dtype` are typed `Any` rather than `pl.DataType`: polars' own stubs
    return `DataTypeClass | DataType` from `.inner`/`.dtype` (a dtype can be an uninstantiated
    class, e.g. bare `pl.Int64` vs `pl.Int64()`), and the only public spelling of that union
    lives in the private `polars._typing` module -- not worth importing a private module over.
    """
    if isinstance(source_dtype, pl.List) and isinstance(target_dtype, pl.List):
        source_inner, target_inner = source_dtype.inner, target_dtype.inner
        if isinstance(target_inner, pl.Struct):
            return expr.list.eval(_harmonise_expr(pl.element(), source_inner, target_inner))
        return expr.list.eval(pl.element().cast(target_inner, strict=False))

    if isinstance(source_dtype, pl.Struct) and isinstance(target_dtype, pl.Struct):
        source_fields = {f.name: f.dtype for f in source_dtype.fields}
        field_exprs = [
            _harmonise_expr(expr.struct.field(f.name), source_fields[f.name], f.dtype).alias(f.name)
            if f.name in source_fields
            else pl.lit(None, dtype=f.dtype).alias(f.name)
            for f in target_dtype.fields
        ]
        return pl.struct(field_exprs)

    return expr.cast(target_dtype, strict=False)


def _harmonise_to_schema(lf: pl.LazyFrame, target_schema: dict[str, pl.DataType]) -> pl.LazyFrame:
    """Cast every column present in both `lf` and `target_schema` to its schema type.

    A column absent from `target_schema` is left alone -- spark's `harmonise_to_schema` only
    logs a warning for it, it never drops or adds top-level columns. Adding a schema field the
    frame lacks only happens one level down, inside a struct (`_harmonise_expr`).
    """
    schema = lf.collect_schema()
    casts = [
        _harmonise_expr(pl.col(name), source_dtype, target_schema[name]).alias(name)
        for name, source_dtype in schema.items()
        if name in target_schema and source_dtype != target_schema[name]
    ]
    return lf.with_columns(casts) if casts else lf


@dataclass
class Evidence:
    """A lazy evidence frame carrying a quality-control column, harmonised to `evidence.json`."""

    lf: pl.LazyFrame

    def __post_init__(self) -> None:
        if QC_COLUMN not in self.lf.collect_schema().names():
            self.lf = self.lf.with_columns(pl.lit([], dtype=pl.List(pl.String)).alias(QC_COLUMN))
        self.lf = _harmonise_to_schema(self.lf, load_spark_schema_as_polars('evidence.json'))

    def validate_diseases(self, disease_lut: pl.DataFrame) -> Evidence:
        """Resolve `diseaseFromSourceMappedId` to `diseaseId`, flagging unmapped rows.

        Args:
            disease_lut: processed disease look-up table (`diseaseFromSourceMappedId`, `diseaseId`).

        Returns:
            Evidence with `diseaseId` joined in, and `EvidenceFlags.INVALID_DISEASE` set for
            evidence without a match.
        """
        return Evidence(
            self.lf.join(disease_lut.lazy(), on='diseaseFromSourceMappedId', how='left').with_columns(
                _flag(pl.col('diseaseId').is_null(), EvidenceFlags.INVALID_DISEASE).alias(QC_COLUMN)
            )
        )

    def validate_target(self, target_lut: pl.DataFrame, invalid_biotypes: list[str] | None = None) -> Evidence:
        """Resolve `targetFromSourceId` to `targetId`, flagging unmapped or invalid-biotype rows.

        Args:
            target_lut: processed target look-up table (`targetId`, `biotype`, `targetFromSourceId`).
            invalid_biotypes: biotypes whose evidence should be flagged `EvidenceFlags.INVALID_BIOTYPE`.
                No flag is added when omitted or empty.

        Returns:
            Evidence with `targetId` joined in and `biotype` dropped.
        """
        joined = self.lf.join(
            target_lut.lazy().select('targetId', 'biotype', 'targetFromSourceId'),
            on='targetFromSourceId',
            how='left',
        ).with_columns(_flag(pl.col('targetId').is_null(), EvidenceFlags.INVALID_TARGET).alias(QC_COLUMN))
        if invalid_biotypes:
            joined = joined.with_columns(
                _flag(pl.col('biotype').is_in(invalid_biotypes), EvidenceFlags.INVALID_BIOTYPE).alias(QC_COLUMN)
            )
        return Evidence(joined.drop('biotype'))

    def validate_datasource(self, datasource_id: str) -> Evidence:
        """Keep only evidence for the given `datasourceId`.

        Args:
            datasource_id: the datasource identifier to keep.

        Returns:
            Evidence filtered to that datasource.
        """
        return Evidence(self.lf.filter(pl.col('datasourceId') == datasource_id))

    def assign_evidence_identifier(self, unique_fields: list[str]) -> Evidence:
        """Assign an `id` column, sha1 of the concatenated source-specific fields.

        Args:
            unique_fields: column names that define evidence uniqueness for this datasource.
                A name absent from the frame is silently skipped, mirroring the pyspark
                `if col in self.df.columns` guard -- not every `unique_fields` entry in
                config.yaml is present on every datasource.

        Returns:
            Evidence with a new `id` column.
        """
        schema = self.lf.collect_schema()
        present = [f for f in unique_fields if f in schema.names()]
        parts = [spark_cast_to_string(f, schema[f]).fill_null('null') for f in present]
        # `pl.concat_str([])` raises ComputeError on an empty list of expressions; spark's
        # `concat_ws('')` over zero columns is well-defined and yields `''`. Unreachable with
        # today's config.yaml (every unique_fields list names at least one column that is
        # always present), but a raw ComputeError here would be a real leak if that ever changes.
        id_input = pl.concat_str(parts) if parts else pl.lit('')
        return Evidence(
            self.lf.with_columns(id_input.alias('_id_input'))
            .with_columns(plh.col('_id_input').nchash.sha1().alias('id'))
            .drop('_id_input')
        )
