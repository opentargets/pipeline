"""Evidence post-processing in polars: harmonisation through scoring and direction of effect.

Polars port of the removed pyspark `Evidence`. `Evidence`'s methods each return a
new `Evidence`, so a caller can chain the full sequence -- `validate_diseases` ->
`validate_target` -> `validate_datasource` -> `assign_evidence_identifier` ->
`validate_uniqueness` -> `resolve_publication_date` ->
`resolve_evidence_date` -> `calculate_evidence_score` -> `assign_direction_on_trait` ->
`assign_direction_on_target` -> `hash_long_variant_identifiers` -> `valid()`/`invalid()` -- exactly
as the removed pyspark post-processing step chained the pyspark class. The shared recipe
(`postprocess`) stops at `calculate_evidence_score`; the steps after it belong to the datasources
that use them.

`calculate_evidence_score`, `assign_direction_on_trait` and `assign_direction_on_target` take a
`pl.Expr | None`; each datasource module supplies its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import polars as pl
import polars_hash as plh

from pts.schemas.evidence import evidence_schema
from pts.transformers.evidence.utils.hashing import render_for_hash
from pts.transformers.utils.variant import VARIANT_HASH_LENGTH, hash_long_variant_ids

QC_COLUMN = 'qualityControls'

#: Columns considered when dating evidence, in `resolve_evidence_date` -- mirrors pyspark's
#: `Evidence.EVIDENCE_DATE_COLUMNS`.
DATE_COLUMNS = ['publicationDate', 'curationDate', 'studyStartDate', 'releaseDate', 'evidenceDate']


class EvidenceFlags(StrEnum):
    INVALID_DISEASE = 'No valid disease'
    INVALID_TARGET = 'No valid target'
    DUPLICATED = 'Duplicated'
    NO_VALID_SCORE = 'No valid score'
    INVALID_BIOTYPE = 'Invalid biotype'


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

    Mirrors the removed pyspark `cast_column_to_target_type`: a `List(Struct)` is
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


def _join_matching_spark_order(lf: pl.LazyFrame, other: pl.LazyFrame, on: str, how: Literal['left']) -> pl.LazyFrame:
    """`lf.join(other, on=on, how=how)`, with spark's `on=<str>` join column order restored.

    Measured (task-9-report.md): spark's `.join(other, on=<str>, how=...)` moves the join key
    column to the FRONT of the result -- `[key] + [left minus key] + [right minus key]`. Polars'
    `.join` instead leaves the left frame's column order untouched and appends right-only columns
    at the end. The two diverge on every one of the 10 staged evidence inputs (none has its join
    key already first), and it is not cosmetic: `validate_uniqueness`'s survivor-selecting content
    hash is built by iterating whatever column order the frame is actually in when it runs, and
    spark's hash input order is `self.df.columns` at the equivalent point.

    `how` is typed `Literal['left']`, the only strategy either caller uses today, rather than
    polars' own `JoinStrategy` -- that alias lives in the private `polars._typing` module (same
    situation as `_harmonise_expr`'s `source_dtype`/`target_dtype`, see that function's docstring).
    """
    left_columns = lf.collect_schema().names()
    right_only = [c for c in other.collect_schema().names() if c != on]
    joined = lf.join(other, on=on, how=how, maintain_order='left_right')
    return joined.select(on, *[c for c in left_columns if c != on], *right_only)


def _require_id_column(schema_names: list[str], method_name: str) -> None:
    """Raise clearly if `id` is missing, mirroring the removed pyspark `@required_columns(['id'])`.

    Neither `validate_uniqueness` nor `resolve_publication_date` ported that decorator; without
    this, a missing `id` leaked a raw, unhelpful `ColumnNotFoundError` from deep inside the
    `.over('id', ...)` / `.select('id', ...)` machinery instead of naming the actual problem.
    """
    if 'id' not in schema_names:
        msg = f"Column 'id' required by '{method_name}' is missing. Available columns: {schema_names}"
        raise ValueError(msg)


@dataclass
class Evidence:
    """A lazy evidence frame carrying a quality-control column, harmonised to `evidence_schema`."""

    lf: pl.LazyFrame

    def __post_init__(self) -> None:
        if QC_COLUMN not in self.lf.collect_schema().names():
            self.lf = self.lf.with_columns(pl.lit([], dtype=pl.List(pl.String)).alias(QC_COLUMN))
        self.lf = _harmonise_to_schema(self.lf, evidence_schema)

    def validate_diseases(self, disease_lut: pl.DataFrame) -> Evidence:
        """Resolve `diseaseFromSourceMappedId` to `diseaseId`, flagging unmapped rows.

        Args:
            disease_lut: processed disease look-up table (`diseaseFromSourceMappedId`, `diseaseId`).

        Returns:
            Evidence with `diseaseId` joined in, and `EvidenceFlags.INVALID_DISEASE` set for
            evidence without a match.
        """
        joined = _join_matching_spark_order(self.lf, disease_lut.lazy(), on='diseaseFromSourceMappedId', how='left')
        return Evidence(
            joined.with_columns(_flag(pl.col('diseaseId').is_null(), EvidenceFlags.INVALID_DISEASE).alias(QC_COLUMN))
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
        joined = _join_matching_spark_order(
            self.lf,
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
        """Assign an `id` column, sha1 of the source-specific fields joined by U+0001.

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
        parts = [render_for_hash(f, schema[f]).fill_null('null') for f in present]
        # `pl.concat_str([])` raises; spark's `concat_ws` over no columns yields `''`.
        id_input = pl.concat_str(parts, separator='\x01') if parts else pl.lit('')
        return Evidence(
            self.lf
            .with_columns(id_input.alias('_id_input'))
            .with_columns(plh.col('_id_input').nchash.sha1().alias('id'))
            .drop('_id_input')
        )

    def validate_uniqueness(self) -> Evidence:
        """Flag every row but one within each `id`, ordered by a content hash.

        The hash input follows spark's:

        * `id` is included in the hash. Spark iterates `self.df.columns`, which by this point
          already includes `id` (added by `assign_evidence_identifier`); an earlier version of
          this method excluded it, reasoning it was constant within the `over('id', ...)`
          partition being ranked and so could not affect which row wins. That reasoning is FALSE
          -- measured over 2,000 two-row same-`id` partitions, excluding `id` changed the
          survivor in 962 of them (48.1%, a coin flip): sha256 ordering is not preserved under a
          constant infix. So the rule is: whatever goes into the hash, keep it stable.
        * Column ORDER matches spark's `self.df.columns` at the equivalent point.
          `validate_diseases`/`validate_target` restore spark's `on=<str>` join-key-to-front
          ordering (`_join_matching_spark_order`) so `schema.names()` below iterates in that same
          order. That ordering also reaches the published parquet, so it is now load-bearing for
          the output's column order regardless of its spark provenance.

        The digests match spark's on JDK 17, the release's JVM, whose `Double.toString` renders the
        doubles, except for a row carrying one of the doubles above `2**63` that java 17 does not
        render shortest (see `java_double_to_string`).

        Returns:
            Evidence with `EvidenceFlags.DUPLICATED` set on every row but the survivor within
            each `id`.

        Raises:
            ValueError: if `id` is missing.
        """
        schema = self.lf.collect_schema()
        _require_id_column(schema.names(), 'validate_uniqueness')
        parts = [render_for_hash(c, schema[c]).fill_null('null') for c in schema.names()]
        # spark's concat_ws separator is U+0001
        content = pl.concat_str(parts, separator='\x01') if parts else pl.lit('')
        return Evidence(
            self.lf
            .with_columns(content.alias('_content'))
            .with_columns(plh.col('_content').chash.sha2_256().alias('_content_hash'))
            .with_columns(pl.int_range(pl.len()).over('id', order_by='_content_hash').alias('_rank'))
            .with_columns(_flag(pl.col('_rank') != 0, EvidenceFlags.DUPLICATED).alias(QC_COLUMN))
            .drop('_rank', '_content', '_content_hash')
        )

    def resolve_publication_date(self, publication_lut: pl.DataFrame) -> Evidence:
        """Date evidence by the earliest publication date among its `literature` references.

        `id` is a mandatory column for this to run; the presence of `literature` is not -- self
        is returned unchanged if the column is absent.

        Args:
            publication_lut: look-up table of `publicationId` -> `publicationDate`
                (`pts.transformers.evidence.utils.luts.build_publication_lut`).

        Returns:
            Evidence with a new `publicationDate` column when `literature` is present.

        Raises:
            ValueError: if `id` is missing.
        """
        schema_names = self.lf.collect_schema().names()
        _require_id_column(schema_names, 'resolve_publication_date')
        if 'literature' not in schema_names:
            return self
        dated = (
            self.lf
            .select('id', pl.col('literature').alias('_lit'))
            .explode('_lit')
            # Spark's `trim` strips the ASCII space only; polars' no-argument `str.strip_chars()`
            # also strips tab/newline/non-breaking-space, which could map two distinct
            # publication ids onto the same lookup key (measured, task-9-report.md).
            .select('id', pl.col('_lit').str.strip_chars(' ').str.to_uppercase().alias('publicationId'))
            .unique()
            .join(publication_lut.lazy(), on='publicationId', how='inner')
            .group_by('id')
            .agg(pl.col('publicationDate').min())
        )
        return Evidence(_join_matching_spark_order(self.lf, dated, on='id', how='left'))

    def resolve_evidence_date(self) -> Evidence:
        """Assign `evidenceDate` as the earliest of the present `DATE_COLUMNS`.

        `evidenceDate` is always added, even when none of `DATE_COLUMNS` are present.

        Returns:
            Evidence with a new `evidenceDate` column.
        """
        present = [c for c in DATE_COLUMNS if c in self.lf.collect_schema().names()]
        # `pl.min_horizontal` already ignores nulls (measured): unlike spark's `array_min`, which
        # returns null if ANY element is null and so needs its inputs pre-filtered, no equivalent
        # filter step is needed here.
        expression = pl.min_horizontal(present) if present else pl.lit(None, dtype=pl.String)
        return Evidence(self.lf.with_columns(expression.alias('evidenceDate')))

    def calculate_evidence_score(self, score_expression: pl.Expr) -> Evidence:
        """Assign `score` from a datasource-specific expression, flagging out-of-range values.

        Unlike `assign_direction_on_trait`/`assign_direction_on_target`, `None` is not accepted:
        every datasource has a score.

        Args:
            score_expression: the datasource's score expression.

        Returns:
            Evidence with a new `score` column, and `EvidenceFlags.NO_VALID_SCORE` set where it
            is missing or outside `[0, 1]`.

        Raises:
            ValueError: if `score_expression` is `None`.
        """
        if score_expression is None:
            msg = 'calculate_evidence_score requires a score_expression -- every datasource has one'
            raise ValueError(msg)
        # Non-strict: spark's cast yields null on an unconvertible value rather than raising.
        score = score_expression.cast(pl.Float64, strict=False)
        return Evidence(
            self.lf.with_columns(score.alias('score')).with_columns(
                _flag(
                    pl.col('score').is_null() | (pl.col('score') < 0) | (pl.col('score') > 1),
                    EvidenceFlags.NO_VALID_SCORE,
                ).alias(QC_COLUMN)
            )
        )

    def assign_direction_on_trait(self, direction_expression: pl.Expr | None) -> Evidence:
        """Assign `directionOnTrait` from a datasource-specific expression.

        Args:
            direction_expression: the datasource's `direction_on_trait` expression, or `None` to
                leave evidence unchanged.

        Returns:
            Evidence with a new `directionOnTrait` column when an expression is given.
        """
        if direction_expression is None:
            return self
        return Evidence(self.lf.with_columns(direction_expression.alias('directionOnTrait')))

    def assign_direction_on_target(
        self, direction_expression: pl.Expr | None, target_lut: pl.DataFrame | None
    ) -> Evidence:
        """Assign `directionOnTarget` from a datasource-specific expression.

        Args:
            direction_expression: the datasource's `direction_on_target` expression, or `None` to
                leave evidence unchanged.
            target_lut: when given, joined in on `targetId` to provide `TSorOncogene` for
                expressions that need it (e.g. `cancer_gene_census`).

        Returns:
            Evidence with a new `directionOnTarget` column when an expression is given, and
            `actionType`/`TSorOncogene` dropped (present on some datasources' raw evidence, or
            joined in here, but not part of the published schema).
        """
        if direction_expression is None:
            return self
        lf = self.lf
        if target_lut is not None:
            lf = _join_matching_spark_order(
                lf, target_lut.lazy().select('targetId', 'TSorOncogene').unique(), on='targetId', how='left'
            )
        return Evidence(
            lf.with_columns(direction_expression.alias('directionOnTarget')).drop(
                'actionType', 'TSorOncogene', strict=False
            )
        )

    def hash_long_variant_identifiers(self) -> Evidence:
        """Hash `variantId` values longer than `VARIANT_HASH_LENGTH`.

        Self is returned unchanged if `variantId` is absent.

        Returns:
            Evidence with long `variantId` values replaced by an `OTVAR_...` hash.
        """
        if 'variantId' not in self.lf.collect_schema().names():
            return self
        return Evidence(
            self.lf.with_columns(
                hash_long_variant_ids(pl.col('variantId'), threshold=VARIANT_HASH_LENGTH).alias('variantId')
            )
        )

    def valid(self) -> pl.LazyFrame:
        """Evidence with an empty `qualityControls` list.

        Returns:
            LazyFrame filtered to evidence that passed every validation.
        """
        return self.lf.filter(pl.col(QC_COLUMN).list.len() == 0)

    def invalid(self) -> pl.LazyFrame:
        """Evidence with a non-empty `qualityControls` list.

        Returns:
            LazyFrame filtered to evidence that failed at least one validation.
        """
        return self.lf.filter(pl.col(QC_COLUMN).list.len() != 0)
