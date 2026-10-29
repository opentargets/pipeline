"""Variant index operations, ported from Gentropy's `VariantIndex` and `VariantEffectNormaliser`."""

from __future__ import annotations

import polars as pl

from pts.schemas.variant import variant_schema
from pts.transformers.utils.dataset import list_parts
from pts.transformers.utils.variant import hash_long_variant_ids, min_max_scaler

ANNOTATION_PREFIX = 'annotation_'
_ARRAYS = [name for name, dtype in variant_schema.items() if isinstance(dtype, pl.List)]


def fill_null_arrays(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Replace null arrays with empty ones, so array unions never nullify a value."""
    return frame.with_columns(pl.col(name).fill_null([]) for name in _ARRAYS if name in frame.collect_schema())


def normalise_score(method: str, score: pl.Expr, assessment: pl.Expr) -> pl.Expr:
    """Map a VEP-derived predictor's raw score onto a comparable scale.

    `score` is single precision, as in the variant schema. Methods other than VEP's own
    predictors are normalised where their annotations are built.
    """
    value = score.cast(pl.Float64)
    match method:
        case 'LOFTEE':
            result = pl.when(assessment == 'HC').then(1.0).when(assessment == 'LC').then(0.85)
        case 'AlphaMissense':
            result = (
                pl
                .when(value < 0.06)
                .then(min_max_scaler(value, 0, 0.06, -1.0, -0.25))
                .when(value < 0.77)
                .then(min_max_scaler(value, 0.06, 0.77, -0.25, 0.25))
                .when(value >= 0.77)
                .then(min_max_scaler(value, 0.77, 1, 0.25, 1))
            )
        case 'CADD':
            result = (
                pl
                .when(value <= 10)
                .then(min_max_scaler(value, 0, 10, -1.0, 0.0))
                .when(value <= 20)
                .then(min_max_scaler(value, 10, 20, 0.0, 0.5))
                .when(value <= 30)
                .then(min_max_scaler(value, 20, 30, 0.5, 0.75))
                .when(value > 30)
                .then(min_max_scaler(value, 30, 81, 0.75, 1))
            )
        case 'GERP':
            result = (
                pl
                .when(value > 6)
                .then(1.0)
                .when(value >= 2)
                .then(min_max_scaler(value, 2, 6, 0.75, 1))
                .when(value >= 0)
                .then(min_max_scaler(value, 0, 2, 0.25, 0.75))
                .when(value >= -3)
                .then(min_max_scaler(value, -3, 0, 0.0, 0.25))
                .when(value < -3)
                .then(0.0)
            )
        case 'VEP':
            result = value
        case _:
            raise ValueError(f'No normalisation for {method!r}')
    return result.cast(pl.Float64)


def require_rows(frame: pl.LazyFrame, path: str) -> pl.LazyFrame:
    """Fail on an empty annotation source rather than annotate nothing from it."""
    if frame.select(pl.len()).collect().item() == 0:
        raise ValueError(f'No annotations found in {path}')
    return frame


def annotation_parts_by_chromosome(path: str) -> dict[str, list[str]]:
    """The Parquet parts of an annotation dataset that hold each chromosome."""
    frame = pl.scan_parquet(
        list_parts(path, '**/*.parquet'),
        schema={'chromosome': pl.String},
        hive_partitioning=False,
        extra_columns='ignore',
        include_file_paths='_part',
    )
    located = (
        require_rows(frame, path)
        .select('_part', 'chromosome')
        .drop_nulls('chromosome')
        .unique()
        .collect(engine='streaming')
    )
    by_chromosome: dict[str, list[str]] = {}
    for chromosome, frame in located.group_by('chromosome'):
        by_chromosome[str(chromosome[0])] = sorted(frame['_part'])
    return by_chromosome


def read_variant_annotations(parts: list[str], hash_threshold: int, loci: pl.LazyFrame) -> pl.LazyFrame:
    """Scan supplementary variant annotation parts with the variant index schema.

    Only annotations at `loci` (distinct `chromosome`, `position` pairs) are kept: identifiers
    embed their coordinates, so no other row could match an indexed variant. Columns and
    nested fields missing from the source become nulls; null arrays become empty and
    identifiers are hashed the same way as the index.
    """
    frame = pl.scan_parquet(
        parts,
        schema=variant_schema,
        hive_partitioning=False,
        missing_columns='insert',
        extra_columns='ignore',
        cast_options=pl.ScanCastOptions(missing_struct_fields='insert', extra_struct_fields='ignore'),
    )
    return (
        frame
        .join(loci, on=['chromosome', 'position'], how='semi')
        .pipe(fill_null_arrays)
        .with_columns(
            hash_long_variant_ids(pl.col('variantId'), pl.col('chromosome'), pl.col('position'), hash_threshold).alias(
                'variantId'
            )
        )
    )


def _sorted(column: pl.Expr, dtype: pl.List) -> pl.Expr:
    """Struct arrays sorted descending with null fields last, as Spark's `sort_array(..., false)`."""
    return column.list.sort(descending=True, nulls_last=True) if isinstance(dtype.inner, pl.Struct) else column


def _has_duplicates(column: pl.Expr) -> pl.Expr:
    """True where an array may repeat an element: distinct element hashes rule repeats out."""
    return column.list.eval(pl.element().hash(seed=0)).list.n_unique() < column.list.len()


def _union(joined: pl.DataFrame, name: str, dtype: pl.List) -> pl.Series:
    """Spark's `safe_array_union` of the index array with the annotation's.

    Struct arrays are sorted descending on both sides, then unioned, which also removes
    repeats within the index's array. Where nothing joined, the index's array is kept as
    sorted. The union is only computed where it can change that array: the annotation has
    elements, or the index's array may repeat one.
    """
    left, right = pl.col(name), pl.col(f'{ANNOTATION_PREFIX}{name}')
    union = pl.concat_list(_sorted(left, dtype), _sorted(right, dtype)).list.unique(maintain_order=True)
    needs_union = right.is_not_null() & ((right.list.len() > 0) | _has_duplicates(left))
    rows = joined.select('_row', left, right, needs_union.alias('_union'))
    if not rows['_union'].any():
        return rows.select(_sorted(left, dtype))[name]
    kept = rows.filter(~pl.col('_union')).select('_row', _sorted(left, dtype))
    unioned = rows.filter(pl.col('_union')).select('_row', union.alias(name))
    return pl.concat([kept, unioned]).sort('_row')[name]


def add_annotation(frame: pl.DataFrame, annotations: pl.DataFrame) -> pl.DataFrame:
    """Left-join annotations by variantId, merging them into the index.

    Arrays are unioned, `variantDescription` values are joined with a space as Spark's
    `concat_ws` does, and other columns are coalesced, preferring the index's value.
    """
    renamed = annotations.rename({name: f'{ANNOTATION_PREFIX}{name}' for name in variant_schema.names()})
    joined = frame.join(
        renamed,
        left_on='variantId',
        right_on=f'{ANNOTATION_PREFIX}variantId',
        how='left',
        coalesce=False,
        maintain_order='left',
    ).with_row_index('_row')
    columns = []
    for name, dtype in variant_schema.items():
        left, right = pl.col(name), pl.col(f'{ANNOTATION_PREFIX}{name}')
        if isinstance(dtype, pl.List):
            columns.append(_union(joined, name, dtype))
        elif name == 'variantDescription':
            columns.append(joined.select(pl.concat_str(left, right, separator=' ', ignore_nulls=True))[name])
        else:
            columns.append(joined.select(pl.coalesce(left, right))[name])
    return pl.DataFrame(columns)


def amino_acid_keys(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Every (accession, amino-acid change) pair any transcript of `frame` carries."""
    return (
        frame
        .select('transcriptConsequences')
        .explode('transcriptConsequences', keep_nulls=False)
        .unnest('transcriptConsequences')
        .select(pl.col('uniprotAccessions').alias('uniprotAccession'), 'aminoAcidChange')
        .drop_nulls('aminoAcidChange')
        .explode('uniprotAccession', keep_nulls=False)
        .unique()
    )


def annotate_with_amino_acid_consequences(frame: pl.DataFrame, annotations: pl.DataFrame) -> pl.DataFrame:
    """Append amino-acid level predictions (e.g. FoldX) from the first transcript with a change.

    Variants whose transcript has no UniProt accession are kept, as Spark's `explode_outer`.
    Exploding one row per accession is undone by keeping, per variant, the row with the most
    effects.
    """
    first = (
        pl
        .col('transcriptConsequences')
        .list.filter(pl.element().struct.field('aminoAcidChange').is_not_null())
        .list.first()
    )
    effects, matched = pl.col('variantEffect'), pl.col('_annotations')
    return (
        frame
        .with_row_index('_row')
        .with_columns(
            first.struct.field('aminoAcidChange').alias('aminoAcidChange'),
            first.struct.field('uniprotAccessions').alias('uniprotAccession'),
        )
        .explode('uniprotAccession', empty_as_null=True)
        .join(
            annotations.select('uniprotAccession', 'aminoAcidChange', effects.alias('_annotations')),
            on=['uniprotAccession', 'aminoAcidChange'],
            how='left',
            maintain_order='left',
        )
        .with_columns(
            pl
            .when(matched.is_not_null())
            .then(pl.concat_list(effects, matched).list.unique(maintain_order=True))
            .otherwise(effects)
            .alias('variantEffect')
        )
        .sort('_row', pl.col('variantEffect').list.len(), descending=[False, True], maintain_order=True)
        .unique('_row', keep='first', maintain_order=True)
        .select(variant_schema.names())
    )
