"""Variant index operations, ported from Gentropy's `VariantIndex` and `VariantEffectNormaliser`."""

from __future__ import annotations

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as f
from pyspark.sql import types as t

from pts.pyspark.common.utils import polars_schema_to_spark
from pts.pyspark.variant_utils.spark import safe_array_union
from pts.schemas.variant import variant_schema

VARIANT_SCHEMA: t.StructType = polars_schema_to_spark(variant_schema)
ANNOTATION_PREFIX = 'annotation_'


def hash_long_variant_ids(variant_id: Column, chromosome: Column, position: Column, threshold: int) -> Column:
    """Hash identifiers following Gentropy's VariantIndex rules.

    Missing coordinates give `OTVAR_<md5>`; identifiers at or above `threshold` characters
    keep their coordinates as `OTVAR_<chromosome>_<position>_<md5>`. Null identifiers stay null.
    """
    digest = f.md5(variant_id).cast(t.StringType())
    return (
        f
        .when(chromosome.isNull() | position.isNull(), f.concat(f.lit('OTVAR_'), digest))
        .when(f.length(variant_id) >= threshold, f.concat_ws('_', f.lit('OTVAR'), chromosome, position, digest))
        .otherwise(variant_id)
    )


def fill_null_arrays(frame: DataFrame) -> DataFrame:
    """Replace null arrays with empty ones, so array unions never nullify a value."""
    return frame.withColumns({
        field.name: f.coalesce(f.col(field.name), f.array().cast(field.dataType))
        for field in frame.schema.fields
        if isinstance(field.dataType, t.ArrayType)
    })


def read_variant_annotations(spark: SparkSession, path: str, hash_threshold: int) -> DataFrame:
    """Read a supplementary variant annotation dataset with the variant index schema.

    Columns and nested fields missing from the source become nulls; null arrays become empty
    and identifiers are hashed the same way as the index.
    """
    frame = spark.read.schema(VARIANT_SCHEMA).option('recursiveFileLookup', 'true').parquet(path)
    return fill_null_arrays(frame).withColumn(
        'variantId', hash_long_variant_ids(f.col('variantId'), f.col('chromosome'), f.col('position'), hash_threshold)
    )


def add_annotation(frame: DataFrame, annotations: DataFrame) -> DataFrame:
    """Left-join annotations by variantId, merging them into the index.

    Arrays are unioned, `variantDescription` values are concatenated and other columns are
    coalesced, preferring the index's value.
    """
    expressions = []
    for field in VARIANT_SCHEMA.fields:
        name, prefixed = field.name, f'{ANNOTATION_PREFIX}{field.name}'
        in_frame, in_annotations = name in frame.columns, name in annotations.columns
        if in_frame and in_annotations:
            if isinstance(field.dataType, t.ArrayType):
                element = field.dataType.elementType
                order = element.fieldNames() if isinstance(element, t.StructType) else None
                expressions.append(safe_array_union(name, prefixed, order).alias(name))
            elif name == 'variantDescription':
                expressions.append(f.concat_ws(' ', f.col(name), f.col(prefixed)).alias(name))
            else:
                expressions.append(f.coalesce(f.col(name), f.col(prefixed)).alias(name))
        elif in_annotations:
            expressions.append(f.col(prefixed).alias(name))
        elif in_frame:
            expressions.append(f.col(name))
    renamed = annotations.toDF(*[f'{ANNOTATION_PREFIX}{column}' for column in annotations.columns])
    return frame.join(renamed, on=f.col('variantId') == f.col(f'{ANNOTATION_PREFIX}variantId'), how='left').select(
        *expressions
    )


def annotate_with_amino_acid_consequences(frame: DataFrame, annotations: DataFrame) -> DataFrame:
    """Append amino-acid level predictions (e.g. FoldX) from the first transcript with a change.

    Variants whose transcript has no UniProt accession are kept (`explode_outer`). Exploding
    one row per accession is undone by keeping, per variant, the row with the most effects.
    """
    first = f.filter('transcriptConsequences', lambda transcript: transcript.aminoAcidChange.isNotNull())[0]
    window = Window.partitionBy('variantId').orderBy(f.size('variantEffect').desc())
    return (
        frame
        .withColumns({
            'aminoAcidChange': first.aminoAcidChange,
            'uniprotAccession': f.explode_outer(first.uniprotAccessions),
        })
        .join(
            annotations.select('uniprotAccession', 'aminoAcidChange', f.col('variantEffect').alias('_annotations')),
            on=['uniprotAccession', 'aminoAcidChange'],
            how='left',
        )
        .withColumn(
            'variantEffect',
            f.when(f.col('_annotations').isNotNull(), f.array_union('variantEffect', '_annotations')).otherwise(
                f.col('variantEffect')
            ),
        )
        .drop('uniprotAccession', 'aminoAcidChange', '_annotations')
        .distinct()
        .withColumn('_rank', f.row_number().over(window))
        .filter(f.col('_rank') == 1)
        .drop('_rank')
    )


def _rescale(column: Column, low: float, high: float, minimum: float, maximum: float) -> Column:
    return (column - low) / (high - low) * (maximum - minimum) + minimum


def normalise_score(score: Column, method: Column, assessment: Column) -> Column:
    """Map each predictor's raw score onto a comparable scale, by predictor.

    SIFT and PolyPhen are not computed (VEP runs without them), so they are not normalised.
    FoldX and LoF curation arrive already normalised by their own PTS steps.
    """
    absolute = f.abs(score)
    return (
        f
        .when(method == 'LOFTEE', f.when(assessment == 'HC', f.lit(1.0)).when(assessment == 'LC', f.lit(0.85)))
        .when(
            method == 'AlphaMissense',
            f
            .when(score < 0.06, _rescale(score, 0, 0.06, -1.0, -0.25))
            .when(score < 0.77, _rescale(score, 0.06, 0.77, -0.25, 0.25))
            .when(score >= 0.77, _rescale(score, 0.77, 1, 0.25, 1)),
        )
        .when(
            method == 'CADD',
            f
            .when(score <= 10, _rescale(score, 0, 10, -1.0, 0.0))
            .when(score <= 20, _rescale(score, 10, 20, 0.0, 0.5))
            .when(score <= 30, _rescale(score, 20, 30, 0.5, 0.75))
            .when(score > 30, _rescale(score, 30, 81, 0.75, 1)),
        )
        .when(
            method == 'Pangolin',
            f.when(absolute > 0.14, _rescale(absolute, 0.14, 1, 0.5, 1)).when(
                absolute <= 0.14, _rescale(absolute, 0, 0.14, 0.0, 0.5)
            ),
        )
        .when(method.isin('SpliceAI', 'VEP'), score)
        .when(
            method == 'GERP',
            f
            .when(score > 6, f.lit(1.0))
            .when(score >= 2, _rescale(score, 2, 6, 0.75, 1))
            .when(score >= 0, _rescale(score, 0, 2, 0.25, 0.75))
            .when(score >= -3, _rescale(score, -3, 0, 0.0, 0.25))
            .when(score < -3, f.lit(0.0)),
        )
        .cast(t.DoubleType())
    )


def normalise_variant_effects(effects: Column) -> Column:
    """Recompute `normalisedScore` for every predictor in a variant effect array."""
    return f.transform(
        effects,
        lambda effect: f.struct(
            effect.method.alias('method'),
            effect.assessment.alias('assessment'),
            effect.score.alias('score'),
            effect.assessmentFlag.alias('assessmentFlag'),
            effect.targetId.alias('targetId'),
            normalise_score(effect.score, effect.method, effect.assessment).alias('normalisedScore'),
        ),
    )
