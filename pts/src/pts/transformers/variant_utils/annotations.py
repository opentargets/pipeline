"""Lazy variant identifier and annotation transformations."""

import polars as pl

from pts.schemas.variant import variant_schema
from pts.transformers.foldx import normalise_foldx_score
from pts.transformers.lof_curation import normalise_lof_score
from pts.transformers.utils.variant import min_max_scaler


def normalize_variant_effects(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Append comparable predictor scores using each predictor's calibration."""
    item = pl.element()
    score = item.struct.field('score').cast(pl.Float64)
    assessment = item.struct.field('assessment')
    method = item.struct.field('method')
    inverted = 1 - score
    rounded = 1 - score.round(2, mode='half_away_from_zero')
    sift = (
        pl
        .when((rounded >= 0.95) & (assessment == 'deleterious'))
        .then(min_max_scaler(inverted, 0.95, 1, 0.5, 1))
        .when((rounded >= 0.95) & (assessment == 'deleterious_low_confidence'))
        .then(min_max_scaler(inverted, 0.95, 1, 0, 0.5))
        .when((rounded <= 0.95) & (assessment == 'tolerated_low_confidence'))
        .then(min_max_scaler(inverted, 0, 0.95, -0.5, 0))
        .when((rounded <= 0.95) & (assessment == 'tolerated'))
        .then(min_max_scaler(inverted, 0, 0.95, -1, -0.5))
    )
    polyphen = (
        pl
        .when(assessment == 'unknown')
        .then(None)
        .when(score <= 0.446)
        .then(min_max_scaler(score, 0, 0.446, -1, -0.25))
        .when(score <= 0.908)
        .then(min_max_scaler(score, 0.446, 0.908, -0.25, 0.25))
        .when(score > 0.908)
        .then(min_max_scaler(score, 0.908, 1, 0.25, 1))
    )
    alpha = (
        pl
        .when(score < 0.06)
        .then(min_max_scaler(score, 0, 0.06, -1, -0.25))
        .when(score < 0.77)
        .then(min_max_scaler(score, 0.06, 0.77, -0.25, 0.25))
        .when(score >= 0.77)
        .then(min_max_scaler(score, 0.77, 1, 0.25, 1))
    )
    cadd = (
        pl
        .when(score <= 10)
        .then(min_max_scaler(score, 0, 10, -1, 0))
        .when(score <= 20)
        .then(min_max_scaler(score, 10, 20, 0, 0.5))
        .when(score <= 30)
        .then(min_max_scaler(score, 20, 30, 0.5, 0.75))
        .when(score > 30)
        .then(min_max_scaler(score, 30, 81, 0.75, 1))
    )
    gerp = (
        pl
        .when(score > 6)
        .then(1.0)
        .when(score >= 2)
        .then(min_max_scaler(score, 2, 6, 0.75, 1))
        .when(score >= 0)
        .then(min_max_scaler(score, 0, 2, 0.25, 0.75))
        .when(score >= -3)
        .then(min_max_scaler(score, -3, 0, 0, 0.25))
        .when(score < -3)
        .then(0.0)
    )
    pangolin = (
        pl
        .when(score.abs() > 0.14)
        .then(min_max_scaler(score.abs(), 0.14, 1, 0.5, 1))
        .when(score.abs() <= 0.14)
        .then(min_max_scaler(score.abs(), 0, 0.14, 0, 0.5))
    )
    normalised = (
        pl
        .when(method == 'LOFTEE')
        .then(assessment.replace_strict({'HC': 1.0, 'LC': 0.85}, default=None))
        .when(method == 'SIFT')
        .then(sift)
        .when(method == 'PolyPhen')
        .then(polyphen)
        .when(method == 'AlphaMissense')
        .then(alpha)
        .when(method == 'CADD')
        .then(cadd)
        .when(method == 'Pangolin')
        .then(pangolin)
        .when(method == 'LossOfFunctionCuration')
        .then(normalise_lof_score(assessment))
        .when(method.is_in(['SpliceAI', 'VEP']))
        .then(score)
        .when(method == 'GERP')
        .then(gerp)
        .when(method == 'FoldX')
        .then(normalise_foldx_score(score))
    )
    return frame.with_columns(
        pl.col('variantEffect').list.eval(
            pl.struct(
                *[item.struct.field(name) for name in ['method', 'assessment', 'score', 'assessmentFlag', 'targetId']],
                normalised.alias('normalisedScore'),
            )
        )
    )


def add_annotation(frame: pl.LazyFrame, annotations: pl.LazyFrame) -> pl.LazyFrame:
    """Left-join annotations, union arrays and preserve existing scalar values."""
    left_schema, right_schema = frame.collect_schema(), annotations.collect_schema()
    expressions = []
    for name, dtype in variant_schema.items():
        left, right = pl.col(name), pl.col(f'annotation_{name}')
        if name in left_schema and name in right_schema:
            if isinstance(dtype, pl.List):
                if isinstance(dtype.inner, pl.Struct):
                    left = left.cast(dtype).list.sort(descending=True)
                    right = right.cast(dtype).list.sort(descending=True)
                expr = (
                    pl
                    .when(left.is_not_null() & right.is_not_null())
                    .then(pl.concat_list(left, right).list.unique(maintain_order=True))
                    .otherwise(pl.coalesce(left, right))
                )
            elif name == 'variantDescription':
                expr = pl.concat_str(left, right, separator=' ', ignore_nulls=True)
            else:
                expr = pl.coalesce(left, right)
        elif name in left_schema:
            expr = left
        elif name in right_schema:
            expr = right
        else:
            continue
        expressions.append(expr.alias(name))
    return frame.join(
        annotations.rename({name: f'annotation_{name}' for name in right_schema}),
        left_on='variantId',
        right_on='annotation_variantId',
        how='left',
        coalesce=False,
        maintain_order='left_right',
    ).select(expressions)


def annotate_with_amino_acid_consequences(frame: pl.LazyFrame, annotation: pl.LazyFrame) -> pl.LazyFrame:
    """Append amino-acid predictions from the first affected transcript."""
    first = (
        pl
        .col('transcriptConsequences')
        .list.eval(pl.element().filter(pl.element().struct.field('aminoAcidChange').is_not_null()))
        .list.first()
    )
    enriched = (
        frame
        .with_columns(first.alias('_transcript'))
        .with_columns(
            pl.col('_transcript').struct.field('aminoAcidChange').alias('aminoAcidChange'),
            pl.col('_transcript').struct.field('uniprotAccessions').alias('uniprotAccession'),
        )
        .drop('_transcript')
        .explode('uniprotAccession')
        .join(
            annotation.rename({'variantEffect': '_annotations'}), on=['uniprotAccession', 'aminoAcidChange'], how='left'
        )
        .with_columns(
            pl
            .when(pl.col('_annotations').is_not_null())
            .then(
                pl
                .when(pl.col('variantEffect').is_not_null())
                .then(pl.concat_list('variantEffect', '_annotations').list.unique(maintain_order=True))
                .otherwise(None)
            )
            .otherwise(pl.col('variantEffect'))
            .alias('variantEffect')
        )
        .drop('uniprotAccession', 'aminoAcidChange', '_annotations')
    )
    return enriched.sort(pl.col('variantEffect').list.len(), descending=True, nulls_last=True).unique(
        subset=['variantId'], keep='first', maintain_order=True
    )
