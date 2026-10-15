"""Ingest curated loss-of-function assessments from OTAR2075."""

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from pydantic import BaseModel, ConfigDict

from pts.schemas.variant_annotation import variant_annotation_schema
from pts.transformers.utils.dataset import scan_dataset, write_dataset
from pts.transformers.utils.variant_identifier import hash_long_variant_ids

_DESCRIPTIONS = {
    'lof': 'Assessed to cause LoF',
    'likely_lof': 'Suspected to cause LoF',
    'uncertain': 'Uncertain LoF assessment',
    'likely_not_lof': 'Suspected not to cause LoF',
    'not_lof': 'Assessed not to cause LoF',
}


class LofCurationSettings(BaseModel):
    """LoF ingestion has no configurable transformation options."""

    model_config = ConfigDict(extra='forbid')


def normalise_lof_score(assessment: pl.Expr) -> pl.Expr:
    """Map the five curated verdicts to ordinal scores; unknowns remain null."""
    return assessment.replace_strict(
        {'lof': 1.0, 'likely_lof': 0.5, 'uncertain': 0.0, 'likely_not_lof': -0.5, 'not_lof': -1.0},
        default=None,
        return_dtype=pl.Float64,
    )


def transform_lof_curation(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Keep GRCh38 coordinates and GRCh37 alleles, then annotate each verdict.

    Records and duplicates are preserved. Missing identifier components are
    omitted from variantId, matching concat_ws in the existing implementation.
    """
    logger.info('Parsing LoF curation identifiers and verdicts')
    parsed = frame.select(
        pl.col('Variant ID GRCh37').str.split_exact('-', 3).alias('_h37'),
        pl.col('Variant ID GRCh38').str.split_exact('-', 3).alias('_h38'),
        pl.col('Verdict').alias('_verdict'),
    )

    def component(build: str, index: int) -> pl.Expr:
        return pl.col(build).struct.field(f'field_{index}').replace('', None)

    chromosome = component('_h38', 0)
    position = component('_h38', 1)
    reference = component('_h37', 2)
    alternate = component('_h37', 3)
    verdict = pl.col('_verdict')
    return (
        parsed
        .select(
            pl.concat_str(chromosome, position, reference, alternate, separator='_', ignore_nulls=True).alias(
                'variantId'
            ),
            chromosome.alias('chromosome'),
            # Spark's non-ANSI string-to-integer cast truncates decimal suffixes.
            position.str.extract(r'^\s*([+-]?\d+)(?:\.\d*)?\s*$', 1).cast(pl.Int32, strict=False).alias('position'),
            reference.alias('referenceAllele'),
            alternate.alias('alternateAllele'),
            pl.concat_list(
                pl.struct(
                    pl.lit('LossOfFunctionCuration').alias('method'),
                    verdict.alias('assessment'),
                    pl.lit(None, dtype=pl.Float32).alias('score'),
                    pl.lit(None, dtype=pl.String).alias('assessmentFlag'),
                    pl.lit(None, dtype=pl.String).alias('targetId'),
                    normalise_lof_score(verdict).alias('normalisedScore'),
                )
            ).alias('variantEffect'),
            pl.concat_str(
                verdict.replace_strict(_DESCRIPTIONS, default=None),
                pl.lit(' by OTAR2075 variant curation effort.'),
            ).alias('variantDescription'),
        )
        .select(variant_annotation_schema.names())
        .pipe(hash_long_variant_ids)
    )


def lof_curation(source: str, destination: str, settings: dict[str, Any], config: Config) -> None:
    """Read the multiline curation CSV and write variant annotation Parquet.

    Args:
        source: Raw OTAR2075 curation CSV.
        destination: Output annotation dataset directory.
        settings: No settings are currently supported.
        config: Pipeline runtime configuration.
    """
    LofCurationSettings.model_validate(settings)
    result = scan_dataset(source, format='csv', infer_schema=False, null_values='').pipe(transform_lof_curation)
    logger.info('Writing LoF curation annotations to {}', destination)
    write_dataset(result, destination)
