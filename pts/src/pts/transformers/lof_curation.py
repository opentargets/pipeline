"""Ingest curated loss-of-function assessments from OTAR2075."""

from typing import Any

import polars as pl
import polars_hash as plh
from loguru import logger
from otter.config.model import Config
from pydantic import BaseModel, ConfigDict

from pts.schemas.variant_annotation import variant_annotation_schema
from pts.transformers.utils.dataset import scan_dataset, write_dataset
from pts.transformers.utils.variant import hash_long_variant_ids

LOF_ASSESSMENTS: dict[str, tuple[str, float]] = {
    'lof': ('Assessed to cause LoF', 1.0),
    'likely_lof': ('Suspected to cause LoF', 0.5),
    'uncertain': ('Uncertain LoF assessment', 0.0),
    'likely_not_lof': ('Suspected not to cause LoF', -0.5),
    'not_lof': ('Assessed not to cause LoF', -1.0),
}


class LofCurationSettings(BaseModel):
    """LoF ingestion has no configurable transformation options."""

    model_config = ConfigDict(extra='forbid')


def _transform_lof_curation(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Keep GRCh38 coordinates and GRCh37 alleles, then annotate each verdict.

    Retain rows with missing coordinates using an OTVAR identifier.
    Reject empty input or multiple annotations for the same variantId, including
    identical annotations.
    """
    logger.info('Parsing LoF curation identifiers and verdicts')
    h37 = pl.col('Variant ID GRCh37').str.split_exact('-', 3)
    h38 = pl.col('Variant ID GRCh38').str.split_exact('-', 3)
    assessments = pl.DataFrame(
        [(verdict, description) for verdict, (description, _) in LOF_ASSESSMENTS.items()],
        schema=['assessment', '_description'],
        orient='row',
    ).lazy()
    result = (
        frame
        .select(
            chromosome=h38.struct[0],
            position=h38.struct[1],
            referenceAllele=h37.struct[2],
            alternateAllele=h37.struct[3],
            assessment=pl.col('Verdict'),
        )
        .with_columns(pl.col('chromosome', 'position', 'referenceAllele', 'alternateAllele').replace('', None))
        .join(assessments, on='assessment', how='left', maintain_order='left')
        .with_columns(
            pl.concat_str(
                'chromosome', 'position', 'referenceAllele', 'alternateAllele', separator='_', ignore_nulls=True
            ).alias('variantId'),
            # Genomic positions must be positive integers; never truncate decimals.
            pl.col('position').str.extract(r'^([1-9][0-9]*)$', 1).cast(pl.Int32, strict=False),
            pl.concat_list(
                pl.struct(
                    pl.lit('LossOfFunctionCuration').alias('method'),
                    pl.col('assessment'),
                    pl.lit(None, dtype=pl.Float32).alias('score'),
                    pl.lit(None, dtype=pl.String).alias('assessmentFlag'),
                    pl.lit(None, dtype=pl.String).alias('targetId'),
                    pl
                    .col('assessment')
                    .replace_strict(
                        {verdict: score for verdict, (_, score) in LOF_ASSESSMENTS.items()},
                        default=None,
                        return_dtype=pl.Float64,
                    )
                    .alias('normalisedScore'),
                )
            ).alias('variantEffect'),
            pl.concat_str(
                pl.col('_description'),
                pl.lit(' by OTAR2075 variant curation effort.'),
            ).alias('variantDescription'),
        )
        .select(variant_annotation_schema.names())
        .collect()
    )
    if result.is_empty():
        raise ValueError('LoF curation input is empty')
    result = result.with_columns(
        pl
        .when(pl.col('chromosome').is_null() | pl.col('position').is_null())
        .then(pl.lit('OTVAR_') + plh.col('variantId').nchash.md5())
        .otherwise(hash_long_variant_ids(pl.col('variantId')))
        .alias('variantId')
    )
    duplicate_ids = result.filter(pl.col('variantId').is_duplicated()).get_column('variantId').unique().to_list()
    if duplicate_ids:
        raise ValueError(f'Multiple LoF annotations for variantId: {duplicate_ids}')
    return result.lazy()


def lof_curation(source: str, destination: str, settings: dict[str, Any], config: Config) -> None:
    """Read the multiline curation CSV and write variant annotation Parquet.

    Args:
        source: Raw OTAR2075 curation CSV.
        destination: Output annotation dataset directory.
        settings: No settings are currently supported.
        config: Pipeline runtime configuration.
    """
    LofCurationSettings.model_validate(settings)
    result = scan_dataset(source, format='csv', infer_schema=False, null_values='').pipe(_transform_lof_curation)
    logger.info('Writing LoF curation annotations to {}', destination)
    write_dataset(result, destination)
