"""Convert raw FoldX predictions into amino-acid variant annotations."""

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from pydantic import BaseModel, ConfigDict, Field

from pts.schemas.amino_acid_variants import amino_acid_variants_schema
from pts.transformers.utils.dataset import scan_dataset, write_dataset


class FoldxSettings(BaseModel):
    """Settings for filtering FoldX structural predictions."""

    model_config = ConfigDict(extra='forbid')

    plddt_threshold: float = Field(default=0.7, allow_inf_nan=False)
    """Keep predictions strictly above this pLDDT threshold, in source units."""


def normalise_foldx_score(score: pl.Expr) -> pl.Expr:
    """Map absolute FoldX ddG to [0, 1], saturating at 2 kcal/mol."""
    absolute = score.cast(pl.Float64).abs()
    return pl.when(absolute >= 2).then(1.0).otherwise(absolute / 2)


def filter_foldx_predictions(frame: pl.LazyFrame, plddt_threshold: float) -> pl.LazyFrame:
    """Exclude ambiguous proteins before applying the strict confidence threshold.

    Multiple distinct non-null pLDDT values for one accession, position and wild
    type invalidate the entire protein, including its otherwise confident sites.
    """
    logger.info('Filtering ambiguous FoldX proteins and low-confidence predictions')
    excluded = (
        frame
        .group_by('protein_acc', 'position', 'wild_type')
        .agg(pl.col('plddt').drop_nulls().n_unique().alias('_plddt_count'))
        .filter(pl.col('_plddt_count') > 1)
        .select('protein_acc')
        .unique()
    )
    return frame.filter(pl.col('plddt') > plddt_threshold).join(excluded, on='protein_acc', how='anti')


def aggregate_foldx_predictions(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Create typed, deduplicated effect lists for each amino-acid substitution."""
    logger.info('Aggregating FoldX amino-acid annotations')
    return (
        frame
        .select(
            pl.col('protein_acc').cast(pl.String).alias('uniprotAccession'),
            pl.concat_str('wild_type', pl.col('position').cast(pl.String), 'mutated_type').alias('aminoAcidChange'),
            # Casting precedes deduplication, matching the published score precision.
            pl.col('foldx_ddg').cast(pl.Float32, strict=False).alias('_score'),
        )
        .with_columns(
            pl.struct(
                pl.lit('FoldX').alias('method'),
                pl.lit(None, dtype=pl.String).alias('assessment'),
                pl.col('_score').alias('score'),
                pl.lit(None, dtype=pl.String).alias('assessmentFlag'),
                pl.lit(None, dtype=pl.String).alias('targetId'),
                normalise_foldx_score(pl.col('_score')).alias('normalisedScore'),
            ).alias('variantEffect')
        )
        .group_by('uniprotAccession', 'aminoAcidChange')
        .agg(pl.col('variantEffect').unique().sort())
        .select(amino_acid_variants_schema.names())
    )


def foldx(source: str, destination: str, settings: dict[str, Any], config: Config) -> None:
    """Read the raw FoldX export and write amino-acid variant annotations.

    Args:
        source: Raw FoldX Parquet input.
        destination: Output annotation dataset directory.
        settings: Confidence filtering options.
        config: Pipeline runtime configuration.
    """
    options = FoldxSettings.model_validate(settings)
    result = (
        scan_dataset(source)
        .select('protein_acc', 'position', 'wild_type', 'mutated_type', 'plddt', 'foldx_ddg')
        .pipe(filter_foldx_predictions, options.plddt_threshold)
        .pipe(aggregate_foldx_predictions)
    )
    logger.info('Writing FoldX annotations to {}', destination)
    write_dataset(result, destination)
