"""Study deconvolution: one record per study ID across sources that share study IDs."""

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.transformers.study import QualityFlag, record_key, sort_study_collections
from pts.transformers.utils.dataset import scan_dataset, write_dataset


def study_deconvolution(
    source: dict[str, list[str]],
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Merge the study records of the given sources into one record per study ID.

    Args:
        source: Study dataset paths under `study`, in order of preference.
        destination: Path of the deconvoluted studies.
        settings: Unused.
        config: Pipeline configuration.
    """
    studies = pl.concat(
        [
            scan_dataset(path).with_columns(pl.lit(rank).alias('_source_rank'))
            for rank, path in enumerate(source['study'])
        ],
        how='diagonal_relaxed',
    )
    write_dataset(_deconvolute_studies(studies), destination)


def _deconvolute_studies(studies: pl.LazyFrame) -> pl.LazyFrame:
    """Merge repeated study IDs and propagate their QC and most granular traits.

    For conflicting unaggregated fields, prefer the first configured source.
    Equally granular traits prefer the last configured source. Within each source,
    canonical record content breaks ties independently of file and row order.
    Callers without source ranks treat all records as belonging to one source.
    """
    logger.info('Deconvoluting study records')
    columns = [name for name in studies.collect_schema().names() if name != '_source_rank']
    if '_source_rank' not in studies.collect_schema():
        studies = studies.with_columns(pl.lit(0).alias('_source_rank'))
    studies = (
        studies
        .pipe(sort_study_collections)
        .with_columns(record_key(columns).alias('_record_key'))
        .sort(['studyId', '_source_rank', '_record_key'])
        .drop('_record_key')
    )
    has_summary = pl.col('hasSumstats').fill_null(False).any().over('studyId')
    trait_count = pl.col('traitFromSourceMappedIds').list.len()
    qc = pl.col('qualityControls').fill_null([])
    prepared = studies.with_columns(
        pl
        .when(~pl.col('hasSumstats'))
        .then(qc.list.eval(pl.element().filter(pl.element() != QualityFlag.NO_OT_CURATION.value)))
        .otherwise(qc)
        .alias('qualityControls'),
        # Only records with mapped traits compete for the most granular (fewest) traits.
        # Gentropy compares size(), where a null list is -1, so a record without mapped
        # traits always won and the study lost its traits; that is intentionally fixed here.
        pl.when(trait_count > 0).then(trait_count).min().over('studyId').alias('_minimum_traits'),
        has_summary.alias('_has_summary'),
    ).with_columns(
        pl
        .when(pl.col('_has_summary'))
        .then(
            pl.col('qualityControls').list.eval(
                pl.element().filter(pl.element() != QualityFlag.SUMSTATS_NOT_AVAILABLE.value)
            )
        )
        .otherwise(pl.col('qualityControls'))
        .alias('qualityControls')
    )
    # When no record has mapped traits, every record is a candidate, as in Gentropy.
    granular = (
        pl.when(pl.col('_minimum_traits').is_null()).then(True).otherwise(trait_count == pl.col('_minimum_traits'))
    )
    aggregated = {
        'qualityControls': pl
        .col('qualityControls')
        .filter(pl.col('qualityControls').list.len() > 0)
        .explode()
        .unique()
        .sort(),
        'sumstatQCValues': pl.col('sumstatQCValues').drop_nulls().first(),
        'analysisFlags': pl.col('analysisFlags').filter(pl.col('analysisFlags').list.len() > 0).explode(),
        'hasSumstats': pl
        .when(pl.col('_has_summary').first())
        .then(True)
        .when((~pl.col('hasSumstats')).any())
        .then(False)
        .otherwise(None),
        'traitFromSourceMappedIds': pl.col('traitFromSourceMappedIds').filter(granular).drop_nulls().last(),
        'traitFromSource': pl.col('traitFromSource').filter(granular).drop_nulls().last(),
        'studyType': pl.col('studyType').drop_nulls().unique().sort().str.join(','),
    }
    return (
        prepared
        .group_by('studyId', maintain_order=True)
        .agg([
            aggregated.get(column, pl.col(column).first()).alias(column) for column in columns if column != 'studyId'
        ])
        .select(columns)
        .pipe(sort_study_collections)
    )
