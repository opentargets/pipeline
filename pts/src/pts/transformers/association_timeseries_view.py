"""Flat association timeseries view: one row per association, aggregation value and year."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.schemas.association import association_timeseries_schema
from pts.transformers.utils.dataset import check_destinations_absent, scan_dataset

DEFAULT_FILE_COUNT = 300
INPUTS = {
    'association_overall_direct': True,
    'association_overall_indirect': False,
    'association_by_datasource_direct': True,
    'association_by_datasource_indirect': False,
}
SORT = ['partitionKey', 'aggregationValue', 'year']
SPLIT_COLUMNS = ['diseaseId', 'targetId', 'aggregationType', 'aggregationValue', 'timeseries', 'isDirect']
TIMESERIES = pl.List(
    pl.Struct({
        'year': pl.Int32(),
        'associationScore': pl.Float64(),
        'novelty': pl.Float64(),
        'yearlyEvidenceCount': pl.Int64(),
    })
)


def association_timeseries_view(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    config: Config | None,
) -> None:
    """Explode the four association datasets into a timeseries view of `file_count` sorted files.

    The inputs are read once and split by targetId hash into local scratch files; each output
    file is then built from its own split only, so every partitionKey lands in one file.

    Args:
        source: paths keyed by the four association datasets.
        destination: output directory.
        settings: optional file_count.
        config: for work_path, where the scratch split goes; the system temp directory when absent.
    """
    file_count = int(settings.get('file_count', DEFAULT_FILE_COUNT))
    check_destinations_absent([destination])
    if '://' not in destination:
        os.makedirs(destination)

    scratch_root = None
    if config is not None:
        scratch_root = Path(config.work_path)
        scratch_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='association_timeseries_view_', dir=scratch_root) as scratch:
        splits = [
            scan_dataset(source[name])
            .select(
                *(c for c in SPLIT_COLUMNS if c != 'isDirect'),
                pl.lit(is_direct).alias('isDirect'),
            )
            .with_columns(
                pl.col('timeseries').cast(TIMESERIES),
                (pl.col('targetId').hash(seed=0) % file_count).cast(pl.Int32).alias('file'),
            )
            for name, is_direct in INPUTS.items()
        ]
        logger.info(f'splitting inputs into {file_count} groups')
        pl.concat(splits).sink_parquet(
            pl.PartitionBy(scratch, key='file', include_key=False), compression='zstd', engine='streaming'
        )

        for index in range(file_count):
            logger.info(f'association_timeseries_view file {index + 1}/{file_count}')
            group = Path(scratch) / f'file={index}'
            if group.exists():
                frame = pl.scan_parquet(group / '*.parquet')
            else:
                frame = pl.LazyFrame(schema=_split_schema())
            (
                frame
                .explode('timeseries')
                .drop_nulls('timeseries')
                .unnest('timeseries')
                .with_columns(
                    pl.concat_str(
                        pl.col('targetId'),
                        pl.col('diseaseId'),
                        pl.col('isDirect').cast(pl.String),
                        separator='_',
                        ignore_nulls=True,
                    ).alias('partitionKey')
                )
                .select(association_timeseries_schema().names())
                .sort(SORT, maintain_order=True)
                .sink_parquet(f'{destination}/part-{index:05d}.parquet', compression='zstd', engine='streaming')
            )


def _split_schema() -> pl.Schema:
    return pl.Schema({
        'diseaseId': pl.String(),
        'targetId': pl.String(),
        'aggregationType': pl.String(),
        'aggregationValue': pl.String(),
        'timeseries': TIMESERIES,
        'isDirect': pl.Boolean(),
    })
