"""Colocalisation by credible set: the served `colocalisation` shape.

Every output row twice: as is (`leftLocusId` as `credibleSetId`) and swapped (`rightLocusId` as `credibleSetId`,
`rightStudyType` `gwas`). Repeated tuples are kept once, then each credible set gets its tuples sorted by every
field. The input is split into `bucket_count` hash buckets of `credibleSetId` on local disk first, and the buckets
are built `concurrency` at a time, one output part each.
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import polars as pl
from loguru import logger
from ot_datamodels.views.colocalisation_by_credible_set import ColocalisationByCredibleSet
from otter.config.model import Config

from pts.schemas.dataset.conform import conform
from pts.task_outputs import writes
from pts.transformers.utils.buckets import for_each_bucket, read_bucket, write_buckets
from pts.transformers.utils.dataset import check_destinations_absent, list_parts

DEFAULT_BUCKET_COUNT = 64
DEFAULT_CONCURRENCY = 8
#: The served tuple, also its sort order.
FIELDS = [
    'credibleSetId',
    'otherCredibleSetId',
    'rightStudyType',
    'chromosome',
    'colocalisationMethod',
    'numberColocalisingVariants',
    'h3',
    'h4',
    'clpp',
    'betaRatioSignAverage',
]
#: The fields copied as they are on both sides.
SHARED = FIELDS[3:]


def both_sides(rows: pl.DataFrame) -> pl.DataFrame:
    """Each colocalisation output row seen from its left set, then from its right set."""
    rows = rows.with_columns(pl.col('colocalisationMethod').cast(pl.String))
    left = rows.select(
        pl.col('leftLocusId').alias('credibleSetId'),
        pl.col('rightLocusId').alias('otherCredibleSetId'),
        'rightStudyType',
        *SHARED,
    )
    right = rows.select(
        pl.col('rightLocusId').alias('credibleSetId'),
        pl.col('leftLocusId').alias('otherCredibleSetId'),
        pl.lit('gwas', dtype=rows.schema['rightStudyType']).alias('rightStudyType'),
        *SHARED,
    )
    return pl.concat([left, right])


def by_credible_set(tuples: pl.DataFrame) -> pl.DataFrame:
    """One row per credible set with its distinct tuples, sorted."""
    return (
        tuples
        .unique()
        .sort(FIELDS, nulls_last=True)
        .rechunk()
        .select('credibleSetId', row=pl.struct(FIELDS))
        .group_by('credibleSetId', maintain_order=True)
        .agg(colocalisation=pl.col('row'))
    )


def colocalisation_view_frame(rows: pl.DataFrame) -> pl.DataFrame:
    """The view of colocalisation output rows, unbucketed."""
    return by_credible_set(both_sides(rows))


@writes(view=ColocalisationByCredibleSet)
def colocalisation_view(
    source: str,
    destination: dict[str, str],
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Build the colocalisation by credible set view in buckets and write it.

    Args:
        source: the `colocalisation` output.
        destination: `view`, the view directory.
        settings: optional `bucket_count` and `concurrency`.
        config: for `work_path`, where the buckets go.
    """
    settings = settings or {}
    bucket_count = int(settings.get('bucket_count', DEFAULT_BUCKET_COUNT))
    concurrency = int(settings.get('concurrency', DEFAULT_CONCURRENCY))
    path = str(destination['view'])
    check_destinations_absent([path])
    parts = list_parts(str(source), '*.parquet')
    work = config.work_path if config is not None else None
    if work is not None:
        Path(work).mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=work, prefix='colocalisation_view_') as tmp:
        directory = Path(tmp)
        tuples = write_buckets(parts, both_sides, 'credibleSetId', bucket_count, directory, concurrency)
        logger.info(f'{tuples} colocalisation tuples from {len(parts)} parts in {bucket_count} buckets')
        if not tuples:
            raise ValueError(f'no colocalisation rows in {source}')
        if '://' not in path:
            os.makedirs(path)

        def build(bucket: int) -> tuple[int, int]:
            rows = read_bucket(directory, bucket)
            if rows is None:
                return 0, 0
            view = conform(by_credible_set(rows), ColocalisationByCredibleSet)
            view.write_parquet(f'{path}/part-{bucket:05d}.parquet', compression='zstd', statistics=True)
            served = int(view['colocalisation'].list.len().sum())
            logger.info(f'colocalisation bucket {bucket + 1}/{bucket_count}: {view.height} sets, {served} tuples')
            return view.height, served

        built = for_each_bucket(build, bucket_count, concurrency)
    sets, served = sum(b[0] for b in built), sum(b[1] for b in built)
    logger.info(f'{sets} credible sets, {served} tuples served, {tuples - served} repeated tuples dropped')
