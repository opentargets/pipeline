"""Association scores per target and disease, computed in targetId buckets."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.transformers.association.evidence import expand_disease, prepare_evidence
from pts.transformers.association.novelty import GROUP, compute_novelty
from pts.transformers.association.scoring import aggregate_by_datasource, aggregate_by_datatype, aggregate_overall
from pts.transformers.utils.dataset import check_destinations_absent, scan_dataset, scan_datasets

DEFAULT_BUCKET_COUNT = 64
DEFAULT_CONCURRENCY = 8
ROW_GROUP_SIZE = 100_000
EVIDENCE_COLUMNS = ['targetId', 'diseaseId', 'datasourceId', 'datatypeId', 'score', 'evidenceDate']
OUTPUTS = [f'{kind}_{way}' for way in ('direct', 'indirect') for kind in ('by_datasource', 'overall', 'by_datatype')]


def association(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config | None,
) -> None:
    """Compute the six association datasets in targetId buckets, `concurrency` buckets at a time.

    Args:
        source: paths keyed by evidence (a glob) and disease.
        destination: paths keyed by the six association outputs.
        settings: novelty parameters, datasource weights and optional bucket_count and concurrency.
        config: unused.
    """
    bucket_count = int(settings.get('bucket_count', DEFAULT_BUCKET_COUNT))
    concurrency = int(settings.get('concurrency', DEFAULT_CONCURRENCY))
    novelty = {
        'window': settings['novelty_window'],
        'scale': settings['novelty_scale'],
        'shift': settings['novelty_shift'],
    }
    weights = pl.DataFrame(settings['datasource_weights'])
    current_year = datetime.now().year

    paths = {name: destination[name] for name in OUTPUTS}
    check_destinations_absent(paths.values())
    for path in paths.values():
        if '://' not in path:
            os.makedirs(path)

    raw = scan_datasets(source['evidence']).select(EVIDENCE_COLUMNS).collect()
    buckets = raw.with_columns((pl.col('targetId').hash(seed=0) % bucket_count).alias('bucket')).partition_by(
        'bucket', as_dict=True, include_key=False
    )
    schema = raw.schema
    del raw
    disease = scan_dataset(source['disease']).collect().lazy()

    def run(bucket: int, evidence: pl.DataFrame) -> None:
        prepared = prepare_evidence(evidence.lazy(), current_year)
        frames = {}
        for way, frame in (('direct', prepared), ('indirect', expand_disease(prepared, disease, weights))):
            by_datasource = aggregate_by_datasource(frame).collect().lazy()
            overall, by_datatype = pl.collect_all([
                aggregate_overall(by_datasource, weights),
                aggregate_by_datatype(by_datasource, weights),
            ])
            for kind, aggregated, dtype in (
                ('by_datasource', by_datasource, pl.Int32()),
                ('overall', overall.lazy(), pl.Int64()),
                ('by_datatype', by_datatype.lazy(), pl.Int64()),
            ):
                frames[f'{kind}_{way}'] = compute_novelty(
                    aggregated, current_year=current_year, evidence_count_dtype=dtype, **novelty
                ).sort(GROUP)
        for name, frame in zip(frames, pl.collect_all(frames.values()), strict=True):
            frame.write_parquet(
                f'{paths[name]}/part-{bucket:05d}.parquet',
                compression='zstd',
                statistics=True,
                row_group_size=ROW_GROUP_SIZE,
            )
        logger.info(f'association bucket {bucket + 1}/{bucket_count} written')

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(run, bucket, buckets.pop((bucket,), pl.DataFrame(schema=schema)))
            for bucket in range(bucket_count)
        ]
        try:
            for future in futures:
                future.result()
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise
