"""Association scores per target and disease, computed in targetId buckets."""

from __future__ import annotations

import os
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
EVIDENCE_COLUMNS = ['targetId', 'diseaseId', 'datasourceId', 'datatypeId', 'score', 'evidenceDate']
OUTPUTS = [f'{kind}_{way}' for way in ('direct', 'indirect') for kind in ('by_datasource', 'overall', 'by_datatype')]


def association(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config | None,
) -> None:
    """Compute the six association datasets, one targetId bucket at a time.

    Args:
        source: paths keyed by evidence (a glob) and disease.
        destination: paths keyed by the six association outputs.
        settings: novelty parameters, datasource weights and an optional bucket_count.
        config: unused.
    """
    bucket_count = int(settings.get('bucket_count', DEFAULT_BUCKET_COUNT))
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
    disease = scan_dataset(source['disease'])

    def novelty_of(frame: pl.LazyFrame, dtype: pl.DataType) -> pl.LazyFrame:
        return (
            compute_novelty(frame, current_year=current_year, evidence_count_dtype=dtype, **novelty).sort(GROUP).lazy()
        )

    for bucket in range(bucket_count):
        logger.info(f'association bucket {bucket + 1}/{bucket_count}')
        in_bucket = buckets.pop((bucket,), pl.DataFrame(schema=schema)).lazy()
        evidence = prepare_evidence(in_bucket, current_year)
        sinks = []
        for way, frame in (('direct', evidence), ('indirect', expand_disease(evidence, disease, weights))):
            by_datasource = aggregate_by_datasource(frame).collect().lazy()
            outputs = {
                f'by_datasource_{way}': (by_datasource, pl.Int32()),
                f'overall_{way}': (aggregate_overall(by_datasource, weights), pl.Int64()),
                f'by_datatype_{way}': (aggregate_by_datatype(by_datasource, weights), pl.Int64()),
            }
            for name, (aggregated, dtype) in outputs.items():
                sinks.append(
                    novelty_of(aggregated, dtype).sink_parquet(
                        f'{paths[name]}/part-{bucket:05d}.parquet', compression='zstd', lazy=True
                    )
                )
        pl.collect_all(sinks)
