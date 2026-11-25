"""Variant view: every variant with its credible-set and pharmacogenomics ids, the served `variants`.

Replaces `variants.sql`. The id lists are held in memory; `variant` is read one part at a time, each written
as one part.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import polars as pl
from loguru import logger
from ot_datamodels.polars import to_polars_schema
from ot_datamodels.views.variant_view import VariantView
from otter.config.model import Config

from pts.schemas.dataset.conform import conform
from pts.task_outputs import writes
from pts.transformers.utils.dataset import check_destinations_absent, list_parts, scan_dataset

DEFAULT_CONCURRENCY = 8
ROW_GROUP_SIZE = 100_000
VIEW_SCHEMA = to_polars_schema(VariantView)
ID_LISTS = ['credibleSetIds', 'pharmacogenomicsIds']


def variant_ids(credible_set_variant: pl.LazyFrame, pharmacogenomics: pl.LazyFrame) -> pl.DataFrame:
    """Per `variantId`, its credible-set ids and pharmacogenomics ids, each sorted; null for none.

    Args:
        credible_set_variant: `credibleSetId`, `variantId` of the credible set variant output.
        pharmacogenomics: `pharmacogenomicsId`, `variantId` of the pharmacogenomics output.
    """
    sets = credible_set_variant.group_by('variantId').agg(credibleSetIds=pl.col('credibleSetId').sort())
    pgx = (
        pharmacogenomics
        .filter(pl.col('variantId').is_not_null())
        .group_by('variantId')
        .agg(pharmacogenomicsIds=pl.col('pharmacogenomicsId').sort())
    )
    return sets.join(pgx, on='variantId', how='full', coalesce=True).collect()


def build_variant_view(variant: pl.DataFrame, ids: pl.DataFrame) -> pl.DataFrame:
    """Every variant row, in variant order, then its id lists, `[]` for none; conformed to `VariantView`."""
    found = ids.filter(pl.col('variantId').is_in(variant['variantId'].implode()))
    view = variant.join(found, on='variantId', how='left', maintain_order='left').with_columns(
        pl.col(name).cast(VIEW_SCHEMA[name]).fill_null(pl.lit([], dtype=VIEW_SCHEMA[name])) for name in ID_LISTS
    )
    return conform(view.select(VIEW_SCHEMA.names()), VariantView)


@writes(view=VariantView)
def variant_view(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Build the variant view and write it, one part per `variant` part, `concurrency` parts at a time.

    Args:
        source: `variant`, `credible_set_variant` and `pharmacogenomics` locations.
        destination: the `view` directory.
        settings: optional `concurrency`.
        config: unused.

    Raises:
        ValueError: if anything already exists at the destination.
    """
    concurrency = int((settings or {}).get('concurrency', DEFAULT_CONCURRENCY))
    path = str(destination['view']).rstrip('/')
    check_destinations_absent([path])
    if '://' not in path:
        os.makedirs(path)
    ids = variant_ids(
        scan_dataset(source['credible_set_variant']).select('credibleSetId', 'variantId'),
        scan_dataset(source['pharmacogenomics']).select('pharmacogenomicsId', 'variantId'),
    )
    parts = list_parts(source['variant'], '*.parquet')
    logger.info(f'{ids.height} variants with ids; {len(parts)} variant parts')

    def run(index: int, part: str) -> int:
        view = build_variant_view(pl.scan_parquet(part).collect(), ids)
        view.write_parquet(
            f'{path}/part-{index:05d}.parquet', compression='zstd', statistics=True, row_group_size=ROW_GROUP_SIZE
        )
        return view.height

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(run, index, part) for index, part in enumerate(parts)]
        try:
            rows = sum(future.result() for future in futures)
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise
    logger.info(f'{rows} variants in the variant view')
