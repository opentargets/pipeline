"""Genetic correlation of every pair of representative studies, one Spark task per tile of pairs."""

from __future__ import annotations

import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from pts.pyspark.common.session import Session
from pts.transformers.genetic_correlation.pairs import Tile, results, run_tile, tiles
from pts.transformers.genetic_correlation.settings import PairsSettings, PairsSource
from pts.transformers.utils.dataset import check_destinations_absent, list_parts, write_dataset

_scratch = Path(tempfile.mkdtemp(prefix='genetic_correlation_'))
"""A Python worker's directory for the munged studies it fetches."""


def _pairs_task(tile: Tile, munged: str, m_snps: dict[str, int], options: PairsSettings) -> list[dict[str, Any]]:
    """Correlate one tile on an executor, with the seconds it took on its first row."""
    start = time.monotonic()
    rows = run_tile(tile, munged, m_snps, options, _scratch)
    rows[0]['_seconds'] = time.monotonic() - start
    return rows


def genetic_correlation_pairs(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Correlate every pair of usable munged studies within an ancestry.

    Args:
        source: the `PairsSource` paths.
        destination: the genetic correlation dataset, one row per pair.
        settings: the `PairsSettings`.
        properties: Spark properties.
    """
    sources = PairsSource.model_validate(source)
    options = PairsSettings.model_validate(settings)
    check_destinations_absent([destination])
    studies = pl.read_parquet(list_parts(sources.studies, '*.parquet')).filter(pl.col('isUsable'))
    panels = pl.read_parquet(list_parts(sources.panels, '*.parquet'))
    m_snps = dict(zip(panels['ldAncestry'], panels['mSnps'], strict=True))
    work = tiles(studies, options.tile_studies)
    n_pairs = sum(len(tile.pairs()) for tile in work)
    logger.info(f'{n_pairs} pairs over {studies.height} studies, in {len(work)} tiles')

    spark = Session(app_name='genetic_correlation_pairs', properties=properties).spark
    task = partial(_pairs_task, munged=sources.munged, m_snps=m_snps, options=options)
    rows = spark.sparkContext.parallelize(work, numSlices=len(work)).flatMap(task).collect()
    seconds = sum(row.pop('_seconds', 0.0) for row in rows)
    output = results(rows)
    logger.info(
        f'{output.height} pairs: {dict(output.group_by("runStatus").len().iter_rows())}; '
        f'{seconds / 3600:.2f} task hours, {seconds / max(output.height, 1):.2f} s per pair'
    )
    write_dataset(output, destination)
