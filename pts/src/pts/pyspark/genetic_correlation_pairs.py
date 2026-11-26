"""Genetic correlation of every pair of representative studies, one Spark task per tile of pairs.

With a previous run computed with the same parameters, the pairs of two studies whose munged files
were reused are copied from it, and the previous pairs no longer current are kept as deprecated.
"""

from __future__ import annotations

import math
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from pts.pyspark.common.session import Session
from pts.transformers.genetic_correlation.pairs import (
    PAIR_KEY,
    Tile,
    assemble,
    parameters,
    results,
    run_tile,
    tiles,
)
from pts.transformers.genetic_correlation.previous import previous_path
from pts.transformers.genetic_correlation.settings import PairsDestination, PairsSettings, PairsSource
from pts.transformers.utils.dataset import check_destinations_absent, list_parts, write_dataset

_scratch = Path(tempfile.mkdtemp(prefix='genetic_correlation_'))
"""A Python worker's directory for the munged studies it fetches."""


def _pairs_task(tile: Tile, munged: str, m_snps: dict[str, int], options: PairsSettings) -> list[dict[str, Any]]:
    """Correlate one tile on an executor, with the seconds it took on its first row."""
    start = time.monotonic()
    rows = run_tile(tile, munged, m_snps, options, _scratch)
    rows[0]['_seconds'] = time.monotonic() - start
    return rows


def _tile_size(studies: pl.DataFrame, largest: int, min_tiles: int) -> int:
    """The largest tile size up to `largest` that gives at least `min_tiles` tiles."""
    biggest_ancestry = int(studies.group_by('ldAncestry').len()['len'].max() or 1)
    # an ancestry of n studies in blocks of s has about (n / s)^2 / 2 tiles
    size = math.floor(biggest_ancestry / math.sqrt(2 * min_tiles))
    return max(1, min(largest, size))


def _read(path: str) -> pl.DataFrame:
    return pl.read_parquet(list_parts(path, '*.parquet'))


def _previous(
    sources: PairsSource, destinations: PairsDestination, studies: pl.DataFrame, options: PairsSettings
) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """The previous run's output, and the studies whose pairs with each other can be copied from it.

    A study qualifies if its munged file was reused and it was correlated in both runs, and only if
    the previous pairs were computed with the same parameters.
    """
    previous_output = previous_path(sources.previous_output, destinations.genetic_correlation)
    if previous_output is None:
        return None, None
    previous = _read(previous_output)
    previous_parameters = _read(str(previous_path(sources.previous_output, destinations.parameters)))
    if not previous_parameters.equals(parameters(options)):
        logger.warning('the previous run used other parameters: every pair is recomputed')
        return previous, None
    previous_studies = _read(str(previous_path(sources.previous_output, sources.studies)))
    reused = (
        studies
        .filter(pl.col('isReused'))
        .join(previous_studies.filter(pl.col('isUsable')), on=['ldAncestry', 'studyId'], how='semi')
        .select('ldAncestry', 'studyId')
    )
    return previous, reused


def genetic_correlation_pairs(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Correlate every pair of usable munged studies within an ancestry.

    Args:
        source: the `PairsSource` paths.
        destination: the `PairsDestination` paths.
        settings: the `PairsSettings`.
        properties: Spark properties.

    Raises:
        ValueError: if the previous run lacks a pair of two studies whose munged files were reused.
    """
    sources = PairsSource.model_validate(source)
    destinations = PairsDestination.model_validate(destination)
    options = PairsSettings.model_validate(settings)
    check_destinations_absent([destinations.genetic_correlation, destinations.parameters])
    index = _read(sources.studies)
    studies = index.filter(pl.col('isUsable'))
    m_snps = dict(_read(sources.panels).select('ldAncestry', 'mSnps').iter_rows())
    previous, reused = _previous(sources, destinations, index, options)
    spark = Session(app_name='genetic_correlation_pairs', properties=properties).spark
    size = _tile_size(studies, options.tile_studies, options.min_tiles)
    work = tiles(studies, size, reused)
    n_pairs = sum(len(tile.pairs()) for tile in work)
    n_reused = 0 if reused is None else reused.height
    logger.info(
        f'{studies.height} studies ({n_reused} reused): computing {n_pairs} pairs in {len(work)} tiles of up '
        f'to {size} studies a side'
    )

    rows: list[dict[str, Any]] = []
    if work:
        task = partial(_pairs_task, munged=sources.munged, m_snps=m_snps, options=options)
        rows = spark.sparkContext.parallelize(work, numSlices=len(work)).flatMap(task).collect()
    seconds = sum(row.pop('_seconds', 0.0) for row in rows)
    computed = results(rows)
    output = assemble(computed, index, previous, reused)

    current = output.filter(~pl.col('isDeprecated'))
    expected = sum(n * (n - 1) // 2 for n in studies.group_by('ldAncestry').len()['len'])
    if current.height != expected or current.select(PAIR_KEY).is_duplicated().any():
        raise ValueError(f'{current.height} current pairs, but {studies.height} studies have {expected} pairs')
    logger.info(
        f'{current.height} current pairs ({computed.height} computed, {current.height - computed.height} '
        f'copied), {output.height - current.height} deprecated; computed: '
        f'{dict(computed.group_by("runStatus").len().iter_rows())}, {seconds / 3600:.2f} task hours, '
        f'{seconds / max(computed.height, 1):.2f} s per pair'
    )
    write_dataset(output, destinations.genetic_correlation)
    write_dataset(parameters(options), destinations.parameters)
