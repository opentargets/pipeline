"""Munge every representative study for genetic correlation, one Spark task per study."""

from __future__ import annotations

import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from otter.storage.storage_context import storage_context

from pts.pyspark.common.session import Session
from pts.transformers.genetic_correlation.munge import PANELS_SCHEMA, STUDIES_SCHEMA, munge_one, panel_size, read_panel
from pts.transformers.genetic_correlation.settings import MungeDestination, MungeSettings, MungeSource
from pts.transformers.utils.dataset import check_destinations_absent, list_parts, write_dataset

_panels: dict[str, pl.DataFrame] = {}
"""The panels a Python worker has read, by path; a worker munges many studies of one ancestry."""


def _panel(path: str) -> pl.DataFrame:
    if path not in _panels:
        _panels.clear()
        with tempfile.TemporaryDirectory() as scratch:
            _panels[path] = read_panel(path, Path(scratch))
    return _panels[path]


def _munge_task(
    study: dict[str, Any], sources: MungeSource, destinations: MungeDestination, options: MungeSettings
) -> dict[str, Any]:
    """Munge one study on an executor; returns its SNP count and the seconds it took."""
    start = time.monotonic()
    panel = _panel(f'{sources.ldscore}/{options.ldscore_template.format(ancestry=study["ldAncestry"])}')
    with tempfile.TemporaryDirectory() as scratch, storage_context(billing_project=options.billing_project):
        n_snps = munge_one(study, panel, sources.sumstats, destinations.munged, Path(scratch))
    return {
        'studyId': study['studyId'],
        'ldAncestry': study['ldAncestry'],
        'nSnps': n_snps,
        'seconds': time.monotonic() - start,
    }


def genetic_correlation_munge(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Munge every representative study, and write each study's SNP count and each panel's M.

    Args:
        source: the `MungeSource` paths.
        destination: the `MungeDestination` paths.
        settings: the `MungeSettings`.
        properties: Spark properties.
    """
    sources = MungeSource.model_validate(source)
    destinations = MungeDestination.model_validate(destination)
    options = MungeSettings.model_validate(settings)
    check_destinations_absent([destinations.munged, destinations.studies, destinations.panels])
    studies = (
        pl
        .read_parquet(list_parts(sources.representatives, '*.parquet'))
        .select('studyId', 'ldAncestry', 'effectiveSampleSize')
        .unique()
        .sort('ldAncestry', 'studyId')
    )
    panels = []
    for ancestry in studies['ldAncestry'].unique().sort():
        with tempfile.TemporaryDirectory() as scratch:
            panel = read_panel(f'{sources.ldscore}/{options.ldscore_template.format(ancestry=ancestry)}', Path(scratch))
        panels.append({'ldAncestry': ancestry, 'mSnps': panel_size(panel, options.maf_threshold)})
        logger.info(f'{ancestry}: {panel.height} panel SNPs, M = {panels[-1]["mSnps"]}')
        del panel

    spark = Session(app_name='genetic_correlation_munge', properties=properties).spark
    rows = list(studies.iter_rows(named=True))
    logger.info(f'munging {len(rows)} studies')
    work = partial(_munge_task, sources=sources, destinations=destinations, options=options)
    done = spark.sparkContext.parallelize(rows, numSlices=len(rows)).map(work).collect()
    seconds = sum(row['seconds'] for row in done)
    logger.info(
        f'munged {len(done)} studies, {sum(row["nSnps"] for row in done)} SNPs, '
        f'{seconds / 3600:.2f} task hours, {seconds / max(len(done), 1):.1f} s per study'
    )
    index = pl.DataFrame(
        [{k: row[k] for k in ('studyId', 'ldAncestry', 'nSnps')} for row in done],
        schema={k: STUDIES_SCHEMA[k] for k in ('studyId', 'ldAncestry', 'nSnps')},
    ).with_columns((pl.col('nSnps') >= options.min_snps).alias('isUsable'))
    excluded = index.filter(~pl.col('isUsable'))
    if excluded.height:
        logger.warning(
            f'{excluded.height} studies have fewer than {options.min_snps} panel SNPs and are not correlated'
        )
    write_dataset(index.sort('ldAncestry', 'studyId'), destinations.studies)
    write_dataset(pl.DataFrame(panels, schema=PANELS_SCHEMA), destinations.panels)
