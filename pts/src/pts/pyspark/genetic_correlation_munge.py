"""Munge every representative study for genetic correlation, one Spark task per study.

A study the previous run munged with the same panel and `MUNGE_VERSION` is copied from it instead.
"""

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
from pts.transformers.genetic_correlation.munge import (
    MUNGE_VERSION,
    PANELS_SCHEMA,
    STUDIES_SCHEMA,
    copy,
    munge_one,
    panel_checksum,
    panel_size,
    read_panel,
)
from pts.transformers.genetic_correlation.previous import previous_path
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
    """Munge one study on an executor, or copy it from the previous run; returns its SNP count and the seconds."""
    start = time.monotonic()
    study_id, ancestry = study['studyId'], study['ldAncestry']
    with tempfile.TemporaryDirectory() as scratch, storage_context(billing_project=options.billing_project):
        if study['previousMunged'] is not None:
            copy(study['previousMunged'], f'{destinations.munged}/{ancestry}/{study_id}.parquet', Path(scratch))
            n_snps = study['previousSnps']
        else:
            panel = _panel(f'{sources.ldscore}/{options.ldscore_template.format(ancestry=ancestry)}')
            n_snps = munge_one(study, panel, sources.sumstats, destinations.munged, Path(scratch))
    return {
        'studyId': study_id,
        'ldAncestry': ancestry,
        'nSnps': n_snps,
        'isReused': study['previousMunged'] is not None,
        'seconds': time.monotonic() - start,
    }


def _reusable(
    studies: pl.DataFrame, panels: pl.DataFrame, sources: MungeSource, destinations: MungeDestination
) -> pl.DataFrame:
    """`studies` with the previous run's munged file and SNP count, where it can be reused.

    `previousMunged` and `previousSnps` are set for each study the previous run munged with the same
    panel and `MUNGE_VERSION`, and null otherwise.
    """
    none = studies.with_columns(
        pl.lit(None, pl.String).alias('previousMunged'), pl.lit(None, pl.Int64).alias('previousSnps')
    )
    previous_munged = previous_path(sources.previous_output, destinations.munged)
    if previous_munged is None:
        return none
    previous_panels = pl.read_parquet(
        list_parts(str(previous_path(sources.previous_output, destinations.panels)), '*.parquet')
    )
    unchanged = panels.join(previous_panels, on=['ldAncestry', 'panelChecksum', 'mungeVersion'], how='semi')
    previous = (
        pl
        .read_parquet(list_parts(str(previous_path(sources.previous_output, destinations.studies)), '*.parquet'))
        .join(unchanged.select('ldAncestry'), on='ldAncestry', how='semi')
        .select(
            'studyId',
            'ldAncestry',
            pl.format('{}/{}/{}.parquet', pl.lit(previous_munged), 'ldAncestry', 'studyId').alias('previousMunged'),
            pl.col('nSnps').alias('previousSnps'),
        )
    )
    return studies.join(previous, on=['studyId', 'ldAncestry'], how='left')


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
    panel_rows = []
    for ancestry in studies['ldAncestry'].unique().sort():
        path = f'{sources.ldscore}/{options.ldscore_template.format(ancestry=ancestry)}'
        with tempfile.TemporaryDirectory() as scratch:
            panel = read_panel(path, Path(scratch))
        panel_rows.append({
            'ldAncestry': ancestry,
            'mSnps': panel_size(panel, options.maf_threshold),
            'panelChecksum': panel_checksum(path),
            'mungeVersion': MUNGE_VERSION,
        })
        logger.info(f'{ancestry}: {panel.height} panel SNPs, M = {panel_rows[-1]["mSnps"]}')
        del panel
    panels = pl.DataFrame(panel_rows, schema=PANELS_SCHEMA)
    studies = _reusable(studies, panels, sources, destinations)

    spark = Session(app_name='genetic_correlation_munge', properties=properties).spark
    rows = list(studies.iter_rows(named=True))
    logger.info(f'{len(rows)} studies, {studies["previousMunged"].is_not_null().sum()} reused from the previous run')
    work = partial(_munge_task, sources=sources, destinations=destinations, options=options)
    done = spark.sparkContext.parallelize(rows, numSlices=len(rows)).map(work).collect()
    munged = [row for row in done if not row['isReused']]
    seconds = sum(row['seconds'] for row in done)
    logger.info(
        f'munged {len(munged)} studies and copied {len(done) - len(munged)}, '
        f'{sum(row["nSnps"] for row in done)} SNPs, {seconds / 3600:.2f} task hours, '
        f'{sum(row["seconds"] for row in munged) / max(len(munged), 1):.1f} s per munged study'
    )
    index = pl.DataFrame(
        [{k: row[k] for k in ('studyId', 'ldAncestry', 'nSnps', 'isReused')} for row in done],
        schema={k: STUDIES_SCHEMA[k] for k in ('studyId', 'ldAncestry', 'nSnps', 'isReused')},
    ).with_columns((pl.col('nSnps') >= options.min_snps).alias('isUsable'))
    excluded = index.filter(~pl.col('isUsable'))
    if excluded.height:
        logger.warning(
            f'{excluded.height} studies have fewer than {options.min_snps} panel SNPs and are not correlated'
        )
    write_dataset(index.select(list(STUDIES_SCHEMA)).sort('ldAncestry', 'studyId'), destinations.studies)
    write_dataset(panels, destinations.panels)
