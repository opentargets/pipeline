"""Genetic correlation of every pair of representative studies within an ancestry.

Pairs are handed to worker processes in chunks, sorted so that consecutive pairs share their first
study and a worker's small cache of loaded studies keeps hitting. The regression is numpy; each
pair takes about a second of one core.
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.transformers.genetic_correlation import ldsc
from pts.transformers.genetic_correlation.munge import VARIANT_KEY, fetch, is_remote
from pts.transformers.genetic_correlation.settings import PairsSettings, PairsSource
from pts.transformers.utils.dataset import check_destinations_absent, list_parts, write_dataset

OUTPUT_SCHEMA = pl.Schema({
    'leftStudyId': pl.String,
    'rightStudyId': pl.String,
    'ldAncestry': pl.String,
    'runStatus': pl.String,
    'skipReason': pl.String,
    'nSnps': pl.Int64,
    'rg': pl.Float64,
    'rgStandardError': pl.Float64,
    'geneticCovariance': pl.Float64,
    'geneticCovarianceStandardError': pl.Float64,
    'intercept': pl.Float64,
    'interceptStandardError': pl.Float64,
    'leftH2': pl.Float64,
    'leftH2StandardError': pl.Float64,
    'rightH2': pl.Float64,
    'rightH2StandardError': pl.Float64,
    'mSnps': pl.Int64,
})
"""One row per pair. `runStatus` is `success`, `skipped` or `error`; estimates are null otherwise,
and null where undefined (rg when a heritability is not positive)."""

_CACHE_SIZE = 3
"""Studies a worker keeps loaded; the second study of a pair changes every time, the first rarely."""

_cache: OrderedDict[tuple[str, str], pl.DataFrame] = OrderedDict()


def study_pairs(representatives: pl.DataFrame) -> pl.DataFrame:
    """Every pair of distinct representative studies within an ancestry, `leftStudyId < rightStudyId`."""
    rows = []
    for (ancestry,), group in representatives.group_by('ldAncestry', maintain_order=True):
        studies = sorted(group['studyId'].unique())
        rows.extend((ancestry, left, right) for left, right in combinations(studies, 2))
    return pl.DataFrame(
        rows, schema={'ldAncestry': pl.String, 'leftStudyId': pl.String, 'rightStudyId': pl.String}, orient='row'
    ).sort('ldAncestry', 'leftStudyId', 'rightStudyId')


def merge_pair(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    """The SNPs two munged studies share, in genomic order, with the right study's effect on the left's alleles.

    A SNP is matched on its alleles as they are, or with ref and alt swapped and the effect negated.
    """
    right = right.select(
        *VARIANT_KEY,
        pl.col('beta').alias('rightBeta'),
        pl.col('standardError').alias('rightStandardError'),
        pl.col('sampleSize').alias('rightSampleSize'),
    )
    flipped = right.select(
        'chromosome',
        'position',
        pl.col('alt').alias('ref'),
        pl.col('ref').alias('alt'),
        -pl.col('rightBeta'),
        'rightStandardError',
        'rightSampleSize',
    )
    return (
        pl
        .concat([left.join(right, on=VARIANT_KEY), left.join(flipped, on=VARIANT_KEY)])
        .unique(VARIANT_KEY, keep='first')
        .sort('chromosome', 'position')
    )


def correlate(left: pl.DataFrame, right: pl.DataFrame, m: int, options: PairsSettings) -> dict[str, Any]:
    """One pair's row of `OUTPUT_SCHEMA`, without the identifiers."""
    merged = merge_pair(left, right)
    n_snps = merged.height
    if n_snps < options.min_overlap_snps:
        return _row('skipped', n_snps, m, skipReason=f'{n_snps} shared SNPs, fewer than {options.min_overlap_snps}')
    if n_snps < options.n_blocks:
        return _row(
            'skipped', n_snps, m, skipReason=f'{n_snps} shared SNPs, fewer than {options.n_blocks} jackknife blocks'
        )
    ld = merged['ldScore'].to_numpy()
    try:
        result = ldsc.genetic_correlation(
            (merged['beta'] / merged['standardError']).to_numpy(),
            (merged['rightBeta'] / merged['rightStandardError']).to_numpy(),
            merged['sampleSize'].to_numpy(),
            merged['rightSampleSize'].to_numpy(),
            ld,
            ld,
            m,
            n_blocks=options.n_blocks,
            twostep=options.twostep,
            intercept=options.intercept,
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        return _row('error', n_snps, m, skipReason=f'{type(error).__name__}: {error}')
    return _row(
        'success',
        n_snps,
        m,
        rg=result.rg,
        rgStandardError=result.rg_se,
        geneticCovariance=result.genetic_covariance,
        geneticCovarianceStandardError=result.genetic_covariance_se,
        intercept=result.intercept,
        interceptStandardError=result.intercept_se,
        leftH2=result.h2_1,
        leftH2StandardError=result.h2_1_se,
        rightH2=result.h2_2,
        rightH2StandardError=result.h2_2_se,
    )


def _row(status: str, n_snps: int, m: int, **values: Any) -> dict[str, Any]:
    if unknown := set(values) - set(OUTPUT_SCHEMA):
        raise ValueError(f'not output columns: {sorted(unknown)}')
    row: dict[str, Any] = dict.fromkeys(OUTPUT_SCHEMA)
    row.update(runStatus=status, nSnps=n_snps, mSnps=m, **values)
    return row


def _load(munged: str, ancestry: str, study_id: str) -> pl.DataFrame:
    key = (ancestry, study_id)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    frame = pl.read_parquet(f'{munged}/{ancestry}/{study_id}.parquet')
    _cache[key] = frame
    while len(_cache) > _CACHE_SIZE:
        _cache.popitem(last=False)
    return frame


def run_chunk(
    chunk: list[tuple[str, str, str]], munged: str, m_snps: dict[str, int], options: PairsSettings
) -> list[dict[str, Any]]:
    """Correlate a chunk of (ancestry, left, right) pairs, reading munged studies from `munged`."""
    rows = []
    for ancestry, left, right in chunk:
        row = correlate(_load(munged, ancestry, left), _load(munged, ancestry, right), m_snps[ancestry], options)
        row.update(leftStudyId=left, rightStudyId=right, ldAncestry=ancestry)
        rows.append(row)
    return rows


def _stage_munged(munged: str, representatives: pl.DataFrame, scratch: Path, workers: int) -> str:
    """The munged studies on local disk: `munged` itself, or a copy of the representatives' files."""
    if not is_remote(munged):
        return munged
    files = [
        (
            f'{munged}/{row["ldAncestry"]}/{row["studyId"]}.parquet',
            scratch / row['ldAncestry'] / f'{row["studyId"]}.parquet',
        )
        for row in representatives.select('ldAncestry', 'studyId').unique().iter_rows(named=True)
    ]
    logger.info('Fetching {} munged studies', len(files))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda pair: fetch(*pair), files))
    return str(scratch)


def pairs(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Correlate every pair of representatives.

    Args:
        source: the `PairsSource` paths.
        destination: the genetic correlation dataset, one row per pair.
        settings: the `PairsSettings`.
        config: for `work_path`, where the munged studies are staged.
    """
    sources = PairsSource.model_validate(source)
    options = PairsSettings.model_validate(settings)
    check_destinations_absent([destination])
    representatives = pl.read_parquet(list_parts(sources.representatives, '*.parquet'))
    panels = pl.read_parquet(list_parts(sources.panels, '*.parquet'))
    m_snps = dict(zip(panels['ldAncestry'], panels['mSnps'], strict=True))
    all_pairs = study_pairs(representatives)
    logger.info('{} pairs over {} studies', all_pairs.height, representatives['studyId'].n_unique())

    with TemporaryDirectory(dir=config.work_path) as directory:
        munged = _stage_munged(sources.munged, representatives, Path(directory), options.parallel_pairs)
        rows = list(all_pairs.iter_rows())
        chunks = [rows[i : i + options.pairs_per_chunk] for i in range(0, len(rows), options.pairs_per_chunk)]
        work = partial(run_chunk, munged=munged, m_snps=m_snps, options=options)
        results: list[dict[str, Any]] = []
        with (
            _worker_threads(options.worker_threads),
            ProcessPoolExecutor(
                max_workers=options.parallel_pairs, mp_context=multiprocessing.get_context('spawn')
            ) as pool,
        ):
            for done, chunk_rows in enumerate(pool.map(work, chunks), start=1):
                results.extend(chunk_rows)
                if done % 50 == 0 or done == len(chunks):
                    logger.info('{} of {} chunks done', done, len(chunks))
        if is_remote(sources.munged):
            shutil.rmtree(munged, ignore_errors=True)

    output = (
        pl
        .DataFrame(results, schema=OUTPUT_SCHEMA)
        .with_columns(pl.col(pl.Float64).fill_nan(None))
        .sort('ldAncestry', 'leftStudyId', 'rightStudyId')
    )
    statuses = dict(output.group_by('runStatus').len().iter_rows())
    logger.info('{} pairs: {}', output.height, statuses)
    write_dataset(output, destination)


@contextmanager
def _worker_threads(threads: int) -> Iterator[None]:
    """Set `POLARS_MAX_THREADS` for the spawned workers, which read it when they import polars."""
    previous = os.environ.get('POLARS_MAX_THREADS')
    os.environ['POLARS_MAX_THREADS'] = str(threads)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('POLARS_MAX_THREADS', None)
        else:
            os.environ['POLARS_MAX_THREADS'] = previous
