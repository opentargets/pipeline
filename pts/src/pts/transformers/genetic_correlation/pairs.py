"""Genetic correlation of every pair of representative studies within an ancestry.

`run_chunk` is one unit of work: a list of pairs, sorted so that consecutive pairs share their
first study and the worker's small cache of loaded studies keeps hitting.
`pts.pyspark.genetic_correlation_pairs` runs it for every chunk.
"""

from __future__ import annotations

from collections import OrderedDict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pts.transformers.genetic_correlation import ldsc
from pts.transformers.genetic_correlation.munge import VARIANT_KEY, fetch
from pts.transformers.genetic_correlation.settings import PairsSettings

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


def _load(munged: str, ancestry: str, study_id: str, scratch: Path) -> pl.DataFrame:
    key = (ancestry, study_id)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    local = fetch(f'{munged}/{ancestry}/{study_id}.parquet', scratch / f'{ancestry}_{study_id}.parquet')
    frame = pl.read_parquet(local)
    if local.is_relative_to(scratch):
        local.unlink()
    _cache[key] = frame
    while len(_cache) > _CACHE_SIZE:
        _cache.popitem(last=False)
    return frame


def chunks(pairs: pl.DataFrame, size: int) -> list[list[tuple[str, str, str]]]:
    """`pairs` from `study_pairs`, in order, as lists of up to `size` (ancestry, left, right) tuples."""
    rows = list(pairs.select('ldAncestry', 'leftStudyId', 'rightStudyId').iter_rows())
    return [rows[i : i + size] for i in range(0, len(rows), size)]


def run_chunk(
    chunk: list[tuple[str, str, str]], munged: str, m_snps: dict[str, int], options: PairsSettings, scratch: Path
) -> list[dict[str, Any]]:
    """Correlate a chunk of (ancestry, left, right) pairs.

    Args:
        chunk: the pairs, from `chunks`.
        munged: the munged studies, as `munge.munge_one` writes them.
        m_snps: the M of each ancestry's panel.
        options: the regression's settings.
        scratch: a local directory for the fetched munged studies.

    Returns:
        One row of `OUTPUT_SCHEMA` per pair.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    rows = []
    for ancestry, left, right in chunk:
        row = correlate(
            _load(munged, ancestry, left, scratch), _load(munged, ancestry, right, scratch), m_snps[ancestry], options
        )
        row.update(leftStudyId=left, rightStudyId=right, ldAncestry=ancestry)
        rows.append(row)
    return rows


def results(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Rows from `run_chunk` as the output dataset, sorted, with undefined estimates null."""
    return (
        pl
        .DataFrame(rows, schema=OUTPUT_SCHEMA)
        .with_columns(pl.col(pl.Float64).fill_nan(None))
        .sort('ldAncestry', 'leftStudyId', 'rightStudyId')
    )
