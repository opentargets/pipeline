"""Genetic correlation of every pair of representative studies within an ancestry.

The pairs of an ancestry are cut into tiles: a tile is two blocks of studies, and every pair with
one study in each block (or two in the one block, on the diagonal). `run_tile` is one unit of
work: it loads each of its studies once and correlates all of its pairs.
`pts.pyspark.genetic_correlation_pairs` runs it for every tile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pts.transformers.genetic_correlation import ldsc
from pts.transformers.genetic_correlation.munge import fetch
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


@dataclass(frozen=True)
class Study:
    """A munged study as arrays, sorted by `snp`."""

    snp: np.ndarray
    z: np.ndarray
    sample_size: np.ndarray
    ld_score: np.ndarray

    @classmethod
    def read(cls, path: Path) -> Study:
        frame = pl.read_parquet(path)
        return cls(
            frame['snp'].to_numpy(),
            frame['z'].to_numpy(),
            frame['sampleSize'].to_numpy(),
            frame['ldScore'].to_numpy(),
        )


@dataclass(frozen=True)
class Tile:
    """The pairs between two blocks of an ancestry's studies; on the diagonal, the pairs within one."""

    ancestry: str
    left: tuple[str, ...]
    right: tuple[str, ...]

    @property
    def diagonal(self) -> bool:
        return self.left == self.right

    def pairs(self) -> list[tuple[str, str]]:
        """The tile's pairs, `left < right`."""
        if self.diagonal:
            return [(a, b) for i, a in enumerate(self.left) for b in self.left[i + 1 :]]
        return [(a, b) for a in self.left for b in self.right]


def tiles(studies: pl.DataFrame, size: int) -> list[Tile]:
    """Every tile of each ancestry's studies, in blocks of `size` sorted study ids.

    Args:
        studies: `studyId` and `ldAncestry`.
        size: studies per block.

    Returns:
        The tiles, together covering every pair within an ancestry once.
    """
    out = []
    for (ancestry,), group in studies.sort('ldAncestry').group_by('ldAncestry', maintain_order=True):
        ids = sorted(group['studyId'].unique())
        blocks = [tuple(ids[i : i + size]) for i in range(0, len(ids), size)]
        out.extend(Tile(ancestry, blocks[i], blocks[j]) for i in range(len(blocks)) for j in range(i, len(blocks)))
    return [tile for tile in out if tile.pairs()]


def shared(left: Study, right: Study) -> tuple[np.ndarray, np.ndarray]:
    """The positions in `left` and in `right` of the SNPs the two studies share, in genomic order."""
    _, i, j = np.intersect1d(left.snp, right.snp, assume_unique=True, return_indices=True)
    return i, j


def correlate(left: Study, right: Study, m: int, options: PairsSettings) -> dict[str, Any]:
    """One pair's row of `OUTPUT_SCHEMA`, without the identifiers."""
    i, j = shared(left, right)
    n_snps = len(i)
    if n_snps < options.min_overlap_snps:
        return _row('skipped', n_snps, m, skipReason=f'{n_snps} shared SNPs, fewer than {options.min_overlap_snps}')
    if n_snps < options.n_blocks:
        return _row(
            'skipped', n_snps, m, skipReason=f'{n_snps} shared SNPs, fewer than {options.n_blocks} jackknife blocks'
        )
    ld = left.ld_score[i]
    try:
        result = ldsc.genetic_correlation(
            left.z[i],
            right.z[j],
            left.sample_size[i],
            right.sample_size[j],
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


def run_tile(
    tile: Tile, munged: str, m_snps: dict[str, int], options: PairsSettings, scratch: Path
) -> list[dict[str, Any]]:
    """Correlate every pair of a tile.

    Args:
        tile: the tile, from `tiles`.
        munged: the munged studies, as `munge.munge_one` writes them.
        m_snps: the M of each ancestry's panel.
        options: the regression's settings.
        scratch: a local directory for the fetched munged studies.

    Returns:
        One row of `OUTPUT_SCHEMA` per pair.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    loaded = {}
    for study_id in dict.fromkeys(tile.left + tile.right):
        local = fetch(f'{munged}/{tile.ancestry}/{study_id}.parquet', scratch / f'{tile.ancestry}_{study_id}.parquet')
        loaded[study_id] = Study.read(local)
        if local.is_relative_to(scratch):
            local.unlink()
    rows = []
    for left, right in tile.pairs():
        row = correlate(loaded[left], loaded[right], m_snps[tile.ancestry], options)
        row.update(leftStudyId=left, rightStudyId=right, ldAncestry=tile.ancestry)
        rows.append(row)
    return rows


def results(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Rows from `run_tile` as the output dataset, sorted, with undefined estimates null."""
    return (
        pl
        .DataFrame(rows, schema=OUTPUT_SCHEMA)
        .with_columns(pl.col(pl.Float64).fill_nan(None))
        .sort('ldAncestry', 'leftStudyId', 'rightStudyId')
    )
