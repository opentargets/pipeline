"""The pilot sample: every early-phase drug the benchmark can score, plus enough of the rest to compare."""

from __future__ import annotations

import polars as pl

BENCHMARKED_STRATA: tuple[tuple[tuple[str, ...], int | None], ...] = (
    (('EARLY_PHASE_1', 'PHASE_1', 'PHASE_1_2'), None),
    (('PHASE_2',), 400),
    (('PHASE_2_3',), 100),
    (('PHASE_3',), 100),
    (('APPROVAL',), 100),
)
"""Stages of linked, benchmarked pools, and how many to take (None for all)."""

UNBENCHMARKED_EARLY = 200
"""Early-phase pools the benchmark cannot score, for the judge."""


def _take(frame: pl.DataFrame, n: int | None, seed: int) -> pl.DataFrame:
    frame = frame.sort('pool_id')
    if n is None or frame.height <= n:
        return frame
    return frame.sample(n=n, seed=seed, shuffle=True)


def pilot_sample(links: pl.DataFrame, bench: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    """Return the pool ids of the pilot.

    Returns:
        pl.DataFrame: ``pool_id``.
    """
    flagged = links.with_columns(
        benchmarked=(pl.col('link') == 'linked') & pl.col('drugId').is_in(bench['drugId'].to_list())
    )
    parts = [
        _take(flagged.filter(pl.col('benchmarked') & pl.col('maximumClinicalStage').is_in(stages)), n, seed)
        for stages, n in BENCHMARKED_STRATA
    ]
    parts.append(
        _take(flagged.filter(~pl.col('benchmarked') & (pl.col('phase_group') == 'early')), UNBENCHMARKED_EARLY, seed)
    )
    return pl.concat([p.select('pool_id') for p in parts]).unique(maintain_order=True)
