"""The pharmacologist judge: what it reads, what it is asked about, and how it adjusts precision.

It judges every grounded mechanism the benchmark cannot score (drugs without a ChEMBL
mechanism), every benchmark miss, and a sample of benchmark hits so its agreement with
the benchmark can be measured.
"""

from __future__ import annotations

from typing import Any

import polars as pl

JUDGEABLE_GROUNDING = ('source', 'suggestion')


def request_id(pool_id: str, mechanism_idx: int) -> str:
    """Return the id one mechanism's judge request is cached under."""
    return f'{pool_id}:{mechanism_idx}'


def select_for_judging(scored: pl.DataFrame, seed: int = 0, n_matches: int = 100) -> pl.DataFrame:
    """Return the keys of the mechanisms to judge.

    Returns:
        pl.DataFrame: ``pool_id``, ``mechanism_idx``.
    """
    keys = ['pool_id', 'mechanism_idx']
    unbenchmarked = scored.filter(
        ~pl.col('benchmarked')
        & pl.col('kept')
        & pl.col('isHumanTarget')
        & pl.col('grounding').is_in(JUDGEABLE_GROUNDING)
        & ~pl.col('class_level')
    )
    misses = scored.filter(pl.col('in_headline') & (pl.col('correct') == False))  # noqa: E712
    matches = scored.filter(pl.col('in_headline') & pl.col('correct')).sort(keys)
    sampled = matches.sample(n=min(n_matches, matches.height), seed=seed, shuffle=True)
    return pl.concat([unbenchmarked.select(keys), misses.select(keys), sampled.select(keys)]).unique().sort(keys)


def judge_input(row: dict[str, Any], pool_text: str, chembl_rows: list[dict[str, Any]]) -> str:
    """Render the judge's input for one mechanism."""
    quotes = '\n'.join(f'- [{e["nctId"]}] "{e["quote"]}"' for e in row['evidence']) or '- none'
    chembl = (
        '\n'.join(f'- {r["mechanismOfAction"]} ({r["actionType"]})' for r in chembl_rows)
        if chembl_rows
        else 'none recorded'
    )
    mechanism = (
        f'Target: {row["targetFromSource"]}\n'
        f'Target type: {row["targetType"]}\n'
        f'Action type: {row["actionType"]}\n'
        f'Mechanism: {row["mechanismOfAction"]}\n'
        f'Quotes:\n{quotes}'
    )
    return f'TRIAL TEXTS\n{pool_text}\n\nMECHANISM\n{mechanism}\n\nCHEMBL MECHANISMS\n{chembl}\n'


def adjust(scored: pl.DataFrame, verdicts: pl.DataFrame) -> pl.DataFrame:
    """Add ``judge_correct`` and ``adjusted_correct`` from the judge's verdicts.

    A mechanism is judge-correct when the judge says its quotes support it and it is the
    drug's primary mechanism. The adjusted verdict keeps every benchmark hit and lets the
    judge rule on everything else it saw.
    """
    judged = verdicts.select('pool_id', 'mechanism_idx', 'quoteSupports', 'primaryMechanism', 'rationale')
    return (
        scored
        .join(judged, on=['pool_id', 'mechanism_idx'], how='left')
        .with_columns(
            judge_correct=pl.when(pl.col('quoteSupports').is_not_null()).then(
                (pl.col('quoteSupports') == 'yes') & (pl.col('primaryMechanism') == 'yes')
            )
        )
        .with_columns(
            adjusted_correct=pl
            .when(pl.col('correct'))
            .then(True)
            .when(pl.col('judge_correct').is_not_null())
            .then(pl.col('judge_correct'))
            .otherwise(pl.col('correct'))
        )
    )


def calibration(adjusted: pl.DataFrame) -> pl.DataFrame:
    """Count benchmark verdict against judge verdict over the judged benchmark mechanisms.

    Returns:
        pl.DataFrame: ``correct``, ``judge_correct``, ``n``, and ``agreement`` (the share on
        the diagonal, the same on every row).
    """
    table = (
        adjusted
        .filter(pl.col('in_headline') & pl.col('judge_correct').is_not_null())
        .group_by('correct', 'judge_correct')
        .agg(n=pl.len())
        .sort('correct', 'judge_correct')
    )
    total = int(table['n'].sum())
    agreeing = int(table.filter(pl.col('correct') == pl.col('judge_correct'))['n'].sum())
    return table.with_columns(agreement=pl.lit(agreeing / total if total else None, dtype=pl.Float64))


def handcheck(adjusted: pl.DataFrame, pools: pl.DataFrame, n: int = 50, seed: int = 0) -> pl.DataFrame:
    """Return a seeded sample of judged mechanisms for reading by hand."""
    judged = adjusted.filter(pl.col('judge_correct').is_not_null()).sort('pool_id', 'mechanism_idx')
    sample = judged.sample(n=min(n, judged.height), seed=seed, shuffle=True)
    return sample.join(pools.select('pool_id', 'drug_name'), on='pool_id', how='left').select(
        'drug_name',
        'pool_id',
        'mechanism_idx',
        'targetFromSource',
        pl.col('evidence').list.eval(pl.element().struct.field('quote')).list.join(' | ').alias('quotes'),
        'benchmarked',
        'correct',
        'quoteSupports',
        'primaryMechanism',
        'rationale',
    )
