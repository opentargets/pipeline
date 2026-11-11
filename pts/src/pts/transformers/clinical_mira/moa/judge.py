"""The pharmacologist judge: what it reads, what it is asked about, and how it adjusts precision.

It judges every grounded mechanism the benchmark cannot score (drugs without a ChEMBL
mechanism), every benchmark miss, and a sample of benchmark hits so its agreement with
the benchmark can be measured.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from pts.transformers.clinical_mira.moa.grounding import GROUNDED

JUDGEABLE_GROUNDING = GROUNDED


def request_id(pool_id: str, mechanism_idx: int) -> str:
    """Return the id one mechanism's judge request is cached under."""
    return f'{pool_id}:{mechanism_idx}'


def select_for_judging(
    scored: pl.DataFrame, seed: int = 0, n_matches: int = 100, strata: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Return the keys of the mechanisms to judge.

    The sample of benchmark hits used for calibration is ``n_matches`` in all; with
    ``strata`` (``pool_id`` and one stratum column) it is split evenly across the strata, so
    the judge's agreement can be read per phase.

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
        & pl.col('eligible')
        & ~pl.col('duplicate')
    )
    misses = scored.filter(pl.col('in_headline') & (pl.col('correct') == False))  # noqa: E712
    matches = scored.filter(pl.col('in_headline') & pl.col('correct')).sort(keys)
    if strata is None:
        sampled = matches.sample(n=min(n_matches, matches.height), seed=seed, shuffle=True)
    else:
        column = next(c for c in strata.columns if c != 'pool_id')
        with_stratum = matches.join(strata, on='pool_id', how='left')
        groups = sorted(with_stratum[column].drop_nulls().unique().to_list())
        per_group = max(1, n_matches // max(1, len(groups)))
        sampled = pl.concat(
            [
                part.sample(n=min(per_group, part.height), seed=seed, shuffle=True)
                for g in groups
                if (part := with_stratum.filter(pl.col(column) == g)).height
            ]
            or [with_stratum.head(0)]
        ).select(matches.columns)
    return pl.concat([unbenchmarked.select(keys), misses.select(keys), sampled.select(keys)]).unique().sort(keys)


def judge_input(row: dict[str, Any], pool_text: str, chembl_rows: list[dict[str, Any]] | None) -> str:
    """Render the judge's input for one mechanism.

    With ``chembl_rows`` None the judge is blind to ChEMBL, so its agreement with the
    benchmark measures what it would do on drugs ChEMBL has no mechanism for.
    """
    quotes = '\n'.join(f'- [{e["nctId"]}] "{e["quote"]}"' for e in row['evidence']) or '- none'
    chembl = (
        '\n'.join(f'- {r["mechanismOfAction"]} ({r["actionType"]})' for r in chembl_rows)
        if chembl_rows
        else 'none recorded'
    )
    genes = ', '.join(row.get('targetSymbols') or []) or 'none'
    mechanism = (
        f'Target: {row["targetFromSource"]}\n'
        f'Genes: {genes}\n'
        f'Target type: {row["targetType"]}\n'
        f'Action type: {row["actionType"]}\n'
        f'Mechanism: {row["mechanismOfAction"]}\n'
        f'Quotes:\n{quotes}'
    )
    text = f'TRIAL TEXTS\n{pool_text}\n\nMECHANISM\n{mechanism}\n'
    return text if chembl_rows is None else f'{text}\nCHEMBL MECHANISMS\n{chembl}\n'


def adjust(scored: pl.DataFrame, verdicts: pl.DataFrame) -> pl.DataFrame:
    """Add ``judge_correct`` and ``adjusted_correct`` from the judge's verdicts.

    A mechanism is judge-correct when the judge says the text supports this drug acting on
    this target and it is the drug's primary mechanism; the action type is reported but not
    required. The adjusted verdict keeps every benchmark hit and lets the judge rule on
    everything else it saw.
    """
    judged = verdicts.select(
        'pool_id', 'mechanism_idx', 'targetSupported', 'actionSupported', 'primaryMechanism', 'rationale'
    )
    return (
        scored
        .join(judged, on=['pool_id', 'mechanism_idx'], how='left')
        .with_columns(
            judge_correct=pl.when(pl.col('targetSupported').is_not_null()).then(
                (pl.col('targetSupported') == 'yes') & (pl.col('primaryMechanism') == 'yes')
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


def handcheck(
    adjusted: pl.DataFrame,
    pools: pl.DataFrame,
    links: pl.DataFrame | None = None,
    n: int = 50,
    seed: int = 0,
    per_stratum: int = 15,
) -> pl.DataFrame:
    """Return a seeded sample of judged mechanisms for reading by hand.

    With ``links``, the sample is of new mechanisms only (drugs the benchmark cannot score),
    ``per_stratum`` per maximum clinical stage: the judge's verdict on them is the only
    evidence for publication, so these are the ones a human should check.
    """
    judged = adjusted.filter(pl.col('judge_correct').is_not_null()).sort('pool_id', 'mechanism_idx')
    if links is None:
        sample = judged.sample(n=min(n, judged.height), seed=seed, shuffle=True).with_columns(
            maximumClinicalStage=pl.lit(None, dtype=pl.String)
        )
    else:
        new = judged.filter(~pl.col('benchmarked')).join(
            links.select('pool_id', 'maximumClinicalStage'), on='pool_id', how='left'
        )
        stages = sorted(new['maximumClinicalStage'].drop_nulls().unique().to_list())
        sample = pl.concat(
            [
                part.sample(n=min(per_stratum, part.height), seed=seed, shuffle=True)
                for stage in stages
                if (part := new.filter(pl.col('maximumClinicalStage') == stage)).height
            ]
            or [new.head(0)]
        )
    return sample.join(pools.select('pool_id', 'drug_name'), on='pool_id', how='left').select(
        'drug_name',
        'maximumClinicalStage',
        'pool_id',
        'mechanism_idx',
        'targetFromSource',
        pl.col('evidence').list.eval(pl.element().struct.field('quote')).list.join(' | ').alias('quotes'),
        'benchmarked',
        'correct',
        'targetSupported',
        'actionSupported',
        'primaryMechanism',
        'rationale',
    )
