"""Link pools to ChEMBL drugs and score their mechanisms against the Platform MoA table.

A pool is linked to the ChEMBL id that the release's ``clinical_report`` gives most of its
single-drug trials, when that id covers at least :py:data:`LINK_SHARE` of them. A predicted
mechanism is correct when its genes overlap the genes of any MoA row of that drug.
"""

from __future__ import annotations

import math

import polars as pl

LINK_SHARE = 0.8

STAGE_RANK = {
    'UNKNOWN': 0,
    'PRECLINICAL': 1,
    'IND': 2,
    'EARLY_PHASE_1': 3,
    'PHASE_1': 4,
    'PHASE_1_2': 5,
    'PHASE_2': 6,
    'PHASE_2_3': 7,
    'PHASE_3': 8,
    'PREAPPROVAL': 9,
    'APPROVAL': 10,
}
EARLY_PHASES = frozenset({'EARLY_PHASE_1', 'PHASE_1', 'PHASE_1_2', 'PHASE_2'})
LATE_PHASES = frozenset({'PHASE_2_3', 'PHASE_3', 'PREAPPROVAL', 'APPROVAL'})

BENCH_ROWS = pl.List(pl.Struct({'targets': pl.List(pl.String), 'actionType': pl.String}))


def _phase_group(stage: pl.Expr) -> pl.Expr:
    return (
        pl
        .when(stage.is_in(sorted(EARLY_PHASES)))
        .then(pl.lit('early'))
        .when(stage.is_in(sorted(LATE_PHASES)))
        .then(pl.lit('late'))
        .otherwise(pl.lit('other'))
    )


def link_pools(pools: pl.DataFrame, clinical_report: pl.DataFrame, molecule: pl.DataFrame) -> pl.DataFrame:
    """Link each pool to a ChEMBL drug, its maximum clinical stage and its drug type.

    Returns:
        pl.DataFrame: ``pool_id``, ``drugId``, ``link`` (``linked``, ``ambiguous`` or
        ``unlinked``), ``maximumClinicalStage``, ``drugType``, ``phase_group``.
    """
    trials = clinical_report.filter(pl.col('source') == 'ClinicalTrials.gov').select(
        pl.col('id').alias('nct_id'),
        pl
        .when(pl.col('drugs').list.len() == 1)
        .then(pl.col('drugs').list.first().struct.field('drugId'))
        .alias('trial_drug'),
        pl.col('clinicalStage').replace({'PHASE_4': 'APPROVAL'}).replace_strict(STAGE_RANK, default=0).alias('rank'),
    )
    per_pool = pools.select('pool_id', 'nct_ids').explode('nct_ids').rename({'nct_ids': 'nct_id'})
    joined = per_pool.join(trials, on='nct_id', how='left')

    stages = joined.group_by('pool_id').agg(trial_rank=pl.col('rank').max().fill_null(0))
    votes = (
        joined
        .drop_nulls('trial_drug')
        .group_by('pool_id', 'trial_drug')
        .agg(n=pl.len())
        .with_columns(total=pl.col('n').sum().over('pool_id'))
        .sort(['pool_id', 'n', 'trial_drug'], descending=[False, True, False])
        .group_by('pool_id', maintain_order=True)
        .first()
        .with_columns(share=pl.col('n') / pl.col('total'))
    )
    stage_names = {rank: stage for stage, rank in STAGE_RANK.items()}
    links = (
        pools
        .select('pool_id')
        .join(votes.select('pool_id', 'trial_drug', 'share'), on='pool_id', how='left')
        .join(stages, on='pool_id', how='left')
        .with_columns(
            link=pl
            .when(pl.col('share').is_null())
            .then(pl.lit('unlinked'))
            .when(pl.col('share') >= LINK_SHARE)
            .then(pl.lit('linked'))
            .otherwise(pl.lit('ambiguous')),
        )
        .with_columns(drugId=pl.when(pl.col('link') == 'linked').then(pl.col('trial_drug')))
        .join(
            molecule.select(pl.col('id').alias('drugId'), 'maximumClinicalStage', 'drugType'), on='drugId', how='left'
        )
        .with_columns(
            maximumClinicalStage=pl.coalesce(
                'maximumClinicalStage',
                pl.col('trial_rank').replace_strict(stage_names, default='UNKNOWN', return_dtype=pl.String),
            )
        )
        .with_columns(phase_group=_phase_group(pl.col('maximumClinicalStage')))
    )
    return links.select('pool_id', 'drugId', 'link', 'maximumClinicalStage', 'drugType', 'phase_group')


def benchmark_targets(moa: pl.DataFrame) -> pl.DataFrame:
    """Return, per ChEMBL drug, its MoA genes and its MoA rows.

    Rows with no target gene (non-human and unmapped targets) are left out.

    Returns:
        pl.DataFrame: ``drugId``, ``targetIds``, ``rows`` (``targets``, ``actionType``).
    """
    return (
        moa
        .filter(pl.col('targets').list.len() > 0)
        .explode('chemblIds')
        .rename({'chemblIds': 'drugId'})
        .group_by('drugId', maintain_order=True)
        .agg(
            targetIds=pl.col('targets').list.explode().unique(maintain_order=True),
            rows=pl.struct('targets', 'actionType'),
        )
        .with_columns(pl.col('rows').cast(BENCH_ROWS))
    )


def _overlaps(predicted: list[str], truth: list[str]) -> bool:
    return bool(set(predicted) & set(truth))


def score(mechanisms: pl.DataFrame, links: pl.DataFrame, bench: pl.DataFrame) -> pl.DataFrame:
    """Score each grounded mechanism against its drug's MoA rows.

    Adds ``benchmarked``, ``duplicate``, ``in_headline``, ``correct`` and ``action_match``.
    A prediction of the same genes for the same drug (from a second name of the drug, or a
    second mention in one pool) is a ``duplicate`` and only its first occurrence is in the
    headline. ``correct`` is null outside the headline; ``action_match`` is null unless the
    mechanism is correct.
    """
    joined = (
        mechanisms
        .join(links.select('pool_id', 'drugId', 'link'), on='pool_id', how='left')
        .join(bench.rename({'targetIds': 'benchTargetIds', 'rows': 'benchRows'}), on='drugId', how='left')
        .with_columns(benchmarked=(pl.col('link') == 'linked') & pl.col('benchTargetIds').is_not_null())
        .with_columns(
            in_headline=pl.col('benchmarked')
            & pl.col('grounding').is_in(['source', 'suggestion'])
            & ~pl.col('class_level')
        )
        .sort('pool_id', 'mechanism_idx')
        .with_columns(
            dup_key=pl.when(pl.col('in_headline')).then(
                pl.concat_str(pl.col('drugId'), pl.col('targetIds').list.sort().list.join(','), separator='|')
            )
        )
        .with_columns(duplicate=pl.col('dup_key').is_not_null() & (pl.col('dup_key').cum_count().over('dup_key') > 1))
        .with_columns(in_headline=pl.col('in_headline') & ~pl.col('duplicate'))
        .drop('dup_key')
    )
    correct, action_match = [], []
    for row in joined.select('in_headline', 'targetIds', 'benchTargetIds', 'benchRows', 'actionType').iter_rows(
        named=True
    ):
        if not row['in_headline']:
            correct.append(None)
            action_match.append(None)
            continue
        hit = _overlaps(row['targetIds'], row['benchTargetIds'])
        correct.append(hit)
        action_match.append(
            any(
                _overlaps(row['targetIds'], r['targets']) and r['actionType'] == row['actionType']
                for r in row['benchRows']
            )
            if hit
            else None
        )
    return joined.drop('benchTargetIds', 'benchRows', 'drugId', 'link').with_columns(
        correct=pl.Series(correct, dtype=pl.Boolean),
        action_match=pl.Series(action_match, dtype=pl.Boolean),
    )


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score interval of ``k`` successes in ``n`` trials."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _ratio(numerator: str, denominator: str) -> pl.Expr:
    return pl.when(pl.col(denominator) > 0).then(pl.col(numerator) / pl.col(denominator))


def summarise(
    scored: pl.DataFrame,
    links: pl.DataFrame,
    statuses: pl.DataFrame,
    bench: pl.DataFrame,
    by: list[str],
    adjusted: bool = False,
) -> pl.DataFrame:
    """Return one row of metrics per stratum of ``by`` (columns of ``links``).

    With ``adjusted``, ``adjusted_correct`` (from the judge) replaces ``correct``, and the
    judged mechanisms the benchmark cannot score are counted too.
    """
    verdict = 'adjusted_correct' if adjusted else 'correct'
    counted = (pl.col('in_headline') | pl.col('judge_correct').is_not_null()) if adjusted else pl.col('in_headline')
    scored = scored.with_columns(counted=counted)
    strata = links.select('pool_id', 'drugId', 'link', *by)
    benchmarked_drugs = bench['drugId'].to_list()

    mechanism_rows = scored.join(strata.select('pool_id', *by), on='pool_id', how='left')
    mechanism_metrics = mechanism_rows.group_by(by).agg(
        mechanisms=pl.len(),
        dropped_by_quote_check=(~pl.col('kept')).sum(),
        non_human=(pl.col('kept') & ~pl.col('isHumanTarget')).sum(),
        unresolved=(pl.col('grounding') == 'unresolved').sum(),
        class_level=pl.col('class_level').sum(),
        scored=pl.col('counted').sum(),
        correct=pl.col(verdict).filter(pl.col('counted')).sum(),
        action_matches=pl.col('action_match').filter(pl.col(verdict) & pl.col('counted')).sum(),
    )

    per_pool = (
        scored
        .filter(pl.col('counted'))
        .group_by('pool_id')
        .agg(all_correct=pl.col(verdict).all(), any_correct=pl.col(verdict).any())
    )
    pools = (
        statuses
        .join(strata, on='pool_id', how='left')
        .join(per_pool, on='pool_id', how='left')
        .with_columns(
            benchmarked=(pl.col('link') == 'linked') & pl.col('drugId').is_in(benchmarked_drugs),
            has_headline=pl.col('all_correct').is_not_null(),
        )
    )
    pool_metrics = pools.group_by(by).agg(
        pools=pl.len(),
        extracted=pl.col('extracted').sum(),
        pools_with_kept_mechanism=(pl.col('not_stated') == False).sum(),  # noqa: E712
        not_stated=pl.col('not_stated').sum(),
        benchmarked_pools=pl.col('benchmarked').sum(),
    )
    drugs = (
        pools
        .filter(pl.col('benchmarked'))
        .group_by([*by, 'drugId'])
        .agg(
            has_headline=pl.col('has_headline').any(),
            all_correct=pl.col('all_correct').drop_nulls().all(),
            any_correct=pl.col('any_correct').fill_null(False).any(),
        )
        .group_by(by)
        .agg(
            benchmarked_drugs=pl.len(),
            drugs_scored=pl.col('has_headline').sum(),
            drugs_all_correct=(pl.col('has_headline') & pl.col('all_correct')).sum(),
            drugs_any_correct=pl.col('any_correct').sum(),
        )
    )
    pool_metrics = pool_metrics.join(drugs, on=by, how='left', nulls_equal=True).with_columns(
        pl.col(c).fill_null(0) for c in drugs.columns if c not in by
    )

    summary = (
        pool_metrics
        .join(mechanism_metrics, on=by, how='left', nulls_equal=True)
        .with_columns(pl.col(c).fill_null(0) for c in mechanism_metrics.columns if c not in by)
        .with_columns(
            not_stated_rate=_ratio('not_stated', 'extracted'),
            precision=_ratio('correct', 'scored'),
            drug_precision=_ratio('drugs_all_correct', 'drugs_scored'),
            recall=_ratio('drugs_any_correct', 'benchmarked_drugs'),
            action_agreement=_ratio('action_matches', 'correct'),
        )
    )
    intervals = [wilson(k, n) for k, n in summary.select('correct', 'scored').iter_rows()]
    return summary.with_columns(
        precision_lo=pl.Series([lo for lo, _ in intervals], dtype=pl.Float64),
        precision_hi=pl.Series([hi for _, hi in intervals], dtype=pl.Float64),
    ).sort(by)


def precision_by_grounding(scored: pl.DataFrame) -> pl.DataFrame:
    """Return headline precision separately for targets named by the text and by the model's suggestion."""
    table = (
        scored
        .filter(pl.col('in_headline'))
        .group_by('grounding')
        .agg(scored=pl.len(), correct=pl.col('correct').sum())
        .with_columns(precision=pl.col('correct') / pl.col('scored'))
    )
    intervals = [wilson(k, n) for k, n in table.select('correct', 'scored').iter_rows()]
    return table.with_columns(
        precision_lo=pl.Series([lo for lo, _ in intervals], dtype=pl.Float64),
        precision_hi=pl.Series([hi for _, hi in intervals], dtype=pl.Float64),
    ).sort('grounding')
