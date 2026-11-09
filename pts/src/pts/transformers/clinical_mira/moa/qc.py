"""Release QC: the inferred mechanisms against ChEMBL's, and the evidence they add."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from pts.transformers.clinical_mira.moa.benchmark import phase_group, wilson
from pts.transformers.clinical_mira.moa.report import GO_PRECISION, markdown_table

if TYPE_CHECKING:
    from pathlib import Path

PAIR = ['targetId', 'diseaseId']


def read_dataset(root: Path, name: str, columns: list[str] | None = None) -> pl.DataFrame:
    """Read a dataset directory of a release root."""
    return pl.read_parquet(str(root / name / '*.parquet'), columns=columns)


def release_filter(candidates: pl.DataFrame, min_confidence: float, exclude_stages: list[str]) -> pl.DataFrame:
    """Keep the candidates the release merge would publish."""
    return candidates.filter(
        (pl.col('confidence') >= min_confidence)
        & ~pl.col('maximumClinicalStage').is_in(exclude_stages).fill_null(False)
    )


def score_candidates(candidates: pl.DataFrame, moa: pl.DataFrame) -> pl.DataFrame:
    """Score the released candidates against the manual mechanisms of their drug.

    A candidate's drug is its parent and every molecule in its ``chemblIds``; its manual targets
    are those of every manual mechanism listing any of them.

    Returns:
        pl.DataFrame: the candidates whose drug has a manual
        mechanism, with ``manualTargets``, ``phase_group`` and boolean ``correct``.
    """
    truth = (
        moa
        .filter(pl.col('curation') == 'manual')
        .select('chemblIds', 'targets')
        .explode('chemblIds')
        .group_by('chemblIds')
        .agg(_targets=pl.col('targets').explode().drop_nulls().unique())
        .rename({'chemblIds': '_drug_id'})
    )
    indexed = candidates.with_row_index('_row')
    manual = (
        indexed
        .select(
            '_row',
            _drug_id=pl.concat_list(pl.col('parentChemblId'), pl.col('chemblIds').fill_null([])),
        )
        .explode('_drug_id')
        .join(truth, on='_drug_id', how='inner')
        .group_by('_row')
        .agg(manualTargets=pl.col('_targets').explode().drop_nulls().unique().sort())
    )
    return (
        indexed
        .join(manual, on='_row', how='inner', maintain_order='left')
        .drop('_row')
        .with_columns(
            phase_group=phase_group(pl.col('maximumClinicalStage')),
            correct=pl.col('targets').list.set_intersection(pl.col('manualTargets')).list.len() > 0,
        )
    )


def precision_table(scored: pl.DataFrame, by: str) -> pl.DataFrame:
    """Return strict precision with its Wilson interval for each value of ``by``."""
    table = (
        scored
        .group_by(by)
        .agg(scored=pl.len(), correct=pl.col('correct').sum())
        .with_columns(precision=pl.col('correct') / pl.col('scored'))
        .sort(by)
    )
    intervals = [wilson(k, n) for k, n in table.select('correct', 'scored').iter_rows()]
    return table.with_columns(
        precision_lo=pl.Series([lo for lo, _ in intervals], dtype=pl.Float64),
        precision_hi=pl.Series([hi for _, hi in intervals], dtype=pl.Float64),
    )


def verdict(scored: pl.DataFrame, min_scorable: int) -> str:
    """Return the go line from the early-phase strict precision.

    ``GO`` or ``NO-GO`` when at least ``min_scorable`` early-phase candidates are scorable,
    ``INCONCLUSIVE`` when fewer are, ``UNSCORED`` when none is.
    """
    early = scored.filter(pl.col('phase_group') == 'early')
    if early.is_empty():
        return 'UNSCORED: no early-phase candidate has a ChEMBL mechanism to be scored against.'
    correct, n = int(early['correct'].sum()), early.height
    low, high = wilson(correct, n)
    if n < min_scorable:
        label = 'INCONCLUSIVE'
    else:
        label = 'GO' if correct / n >= GO_PRECISION else 'NO-GO'
    return (
        f'{label}: early-phase strict target precision {correct / n:.3f} ({correct}/{n}; '
        f'Wilson 95% {low:.3f} to {high:.3f}) against a bar of {GO_PRECISION:.2f} '
        f'with at least {min_scorable} scorable.'
    )


def new_evidence_pairs(release: pl.DataFrame, without: pl.DataFrame) -> pl.DataFrame:
    """Return the (target, disease) pairs of ``release`` absent from ``without``."""
    return release.select(PAIR).unique().join(without.select(PAIR).unique(), on=PAIR, how='anti').sort(PAIR)


def run_qc(
    release: Path,
    without: Path,
    out: Path,
    min_confidence: float,
    exclude_stages: list[str],
    min_scorable: int,
) -> str:
    """Write ``qc.md`` and ``new_evidence_pairs.csv`` to ``out``; return the go line."""
    out.mkdir(parents=True, exist_ok=True)
    moa = read_dataset(release, 'output/drug_mechanism_of_action', ['chemblIds', 'targets', 'curation'])
    candidates = read_dataset(release, 'intermediate/drug_mechanism_of_action_aact')
    released = release_filter(candidates, min_confidence, exclude_stages)
    scored = score_candidates(released, moa)
    go = verdict(scored, min_scorable)

    precedence = {
        'release': read_dataset(release, 'output/evidence_clinical_precedence', PAIR),
        'without': read_dataset(without, 'output/evidence_clinical_precedence', PAIR),
    }
    associations = {
        'release': read_dataset(release, 'output/association_overall_direct', PAIR),
        'without': read_dataset(without, 'output/association_overall_direct', PAIR),
    }
    new_pairs = new_evidence_pairs(precedence['release'], precedence['without'])
    new_pairs.write_csv(out / 'new_evidence_pairs.csv')
    counts = pl.DataFrame({
        'dataset': ['evidence_clinical_precedence', 'association_overall_direct'],
        'release': [precedence['release'].height, associations['release'].height],
        'without': [precedence['without'].height, associations['without'].height],
    }).with_columns(difference=pl.col('release') - pl.col('without'))

    sections = [
        go + '\n',
        f'electronic rows published: {moa.filter(pl.col("curation") == "electronic").height}\n',
        (
            f'candidates passing the release filter: {released.height}; '
            f'scorable against a manual mechanism: {scored.height}\n'
        ),
    ]
    for by in ('phase_group', 'drugType'):
        sections.append(f'## Strict precision by {by}\n')
        sections.append(markdown_table(precision_table(scored, by)) if scored.height else 'no scorable rows\n')
    sections += [
        '## Evidence with and without the source\n',
        markdown_table(counts),
        f'new evidence pairs: {new_pairs.height} (see new_evidence_pairs.csv)\n',
    ]
    (out / 'qc.md').write_text('\n'.join(sections), encoding='utf-8')
    return go


def _pair_sets(snapshot: pl.DataFrame) -> pl.DataFrame:
    """Return the sorted set of ``"<symbols>|<actionType>"`` pairs of each drug."""
    return (
        snapshot
        .select(
            'drugName',
            pair=pl.concat_str(
                pl.col('targetSymbolSuggestion').fill_null([]).list.sort().list.join(','),
                pl.col('actionType').fill_null(''),
                separator='|',
            ),
        )
        .group_by('drugName')
        .agg(pairs=pl.col('pair').unique().sort())
    )


def diff_snapshots(old: pl.DataFrame, new: pl.DataFrame) -> pl.DataFrame:
    """Return one row per drug whose (targets, action type) pairs differ between two snapshots.

    Returns:
        pl.DataFrame: ``drugName``, and its ``gained`` and ``lost`` pair lists.
    """
    return (
        _pair_sets(old)
        .rename({'pairs': 'old'})
        .join(_pair_sets(new).rename({'pairs': 'new'}), on='drugName', how='full', coalesce=True)
        .with_columns(pl.col('old', 'new').fill_null([]))
        .select(
            'drugName',
            gained=pl.col('new').list.set_difference(pl.col('old')).list.sort(),
            lost=pl.col('old').list.set_difference(pl.col('new')).list.sort(),
        )
        .filter((pl.col('gained').list.len() > 0) | (pl.col('lost').list.len() > 0))
        .sort('drugName')
    )
