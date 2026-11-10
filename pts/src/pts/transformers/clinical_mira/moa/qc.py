"""Release QC: the inferred mechanisms against ChEMBL's, and the evidence they add."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import polars as pl

from pts.transformers.clinical_mira.moa.benchmark import phase_group, wilson
from pts.transformers.clinical_mira.moa.report import GO_PRECISION, markdown_table
from pts.transformers.drug_mechanism_of_action_candidates import SOURCE_AACT, SOURCE_LITERATURE, stage_excluded

if TYPE_CHECKING:
    from pathlib import Path

PAIR = ['targetId', 'diseaseId']
SOURCES = (SOURCE_AACT, SOURCE_LITERATURE)


def read_dataset(root: Path, name: str, columns: list[str] | None = None) -> pl.DataFrame:
    """Read a dataset directory of a release root."""
    return pl.read_parquet(str(root / name / '*.parquet'), columns=columns)


def release_filter(
    candidates: pl.DataFrame, min_confidence: float, exclude_stages: dict[str, list[str]]
) -> pl.DataFrame:
    """Keep the candidates the release merge would publish.

    Args:
        candidates: The candidates of every source.
        min_confidence: The lowest confidence published.
        exclude_stages: For each source, the maximum clinical stages not published.

    Returns:
        pl.DataFrame: The candidates at or above ``min_confidence`` whose stage their source does not exclude.

    Raises:
        ValueError: If a candidate source has no ``exclude_stages`` entry.
    """
    excluded = stage_excluded(candidates, exclude_stages)
    return candidates.filter((pl.col('confidence') >= min_confidence) & ~excluded)


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


class Verdict(NamedTuple):
    """A go label, ``GO``, ``NO-GO``, ``INCONCLUSIVE`` or ``UNSCORED``, and the line reporting it."""

    label: str
    line: str


def _verdict(scored: pl.DataFrame, min_scorable: int) -> Verdict:
    early = scored.filter(pl.col('phase_group') == 'early')
    if early.is_empty():
        return Verdict('UNSCORED', 'UNSCORED: no early-phase candidate has a ChEMBL mechanism to be scored against.')
    correct, n = int(early['correct'].sum()), early.height
    low, high = wilson(correct, n)
    if n < min_scorable:
        label = 'INCONCLUSIVE'
    else:
        label = 'GO' if correct / n >= GO_PRECISION else 'NO-GO'
    return Verdict(
        label,
        f'{label}: early-phase strict target precision {correct / n:.3f} ({correct}/{n}; '
        f'Wilson 95% {low:.3f} to {high:.3f}) against a bar of {GO_PRECISION:.2f} '
        f'with at least {min_scorable} scorable.',
    )


def verdict(scored: pl.DataFrame, min_scorable: int) -> str:
    """Return the go line from the early-phase strict precision of ``scored`` as a whole.

    ``GO`` or ``NO-GO`` when at least ``min_scorable`` early-phase candidates are scorable,
    ``INCONCLUSIVE`` when fewer are, ``UNSCORED`` when none is.
    """
    return _verdict(scored, min_scorable).line


def source_verdicts(scored: pl.DataFrame, min_scorable: int) -> dict[str, Verdict]:
    """Return the verdict of each source on its own scored candidates, by the rule of :py:func:`verdict`.

    The sources are ``aact`` and ``literature``, then any other source of ``scored`` in name order.
    Each line starts with its source.
    """
    others = sorted(set(scored['source'].to_list()) - set(SOURCES))
    verdicts = {}
    for source in [*SOURCES, *others]:
        label, line = _verdict(scored.filter(pl.col('source') == source), min_scorable)
        verdicts[source] = Verdict(label, f'{source} {line}')
    return verdicts


def overall_verdict(verdicts: dict[str, Verdict]) -> str:
    """Return the overall go line of the per-source verdicts.

    ``NO-GO`` when any source is ``NO-GO``, else ``GO`` when any source is ``GO``; a source
    ``INCONCLUSIVE`` or ``UNSCORED`` does not block. Without a ``GO`` or ``NO-GO`` source it is
    ``INCONCLUSIVE`` when any source is, else ``UNSCORED``.
    """
    labels = {v.label for v in verdicts.values()}
    label = next(candidate for candidate in ('NO-GO', 'GO', 'INCONCLUSIVE', 'UNSCORED') if candidate in labels)
    sources = ', '.join(f'{source} {v.label}' for source, v in verdicts.items())
    return f'{label}: overall, GO only if every source with enough scorable early-phase candidates is GO ({sources}).'


def new_evidence_pairs(release: pl.DataFrame, without: pl.DataFrame) -> pl.DataFrame:
    """Return the (target, disease) pairs of ``release`` absent from ``without``."""
    return release.select(PAIR).unique().join(without.select(PAIR).unique(), on=PAIR, how='anti').sort(PAIR)


def run_qc(
    release: Path,
    without: Path,
    out: Path,
    min_confidence: float,
    exclude_stages: dict[str, list[str]],
    min_scorable: int,
) -> str:
    """Write ``qc.md`` and ``new_evidence_pairs.csv`` to ``out``; return the overall go line then one per source."""
    out.mkdir(parents=True, exist_ok=True)
    moa = read_dataset(release, 'output/drug_mechanism_of_action', ['chemblIds', 'targets', 'curation'])
    candidates = read_dataset(release, 'intermediate/drug_mechanism_of_action_candidates')
    released = release_filter(candidates, min_confidence, exclude_stages)
    scored = score_candidates(released, moa)
    per_source = source_verdicts(scored, min_scorable)
    go = '\n'.join([overall_verdict(per_source), *(v.line for v in per_source.values())])

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
    for by in ('phase_group', 'drugType', 'source'):
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
