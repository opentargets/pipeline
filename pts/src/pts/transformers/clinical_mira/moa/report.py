"""The markdown report of a run."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import polars as pl

from pts.transformers.clinical_mira.moa.grounding import GROUNDED

if TYPE_CHECKING:
    from pathlib import Path

GO_PRECISION = 0.90
"""David's go criterion: strict target precision on early-phase drugs."""

CONCLUSIVE_LOWER_BOUND = 0.85
"""Below this lower bound of the interval, a precision at or above the bar is not conclusive."""

SUMMARY_COLUMNS = [
    'pools',
    'extracted',
    'not_stated_rate',
    'mechanisms',
    'dropped_by_quote_check',
    'non_human',
    'unresolved',
    'class_level',
    'scored',
    'correct',
    'precision',
    'precision_lo',
    'precision_hi',
    'benchmarked_drugs',
    'drugs_scored',
    'drug_precision',
    'recall',
    'action_agreement',
]


def go_line(early: dict[str, Any] | None) -> str:
    """Return the go/no-go verdict on the early-phase strict precision."""
    if not early or not early.get('scored'):
        return '**Verdict: no early-phase predictions scored.**'
    precision, low, high = early['precision'], early['precision_lo'], early['precision_hi']
    if precision >= GO_PRECISION:
        verdict = 'GO' if low >= CONCLUSIVE_LOWER_BOUND else 'INCONCLUSIVE'
    else:
        verdict = 'INCONCLUSIVE' if high >= GO_PRECISION and low < CONCLUSIVE_LOWER_BOUND else 'NO GO'
    return (
        f'**Verdict: {verdict}.** Early-phase strict target precision {precision:.3f} '
        f'({early["correct"]}/{early["scored"]}; Wilson 95% {low:.3f} to {high:.3f}) '
        f'against a bar of {GO_PRECISION:.2f}.'
    )


def _cell(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.3f}'
    if isinstance(value, list):
        return ' | '.join(str(v) for v in value)
    return str(value).replace('|', '/').replace('\n', ' ')


def markdown_table(frame: pl.DataFrame, columns: list[str] | None = None) -> str:
    """Render a frame as a markdown table."""
    columns = [c for c in (columns or frame.columns) if c in frame.columns]
    if frame.is_empty():
        return '_none_\n'
    lines = ['| ' + ' | '.join(columns) + ' |', '|' + '---|' * len(columns)]
    lines += ['| ' + ' | '.join(_cell(v) for v in row) + ' |' for row in frame.select(columns).iter_rows()]
    return '\n'.join(lines) + '\n'


def _read(run_dir: Path, name: str) -> pl.DataFrame | None:
    path = run_dir / name
    return pl.read_parquet(path) if path.exists() else None


def _examples(adjusted: pl.DataFrame, pools: pl.DataFrame, condition: pl.Expr, n: int = 10) -> pl.DataFrame:
    return (
        adjusted
        .filter(condition)
        .sort('pool_id', 'mechanism_idx')
        .head(n)
        .join(pools.select('pool_id', 'drug_name'), on='pool_id', how='left')
        .with_columns(quotes=pl.col('evidence').list.eval(pl.element().struct.field('quote')).list.join(' … '))
    )


def render_report(run_dir: Path) -> str:
    """Return the run's report, from the files ``evaluate`` wrote."""
    counts: dict[str, Any] = {}
    for name in ('text_counts.json', 'pool_counts.json'):
        path = run_dir / name
        if path.exists():
            counts.update(json.loads(path.read_text()))

    strict = _read(run_dir, 'summary_phase_group.parquet')
    early = strict.filter(pl.col('phase_group') == 'early').to_dicts() if strict is not None else []
    adjusted = pl.read_parquet(run_dir / 'scored.parquet')
    pools = pl.read_parquet(run_dir / 'pools.parquet')

    funnel = {
        **counts,
        'pools_extracted': int(pl.read_parquet(run_dir / 'statuses.parquet')['extracted'].sum()),
        'mechanisms': adjusted.height,
        'kept_after_quote_check': int(adjusted['kept'].sum()),
        'grounded': int(adjusted['grounding'].is_in(GROUNDED).sum()),
        'headline': int(adjusted['in_headline'].sum()),
    }

    sections = [
        '# AACT MoA extraction\n',
        go_line(early[0] if early else None) + '\n',
        '## Funnel\n',
        markdown_table(pl.DataFrame({'step': list(funnel), 'count': [str(v) for v in funnel.values()]})),
    ]
    for by in ('phase_group', 'maximumClinicalStage', 'drugType'):
        for kind, suffix in (('strict', ''), ('judge-adjusted', '_adjusted')):
            frame = _read(run_dir, f'summary_{by}{suffix}.parquet')
            if frame is not None:
                sections += [f'## By {by}, {kind}\n', markdown_table(frame, [by, *SUMMARY_COLUMNS])]

    by_grounding = _read(run_dir, 'precision_by_grounding.parquet')
    if by_grounding is not None:
        sections += ['## Headline precision by how the target was grounded\n', markdown_table(by_grounding)]

    for label in ('relation', 'statementType', 'effect_wording', 'action_other'):
        table = _read(run_dir, f'precision_by_label_{label}.parquet')
        if table is not None:
            sections += [f'## Precision of every scorable mechanism by {label}\n', markdown_table(table)]

    calibration = _read(run_dir, 'calibration.parquet')
    if calibration is not None:
        sections += ['## Judge calibration against the benchmark\n', markdown_table(calibration)]

    example_columns = [
        'drug_name',
        'targetFromSource',
        'actionType',
        'quotes',
        'targetSupported',
        'actionSupported',
        'rationale',
    ]
    sections += [
        '## Example hits\n',
        markdown_table(_examples(adjusted, pools, pl.col('correct') == True), example_columns),  # noqa: E712
        '## Example misses\n',
        markdown_table(_examples(adjusted, pools, pl.col('correct') == False), example_columns),  # noqa: E712
    ]
    if 'judge_correct' in adjusted.columns:
        candidates = _examples(
            adjusted,
            pools,
            (pl.col('correct') == False) | ~pl.col('benchmarked'),  # noqa: E712
            n=10_000,
        ).filter(pl.col('judge_correct') == True)  # noqa: E712
        sections += [
            '## Not in ChEMBL, supported by the judge\n',
            markdown_table(candidates, ['drug_name', 'benchmarked', *example_columns[1:]]),
        ]
    return '\n'.join(sections)
