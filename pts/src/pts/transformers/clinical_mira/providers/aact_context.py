"""A ClinicalTrials.gov trial as the extraction sees it: registry text plus arms, interventions and outcomes.

`trial_context` joins the AACT tables into one row per trial; `render_trial` turns a row into the prompt text,
deterministically and within a character budget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from pts.postgres import read_dump_tables
from pts.transformers.clinical_mira.providers.aact import AACT_ARCHIVE_MEMBER, AACT_SCHEMA_NAME

if TYPE_CHECKING:
    from pathlib import Path

CONTEXT_TABLES = {
    'studies': ['nct_id', 'official_title', 'phase'],
    'brief_summaries': ['nct_id', 'description'],
    'detailed_descriptions': ['nct_id', 'description'],
    'interventions': ['id', 'nct_id', 'intervention_type', 'name', 'description'],
    'design_groups': ['id', 'nct_id', 'group_type', 'title', 'description'],
    'design_group_interventions': ['id', 'nct_id', 'design_group_id', 'intervention_id'],
    'design_outcomes': ['nct_id', 'outcome_type', 'measure'],
    'designs': ['nct_id', 'primary_purpose', 'allocation', 'intervention_model', 'masking'],
}
"""AACT tables and columns of the trial context; separate from the release catalogue `AACT_TABLES`."""

MAX_CHARS = 12000
"""Character budget of a rendered trial."""

ELLIPSIS = ' […]'
MIN_KEPT = 40
"""A truncated text keeps at least this many characters, or is emptied."""


def read_context_tables(archive: str, scratch_root: str | Path) -> dict[str, pl.DataFrame]:
    """Restore the `CONTEXT_TABLES` from an AACT dump, keyed as `trial_context` takes them."""
    tables = read_dump_tables(
        archive,
        CONTEXT_TABLES,
        schema_name=AACT_SCHEMA_NAME,
        archive_member=AACT_ARCHIVE_MEMBER,
        scratch_root=scratch_root,
    )
    renamed = {'brief_summaries': 'summaries', 'detailed_descriptions': 'details'}
    return {renamed.get(name, name): frame for name, frame in tables.items()}


def _first_text(frame: pl.DataFrame, alias: str) -> pl.DataFrame:
    return frame.group_by('nct_id').agg(pl.col('description').sort().first().alias(alias))


def trial_context(
    studies: pl.DataFrame,
    summaries: pl.DataFrame,
    details: pl.DataFrame,
    interventions: pl.DataFrame,
    design_groups: pl.DataFrame,
    design_group_interventions: pl.DataFrame,
    design_outcomes: pl.DataFrame,
    designs: pl.DataFrame,
) -> pl.DataFrame:
    """One row per trial of `studies`, with its texts, interventions, arms and primary outcomes."""
    interventions = interventions.filter(pl.col('name').is_not_null())
    intervention_names = interventions.select(intervention_id='id', name='name')
    arm_interventions = (
        design_group_interventions
        .join(intervention_names, on='intervention_id')
        .group_by('design_group_id')
        .agg(pl.col('name').unique().sort().alias('interventions'))
    )
    arms = (
        design_groups
        .join(arm_interventions, left_on='id', right_on='design_group_id', how='left')
        .with_columns(pl.col('interventions').fill_null(pl.lit([], dtype=pl.List(pl.String))))
        .sort('nct_id', 'title', 'group_type')
        .group_by('nct_id', maintain_order=True)
        .agg(arms=pl.struct(type='group_type', title='title', description='description', interventions='interventions'))
    )
    trial_interventions = (
        interventions
        .sort('nct_id', 'name', 'intervention_type', 'description', nulls_last=True)
        .group_by('nct_id', maintain_order=True)
        .agg(interventions=pl.struct(name='name', type='intervention_type', description='description'))
    )
    outcomes = (
        design_outcomes
        .filter(pl.col('outcome_type') == 'PRIMARY')
        .group_by('nct_id')
        .agg(pl.col('measure').drop_nulls().unique().sort().alias('primary_outcomes'))
    )
    design_columns = ['nct_id', 'primary_purpose', 'allocation', 'intervention_model', 'masking']
    design = designs.select(design_columns).unique('nct_id', keep='first')
    return (
        studies
        .select('nct_id', 'official_title', 'phase')
        .join(_first_text(summaries, 'brief_summary'), on='nct_id', how='left')
        .join(_first_text(details, 'detailed_description'), on='nct_id', how='left')
        .join(design, on='nct_id', how='left')
        .join(trial_interventions, on='nct_id', how='left')
        .join(arms, on='nct_id', how='left')
        .join(outcomes, on='nct_id', how='left')
        .sort('nct_id')
    )


def _cut(text: str, length: int) -> str:
    if length < MIN_KEPT:
        return ''
    return text[: length - len(ELLIPSIS)].rstrip() + ELLIPSIS


def _render(row: dict, texts: dict[tuple, str]) -> str:
    lines = [
        f'Trial: {row["nct_id"]}',
        f'Title: {row.get("official_title") or ""}',
        f'Phase: {row.get("phase") or "not given"}',
        f'Purpose: {row.get("primary_purpose") or "not given"}',
        'Design: '
        + ', '.join(f'{key} {row[key]}' for key in ('allocation', 'intervention_model', 'masking') if row.get(key)),
        f'Summary: {texts[("summary",)]}',
        f'Detailed description: {texts[("detail",)]}',
    ]
    arms = row.get('arms') or []
    if arms:
        lines.append('Arms:')
        for i, arm in enumerate(arms):
            lines.append(f'- {arm["title"] or "untitled"} ({arm["type"] or "type not given"})')
            if texts[('arm', i)]:
                lines.append(f'  Description: {texts[("arm", i)]}')
            lines.append('  Interventions: ' + ('; '.join(arm['interventions']) or 'none linked'))
    else:
        lines.append('Arms: none listed')
    interventions = row.get('interventions') or []
    lines.append('Interventions:' if interventions else 'Interventions: none listed')
    for i, intervention in enumerate(interventions):
        description = texts[('intervention', i)]
        lines.append(f'- {intervention["name"]} ({intervention["type"]})' + (f': {description}' if description else ''))
    outcomes = row.get('primary_outcomes') or []
    lines.append('Primary outcomes:' if outcomes else 'Primary outcomes: none listed')
    lines.extend(f'- {texts[("outcome", i)]}' for i in range(len(outcomes)) if texts[('outcome', i)])
    return '\n'.join(lines)


def render_trial(row: dict, max_chars: int = MAX_CHARS) -> str:
    """The prompt text of one `trial_context` row, at most `max_chars` long when the untruncatable parts fit.

    Descriptions, the summaries and outcome measures are cut, the longest first, until the text fits; the
    title, arm titles, arm intervention lists and intervention names are kept whole.
    """
    texts: dict[tuple, str] = {
        ('summary',): row.get('brief_summary') or '',
        ('detail',): row.get('detailed_description') or '',
    }
    texts.update({('arm', i): arm['description'] or '' for i, arm in enumerate(row.get('arms') or [])})
    texts.update({('intervention', i): it['description'] or '' for i, it in enumerate(row.get('interventions') or [])})
    texts.update({('outcome', i): measure or '' for i, measure in enumerate(row.get('primary_outcomes') or [])})
    text = _render(row, texts)
    while len(text) > max_chars:
        key = max(texts, key=lambda k: (len(texts[k]), k))
        if not texts[key]:
            break
        excess = len(text) - max_chars
        texts[key] = _cut(texts[key], max(len(texts[key]) - excess, len(texts[key]) // 2))
        text = _render(row, texts)
    return text
