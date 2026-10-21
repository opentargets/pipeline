"""Domain-specific LLM extraction helpers for AACT clinical trials."""

from __future__ import annotations

import polars as pl


def sample_report(report: pl.DataFrame, sample_size: int | None, seed: int = 42) -> pl.DataFrame:
    """Sample trials before expensive downstream steps.

    Returns full report when sample_size is None or >= row count.
    """
    if sample_size is None or sample_size >= report.height:
        return report
    return report.sample(n=sample_size, seed=seed, shuffle=True)


def build_prompts(report: pl.DataFrame, trial_fields: dict) -> list[dict]:
    """Build prompts for each record in the report.

    Returns a list of dicts: [{"id": ..., "prompt": ...}, ...]
    """
    records = report.to_dicts()
    results = []
    for row in records:
        prompt = build_prompt(row, trial_fields=trial_fields)
        results.append({'id': row['id'], 'prompt': prompt})
    return results


def build_prompt(row: dict, trial_fields: dict) -> str:
    """Build prompt from a single clinical trial record."""
    lines = [f'Trial ID: {row["id"]}']
    for field, label in trial_fields.items():
        value = row.get(field)
        lines.append(f'{label}: {value if value is not None else "null"}')
    drugs = [d['drugFromSource'] for d in (row.get('drugs') or []) if d.get('drugFromSource')]
    if drugs:
        lines.append(f'Interventions: {"; ".join(drugs)}')
    return '\n'.join(lines)
