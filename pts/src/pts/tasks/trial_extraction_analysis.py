"""Systematic analysis of every accepted AACT extraction in the current schema cache."""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from importlib import import_module
from statistics import mean, median
from typing import Any, Self

import polars as pl
from loguru import logger
from otter.manifest.model import Artifact
from otter.storage.synchronous.handle import StorageHandle
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.errors import TaskValidationError

from pts.result_cache import read_cache


def schema_cache_uri(cache_uri: str, model_class: str) -> str:
    """Select the cache namespace used by the current extraction schema."""
    module_name, class_name = model_class.rsplit('.', 1)
    model = getattr(import_module(module_name), class_name)
    schema = model.model_json_schema(by_alias=True)
    digest = hashlib.sha256(json.dumps(schema, sort_keys=True).encode('utf-8')).hexdigest()
    return f'{cache_uri.rstrip("/")}/{digest}'


class TrialExtractionAnalysisSpec(Spec):
    """Inputs and outputs for the post-extraction PTS step."""

    cache_uri: str
    """Shared cache root, before the schema digest."""
    model_class: str = 'mira.schemas.ClinicalReportExtractionSchema'
    """Must match the model class used by the extraction step."""
    current_prompts: str
    """Prompts published by the extraction step for this AACT snapshot."""
    destination: dict[str, str]
    """Keys: ``summary`` (JSON), ``flags`` and ``trials`` (Parquet)."""


def _read_parquet(uri: str, config: Any) -> pl.DataFrame:
    data, _ = StorageHandle(uri, config=config).read()
    return pl.read_parquet(io.BytesIO(data))


def _percent(numerator: int, denominator: int) -> float | None:
    return round(100 * numerator / denominator, 2) if denominator else None


def _items(row: dict[str, Any], field: str) -> list[dict[str, Any]]:
    return [item for item in (row.get(field) or []) if isinstance(item, dict)]


def analyse_cache(
    cache: pl.DataFrame,
    current_prompts: pl.DataFrame,
) -> tuple[dict[str, Any], pl.DataFrame]:
    """Return cache-wide metrics and individual audit flags.

    Label checks use each entity's own quote, independent of prompt snapshots.
    """
    required = {
        'id',
        'prompt_sha256',
        'drug_intent',
        'drug_intent_confidence',
        'investigated_drugs',
        'primary_indications',
    }
    missing = required - set(cache.columns)
    if missing and cache.height:
        raise ValueError(f'cache lacks extraction columns: {sorted(missing)}')

    intents: Counter[str] = Counter()
    shapes: Counter[str] = Counter()
    confidence_bins: Counter[str] = Counter()
    confidence_by_intent: dict[str, list[float]] = {}
    quote_counts: Counter[str] = Counter()
    fields = {
        'investigated_drugs': 'drug',
        'comparator_drugs': 'drug',
        'supportive_drugs': 'drug',
        'primary_indications': 'name',
        'background_conditions': 'name',
    }
    field_counts: dict[str, Counter[str]] = {field: Counter() for field in fields}
    name_mismatch_trials_by_field: dict[str, set[str]] = {}
    quote_issue_trials: dict[str, set[str]] = {
        'missing_quote': set(),
        'name_not_in_quote': set(),
    }
    flags: list[dict[str, str]] = []
    current_ids = set(current_prompts['id'].to_list())
    cached_ids: set[str] = set()
    cache_hashes: dict[str, str] = {}

    def flag(trial_id: str, category: str, field: str = '', entity: str = '', quote: str = '') -> None:
        flags.append({'id': trial_id, 'category': category, 'field': field, 'entity': entity, 'quote': quote})

    for row in cache.iter_rows(named=True):
        trial_id = row['id']
        cached_ids.add(trial_id)
        cache_hashes[trial_id] = row['prompt_sha256']
        intent = row['drug_intent'] or 'missing'
        intents[intent] += 1
        confidence = row['drug_intent_confidence']
        if confidence is None:
            confidence_bins['missing'] += 1
        elif confidence < 0.7:
            confidence_bins['<0.7'] += 1
        elif confidence < 0.9:
            confidence_bins['0.7-<0.9'] += 1
        else:
            confidence_bins['>=0.9'] += 1
        if confidence is not None:
            confidence_by_intent.setdefault(intent, []).append(confidence)

        drugs = _items(row, 'investigated_drugs')
        diseases = _items(row, 'primary_indications')
        nd, nc = len(drugs), len(diseases)
        if nd == 0 or nc == 0:
            shape = 'zero_drug_or_indication'
        elif nd == nc == 1:
            shape = 'one_drug_one_indication'
        elif nd == 1:
            shape = 'one_drug_multiple_indications'
        elif nc == 1:
            shape = 'multiple_drugs_one_indication'
        else:
            shape = 'multiple_drugs_multiple_indications'
        shapes[shape] += 1
        if nd == 0:
            flag(trial_id, 'no_investigated_drug')
        if nc == 0:
            flag(trial_id, 'no_primary_indication')
        if nd > 1 and nc > 1:
            flag(trial_id, 'multiple_drug_indication_combinations')

        for field, name_key in fields.items():
            if row.get(field) is not None:
                field_counts[field]['non_null_trials'] += 1
            entities = _items(row, field)
            if entities:
                field_counts[field]['trials_with_entities'] += 1
            for entity in entities:
                name = entity.get(name_key) or ''
                quote = entity.get('evidence_quote') or ''
                quote_counts['entities_checked'] += 1
                field_counts[field]['entities'] += 1
                if not quote:
                    quote_counts['missing_quote'] += 1
                    field_counts[field]['missing_quote'] += 1
                    quote_issue_trials['missing_quote'].add(trial_id)
                    flag(trial_id, 'missing_quote', field, name)
                    continue
                quote_counts['with_quote'] += 1
                field_counts[field]['with_quote'] += 1
                if name and name.casefold() not in quote.casefold():
                    quote_counts['name_not_in_quote'] += 1
                    field_counts[field]['name_not_in_quote'] += 1
                    name_mismatch_trials_by_field.setdefault(field, set()).add(trial_id)
                    quote_issue_trials['name_not_in_quote'].add(trial_id)
                    flag(trial_id, 'name_not_in_quote', field, name, quote)

    stale_current = 0
    for row in current_prompts.select('id', 'prompt_sha256').iter_rows(named=True):
        if row['id'] in cache_hashes and cache_hashes[row['id']] != row['prompt_sha256']:
            stale_current += 1
            flag(row['id'], 'current_prompt_differs_from_cached_prompt')

    total = cache.height
    summary = {
        'scope': 'all accepted rows in the latest snapshot of the current schema cache',
        'cache_rows': total,
        'unique_trial_ids': len(cached_ids),
        'duplicate_trial_id_rows': total - len(cached_ids),
        'current_snapshot': {
            'prompt_count': current_prompts.height,
            'cached_count': len(current_ids & cached_ids),
            'missing_extraction_count': len(current_ids - cached_ids),
            'coverage_percent': _percent(len(current_ids & cached_ids), len(current_ids)),
            'cached_prompt_differs_count': stale_current,
        },
        'drug_intent_counts': dict(sorted(intents.items())),
        'confidence_counts': dict(sorted(confidence_bins.items())),
        'confidence_by_intent': {
            intent: {
                'count': len(values),
                'mean': round(mean(values), 3),
                'median': round(median(values), 3),
                'min': min(values),
                'max': max(values),
            }
            for intent, values in sorted(confidence_by_intent.items())
        },
        'entity_shape_counts': dict(sorted(shapes.items())),
        'entity_shape_percent_of_cache': {shape: _percent(count, total) for shape, count in sorted(shapes.items())},
        'label_quote_by_field': {
            field: {
                **{
                    key: counts[key]
                    for key in (
                        'non_null_trials',
                        'trials_with_entities',
                        'entities',
                        'with_quote',
                        'missing_quote',
                        'name_not_in_quote',
                    )
                },
                'trials_with_name_not_in_quote': len(name_mismatch_trials_by_field.get(field, set())),
                'percent_of_quoted_entities_with_name_not_in_quote': _percent(
                    counts['name_not_in_quote'], counts['with_quote']
                ),
                'percent_of_trials_with_entities_affected': _percent(
                    len(name_mismatch_trials_by_field.get(field, set())), counts['trials_with_entities']
                ),
            }
            for field, counts in field_counts.items()
        },
        'quote_checks': {
            **dict(sorted(quote_counts.items())),
            'name_not_in_quote_percent_of_quoted': _percent(
                quote_counts['name_not_in_quote'], quote_counts['with_quote']
            ),
            'affected_trial_counts': {issue: len(ids) for issue, ids in quote_issue_trials.items()},
            'affected_trial_percent': {issue: _percent(len(ids), total) for issue, ids in quote_issue_trials.items()},
        },
        'flag_counts': dict(sorted(Counter(item['category'] for item in flags).items())),
    }
    flag_schema = {
        'id': pl.String,
        'category': pl.String,
        'field': pl.String,
        'entity': pl.String,
        'quote': pl.String,
    }
    return summary, pl.DataFrame(flags, schema=flag_schema)


def trial_audit(
    cache: pl.DataFrame,
    current_prompts: pl.DataFrame,
    flags: pl.DataFrame,
) -> pl.DataFrame:
    """One row per cached trial with field-specific label and quote flags."""
    audit = cache.select(
        'id',
        pl.col('prompt_sha256').alias('cached_prompt_sha256'),
        pl.col('drug_intent'),
        pl.col('drug_intent_confidence'),
        pl.col('investigated_drugs').list.len().fill_null(0).alias('investigated_drug_count'),
        pl.col('primary_indications').list.len().fill_null(0).alias('primary_indication_count'),
    )
    current = current_prompts.select('id', pl.col('prompt_sha256').alias('current_prompt_sha256'))
    audit = audit.join(current, on='id', how='left').with_columns(
        current_prompt_differs=pl
        .when(pl.col('current_prompt_sha256').is_null())
        .then(None)
        .otherwise(pl.col('cached_prompt_sha256') != pl.col('current_prompt_sha256'))
    )
    for category, column in (
        ('missing_quote', 'missing_quote_count'),
        ('name_not_in_quote', 'name_not_in_quote_count'),
    ):
        counts = flags.filter(pl.col('category') == category).group_by('id').len(name=column)
        audit = audit.join(counts, on='id', how='left').with_columns(pl.col(column).fill_null(0))
    for field in (
        'investigated_drugs',
        'comparator_drugs',
        'supportive_drugs',
        'primary_indications',
        'background_conditions',
    ):
        column = f'{field}_name_not_in_quote_count'
        counts = (
            flags
            .filter((pl.col('category') == 'name_not_in_quote') & (pl.col('field') == field))
            .group_by('id')
            .len(name=column)
        )
        audit = audit.join(counts, on='id', how='left').with_columns(pl.col(column).fill_null(0))
    return audit


class TrialExtractionAnalysis(Task):
    """Produce a cache-wide audit after the extraction DAG step completes."""

    def __init__(self, spec: TrialExtractionAnalysisSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: TrialExtractionAnalysisSpec
        self.stats: dict[str, Any] = {}

    @report
    def run(self) -> Self:
        namespace = schema_cache_uri(self.spec.cache_uri, self.spec.model_class)
        cache = read_cache(namespace, self.context.config)
        if cache.is_empty():
            raise TaskValidationError(f'no accepted extraction rows in {namespace}')
        current = _read_parquet(self.spec.current_prompts, self.context.config)
        self.stats, flags = analyse_cache(cache, current)
        trials = trial_audit(cache, current, flags)

        summary_uri = StorageHandle(self.spec.destination['summary'], config=self.context.config).absolute
        flags_uri = StorageHandle(self.spec.destination['flags'], config=self.context.config).absolute
        trials_uri = StorageHandle(self.spec.destination['trials'], config=self.context.config).absolute
        StorageHandle(summary_uri, config=self.context.config).write_text(
            json.dumps(self.stats, indent=2, sort_keys=True)
        )
        buf = io.BytesIO()
        flags.write_parquet(buf, compression='zstd')
        StorageHandle(flags_uri, config=self.context.config).write(buf.getvalue())
        buf = io.BytesIO()
        trials.write_parquet(buf, compression='zstd')
        StorageHandle(trials_uri, config=self.context.config).write(buf.getvalue())
        self.artifacts = [
            Artifact(source=namespace, destination=summary_uri),
            Artifact(source=namespace, destination=flags_uri),
            Artifact(source=namespace, destination=trials_uri),
        ]
        logger.info(f'analysed {cache.height} cached trials; wrote {flags.height} flags')
        return self

    @report
    def validate(self) -> Self:
        if not self.stats:
            raise TaskValidationError('analysis produced no statistics')
        return self
