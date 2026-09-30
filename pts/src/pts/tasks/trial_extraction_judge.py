"""Judge a bounded sample of AACT extractions against their trial text."""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from importlib import import_module
from pathlib import Path
from typing import Any, Self

import polars as pl
from loguru import logger
from otter.manifest.model import Artifact
from otter.storage.synchronous.handle import StorageHandle
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report
from otter.util.errors import TaskValidationError
from pydantic import BaseModel, Field, SecretStr

from pts.result_cache import cache_key, cached_map, read_cache
from pts.tasks.trial_extraction_analysis import _read_parquet, schema_cache_uri

RUBRIC_VERSION = 'aact-roles-v5-intent-timing'
TRAITS = {
    'investigated_drugs_correct': (
        'Compare the EXTRACTION with the TRIAL TEXT. The investigated_drugs field should contain '
        'the experimental intervention being evaluated. Standard treatment used as a benchmark '
        'belongs in comparator_drugs, not investigated_drugs, even if it is a trial arm. Return '
        'true only if all experimental drugs are present and no comparator is misclassified there. '
        'Do not use outside knowledge. If the roles are ambiguous, return false.'
    ),
    'primary_indications_correct': (
        'Compare the EXTRACTION with the TRIAL TEXT. Return true only if every primary indication '
        'is a disease, condition, or event that the investigated intervention aims to treat, detect, '
        'prevent, or relieve according to drug_intent, and no clear primary indication is omitted. '
        'Do not confuse eligibility or background conditions with primary indications. Do not use '
        'outside knowledge.'
    ),
    'drug_intent_correct': (
        'Compare drug_intent with what the investigated drugs are evaluated to do in TRIAL TEXT. '
        'Use therapeutic for treating an existing disease or condition; diagnostic for detecting '
        'or localizing it; prevention for preventing a future condition or event; supportive_care '
        'for relieving symptoms or side effects without treating their underlying disease; and '
        'other when none apply. The trial title or administrative Primary Purpose is only a hint. '
        'Use the timing of intervention and outcome measurement to distinguish treatment from '
        'prevention; do not assume a condition already exists merely because it is measured later. '
        'If the text cannot distinguish the categories, return false and explain the uncertainty. '
        'Do not use outside knowledge.'
    ),
    'other_roles_correct': (
        'Compare the EXTRACTION with the TRIAL TEXT. Return true only if comparator drugs, supportive '
        'drugs, and background conditions have the right roles where present. An empty '
        'field is acceptable when that role is not supported by the trial text. Do not use outside '
        'knowledge.'
    ),
}


class TrialJudgeFindings(BaseModel):
    """One-call verdicts and review notes for the four clinical role checks."""

    investigated_drugs_correct: bool = Field(description=TRAITS['investigated_drugs_correct'])
    investigated_drugs_issue: str = Field(
        description='If false, name the missing, extra, or misclassified drug; otherwise empty.'
    )
    investigated_drugs_source_quote: str = Field(
        description='If false, copy a short verbatim quote from TRIAL TEXT supporting the issue; otherwise empty.'
    )
    investigated_drugs_suggested_fix: str = Field(
        description='If false, suggest a concrete correction to investigated_drugs; otherwise empty.'
    )
    primary_indications_correct: bool = Field(description=TRAITS['primary_indications_correct'])
    primary_indications_issue: str = Field(
        description='If false, name the missing, extra, or misclassified condition; otherwise empty.'
    )
    primary_indications_source_quote: str = Field(
        description='If false, copy a short verbatim quote from TRIAL TEXT supporting the issue; otherwise empty.'
    )
    primary_indications_suggested_fix: str = Field(
        description='If false, suggest a concrete correction to primary_indications; otherwise empty.'
    )
    drug_intent_correct: bool = Field(description=TRAITS['drug_intent_correct'])
    drug_intent_recommended_label: str = Field(
        description='Choose exactly one of therapeutic, diagnostic, prevention, supportive_care, or other.'
    )
    drug_intent_reason: str = Field(
        description=(
            'Always explain why that intent category best describes the investigated intervention, '
            'including when the extracted label is correct.'
        )
    )
    drug_intent_issue: str = Field(
        description='If false, explain why the extracted intent is unsupported or ambiguous; otherwise empty.'
    )
    drug_intent_source_quote: str = Field(
        description='Always copy a short verbatim quote from TRIAL TEXT supporting the intent classification.'
    )
    drug_intent_suggested_fix: str = Field(
        description=(
            'If false, suggest one of therapeutic, diagnostic, prevention, supportive_care, '
            'or other, or say needs review; otherwise empty.'
        )
    )
    other_roles_correct: bool = Field(description=TRAITS['other_roles_correct'])
    other_roles_issue: str = Field(description='If false, identify the affected role and value; otherwise empty.')
    other_roles_source_quote: str = Field(
        description='If false, copy a short verbatim quote from TRIAL TEXT supporting the issue; otherwise empty.'
    )
    other_roles_suggested_fix: str = Field(
        description='If false, suggest a concrete correction to that role; otherwise empty.'
    )


JUDGE_INSTRUCTIONS = (
    'Evaluate each clinical role using only TRIAL TEXT. Give a concise issue and suggested '
    'correction for every false verdict. Always explain the drug_intent verdict. '
    'Source quotes must be copied verbatim from TRIAL TEXT, '
    'never from EXTRACTION TO EVALUATE. If evidence is insufficient, mark the criterion false '
    'and explain the uncertainty. For true verdicts, leave issue and fix empty; '
    'leave quote empty except for drug_intent, which always needs a quote.'
)
RUBRIC_DIGEST = hashlib.sha256(
    json.dumps(
        {'schema': TrialJudgeFindings.model_json_schema(), 'instructions': JUDGE_INSTRUCTIONS}, sort_keys=True
    ).encode()
).hexdigest()


def parse_findings(scores: dict[str, Any], prompt: str) -> dict[str, Any]:
    """Validate Karenina's flattened template output and its source quotes."""
    prefix = 'clinical_roles.'
    values = {name: scores.get(prefix + name) for name in TrialJudgeFindings.model_fields}
    findings = TrialJudgeFindings.model_validate(values, strict=True).model_dump()
    if findings['drug_intent_recommended_label'] not in {
        'therapeutic',
        'diagnostic',
        'prevention',
        'supportive_care',
        'other',
    }:
        raise ValueError('Karenina returned an invalid drug_intent label')
    if not findings['drug_intent_reason'].strip() or not findings['drug_intent_source_quote'].strip():
        raise ValueError('Karenina returned no explanation or source quote for drug_intent')
    for name in TRAITS:
        field = name.removesuffix('_correct')
        quote = findings[f'{field}_source_quote']
        findings[f'{field}_quote_in_prompt'] = bool(quote and quote in prompt)
    return findings


class TrialExtractionJudgeSpec(Spec):
    """Bounded Karenina evaluation after the systematic analysis step."""

    cache_uri: str
    model_class: str = 'mira.schemas.ClinicalReportExtractionSchema'
    current_prompts: str
    trials_needing_extraction: str
    """IDs needing extraction when this snapshot started, published by llm_extract."""
    analysis_flags: str
    judge_cache_uri: str
    snapshot: str
    destination: dict[str, str]
    sample_size: int = Field(default=100, ge=1)
    flagged_fraction: float = Field(default=0.5, ge=0, le=1)
    seed: str = 'aact-trial-judge-v1'
    model: str = 'gpt-6-luna'
    openai_key_path: str = '/var/run/secrets/openai-api-key'


def _rank(seed: str, trial_id: str) -> str:
    return hashlib.sha256(f'{seed}\x1f{trial_id}'.encode()).hexdigest()


def select_sample(
    cache: pl.DataFrame,
    prompts: pl.DataFrame,
    trials_needing_extraction: pl.DataFrame,
    flags: pl.DataFrame,
    *,
    sample_size: int,
    flagged_fraction: float,
    seed: str,
) -> list[dict[str, Any]]:
    """Sample eligible extractions, balancing flagged and unflagged cases."""
    if sample_size < 1 or not 0 <= flagged_fraction <= 1:
        raise ValueError('sample_size must be positive and flagged_fraction must be in [0, 1]')
    flagged_ids = set(
        flags.filter(
            pl.col('category').is_in([
                'name_not_in_quote',
                'missing_quote',
                'no_investigated_drug',
                'no_primary_indication',
            ])
        )['id'].to_list()
    )
    current = prompts.select('id', 'prompt', pl.col('prompt_sha256').alias('current_prompt_sha256'))
    eligible = (
        cache
        .join(trials_needing_extraction.select('id', 'prompt_sha256').unique(), on=['id', 'prompt_sha256'], how='inner')
        .join(current, on='id', how='inner')
        .filter(pl.col('prompt_sha256') == pl.col('current_prompt_sha256'))
        .unique(subset='id')
    )
    rows = eligible.to_dicts()
    flagged = sorted((r for r in rows if r['id'] in flagged_ids), key=lambda r: _rank(seed, r['id']))
    random = sorted((r for r in rows if r['id'] not in flagged_ids), key=lambda r: _rank(seed, r['id']))
    n_flagged = min(round(sample_size * flagged_fraction), len(flagged))
    selected = [(r, 'flagged') for r in flagged[:n_flagged]]
    selected += [(r, 'random') for r in random[: sample_size - len(selected)]]
    if len(selected) < sample_size:
        selected += [(r, 'flagged') for r in flagged[n_flagged : n_flagged + sample_size - len(selected)]]
    return [{**row, 'source': source} for row, source in selected]


def _extraction(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        'drug_intent',
        'drug_intent_confidence',
        'investigated_drugs',
        'primary_indications',
        'comparator_drugs',
        'supportive_drugs',
        'background_conditions',
    )
    return {field: row.get(field) for field in fields}


def judge_cache_key(row: dict[str, Any], model: str) -> str:
    """Identify a verdict by trial text, extraction, judge model, and rubric."""
    extraction_digest = hashlib.sha256(json.dumps(_extraction(row), sort_keys=True, default=str).encode()).hexdigest()
    return cache_key(row['id'], row['prompt_sha256'], extraction_digest, model, RUBRIC_VERSION, RUBRIC_DIGEST)


def unjudged_previous_trials(
    cache: pl.DataFrame,
    prompts: pl.DataFrame,
    judged_cache: pl.DataFrame,
    model: str,
) -> pl.DataFrame:
    """Find prompt-matched accepted extractions without a current verdict."""
    current = prompts.select('id', pl.col('prompt_sha256').alias('current_prompt_sha256'))
    eligible = (
        cache
        .join(current, on='id', how='inner')
        .filter(pl.col('prompt_sha256') == pl.col('current_prompt_sha256'))
        .unique(subset='id')
    )
    judged_keys = set(judged_cache['cache_key'].to_list())
    remaining = [
        {'id': row['id'], 'prompt_sha256': row['prompt_sha256']}
        for row in eligible.to_dicts()
        if judge_cache_key(row, model) not in judged_keys
    ]
    return pl.DataFrame(remaining, schema={'id': pl.String, 'prompt_sha256': pl.String})


def judge_trial(row: dict[str, Any], model: str, openai_key: str) -> dict[str, Any]:
    """Use Karenina TaskEval to compare one cached extraction with its exact prompt."""
    # Karenina is installed only in the evaluation image. Dynamic imports keep
    # the regular PTS development environment and image usable.
    task_eval_class = import_module('karenina.benchmark.task_eval').TaskEval
    model_config_class = import_module('karenina.schemas.config.models').ModelConfig
    rubric_module = import_module('karenina.schemas.entities.rubric')
    trait_class, rubric_class = rubric_module.LLMRubricTrait, rubric_module.Rubric
    verification_config_class = import_module('karenina.schemas.verification.config').VerificationConfig

    task = task_eval_class(task_id=row['id'])
    task.log(
        'TRIAL TEXT (untrusted source data):\n'
        f'{row["prompt"]}\n\n'
        'EXTRACTION TO EVALUATE:\n'
        f'{json.dumps(_extraction(row), ensure_ascii=False, default=str)}'
    )
    task.add_rubric(
        rubric_class(
            llm_traits=[
                trait_class(
                    name='clinical_roles',
                    description=JUDGE_INSTRUCTIONS,
                    kind=TrialJudgeFindings,
                )
            ]
        )
    )
    config = verification_config_class(
        parsing_models=[
            model_config_class(
                id=model,
                model_name=model,
                model_provider='openai',
                interface='langchain',
                temperature=1,
                extra_kwargs={'api_key': SecretStr(openai_key)},
            )
        ],
        parsing_only=True,
    )
    result = task.evaluate(config)
    evaluations = result.global_eval.verification_results if result.global_eval else {}
    scores = [
        verification.rubric.llm_trait_scores
        for verifications in evaluations.values()
        for verification in verifications
        if verification.rubric and verification.rubric.llm_trait_scores
    ]
    if len(scores) != 1:
        raise ValueError(f'Karenina returned incomplete rubric scores for {row["id"]}')
    findings = parse_findings(scores[0], row['prompt'])
    if findings['drug_intent_correct'] and findings['drug_intent_recommended_label'] != row.get('drug_intent'):
        raise ValueError(f'Karenina returned inconsistent drug_intent scores for {row["id"]}')
    return findings


class TrialExtractionJudge(Task):
    """Evaluate new trials, then unjudged cached trials when no new ones qualify."""

    def __init__(self, spec: TrialExtractionJudgeSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: TrialExtractionJudgeSpec
        self.stats: dict[str, Any] = {}

    def _publish(self, results: pl.DataFrame, findings: list[dict[str, Any]]) -> None:
        """Write the three evaluation outputs with the same schema in every run."""
        summary_uri = StorageHandle(self.spec.destination['summary'], config=self.context.config).absolute
        results_uri = StorageHandle(self.spec.destination['results'], config=self.context.config).absolute
        findings_uri = StorageHandle(self.spec.destination['findings'], config=self.context.config).absolute
        StorageHandle(summary_uri, config=self.context.config).write_text(
            json.dumps(self.stats, indent=2, sort_keys=True)
        )
        buffer = io.BytesIO()
        results.write_parquet(buffer, compression='zstd')
        StorageHandle(results_uri, config=self.context.config).write(buffer.getvalue())
        StorageHandle(findings_uri, config=self.context.config).write_text(
            ''.join(json.dumps(finding, ensure_ascii=False) + '\n' for finding in findings)
        )
        self.artifacts = [
            Artifact(source=self.spec.judge_cache_uri, destination=summary_uri),
            Artifact(source=self.spec.judge_cache_uri, destination=results_uri),
            Artifact(source=self.spec.judge_cache_uri, destination=findings_uri),
        ]

    def _write_empty_evaluation(self) -> None:
        """Publish an explicit zero-trial result when no eligible trial remains."""
        self.stats = {
            'scope': 'no eligible new or previously unjudged extraction with a matching current prompt hash',
            'selection_pool': None,
            'sample_size': 0,
            'eligible_count': 0,
            'source_counts': {},
            'source_stats': {},
            'model': self.spec.model,
            'rubric_version': RUBRIC_VERSION,
            'sample_seed': self.spec.seed,
            'trait_pass_counts': dict.fromkeys(TRAITS, 0),
            'all_correct_count': 0,
            'all_correct_percent': None,
            'finding_count': 0,
            'findings_by_criterion': {},
            'findings_with_source_quote_in_prompt': 0,
        }
        schema = {
            'id': pl.String,
            'source': pl.String,
            'selection_pool': pl.String,
            'prompt_sha256': pl.String,
            'extracted_drug_intent': pl.String,
            'extracted_drug_intent_confidence': pl.Float64,
            **{
                name: pl.Boolean if name.endswith('_correct') else pl.String for name in TrialJudgeFindings.model_fields
            },
            **{f'{name.removesuffix("_correct")}_quote_in_prompt': pl.Boolean for name in TRAITS},
            'all_correct': pl.Boolean,
        }
        self._publish(pl.DataFrame(schema=schema), [])

    @report
    def run(self) -> Self:
        namespace = schema_cache_uri(self.spec.cache_uri, self.spec.model_class)
        cache = read_cache(namespace, self.context.config)
        prompts = _read_parquet(self.spec.current_prompts, self.context.config)
        trials_needing_extraction = _read_parquet(self.spec.trials_needing_extraction, self.context.config)
        flags = _read_parquet(self.spec.analysis_flags, self.context.config)
        eligible_new_count = len(
            select_sample(
                cache,
                prompts,
                trials_needing_extraction,
                flags,
                sample_size=cache.height or 1,
                flagged_fraction=self.spec.flagged_fraction,
                seed=self.spec.seed,
            )
        )
        selection_pool = 'trials_needing_extraction'
        candidates = trials_needing_extraction
        eligible_count = eligible_new_count
        if not eligible_new_count:
            selection_pool = 'previous_unjudged'
            candidates = unjudged_previous_trials(
                cache, prompts, read_cache(self.spec.judge_cache_uri, self.context.config), self.spec.model
            )
            eligible_count = candidates.height
        sample = select_sample(
            cache,
            prompts,
            candidates,
            flags,
            sample_size=self.spec.sample_size,
            flagged_fraction=self.spec.flagged_fraction,
            seed=self.spec.seed,
        )
        if not sample:
            logger.info('no new or previously unjudged trials with matching prompts; writing an empty evaluation')
            self._write_empty_evaluation()
            return self

        openai_key = Path(self.spec.openai_key_path).read_text(encoding='utf-8').strip()
        records = [
            {
                **row,
                'selection_pool': selection_pool,
                'cache_key': judge_cache_key(row, self.spec.model),
            }
            for row in sample
        ]
        selected = pl.DataFrame(records)

        def evaluate_one(shard: pl.DataFrame) -> pl.DataFrame:
            row = shard.to_dicts()[0]
            scores = judge_trial(row, self.spec.model, openai_key)
            return pl.DataFrame([
                {
                    'cache_key': row['cache_key'],
                    'id': row['id'],
                    'source': row['source'],
                    'prompt_sha256': row['prompt_sha256'],
                    'extracted_drug_intent': row['drug_intent'],
                    'extracted_drug_intent_confidence': row['drug_intent_confidence'],
                    **scores,
                }
            ])

        registry = import_module('karenina.adapters.registry').AdapterRegistry
        if 'langchain' not in registry.get_interfaces():
            raise TaskValidationError('Karenina langchain adapter is unavailable')

        judged = cached_map(
            records=selected,
            compute=evaluate_one,
            cache_uri=self.spec.judge_cache_uri,
            config=self.context.config,
            run_id=self.spec.snapshot,
            timestamp=self.spec.snapshot,
            shard_size=1,
        )
        if judged.height != len(sample):
            raise TaskValidationError(f'judged {judged.height} of {len(sample)} selected trials')

        # The judge cache can contain a verdict from an earlier pilot with a
        # different sample source. Source describes this run's selection, so
        # take it from selected rather than from the cached verdict.
        judged = judged.drop('source', 'selection_pool', strict=False).join(
            selected.select('cache_key', 'source', 'selection_pool'), on='cache_key'
        )

        results = judged.select(
            'id',
            'source',
            'selection_pool',
            'prompt_sha256',
            'extracted_drug_intent',
            'extracted_drug_intent_confidence',
            *TrialJudgeFindings.model_fields,
            *(f'{name.removesuffix("_correct")}_quote_in_prompt' for name in TRAITS),
        )
        all_correct = pl.all_horizontal([pl.col(name) for name in TRAITS])
        results = results.with_columns(all_correct.alias('all_correct')).sort('id')
        findings = [
            {
                'id': row['id'],
                'source': row['source'],
                'selection_pool': row['selection_pool'],
                'criterion': name,
                'issue': row[f'{field}_issue'],
                'source_quote': row[f'{field}_source_quote'],
                'source_quote_in_prompt': row[f'{field}_quote_in_prompt'],
                'suggested_fix': row[f'{field}_suggested_fix'],
                'recommended_label': row['drug_intent_recommended_label'] if field == 'drug_intent' else None,
                'extracted_label': row['extracted_drug_intent'] if field == 'drug_intent' else None,
                'reason': row['drug_intent_reason'] if field == 'drug_intent' else None,
            }
            for row in results.to_dicts()
            for name in TRAITS
            for field in [name.removesuffix('_correct')]
            if not row[name]
        ]
        all_correct_count = int(results['all_correct'].sum())
        source_stats = {}
        for source in ('random', 'flagged'):
            group = results.filter(pl.col('source') == source)
            group_passed = int(group['all_correct'].sum()) if group.height else 0
            source_stats[source] = {
                'sample_size': group.height,
                'trait_pass_counts': {name: int(group[name].sum()) if group.height else 0 for name in TRAITS},
                'all_correct_count': group_passed,
                'all_correct_percent': round(100 * group_passed / group.height, 2) if group.height else None,
            }
        self.stats = {
            'scope': 'deterministic sample with matching current prompt hashes',
            'selection_pool': selection_pool,
            'sample_size': results.height,
            'eligible_count': eligible_count,
            'source_counts': dict(Counter(results['source'].to_list())),
            'source_stats': source_stats,
            'model': self.spec.model,
            'rubric_version': RUBRIC_VERSION,
            'sample_seed': self.spec.seed,
            'trait_pass_counts': {name: int(results[name].sum()) for name in TRAITS},
            'all_correct_count': all_correct_count,
            'all_correct_percent': round(100 * all_correct_count / results.height, 2),
            'finding_count': len(findings),
            'findings_by_criterion': dict(Counter(finding['criterion'] for finding in findings)),
            'findings_with_source_quote_in_prompt': sum(1 for finding in findings if finding['source_quote_in_prompt']),
        }
        self._publish(results, findings)
        logger.info(f'judged {results.height} trials; {self.stats["all_correct_count"]} passed all traits')
        return self

    @report
    def validate(self) -> Self:
        if not self.stats:
            raise TaskValidationError('judge produced no results')
        return self
