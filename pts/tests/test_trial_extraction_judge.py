"""Checks for the bounded, reproducible new-trial judge sample."""

import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from pts.tasks.trial_extraction_judge import (
    TrialExtractionJudge,
    TrialExtractionJudgeSpec,
    TrialJudgeFindings,
    judge_cache_key,
    parse_findings,
    select_sample,
    unjudged_previous_trials,
)


def test_sample_uses_only_new_extractions_with_exact_prompts_and_flags() -> None:
    cache = pl.DataFrame({
        'id': ['nct1', 'nct2', 'nct3', 'nct4', 'nct5', 'nct6'],
        'prompt_sha256': ['a', 'b', 'c', 'd', 'e', 'old'],
    })
    prompts = pl.DataFrame({
        'id': ['nct1', 'nct2', 'nct3', 'nct4', 'nct5', 'nct6'],
        'prompt': ['text'] * 6,
        'prompt_sha256': ['a', 'b', 'c', 'd', 'e', 'new'],
    })
    trials_needing_extraction = pl.DataFrame({
        'id': ['nct1', 'nct2', 'nct6'],
        'prompt_sha256': ['a', 'b', 'new'],
    })
    flags = pl.DataFrame({'id': ['nct1'], 'category': ['name_not_in_quote']})
    kwargs = {'sample_size': 4, 'flagged_fraction': 0.5, 'seed': 'fixed'}
    first = select_sample(cache, prompts, trials_needing_extraction, flags, **kwargs)
    second = select_sample(cache, prompts, trials_needing_extraction, flags, **kwargs)

    assert first == second
    assert len(first) == 2
    assert {row['id'] for row in first}.isdisjoint({'nct6'})
    assert {row['id'] for row in first} == {'nct1', 'nct2'}
    assert {row['source'] for row in first} == {'random', 'flagged'}


def test_sample_balances_100_from_trials_needing_extraction() -> None:
    ids = [f'nct{i:03d}' for i in range(140)]
    cache = pl.DataFrame({'id': ids, 'prompt_sha256': ['same'] * len(ids)})
    prompts = pl.DataFrame({'id': ids, 'prompt': ['text'] * len(ids), 'prompt_sha256': ['same'] * len(ids)})
    trials_needing_extraction = pl.DataFrame({'id': ids[:130], 'prompt_sha256': ['same'] * 130})
    flags = pl.DataFrame({'id': ids[:60], 'category': ['missing_quote'] * 60})

    sample = select_sample(
        cache, prompts, trials_needing_extraction, flags, sample_size=100, flagged_fraction=0.5, seed='fixed'
    )

    assert len(sample) == 100
    assert sum(row['source'] == 'flagged' for row in sample) == 50
    assert sum(row['source'] == 'random' for row in sample) == 50
    assert {row['id'] for row in sample}.issubset(set(ids[:130]))


def test_fallback_excludes_only_verdicts_for_current_input_and_judge() -> None:
    cache = pl.DataFrame({
        'id': ['nct1', 'nct2', 'nct3'],
        'prompt_sha256': ['a', 'b', 'old'],
    })
    prompts = pl.DataFrame({
        'id': ['nct1', 'nct2', 'nct3'],
        'prompt': ['text'] * 3,
        'prompt_sha256': ['a', 'b', 'new'],
    })
    judged_cache = pl.DataFrame({'cache_key': [judge_cache_key(cache.row(0, named=True), 'gpt-6-luna')]})

    remaining = unjudged_previous_trials(cache, prompts, judged_cache, 'gpt-6-luna')
    assert remaining['id'].to_list() == ['nct2']

    other_model = unjudged_previous_trials(cache, prompts, judged_cache, 'another-model')
    assert set(other_model['id'].to_list()) == {'nct1', 'nct2'}

    both_judged = pl.DataFrame({'cache_key': [judge_cache_key(row, 'gpt-6-luna') for row in cache.to_dicts()[:2]]})
    none_remaining = unjudged_previous_trials(cache, prompts, both_judged, 'gpt-6-luna')
    assert none_remaining.is_empty()
    assert none_remaining.columns == ['id', 'prompt_sha256']


def test_structured_findings_check_quotes_against_trial_prompt() -> None:
    values = {name: (True if name.endswith('_correct') else '') for name in TrialJudgeFindings.model_fields}
    values.update({
        'investigated_drugs_correct': False,
        'investigated_drugs_issue': 'Aspirin is missing',
        'investigated_drugs_source_quote': 'Aspirin 100 mg daily',
        'investigated_drugs_suggested_fix': 'Add aspirin to investigated_drugs',
        'drug_intent_correct': False,
        'drug_intent_recommended_label': 'prevention',
        'drug_intent_reason': 'The intervention aims to prevent infection.',
        'drug_intent_issue': 'The intervention prevents a future event',
        'drug_intent_source_quote': 'prevent infection',
        'drug_intent_suggested_fix': 'Use prevention',
    })
    scores = {f'clinical_roles.{name}': value for name, value in values.items()}
    parsed = parse_findings(scores, 'Study arms include Aspirin 100 mg daily to prevent infection.')
    assert parsed['investigated_drugs_correct'] is False
    assert parsed['investigated_drugs_quote_in_prompt'] is True
    assert parsed['primary_indications_quote_in_prompt'] is False
    assert parsed['drug_intent_correct'] is False
    assert parsed['drug_intent_recommended_label'] == 'prevention'
    assert parsed['drug_intent_quote_in_prompt'] is True


def test_empty_evaluation_publishes_readable_outputs(tmp_path: Path) -> None:
    destination = {
        'summary': str(tmp_path / 'summary.json'),
        'results': str(tmp_path / 'results.parquet'),
        'findings': str(tmp_path / 'findings.jsonl'),
    }
    task = TrialExtractionJudge(
        TrialExtractionJudgeSpec(
            name='trial_extraction_judge test',
            cache_uri=str(tmp_path / 'extraction-cache'),
            current_prompts=str(tmp_path / 'prompts.parquet'),
            trials_needing_extraction=str(tmp_path / 'trials_needing_extraction.parquet'),
            analysis_flags=str(tmp_path / 'flags.parquet'),
            judge_cache_uri=str(tmp_path / 'judge-cache'),
            snapshot='snapshot',
            destination=destination,
        ),
        SimpleNamespace(config=None),
    )

    task._write_empty_evaluation()

    assert json.loads(Path(destination['summary']).read_text())['sample_size'] == 0
    assert pl.read_parquet(destination['results']).is_empty()
    assert Path(destination['findings']).read_text() == ''
