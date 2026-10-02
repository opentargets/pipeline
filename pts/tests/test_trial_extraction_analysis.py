"""The cache audit compares labels with their own quotes across all cache rows."""

import hashlib

import polars as pl

from pts.tasks.trial_extraction_analysis import analyse_cache, trial_audit


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def test_label_quote_metrics_use_field_denominators_and_separate_missing_quotes() -> None:
    cache = pl.DataFrame([
        {
            'id': 'NCT1',
            'prompt_sha256': _digest('old'),
            'drug_intent': 'therapeutic',
            'drug_intent_confidence': 0.95,
            'investigated_drugs': [{'drug': 'aspirin', 'evidence_quote': 'ASPIRIN for headache'}],
            'primary_indications': [{'name': 'headache', 'evidence_quote': 'aspirin for headache'}],
        },
        {
            'id': 'NCT2',
            'prompt_sha256': _digest('second'),
            'drug_intent': 'therapeutic',
            'drug_intent_confidence': 0.65,
            'investigated_drugs': [{'drug': 'metformin', 'evidence_quote': 'brand name only'}],
            'primary_indications': [{'name': 'diabetes', 'evidence_quote': ''}],
        },
    ])
    current = pl.DataFrame({
        'id': ['NCT1', 'NCT3'],
        'prompt_sha256': [_digest('new'), _digest('third')],
    })

    summary, flags = analyse_cache(cache, current)
    drugs = summary['label_quote_by_field']['investigated_drugs']
    indications = summary['label_quote_by_field']['primary_indications']

    assert summary['cache_rows'] == 2
    assert summary['current_snapshot']['missing_extraction_count'] == 1
    assert summary['current_snapshot']['cached_prompt_differs_count'] == 1
    assert drugs['trials_with_entities'] == 2
    assert drugs['name_not_in_quote'] == 1
    assert drugs['percent_of_quoted_entities_with_name_not_in_quote'] == 50.0
    assert drugs['percent_of_trials_with_entities_affected'] == 50.0
    assert indications['missing_quote'] == 1
    assert indications['name_not_in_quote'] == 0
    assert indications['percent_of_quoted_entities_with_name_not_in_quote'] == 0.0
    assert set(flags['category'].to_list()) == {
        'name_not_in_quote',
        'missing_quote',
        'current_prompt_differs_from_cached_prompt',
    }

    audit = trial_audit(cache, current, flags)
    assert audit['investigated_drugs_name_not_in_quote_count'].to_list() == [0, 1]
    assert audit['primary_indications_name_not_in_quote_count'].to_list() == [0, 0]
    assert audit['missing_quote_count'].to_list() == [0, 1]
