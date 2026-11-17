import polars as pl

from pts.transformers.clinical_mira.moa.pooling import (
    build_pools,
    link_single_drug_trials,
    match_texts,
    single_drug_names,
    texts_by_id,
)
from tests.clinical_mira.extraction_v2_fixtures import v2_frame, v2_link

DRUG = pl.Struct({'drug': pl.String, 'synonyms': pl.List(pl.String)})


def _drugs(*names: str) -> list[dict]:
    return [{'drug': n, 'synonyms': []} for n in names]


def test_match_texts_excludes_hash_mismatches():
    texts = pl.DataFrame({
        'id': ['nct1', 'nct2', 'nct4'],
        'prompt': ['a', 'b', 'd'],
        'prompt_sha256': ['h1', 'h2-other', 'h4'],
    })
    extraction = pl.DataFrame({'id': ['nct1', 'nct2', 'nct3'], 'prompt_sha256': ['h1', 'h2', 'h3']})

    kept, counts = match_texts(texts, extraction)

    assert kept['id'].to_list() == ['nct1']
    assert counts == {'matched': 1, 'hash_mismatch': 1, 'missing_text': 1, 'missing_hash': 0}


def test_single_drug_names_keep_only_one_investigated_drug():
    extraction = pl.DataFrame(
        {
            'id': ['nct1', 'nct2', 'nct3', 'nct4'],
            'investigated_drugs': [_drugs('Sotorasib '), _drugs('a', 'b'), _drugs('Placebo'), []],
        },
        schema={'id': pl.String, 'investigated_drugs': pl.List(DRUG)},
    )

    assert single_drug_names(extraction).to_dicts() == [{'nct_id': 'nct1', 'drug_name': 'sotorasib'}]


def _texts(rows: dict[str, str]) -> pl.DataFrame:
    return pl.DataFrame({'id': list(rows), 'prompt': list(rows.values())})


def test_build_pools_orders_mechanism_trials_first_and_respects_budget():
    texts = _texts({
        'nct_a': 'x' * 50,
        'nct_b': 'a KRAS inhibitor ' + 'y' * 23,
        'nct_c': 'z' * 30,
    })
    trials = pl.DataFrame({'nct_id': ['nct_a', 'nct_b', 'nct_c'], 'drug_name': ['drugx'] * 3})

    row = build_pools(trials, texts, char_budget=100).row(0, named=True)

    assert row['nct_ids'] == ['nct_b', 'nct_a']
    assert row['n_trials'] == 3
    assert row['n_trials_used'] == 2
    assert row['text'].startswith('a KRAS inhibitor')


def test_build_pools_always_keeps_first_trial_over_budget():
    texts = _texts({'nct_a': 'x' * 50})
    trials = pl.DataFrame({'nct_id': ['nct_a'], 'drug_name': ['drugx']})

    assert build_pools(trials, texts, char_budget=10)['n_trials_used'].to_list() == [1]


def test_build_pools_one_pool_per_drug_with_stable_ids():
    texts = _texts({'n1': 'a', 'n2': 'b', 'n3': 'c'})
    trials = pl.DataFrame({'nct_id': ['n1', 'n2', 'n3'], 'drug_name': ['d1', 'd2', 'd1']})

    first = build_pools(trials, texts).sort('drug_name')
    second = build_pools(trials, texts).sort('drug_name')

    assert first['drug_name'].to_list() == ['d1', 'd2']
    assert first['pool_id'].to_list() == second['pool_id'].to_list()
    assert first['pool_sha256'].to_list() == second['pool_sha256'].to_list()


def test_punctuation_variants_of_a_name_share_a_pool():
    extraction = pl.DataFrame(
        {
            'id': ['nct1', 'nct2', 'nct3'],
            'investigated_drugs': [_drugs('MK-3475'), _drugs('mk3475'), _drugs('Anti-IL-20')],
        },
        schema={'id': pl.String, 'investigated_drugs': pl.List(DRUG)},
    )

    assert single_drug_names(extraction)['drug_name'].to_list() == ['mk3475', 'mk3475']


def test_rows_without_a_hash_are_counted():
    texts = pl.DataFrame({'id': ['nct1'], 'prompt': ['a'], 'prompt_sha256': ['h1']})
    extraction = pl.DataFrame({'id': ['nct1', 'nct2'], 'prompt_sha256': ['h1', None]})

    _, counts = match_texts(texts, extraction)

    assert counts == {'matched': 1, 'hash_mismatch': 0, 'missing_text': 0, 'missing_hash': 1}


def test_class_names_are_not_pools():
    extraction = pl.DataFrame(
        {
            'id': ['n1', 'n2', 'n3', 'n4', 'n5'],
            'investigated_drugs': [
                _drugs('Oral anticoagulants'),
                _drugs('Selective serotonin reuptake inhibitors'),
                _drugs('chemotherapy'),
                _drugs('Beta-blockers'),
                _drugs('pembrolizumab'),
            ],
        },
        schema={'id': pl.String, 'investigated_drugs': pl.List(DRUG)},
    )

    assert single_drug_names(extraction)['drug_name'].to_list() == ['pembrolizumab']


def test_content_key_ignores_formatting_but_not_content():
    trials = pl.DataFrame({'nct_id': ['n1', 'n2'], 'drug_name': ['d1', 'd1']})
    base = {
        'n1': 'Trial ID: n1\nOfficial Title: T1\nDescription: a KRAS inhibitor\nInterventions: a; b',
        'n2': 'Trial ID: n2\nOfficial Title: T2\nDescription: something',
    }
    reordered = base | {'n1': 'Trial ID: n1\nOfficial Title: T1\nDescription: a KRAS inhibitor\nInterventions: b; a'}
    edited = base | {'n2': 'Trial ID: n2\nOfficial Title: T2\nDescription: something else'}

    key = build_pools(trials, _texts(base))['content_key'][0]

    assert build_pools(trials, _texts(reordered))['content_key'][0] == key
    assert build_pools(trials, _texts(edited))['content_key'][0] != key


def test_texts_by_id_match_ids_whatever_the_case_and_ignore_the_hash():
    texts = pl.DataFrame({
        'id': ['nct1', 'nct2', 'nct4'],
        'prompt': ['a', 'b', 'd'],
        'prompt_sha256': ['h1', 'h2', 'h4'],
    })
    extraction = v2_frame({'NCT1': {}, 'NCT2': {}, 'NCT3': {}})

    kept, counts = texts_by_id(texts, extraction)

    assert kept['id'].to_list() == ['nct1', 'nct2']
    assert counts == {'matched': 2, 'missing_text': 1}


def test_link_single_drug_trials_take_the_text_id_and_the_normalised_drug():
    texts = _texts({'nct1': 'a', 'nct2': 'b', 'nct3': 'c'})
    extraction = v2_frame({
        'NCT1': {'links': [v2_link('Sotorasib ', 'x'), v2_link('sotorasib', 'y')]},
        'NCT2': {'links': [v2_link('a', 'x'), v2_link('b', 'x')]},
        'NCT3': {'links': [v2_link('Placebo', 'x')]},
        'NCT9': {'links': [v2_link('aspirin', 'x')]},
    })

    assert link_single_drug_trials(extraction, texts).to_dicts() == [{'nct_id': 'nct1', 'drug_name': 'sotorasib'}]
