import polars as pl

from pts.transformers.clinical_mira.moa.pooling import build_pools, match_texts, single_drug_trials

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
    assert counts == {'matched': 1, 'hash_mismatch': 1, 'missing_text': 1}


def test_single_drug_trials_keeps_only_one_investigated_drug():
    extraction = pl.DataFrame(
        {
            'id': ['nct1', 'nct2', 'nct3', 'nct4'],
            'investigated_drugs': [_drugs('Sotorasib '), _drugs('a', 'b'), _drugs('Placebo'), []],
        },
        schema={'id': pl.String, 'investigated_drugs': pl.List(DRUG)},
    )

    assert single_drug_trials(extraction).to_dicts() == [{'nct_id': 'nct1', 'drug_name': 'sotorasib'}]


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

    assert single_drug_trials(extraction)['drug_name'].to_list() == ['mk3475', 'mk3475']
