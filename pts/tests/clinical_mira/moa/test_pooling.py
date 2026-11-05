import polars as pl

from pts.transformers.clinical_mira.moa.pooling import (
    TRIAL_SEPARATOR,
    add_papers,
    build_pools,
    content_key,
    match_texts,
    single_drug_trials,
    trial_content_sha256,
)

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

    assert single_drug_trials(extraction)['drug_name'].to_list() == ['pembrolizumab']


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


def _trial_prompt(nct: str, body: str) -> str:
    return f'Trial ID: {nct}\nOfficial Title: title {nct}\nDescription: {body}'


def _fixture() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    prompts = {
        'NCT1': _trial_prompt('NCT1', 'a KRAS inhibitor'),
        'NCT2': _trial_prompt('NCT2', 'second ' + 'x' * 80),
        'NCT3': _trial_prompt('NCT3', 'other drug trial'),
    }
    texts = pl.DataFrame({
        'id': list(prompts),
        'prompt': list(prompts.values()),
        'prompt_sha256': [str(i) for i in range(3)],
    })
    trials = pl.DataFrame({'nct_id': list(prompts), 'drug_name': ['drugx', 'drugx', 'drugy']})
    return trials, texts, build_pools(trials, texts, char_budget=60)


def _selected(*rows: tuple[str, str, str]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={'drug_name': pl.String, 'pmid': pl.String, 'nct_id': pl.String}, orient='row')


def _abstracts(*rows: tuple[str, str | None]) -> pl.DataFrame:
    return pl.DataFrame(
        [(pmid, f'T{pmid}', abstract) for pmid, abstract in rows],
        schema={'pmid': pl.String, 'title': pl.String, 'abstract': pl.String},
        orient='row',
    )


def test_content_key_matches_build_pools():
    _, texts, pools = _fixture()
    entries = [(row['id'], trial_content_sha256(row['prompt'])) for row in texts.iter_rows(named=True)]

    assert content_key('drugx', entries[:2][::-1]) == pools.filter(pl.col('drug_name') == 'drugx')['content_key'][0]
    assert content_key('drugy', entries[2:]) == pools.filter(pl.col('drug_name') == 'drugy')['content_key'][0]


def test_add_papers_leaves_pools_without_papers_unchanged():
    trials, texts, pools = _fixture()

    out, out_trials, out_texts = add_papers(
        pools, trials, texts, _selected(('drugx', '5', 'NCT1')), _abstracts(('5', 'an abstract')), 4000
    )

    drugy = pl.col('drug_name') == 'drugy'
    assert out.filter(drugy).to_dicts() == pools.filter(drugy).to_dicts()
    assert out.columns == pools.columns
    assert out['drug_name'].to_list() == pools['drug_name'].to_list()
    assert out_trials.height == trials.height + 1
    assert out_texts.height == texts.height + 1


def test_add_papers_leaves_pools_unchanged_when_nothing_is_selected():
    trials, texts, pools = _fixture()

    out, out_trials, out_texts = add_papers(pools, trials, texts, _selected(), _abstracts(), 4000)

    assert out.equals(pools)
    assert out_trials.equals(trials)
    assert out_texts.equals(texts)


def test_add_papers_appends_after_trials_and_changes_key():
    trials, texts, pools = _fixture()
    selected = _selected(('drugx', '9', 'NCT2'), ('drugx', '100', 'NCT1'), ('drugx', '20', 'NCT1'))
    abstracts = _abstracts(('9', 'nine'), ('100', 'hundred'), ('20', 'twenty'))

    out, out_trials, out_texts = add_papers(pools, trials, texts, selected, abstracts, 4000)

    old = pools.filter(pl.col('drug_name') == 'drugx').row(0, named=True)
    new = out.filter(pl.col('drug_name') == 'drugx').row(0, named=True)
    assert new['nct_ids'] == [*old['nct_ids'], 'pmid100', 'pmid20', 'pmid9']
    assert new['text'].startswith(old['text'] + TRIAL_SEPARATOR + 'Trial ID: pmid100\n')
    assert new['text'].endswith('Description: nine')
    assert new['pool_sha256'] != old['pool_sha256']
    assert new['n_trials'] == old['n_trials']
    assert new['n_trials_used'] == old['n_trials_used']

    entries = [(r['id'], trial_content_sha256(r['prompt'])) for r in texts.iter_rows(named=True) if r['id'] != 'NCT3']
    entries += [
        (r['id'], trial_content_sha256(r['prompt']))
        for r in out_texts.filter(pl.col('id').str.starts_with('pmid')).iter_rows(named=True)
    ]
    assert new['content_key'] == content_key('drugx', entries)
    assert new['content_key'] != old['content_key']

    added = out_trials.filter(pl.col('nct_id').str.starts_with('pmid'))
    assert sorted(added['nct_id'].to_list()) == ['pmid100', 'pmid20', 'pmid9']
    assert added['drug_name'].unique().to_list() == ['drugx']
    row = out_texts.filter(pl.col('id') == 'pmid9').row(0, named=True)
    assert row['prompt'] == 'Trial ID: pmid9\nOfficial Title: T9\nDescription: nine'


def test_add_papers_skips_papers_without_abstract():
    trials, texts, pools = _fixture()
    selected = _selected(('drugx', '1', 'NCT1'), ('drugx', '2', 'NCT1'), ('drugx', '3', 'NCT1'))
    abstracts = _abstracts(('1', None), ('2', ''))

    out, out_trials, out_texts = add_papers(pools, trials, texts, selected, abstracts, 4000)

    assert out.equals(pools)
    assert out_trials.equals(trials)
    assert out_texts.equals(texts)
