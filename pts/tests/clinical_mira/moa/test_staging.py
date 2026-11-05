import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest

from pts.transformers.clinical_mira.moa import llm, pooling, schemas, staging
from pts.transformers.clinical_mira.moa.staging import (
    adopt_records,
    carry_forward,
    import_experiment_cache,
    parse_output,
    pool_trials,
    record_output,
    request_text,
)
from pts.transformers.clinical_mira.record_store import read_records, record_schema, text_sha256
from pts.transformers.clinical_mira.staged_outputs import MOA

PROMPT = 'You extract mechanisms.'
EXTRACTION = {'mechanisms': []}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _marked(name: str, *cited: str) -> dict:
    """An extraction with one mechanism named ``name`` whose evidence cites ``cited``."""
    return {'mechanisms': [{'mechanismOfAction': name, 'evidence': [{'nctId': c, 'quote': 'q'} for c in cited]}]}


def _inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    texts = pl.DataFrame({
        'id': ['nct1', 'nct2', 'nct3', 'nct4'],
        'prompt': [
            'Official Title: one\nPhase: 2',
            'Official Title: two\nPhase: 2',
            'Official Title: three\nPhase: 2',
            'Official Title: four\nPhase: 2',
        ],
    })
    trials = pl.DataFrame({
        'nct_id': ['nct1', 'nct2', 'nct3', 'nct4'],
        'drug_name': ['drugx', 'drugx', 'drugx', 'drugy'],
    })
    return trials, texts


def _write_cache(path: Path, pools: pl.DataFrame, system_prompt: str, which: str) -> str:
    row = pools.filter(pl.col('drug_name') == which).row(0, named=True)
    key = llm.result_key(request_text(row['drug_name'], row['text']), system_prompt, schemas.MoaExtraction, MOA.model)
    path.write_text(json.dumps({'key': key, 'request_id': row['pool_id'], 'response': EXTRACTION}) + '\n')
    return str(path)


def test_pool_trials_cover_trials_beyond_the_budget() -> None:
    trials, texts = _inputs()
    pools = pooling.build_pools(trials, texts, char_budget=10)

    listed = pool_trials(pools, texts, trials).filter(pl.col('drug_name') == 'drugx').row(0, named=True)

    assert [t['nct_id'] for t in listed['trials']] == ['nct1', 'nct2', 'nct3']
    assert listed['trials'][0]['content_sha256'] == pooling.trial_content_sha256(texts['prompt'][0])
    assert listed['content_key'] == pools.filter(pl.col('drug_name') == 'drugx')['content_key'][0]


def test_record_output_round_trips() -> None:
    trials = [{'nct_id': 'nct1', 'content_sha256': 'h'}]
    output = record_output('drugx', trials, EXTRACTION, None, None, 'abc')

    assert parse_output(output) == {
        'drug_name': 'drugx',
        'trials': trials,
        'extraction': EXTRACTION,
        'usage': None,
        'carried_from': None,
        'prompt_sha256': 'abc',
    }


def test_parse_output_accepts_an_output_without_the_prompt() -> None:
    old = json.dumps({
        'drug_name': 'drugx',
        'trials': [],
        'extraction': EXTRACTION,
        'usage': None,
        'carried_from': None,
    })

    assert parse_output(old)['prompt_sha256'] is None
    assert parse_output(old)['extraction'] == EXTRACTION


def test_import_matches_by_prompt_and_writes_content_key_records(tmp_path: Path) -> None:
    trials, texts = _inputs()
    pools = pooling.build_pools(trials, texts)
    cache = _write_cache(tmp_path / 'cache.jsonl', pools, PROMPT, 'drugx')
    store = str(tmp_path / 'store')

    written = import_experiment_cache(cache, pools, pool_trials(pools, texts, trials), PROMPT, store, cast(Any, None))

    assert written == 1
    records = read_records(store, cast(Any, None), output_dtype=pl.String())
    key = pools.filter(pl.col('drug_name') == 'drugx')['content_key'][0]
    assert records['text'].to_list() == [key]
    assert records['text_sha256'].to_list() == [text_sha256(key)]
    assert records['model'].to_list() == [MOA.name]
    assert records['model_revision'].to_list() == [MOA.tag()]
    assert records['status'].to_list() == ['ok']
    assert records['run_id'].to_list() == [f'import-moa-experiment-{MOA.digest}']
    parsed = parse_output(records['output'][0])
    assert parsed['extraction'] == EXTRACTION
    assert parsed['drug_name'] == 'drugx'
    assert [t['nct_id'] for t in parsed['trials']] == ['nct1', 'nct2', 'nct3']
    assert parsed['carried_from'] is None
    assert parsed['prompt_sha256'] == _sha256(PROMPT)


def test_import_refuses_a_cache_for_another_prompt(tmp_path: Path) -> None:
    trials, texts = _inputs()
    pools = pooling.build_pools(trials, texts)
    cache = _write_cache(tmp_path / 'cache.jsonl', pools, PROMPT, 'drugx')
    store = str(tmp_path / 'store')
    listed = pool_trials(pools, texts, trials)

    assert import_experiment_cache(cache, pools, listed, 'another prompt', store, cast(Any, None)) == 0
    assert import_experiment_cache(cache, pools, listed, PROMPT, store, cast(Any, None)) == 0


def test_import_runs_once(tmp_path: Path) -> None:
    trials, texts = _inputs()
    pools = pooling.build_pools(trials, texts)
    cache = _write_cache(tmp_path / 'cache.jsonl', pools, PROMPT, 'drugy')
    store = str(tmp_path / 'store')
    listed = pool_trials(pools, texts, trials)

    assert import_experiment_cache(cache, pools, listed, PROMPT, store, cast(Any, None)) == 1
    assert import_experiment_cache(cache, pools, listed, PROMPT, store, cast(Any, None)) == 0


def test_import_raises_when_the_cache_is_missing(tmp_path: Path) -> None:
    trials, texts = _inputs()
    pools = pooling.build_pools(trials, texts)
    store = str(tmp_path / 'store')

    with pytest.raises(Exception):  # noqa: B017
        import_experiment_cache(
            str(tmp_path / 'absent.jsonl'), pools, pool_trials(pools, texts, trials), PROMPT, store, cast(Any, None)
        )


QUIET = 'Official Title: a quiet trial\nPhase: 2'
LOUD = 'Official Title: a kinase inhibitor trial\nPhase: 2'


PREVIOUS_PROMPT_SHA = _sha256('the previous prompt')


def _record_for(
    drug: str,
    prompts: dict[str, str],
    key: str,
    extraction: dict,
    minute: int,
    tag: str | None = None,
    legacy: bool = False,
) -> pl.DataFrame:
    """A record of ``key``; a ``legacy`` one has no ``prompt_sha256`` in its output."""
    listed = [{'nct_id': n, 'content_sha256': pooling.trial_content_sha256(p)} for n, p in sorted(prompts.items())]
    output = record_output(drug, listed, extraction, {'in': 1}, None, PREVIOUS_PROMPT_SHA)
    if legacy:
        output = json.dumps({k: v for k, v in json.loads(output).items() if k != 'prompt_sha256'})
    return pl.DataFrame(
        [
            {
                'text_sha256': text_sha256(key),
                'text': key,
                'model': MOA.name,
                'model_revision': tag or MOA.tag(),
                'output': output,
                'status': 'ok',
                'error': None,
                'computed_at': datetime(2026, 1, 1, 0, minute, tzinfo=UTC),
                'run_id': 'r0',
            }
        ],
        schema=record_schema(pl.String()),
    )


def _current(prompts: dict[str, str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    texts = pl.DataFrame({'id': list(prompts), 'prompt': list(prompts.values())})
    trials = pl.DataFrame({'nct_id': list(prompts), 'drug_name': ['drugx'] * len(prompts)})
    return texts, pool_trials(pooling.build_pools(trials, texts), texts, trials)


def test_added_trial_without_vocabulary_is_carried() -> None:
    texts, current = _current({'nct1': LOUD, 'nct2': QUIET})
    records = _record_for('drugx', {'nct1': LOUD}, 'k-prev', _marked('old', 'nct1'), 1)

    out = carry_forward(current, texts, records, MOA.tag(), 'run1')

    assert out.schema == record_schema(pl.String())
    row = out.row(0, named=True)
    assert out.height == 1
    assert row['text'] == current['content_key'][0]
    assert row['text_sha256'] == text_sha256(current['content_key'][0])
    assert (row['status'], row['run_id'], row['model'], row['model_revision']) == (
        'ok',
        'run1',
        MOA.name,
        MOA.tag(),
    )
    parsed = parse_output(row['output'])
    assert parsed['extraction'] == _marked('old', 'nct1')
    assert parsed['usage'] is None
    assert parsed['carried_from'] == 'k-prev'
    assert parsed['prompt_sha256'] == PREVIOUS_PROMPT_SHA
    assert [t['nct_id'] for t in parsed['trials']] == ['nct1', 'nct2']


def test_carrying_a_record_without_a_prompt_leaves_it_null() -> None:
    texts, current = _current({'nct1': LOUD, 'nct2': QUIET})
    records = _record_for('drugx', {'nct1': LOUD}, 'k-prev', EXTRACTION, 1, legacy=True)

    out = carry_forward(current, texts, records, MOA.tag(), 'run1')

    assert out.height == 1
    assert json.loads(out['output'][0])['prompt_sha256'] is None


PAPER_OLD = 'Trial ID: pmid1\nOfficial Title: an old report\nDescription: drugx is a KRAS inhibitor.'
PAPER_NEW = 'Trial ID: pmid2\nOfficial Title: a new report\nDescription: Adults were enrolled.'


def test_a_dropped_paper_the_evidence_cites_blocks_the_carry() -> None:
    texts, current = _current({'nct1': QUIET, 'pmid2': PAPER_NEW})
    records = _record_for('drugx', {'nct1': QUIET, 'pmid1': PAPER_OLD}, 'k-prev', _marked('KRAS', 'pmid1'), 1)

    assert carry_forward(current, texts, records, MOA.tag(), 'run1').is_empty()


def test_a_dropped_paper_the_evidence_does_not_cite_is_carried() -> None:
    texts, current = _current({'nct1': QUIET, 'pmid2': PAPER_NEW})
    records = _record_for('drugx', {'nct1': QUIET, 'pmid1': PAPER_OLD}, 'k-prev', _marked('KRAS', 'nct1'), 1)

    assert carry_forward(current, texts, records, MOA.tag(), 'run1').height == 1


def test_a_removed_trial_the_evidence_cites_blocks_the_carry() -> None:
    texts, current = _current({'nct1': LOUD})
    records = _record_for('drugx', {'nct1': LOUD, 'nct2': LOUD}, 'k-prev', _marked('KRAS', 'nct1', 'nct2'), 1)

    assert carry_forward(current, texts, records, MOA.tag(), 'run1').is_empty()


def test_a_cited_id_that_was_never_an_entry_does_not_block_the_carry() -> None:
    texts, current = _current({'nct1': LOUD, 'nct2': QUIET})
    records = _record_for('drugx', {'nct1': LOUD}, 'k-prev', _marked('KRAS', 'nct1', 'pmid999'), 1)

    assert carry_forward(current, texts, records, MOA.tag(), 'run1').height == 1


def test_added_trial_with_vocabulary_is_computed() -> None:
    texts, current = _current({'nct1': QUIET, 'nct2': LOUD})
    records = _record_for('drugx', {'nct1': QUIET}, 'k-prev', EXTRACTION, 1)

    assert carry_forward(current, texts, records, MOA.tag(), 'run1').is_empty()


def test_edited_trial_with_vocabulary_is_computed() -> None:
    texts, current = _current({'nct1': LOUD.replace('trial', 'edited trial')})
    records = _record_for('drugx', {'nct1': LOUD}, 'k-prev', EXTRACTION, 1)

    assert carry_forward(current, texts, records, MOA.tag(), 'run1').is_empty()


def test_removed_trial_only_is_carried() -> None:
    texts, current = _current({'nct1': LOUD})
    records = _record_for('drugx', {'nct1': LOUD, 'nct2': LOUD}, 'k-prev', EXTRACTION, 1)

    out = carry_forward(current, texts, records, MOA.tag(), 'run1')

    assert out.height == 1
    assert [t['nct_id'] for t in parse_output(out['output'][0])['trials']] == ['nct1']


def test_pool_with_a_record_is_not_carried_again() -> None:
    texts, current = _current({'nct1': LOUD})
    records = _record_for('drugx', {'nct1': LOUD}, current['content_key'][0], EXTRACTION, 1)

    assert carry_forward(current, texts, records, MOA.tag(), 'run1').is_empty()


def test_new_drug_name_is_never_carried() -> None:
    texts, current = _current({'nct1': QUIET})
    records = _record_for('drugz', {'nct1': QUIET}, 'k-prev', EXTRACTION, 1)

    assert carry_forward(current, texts, records, MOA.tag(), 'run1').is_empty()


def test_carry_chains_from_the_latest_record() -> None:
    quiet2 = 'Official Title: another quiet one\nPhase: 1'
    texts, current = _current({'nct1': LOUD, 'nct2': QUIET, 'nct3': quiet2})
    first = _record_for('drugx', {'nct1': LOUD}, 'k1', _marked('first'), 1)
    second = _record_for('drugx', {'nct1': LOUD, 'nct2': QUIET}, 'k2', _marked('first'), 2)

    out = carry_forward(current, texts, pl.concat([second, first]), MOA.tag(), 'run1')

    parsed = parse_output(out['output'][0])
    assert parsed['carried_from'] == 'k2'
    assert parsed['extraction'] == _marked('first')
    assert [t['nct_id'] for t in parsed['trials']] == ['nct1', 'nct2', 'nct3']


def test_carry_diffs_against_the_latest_trials_not_the_first() -> None:
    texts, current = _current({'nct1': LOUD, 'nct2': LOUD})
    first = _record_for('drugx', {'nct1': LOUD}, 'k1', EXTRACTION, 1)
    second = _record_for('drugx', {'nct1': LOUD, 'nct2': LOUD}, 'k2', EXTRACTION, 2)
    current = current.with_columns(content_key=pl.lit('k3'))

    out = carry_forward(current, texts, pl.concat([first, second]), MOA.tag(), 'run1')

    assert parse_output(out['output'][0])['carried_from'] == 'k2'


def test_records_at_another_tag_are_ignored() -> None:
    texts, current = _current({'nct1': LOUD, 'nct2': QUIET})
    records = _record_for('drugx', {'nct1': LOUD}, 'k-prev', EXTRACTION, 1, tag='other-tag')

    assert carry_forward(current, texts, records, MOA.tag(), 'run1').is_empty()


def test_moa_compute_has_a_cache_and_state_per_prompt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    trials, texts = _inputs()
    pools = pooling.build_pools(trials, texts)
    listed = pool_trials(pools, texts, trials)
    keys = pools['content_key'].to_list()
    calls: list[dict[str, Any]] = []

    def fake_batch(requests, *, system_prompt, cache_path, state_uri, **kwargs):
        calls.append({
            'ids': [r[0] for r in requests],
            'prompt': system_prompt,
            'cache': cache_path,
            'state': state_uri,
        })
        extraction = schemas.MoaExtraction.model_validate({'drugName': 'x', 'notStated': True, 'mechanisms': []})
        return {r[0]: extraction for r in requests}, {}

    monkeypatch.setattr(staging, 'run_batch_cached', fake_batch)
    state = str(tmp_path / 'batches' / 's.json')

    out_b = staging.moa_compute(pools, listed, 'prompt b', state, tmp_path)([keys[1], keys[0]])
    out_a = staging.moa_compute(pools, listed, 'prompt a', state, tmp_path)([keys[0]])

    assert [c['prompt'] for c in calls] == ['prompt b', 'prompt a']
    assert [c['ids'] for c in calls] == [[keys[1], keys[0]], [keys[0]]]
    assert [c['cache'] for c in calls] == [
        tmp_path / f'moa_cache-{_sha256("prompt b")[:8]}.jsonl',
        tmp_path / f'moa_cache-{_sha256("prompt a")[:8]}.jsonl',
    ]
    assert [c['state'] for c in calls] == [
        str(tmp_path / 'batches' / f's-{_sha256("prompt b")[:8]}.json'),
        str(tmp_path / 'batches' / f's-{_sha256("prompt a")[:8]}.json'),
    ]
    assert [parse_output(o)['drug_name'] for o, _ in out_b] == [
        pools.filter(pl.col('content_key') == k)['drug_name'][0] for k in [keys[1], keys[0]]
    ]
    assert all(error is None for _, error in out_b + out_a)
    assert {parse_output(o)['prompt_sha256'] for o, _ in out_b} == {_sha256('prompt b')}
    assert parse_output(out_a[0][0])['prompt_sha256'] == _sha256('prompt a')


def test_adopt_records_copies_only_listed_keys_and_skips_done() -> None:
    old = pl.concat([
        _record_for('drugx', {'nct1': LOUD}, 'k-a', {'mechanisms': ['a']}, 1, tag='old'),
        _record_for('drugx', {'nct1': LOUD}, 'k-b', {'mechanisms': ['b']}, 2, tag='old'),
        _record_for('drugx', {'nct1': LOUD}, 'k-c', {'mechanisms': ['c']}, 3, tag='old'),
        _record_for('drugx', {'nct1': LOUD}, 'k-d', {'mechanisms': ['d']}, 4, tag='other'),
    ])
    done = _record_for('drugx', {'nct1': LOUD}, 'k-b', {'mechanisms': ['b']}, 5, tag='new')

    out = adopt_records(pl.concat([old, done]), ['k-a', 'k-b', 'k-d'], 'old', 'new', 'run9')

    assert out.schema == record_schema(pl.String())
    assert out['text'].to_list() == ['k-a']
    row = out.row(0, named=True)
    assert (row['model_revision'], row['run_id'], row['status']) == ('new', 'run9', 'ok')
    assert parse_output(row['output'])['extraction'] == {'mechanisms': ['a']}
    assert parse_output(row['output'])['prompt_sha256'] == PREVIOUS_PROMPT_SHA
