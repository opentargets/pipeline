import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest
from loguru import logger

from pts.transformers.clinical_mira.moa import literature, literature_staging, llm, schemas
from pts.transformers.clinical_mira.moa.literature_staging import (
    carry_forward,
    import_experiment_cache,
    import_run_id,
    parse_output,
    record_output,
)
from pts.transformers.clinical_mira.record_store import read_records, record_schema, run_exists, text_sha256
from pts.transformers.clinical_mira.staged_outputs import MOA_LITERATURE
from pts.transformers.clinical_mira.staging_errors import SeedTooSmall

PROMPT = 'You extract mechanisms from sentences.'
EXTRACTION = {'drugName': 'x', 'notStated': True, 'mechanisms': []}
SENTENCE = pl.Struct({'doc': pl.String, 'sha256': pl.String, 'cue': pl.Boolean})
DOC = pl.Struct({'doc': pl.String, 'text': pl.String})


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _marked(name: str, *cited: str) -> dict:
    """An extraction with one mechanism named ``name`` whose evidence cites ``cited``."""
    return {'mechanisms': [{'mechanismOfAction': name, 'evidence': [{'nctId': c, 'quote': 'q'} for c in cited]}]}


def _sentences(*items: tuple[str, str, bool]) -> list[dict]:
    """Sentences ``(doc, text, cue)`` as ``{doc, sha256, cue}`` sorted by ``(doc, sha256)``."""
    out = [{'doc': d, 'sha256': _sha256(t), 'cue': c} for d, t, c in items]
    return sorted(out, key=lambda s: (s['doc'], s['sha256']))


def _requests(drugs: dict[str, list[dict]]) -> pl.DataFrame:
    """A frame shaped like :py:func:`literature.requests` for ``{parent: sentences}``."""
    rows = []
    for parent, sentences in drugs.items():
        docs = sorted({s['doc'] for s in sentences})
        rows.append({
            'parent': parent,
            'name': parent.lower(),
            'request': f'Drug: {parent.lower()}\n\n' + '\n'.join(s['sha256'] for s in sentences),
            'content_key': literature.content_key(parent, sentences),
            'sentences': sentences,
            'docs': [{'doc': d, 'text': 't'} for d in docs],
        })
    return pl.DataFrame(
        rows,
        schema={
            'parent': pl.String,
            'name': pl.String,
            'request': pl.String,
            'content_key': pl.String,
            'sentences': pl.List(SENTENCE),
            'docs': pl.List(DOC),
        },
    )


def _seed_row(parent: str, key: str, prompt: str = PROMPT) -> dict:
    return {
        'content_key': key,
        'parent': parent,
        'response': EXTRACTION,
        'usage': {'input_tokens': 3},
        'prompt_sha256': _sha256(prompt),
        'run': 'pilot',
    }


def _write_seed(path: Path, rows: list[dict]) -> str:
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return str(path)


def test_record_output_round_trips() -> None:
    sentences = _sentences(('pmid1', 'a sentence', True))
    output = record_output('CHEMBL1', sentences, EXTRACTION, None, None, 'abc')

    assert parse_output(output) == {
        'parent': 'CHEMBL1',
        'sentences': sentences,
        'extraction': EXTRACTION,
        'usage': None,
        'carried_from': None,
        'prompt_sha256': 'abc',
    }


def test_import_run_id_names_the_literature_digest() -> None:
    assert import_run_id() == f'import-moa-literature-experiment-{MOA_LITERATURE.digest}'


def _import_inputs(tmp_path: Path) -> tuple[pl.DataFrame, str]:
    matched = _sentences(('pmid1', 'matched sentence', True))
    changed = _sentences(('pmid2', 'changed sentence now', False))
    other = _sentences(('pmid3', 'other prompt sentence', True))
    current = _requests({'CHEMBL1': matched, 'CHEMBL2': changed, 'CHEMBL3': other})
    old_changed = _sentences(('pmid2', 'changed sentence before', False))
    seed = _write_seed(
        tmp_path / 'seed.jsonl',
        [
            _seed_row('CHEMBL1', literature.content_key('CHEMBL1', matched)),
            _seed_row('CHEMBL2', literature.content_key('CHEMBL2', old_changed)),
            _seed_row('CHEMBL3', literature.content_key('CHEMBL3', other), prompt='another prompt'),
        ],
    )
    return current, seed


def test_import_matches_only_equal_content_keys(tmp_path: Path) -> None:
    current, seed = _import_inputs(tmp_path)
    store = str(tmp_path / 'store')

    written = import_experiment_cache(seed, current, PROMPT, store, cast(Any, None))

    assert written == 1
    records = read_records(store, cast(Any, None), output_dtype=pl.String())
    key = current.filter(pl.col('parent') == 'CHEMBL1')['content_key'][0]
    assert records['text'].to_list() == [key]
    assert records['text_sha256'].to_list() == [text_sha256(key)]
    assert records['model'].to_list() == [MOA_LITERATURE.name]
    assert records['model_revision'].to_list() == [MOA_LITERATURE.tag()]
    assert records['status'].to_list() == ['ok']
    assert records['run_id'].to_list() == [import_run_id()]
    parsed = parse_output(records['output'][0])
    assert parsed['parent'] == 'CHEMBL1'
    assert parsed['sentences'] == current.filter(pl.col('parent') == 'CHEMBL1')['sentences'].to_list()[0]
    assert parsed['extraction'] == EXTRACTION
    assert parsed['usage'] == {'input_tokens': 3}
    assert parsed['carried_from'] is None
    assert parsed['prompt_sha256'] == _sha256(PROMPT)


def test_import_is_idempotent(tmp_path: Path) -> None:
    current, seed = _import_inputs(tmp_path)
    store = str(tmp_path / 'store')

    assert import_experiment_cache(seed, current, PROMPT, store, cast(Any, None)) == 1
    assert import_experiment_cache(seed, current, PROMPT, store, cast(Any, None)) == 0


def test_import_below_min_share_raises_and_writes_no_run(tmp_path: Path) -> None:
    current, seed = _import_inputs(tmp_path)
    store = str(tmp_path / 'store')

    with pytest.raises(SeedTooSmall, match='1 of 3'):
        import_experiment_cache(seed, current, PROMPT, store, cast(Any, None), min_share=0.5)

    assert not run_exists(store, import_run_id(), cast(Any, None))
    assert import_experiment_cache(seed, current, PROMPT, store, cast(Any, None), min_share=0.3) == 1


def test_import_rejects_duplicate_content_keys(tmp_path: Path) -> None:
    sentences = _sentences(('pmid1', 'matched sentence', True))
    current = _requests({'CHEMBL1': sentences})
    key = literature.content_key('CHEMBL1', sentences)
    seed = _write_seed(tmp_path / 'seed.jsonl', [_seed_row('CHEMBL1', key), _seed_row('CHEMBL1', key)])

    with pytest.raises(ValueError, match=r'CHEMBL1\|'):
        import_experiment_cache(seed, current, PROMPT, str(tmp_path / 'store'), cast(Any, None))


def test_import_rejects_a_malformed_seed_response(tmp_path: Path) -> None:
    sentences = _sentences(('pmid1', 'matched sentence', True))
    current = _requests({'CHEMBL1': sentences})
    row = _seed_row('CHEMBL1', literature.content_key('CHEMBL1', sentences)) | {'response': {'mechanisms': 'x'}}
    seed = _write_seed(tmp_path / 'seed.jsonl', [row])

    with pytest.raises(ValueError, match='CHEMBL1'):
        import_experiment_cache(seed, current, PROMPT, str(tmp_path / 'store'), cast(Any, None))


def test_import_logs_seed_rows_without_a_current_request(tmp_path: Path) -> None:
    current, _ = _import_inputs(tmp_path)
    absent = _sentences(('pmid9', 'absent sentence', True))
    seed = _write_seed(tmp_path / 'seed.jsonl', [_seed_row('CHEMBL9', literature.content_key('CHEMBL9', absent))])
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level='INFO')
    try:
        assert import_experiment_cache(seed, current, PROMPT, str(tmp_path / 'store'), cast(Any, None)) == 0
    finally:
        logger.remove(sink)

    assert any('1 without a current request' in m for m in messages)


SENTENCE_INPUT = {
    'parent': pl.String,
    'mappedId': pl.String,
    'pub': pl.String,
    'pmid': pl.String,
    'pmcid': pl.String,
    'section': pl.String,
    'year': pl.Int64,
    'label': pl.String,
    'text': pl.String,
    'h': pl.UInt64,
    'cue': pl.Boolean,
    'srank': pl.Int8,
    'genes': pl.List(pl.String),
}
NAMES = pl.DataFrame({'parent': ['CHEMBL1'], 'name': ['Aspirin']})


def _real_requests(*items: tuple[str, str, bool]) -> pl.DataFrame:
    """:py:func:`literature.requests` of CHEMBL1 sentences ``(pmid, text, cue)``."""
    rows = [
        {
            'parent': 'CHEMBL1',
            'mappedId': 'CHEMBL1',
            'pub': pmid,
            'pmid': pmid,
            'pmcid': None,
            'section': 'abstract',
            'year': 2020,
            'label': 'Aspirin',
            'text': text,
            'h': i,
            'cue': cue,
            'srank': 1,
            'genes': None,
        }
        for i, (pmid, text, cue) in enumerate(items)
    ]
    return literature.requests(literature.select_sentences(pl.DataFrame(rows, schema=SENTENCE_INPUT)), NAMES)


def test_import_and_carry_forward_accept_real_requests(tmp_path: Path) -> None:
    kept = ('1', 'aspirin is an inhibitor of cyclooxygenase enzymes', True)
    first = _real_requests(kept)
    assert first.schema == _requests({'CHEMBL1': _sentences(('pmid1', 'x', True))}).schema
    store = str(tmp_path / 'store')
    seed = _write_seed(tmp_path / 'seed.jsonl', [_seed_row('CHEMBL1', first['content_key'][0])])

    assert import_experiment_cache(seed, first, PROMPT, store, cast(Any, None)) == 1
    imported = read_records(store, cast(Any, None), output_dtype=pl.String())
    assert parse_output(imported['output'][0])['sentences'] == first['sentences'].to_list()[0]

    second = _real_requests(kept, ('2', 'aspirin was given to patients after surgery', False))
    previous = imported.with_columns(run_id=pl.lit('r0'))
    out = carry_forward(second, previous, MOA_LITERATURE.tag(), 'run1')

    assert out['text'].to_list() == second['content_key'].to_list()
    assert parse_output(out['output'][0])['carried_from'] == first['content_key'][0]


def test_carry_forward_breaks_computed_at_ties_by_run_id() -> None:
    current = _requests({'CHEMBL1': _sentences(KEPT, QUIET)})
    later = _record('CHEMBL1', _sentences(KEPT), 'k-b', EXTRACTION, 1, run_id='rb')
    earlier = _record('CHEMBL1', _sentences(KEPT), 'k-a', EXTRACTION, 1, run_id='ra')

    for records in (pl.concat([later, earlier]), pl.concat([earlier, later])):
        out = carry_forward(current, records, MOA_LITERATURE.tag(), 'run1')
        assert parse_output(out['output'][0])['carried_from'] == 'k-b'


def _record(
    parent: str,
    sentences: list[dict],
    key: str,
    extraction: dict,
    minute: int,
    run_id: str = 'r0',
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                'text_sha256': text_sha256(key),
                'text': key,
                'model': MOA_LITERATURE.name,
                'model_revision': MOA_LITERATURE.tag(),
                'output': record_output(parent, sentences, extraction, {'in': 1}, None, 'prev-sha'),
                'status': 'ok',
                'error': None,
                'computed_at': datetime(2026, 1, 1, 0, minute, tzinfo=UTC),
                'run_id': run_id,
            }
        ],
        schema=record_schema(pl.String()),
    )


KEPT = ('pmid1', 'an inhibitor sentence', True)
QUIET = ('pmid2', 'a quiet new sentence', False)
LOUD = ('pmid2', 'a new antagonist sentence', True)


def test_carry_forward_when_new_sentences_have_no_cue() -> None:
    current = _requests({'CHEMBL1': _sentences(KEPT, QUIET)})
    records = _record('CHEMBL1', _sentences(KEPT), 'k-prev', _marked('old', 'pmid1'), 1)

    out = carry_forward(current, records, MOA_LITERATURE.tag(), 'run1')

    assert out.schema == record_schema(pl.String())
    assert out.height == 1
    row = out.row(0, named=True)
    assert row['text'] == current['content_key'][0]
    assert row['text_sha256'] == text_sha256(current['content_key'][0])
    assert (row['status'], row['run_id'], row['model'], row['model_revision']) == (
        'ok',
        'run1',
        MOA_LITERATURE.name,
        MOA_LITERATURE.tag(),
    )
    parsed = parse_output(row['output'])
    assert parsed['extraction'] == _marked('old', 'pmid1')
    assert parsed['usage'] is None
    assert parsed['carried_from'] == 'k-prev'
    assert parsed['prompt_sha256'] == 'prev-sha'
    assert parsed['sentences'] == _sentences(KEPT, QUIET)


def test_no_carry_forward_when_a_new_sentence_has_a_cue() -> None:
    current = _requests({'CHEMBL1': _sentences(KEPT, LOUD)})
    records = _record('CHEMBL1', _sentences(KEPT), 'k-prev', EXTRACTION, 1)

    assert carry_forward(current, records, MOA_LITERATURE.tag(), 'run1').is_empty()


def test_no_carry_forward_from_seed_records() -> None:
    current = _requests({'CHEMBL1': _sentences(KEPT, QUIET)})
    records = _record('CHEMBL1', _sentences(KEPT), 'k-seed', EXTRACTION, 1, run_id=import_run_id())

    assert carry_forward(current, records, MOA_LITERATURE.tag(), 'run1').is_empty()


def test_carry_forward_skips_seed_records_for_an_earlier_run() -> None:
    current = _requests({'CHEMBL1': _sentences(KEPT, QUIET)})
    earlier = _record('CHEMBL1', _sentences(KEPT), 'k-run', EXTRACTION, 1)
    seed = _record('CHEMBL1', _sentences(KEPT), 'k-seed', EXTRACTION, 2, run_id=import_run_id())

    out = carry_forward(current, pl.concat([seed, earlier]), MOA_LITERATURE.tag(), 'run1')

    assert parse_output(out['output'][0])['carried_from'] == 'k-run'


def test_no_carry_forward_when_a_cited_doc_left_the_selection() -> None:
    current = _requests({'CHEMBL1': _sentences(KEPT)})
    records = _record('CHEMBL1', _sentences(KEPT, QUIET), 'k-prev', _marked('X', 'pmid1', 'pmid2'), 1)

    assert carry_forward(current, records, MOA_LITERATURE.tag(), 'run1').is_empty()


def test_carry_forward_when_an_uncited_doc_left_the_selection() -> None:
    current = _requests({'CHEMBL1': _sentences(KEPT)})
    records = _record('CHEMBL1', _sentences(KEPT, QUIET), 'k-prev', _marked('X', 'pmid1'), 1)

    assert carry_forward(current, records, MOA_LITERATURE.tag(), 'run1').height == 1


def test_no_carry_forward_for_a_content_key_with_a_record() -> None:
    current = _requests({'CHEMBL1': _sentences(KEPT)})
    records = _record('CHEMBL1', _sentences(KEPT), current['content_key'][0], EXTRACTION, 1)

    assert carry_forward(current, records, MOA_LITERATURE.tag(), 'run1').is_empty()


def test_compute_records_usage_and_prompt_sha(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    current = _requests({'CHEMBL1': _sentences(KEPT), 'CHEMBL2': _sentences(QUIET)})
    keys = current['content_key'].to_list()
    requests_by_key = dict(zip(keys, current['request'].to_list(), strict=True))
    calls: list[dict[str, Any]] = []

    def fake_batch(requests, *, system_prompt, response_model, model, effort, cache_path, state_uri, **kwargs):
        calls.append({
            'requests': requests,
            'prompt': system_prompt,
            'model': model,
            'effort': effort,
            'cache': cache_path,
            'state': state_uri,
        })
        with cache_path.open('w') as f:
            for _, text in requests:
                key = llm.result_key(text, system_prompt, response_model, model)
                f.write(json.dumps({'key': key, 'usage': {'input_tokens': len(text)}}) + '\n')
        extraction = schemas.MoaExtraction.model_validate(EXTRACTION)
        return {r[0]: extraction for r in requests[:1]}, {requests[1][0]: 'boom'}

    monkeypatch.setattr(literature_staging, 'run_batch_cached', fake_batch)
    state = str(tmp_path / 'batches' / 's.json')

    out = literature_staging.literature_compute(current, PROMPT, state, tmp_path)([keys[1], keys[0]])

    assert calls == [
        {
            'requests': [(keys[1], requests_by_key[keys[1]]), (keys[0], requests_by_key[keys[0]])],
            'prompt': PROMPT,
            'model': MOA_LITERATURE.model,
            'effort': MOA_LITERATURE.params['effort'],
            'cache': tmp_path / 'moa_literature_cache.jsonl',
            'state': state,
        }
    ]
    output, error = out[0]
    assert error is None
    parsed = parse_output(output)
    assert parsed['parent'] == 'CHEMBL2'
    assert parsed['sentences'] == _sentences(QUIET)
    assert parsed['usage'] == {'input_tokens': len(requests_by_key[keys[1]])}
    assert parsed['prompt_sha256'] == _sha256(PROMPT)
    assert parsed['carried_from'] is None
    assert out[1] == (None, 'boom')
