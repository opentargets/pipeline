import json
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest

from pts.transformers.clinical_mira.moa import llm, pooling, schemas
from pts.transformers.clinical_mira.moa.staging import (
    import_experiment_cache,
    parse_output,
    pool_trials,
    record_output,
)
from pts.transformers.clinical_mira.record_store import read_records, text_sha256
from pts.transformers.clinical_mira.staged_outputs import MOA

PROMPT = 'You extract mechanisms.'
EXTRACTION = {'mechanisms': []}


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
    key = llm.result_key(row['text'], system_prompt, schemas.MoaExtraction, MOA.model)
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
    output = record_output('drugx', trials, EXTRACTION, None, None)

    assert parse_output(output) == {
        'drug_name': 'drugx',
        'trials': trials,
        'extraction': EXTRACTION,
        'usage': None,
        'carried_from': None,
    }


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
