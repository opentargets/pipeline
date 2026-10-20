"""Tests for the append-only record store and write-once snapshots."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from pts import record_store
from pts.record_store import (
    RECORD_SCHEMA,
    ComputeAborted,
    RecordStoreError,
    SnapshotConflictError,
    map_texts,
    publish_snapshot,
    read_records,
    text_sha256,
    write_run,
)

MODEL = 'org/model'
REVISION = 'rev1'


class Recorder:
    """A compute function that remembers what it was asked to compute.

    Texts in ``fails`` come back as failures, which is how a model reports one.
    """

    def __init__(self, fails: set[str] | None = None, raises: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.fails = fails or set()
        self.raises = raises

    def __call__(self, texts: list[str]) -> list[tuple[list[str] | None, str | None]]:
        self.calls.append(list(texts))
        if self.raises:
            msg = 'model exploded'
            raise RuntimeError(msg)
        return [(None, 'bad text') if t in self.fails else ([t.upper()], None) for t in texts]

    @property
    def computed(self) -> list[str]:
        return [t for call in self.calls for t in call]


@pytest.fixture
def store_uri(tmp_path):
    return str(tmp_path / 'store')


def run(texts, compute, store_uri, run_id, revision=REVISION, **kwargs):
    return map_texts(
        pl.Series('text', texts, dtype=pl.String), compute, store_uri, MODEL, revision, run_id, None, **kwargs
    )


def record(run_id, output, when, text='a'):
    return pl.DataFrame(
        {
            'text_sha256': [text_sha256(text)],
            'text': [text],
            'model': [MODEL],
            'model_revision': [REVISION],
            'output': [[output]],
            'status': ['ok'],
            'error': [None],
            'computed_at': [when],
            'run_id': [run_id],
        },
        schema=RECORD_SCHEMA,
    )


def test_text_sha256_is_sha256_of_utf8():
    assert text_sha256('café') == hashlib.sha256('café'.encode()).hexdigest()


def test_first_run_computes_each_distinct_text_once(store_uri):
    compute = Recorder()
    out = run(['b', 'a', 'b', ''], compute, store_uri, 'run1')

    assert compute.calls == [['', 'a', 'b']]
    assert out.columns == ['text', 'text_sha256', 'output']
    assert out['text'].to_list() == ['', 'a', 'b']
    assert out['output'].to_list() == [[''], ['A'], ['B']]
    assert out['text_sha256'].to_list() == [text_sha256(t) for t in ['', 'a', 'b']]


def test_second_run_computes_nothing(store_uri):
    run(['a', 'b'], Recorder(), store_uri, 'run1')

    compute = Recorder()
    out = run(['a', 'b'], compute, store_uri, 'run2')

    assert compute.calls == []
    assert out['output'].to_list() == [['A'], ['B']]
    assert not Path(store_uri, 'runs', 'run2').exists()


def test_new_revision_recomputes_and_keeps_old_records(store_uri):
    run(['a'], Recorder(), store_uri, 'run1')

    compute = Recorder()
    run(['a'], compute, store_uri, 'run2', revision='rev2')

    assert compute.computed == ['a']
    records = read_records(store_uri, None)
    assert sorted(records['model_revision'].to_list()) == ['rev1', 'rev2']


def test_failed_text_is_recorded_and_the_run_fails(store_uri):
    with pytest.raises(RecordStoreError, match=r'1 text'):
        run(['a', 'b'], Recorder(fails={'b'}), store_uri, 'run1')

    records = read_records(store_uri, None).sort('text')
    assert records['status'].to_list() == ['ok', 'failed']
    assert records['error'].to_list() == [None, 'bad text']
    assert records['output'].to_list() == [['A'], None]


def test_failed_text_is_retried_and_succeeds(store_uri):
    with pytest.raises(RecordStoreError):
        run(['a', 'b'], Recorder(fails={'b'}), store_uri, 'run1')

    compute = Recorder()
    out = run(['a', 'b'], compute, store_uri, 'run2')

    assert compute.computed == ['b']
    assert out['output'].to_list() == [['A'], ['B']]


def test_retries_stop_after_max_attempts(store_uri):
    for i in range(3):
        with pytest.raises(RecordStoreError):
            run(['a', 'b'], Recorder(fails={'b'}), store_uri, f'run{i}')

    compute = Recorder()
    with pytest.raises(RecordStoreError, match=r'1 text'):
        run(['a', 'b'], compute, store_uri, 'run3')

    assert compute.computed == []
    assert not Path(store_uri, 'runs', 'run3').exists()


def test_error_names_the_count_and_at_most_five_examples(store_uri):
    texts = [f't{i}' for i in range(8)]
    with pytest.raises(RecordStoreError) as exc:
        run(texts, Recorder(fails=set(texts)), store_uri, 'run1')

    message = str(exc.value)
    assert '8 texts' in message
    assert 't4' in message
    assert 't5' not in message


def test_compute_raising_fails_every_text_of_the_call(store_uri):
    with pytest.raises(RecordStoreError, match=r'2 texts'):
        run(['a', 'b'], Recorder(raises=True), store_uri, 'run1')

    records = read_records(store_uri, None)
    assert records['status'].to_list() == ['failed', 'failed']
    assert records['error'].to_list() == ['model exploded', 'model exploded']


def test_run_without_manifest_is_ignored(store_uri):
    run(['a'], Recorder(), store_uri, 'run1')
    Path(store_uri, 'runs', 'run1', 'manifest.json').unlink()

    assert read_records(store_uri, None).is_empty()

    compute = Recorder()
    run(['a'], compute, store_uri, 'run1')
    assert compute.computed == ['a']


def test_empty_store_reads_as_empty_frame(store_uri):
    records = read_records(store_uri, None)

    assert records.is_empty()
    assert records.schema == RECORD_SCHEMA


def test_write_run_refuses_an_existing_run(store_uri):
    records = record('run1', 'A', datetime(2026, 1, 1, tzinfo=UTC))
    write_run(records, store_uri, 'run1', None)

    manifest = json.loads(Path(store_uri, 'runs', 'run1', 'manifest.json').read_text())
    assert manifest['run_id'] == 'run1'
    assert manifest['ok'] == 1
    assert manifest['failed'] == 0
    assert manifest['models'] == [{'model': MODEL, 'model_revision': REVISION}]
    assert 'written_at' in manifest

    with pytest.raises(RecordStoreError, match='run1'):
        write_run(records, store_uri, 'run1', None)


def test_newest_ok_wins_over_older_ok(store_uri):
    write_run(record('run1', 'old', datetime(2026, 1, 1, tzinfo=UTC)), store_uri, 'run1', None)
    write_run(record('run2', 'new', datetime(2026, 2, 1, tzinfo=UTC)), store_uri, 'run2', None)

    out = run(['a'], Recorder(), store_uri, 'run3')
    assert out['output'].to_list() == [['new']]


def test_null_text_is_rejected(store_uri):
    with pytest.raises(ValueError, match='null'):
        run(['a', None], Recorder(), store_uri, 'run1')


def snapshot(rows=('a', 'b')) -> pl.DataFrame:
    return pl.DataFrame({'text': list(rows), 'output': [[r.upper()] for r in rows]})


def test_publish_writes_data_then_metadata(tmp_path):
    uri = str(tmp_path / 'snap')

    assert publish_snapshot(snapshot(), uri, {'version': '1'}, None) is True

    assert pl.read_parquet(Path(uri, 'data.parquet')).equals(snapshot())
    assert json.loads(Path(uri, 'metadata.json').read_text()) == {'version': '1'}
    assert Path(uri, 'data.parquet').stat().st_mtime_ns <= Path(uri, 'metadata.json').stat().st_mtime_ns


def test_publish_identical_is_a_noop(tmp_path):
    uri = str(tmp_path / 'snap')
    publish_snapshot(snapshot(), uri, {'version': '1'}, None)
    before = Path(uri, 'metadata.json').stat().st_mtime_ns

    assert publish_snapshot(snapshot(('b', 'a')), uri, {'version': '1'}, None) is False
    assert Path(uri, 'metadata.json').stat().st_mtime_ns == before


def test_publish_refuses_different_data(tmp_path):
    uri = str(tmp_path / 'snap')
    publish_snapshot(snapshot(), uri, {'version': '1'}, None)

    with pytest.raises(SnapshotConflictError):
        publish_snapshot(snapshot(('a', 'c')), uri, {'version': '1'}, None)


def test_publish_refuses_different_metadata(tmp_path):
    uri = str(tmp_path / 'snap')
    publish_snapshot(snapshot(), uri, {'model_revision': 'rev1'}, None)

    with pytest.raises(SnapshotConflictError):
        publish_snapshot(snapshot(), uri, {'model_revision': 'rev2'}, None)


def test_publish_snapshot_with_a_trailing_slash_writes_single_slash_names(tmp_path, monkeypatch):
    uris = []
    handle = record_store.StorageHandle
    monkeypatch.setattr(record_store, 'StorageHandle', lambda uri, **kw: uris.append(uri) or handle(uri, **kw))

    assert publish_snapshot(snapshot(), f'{tmp_path}/snap/', {'version': '1'}, None) is True

    assert sorted(p.name for p in (tmp_path / 'snap').iterdir()) == ['data.parquet', 'metadata.json']
    assert publish_snapshot(snapshot(), f'{tmp_path}/snap', {'version': '1'}, None) is False
    assert not [u for u in uris if '//' in u.removeprefix(str(tmp_path))]


class Crash(BaseException):
    """Stands in for a process dying, which `map_texts` must not catch."""


class JsonRecorder(Recorder):
    """A compute function whose outputs are JSON strings, as an LLM extraction's are.

    A text in ``fail_times`` fails that many times before it succeeds.
    """

    def __init__(self, fails=None, fail_times=None, crash_on=None) -> None:
        super().__init__(fails)
        self.fail_times = fail_times or {}
        self.crash_on = crash_on

    def __call__(self, texts):
        before = self.computed
        self.calls.append(list(texts))
        if self.crash_on in texts:
            raise Crash
        return [
            (None, 'bad text')
            if t in self.fails or before.count(t) < self.fail_times.get(t, 0)
            else (json.dumps({'text': t}), None)
            for t in texts
        ]


def test_list_and_string_outputs_round_trip(store_uri):
    listed = run(['a', 'b'], Recorder(), store_uri + '/list', 'r1')
    strings = run(['a', 'b'], JsonRecorder(), store_uri + '/json', 'r1', output_dtype=pl.String)

    assert listed['output'].to_list() == [['A'], ['B']]
    assert strings['output'].to_list() == ['{"text": "a"}', '{"text": "b"}']
    assert read_records(store_uri + '/json', None, pl.String).schema == record_store.record_schema(pl.String)
    again = JsonRecorder()
    assert run(['a', 'b'], again, store_uri + '/json', 'r2', output_dtype=pl.String).equals(strings)
    assert again.calls == []


def test_tolerated_failures_retry_in_run_then_publish_the_rest(store_uri):
    compute = JsonRecorder(fails={'bad'})

    result = run(['good', 'bad'], compute, store_uri, 'r1', output_dtype=pl.String, allow_failures=True)

    assert result['text'].to_list() == ['good']
    assert compute.calls == [['bad', 'good'], ['bad'], ['bad']]
    again = JsonRecorder(fails={'bad'})
    assert run(['good', 'bad'], again, store_uri, 'r2', output_dtype=pl.String, allow_failures=True).equals(result)
    assert again.calls == []


def test_a_later_success_in_the_same_run_counts(store_uri):
    compute = JsonRecorder(fail_times={'a': 1})

    result = run(['a', 'b'], compute, store_uri, 'r1', output_dtype=pl.String, allow_failures=True)

    assert result['text'].to_list() == ['a', 'b']
    assert compute.calls == [['a', 'b'], ['a']]
    records = read_records(store_uri, None, pl.String)
    assert records.filter(pl.col('text') == 'a')['status'].sort().to_list() == ['failed', 'ok']
    assert set(records['run_id']) == {'r1'}


def test_earlier_failures_count_toward_the_attempts_of_a_tolerant_run(store_uri):
    with pytest.raises(RecordStoreError):
        run(['bad'], JsonRecorder(fails={'bad'}), store_uri, 'r1', output_dtype=pl.String, max_attempts=2)
    compute = JsonRecorder(fails={'bad'})

    result = run(['bad'], compute, store_uri, 'r2', output_dtype=pl.String, allow_failures=True, max_attempts=3)

    assert result.height == 0
    assert compute.calls == [['bad'], ['bad']]


def test_finished_shards_survive_a_crash(store_uri):
    texts = [f't{i}' for i in range(6)]
    with pytest.raises(Crash):
        run(texts, JsonRecorder(crash_on='t4'), store_uri, 'r1', output_dtype=pl.String, shard_size=2)

    assert sorted(p.name for p in Path(store_uri, 'runs').iterdir()) == ['r1-0000', 'r1-0001']
    rerun = JsonRecorder()
    result = run(texts, rerun, store_uri, 'r2', output_dtype=pl.String, shard_size=2)

    assert rerun.computed == ['t4', 't5']
    assert result['text'].to_list() == texts
    assert sorted(p.name for p in Path(store_uri, 'runs').iterdir()) == ['r1-0000', 'r1-0001', 'r2-0000']


def test_compute_aborted_writes_nothing_for_that_call(store_uri):
    texts = [f't{i}' for i in range(4)]

    def abort_on_t2(batch):
        if 't2' in batch:
            msg = 'no key'
            raise ComputeAborted(msg)
        return [(json.dumps({'text': t}), None) for t in batch]

    with pytest.raises(ComputeAborted, match='no key'):
        run(texts, abort_on_t2, store_uri, 'r1', output_dtype=pl.String, allow_failures=True, shard_size=2)

    assert sorted(p.name for p in Path(store_uri, 'runs').iterdir()) == ['r1-0000']
    assert read_records(store_uri, None, pl.String)['text'].to_list() == ['t0', 't1']


def test_strict_store_is_unchanged(store_uri):
    with pytest.raises(RecordStoreError):
        run(['a', 'bad'], Recorder(fails={'bad'}), store_uri, 'r1')
    assert sorted(p.name for p in Path(store_uri, 'runs').iterdir()) == ['r1']
