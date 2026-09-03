"""Tests for the content-addressed cache."""

import polars as pl
import pytest

from pts.result_cache import cache_key, cached_map, read_cache


def records(*keys: str) -> pl.DataFrame:
    """Build the input frame cached_map expects."""
    return pl.DataFrame({'cache_key': list(keys), 'payload': [f'payload-{k}' for k in keys]})


class Recorder:
    """A compute function that remembers what it was asked to compute.

    Anything in ``fails`` is left out of the result, which is how a compute
    function reports a failure.
    """

    def __init__(self, fails: set[str] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fails = fails or set()

    def __call__(self, shard: pl.DataFrame) -> pl.DataFrame:
        keys = shard['cache_key'].to_list()
        self.calls.append(keys)
        got = [k for k in keys if k not in self.fails]
        return pl.DataFrame({'cache_key': got, 'result': [f'result-{k}' for k in got]})

    @property
    def computed(self) -> list[str]:
        return [k for call in self.calls for k in call]


@pytest.fixture
def cache_uri(tmp_path):
    return str(tmp_path / 'cache')


class TestCacheKey:
    def test_is_stable_and_order_sensitive(self):
        assert cache_key('a', 'b') == cache_key('a', 'b')
        assert cache_key('a', 'b') != cache_key('b', 'a')

    def test_every_part_changes_the_key(self):
        base = cache_key('model', 'prompt', 'schema')
        assert base != cache_key('model2', 'prompt', 'schema')
        assert base != cache_key('model', 'prompt2', 'schema')
        assert base != cache_key('model', 'prompt', 'schema2')

    def test_parts_cannot_be_confused_by_concatenation(self):
        # without a separator 'ab' + 'c' and 'a' + 'bc' would collide
        assert cache_key('ab', 'c') != cache_key('a', 'bc')


class TestCachedMap:
    def test_empty_cache_computes_everything(self, cache_uri):
        compute = Recorder()
        out = cached_map(records('k1', 'k2'), compute, cache_uri, None, 'run1', 'snap1')

        assert sorted(compute.computed) == ['k1', 'k2']
        assert out.height == 2

    def test_second_run_computes_nothing(self, cache_uri):
        cached_map(records('k1', 'k2'), Recorder(), cache_uri, None, 'run1', 'snap1')

        compute = Recorder()
        out = cached_map(records('k1', 'k2'), compute, cache_uri, None, 'run2', 'snap2')

        assert compute.computed == []
        assert out.height == 2

    def test_only_new_records_are_computed(self, cache_uri):
        cached_map(records('k1'), Recorder(), cache_uri, None, 'run1', 'snap1')

        compute = Recorder()
        cached_map(records('k1', 'k2'), compute, cache_uri, None, 'run2', 'snap2')

        assert compute.computed == ['k2']

    def test_returns_only_requested_keys(self, cache_uri):
        cached_map(records('k1', 'k2'), Recorder(), cache_uri, None, 'run1', 'snap1')

        out = cached_map(records('k1'), Recorder(), cache_uri, None, 'run2', 'snap2')

        assert out['cache_key'].to_list() == ['k1']

    def test_cache_accumulates_across_runs(self, cache_uri):
        cached_map(records('k1'), Recorder(), cache_uri, None, 'run1', 'snap1')
        cached_map(records('k2'), Recorder(), cache_uri, None, 'run2', 'snap2')

        assert sorted(read_cache(cache_uri, None)['cache_key'].to_list()) == ['k1', 'k2']

    def test_shards_split_the_work(self, cache_uri):
        compute = Recorder()
        cached_map(records('k1', 'k2', 'k3'), compute, cache_uri, None, 'run1', 'snap1', shard_size=2)

        assert [len(call) for call in compute.calls] == [2, 1]


class TestFailures:
    def test_a_failure_is_not_cached(self, cache_uri):
        cached_map(records('k1'), Recorder(fails={'k1'}), cache_uri, None, 'run1', 'snap1')

        assert read_cache(cache_uri, None).is_empty()

    def test_a_failure_is_retried_on_the_next_run(self, cache_uri):
        cached_map(records('k1'), Recorder(fails={'k1'}), cache_uri, None, 'run1', 'snap1')

        compute = Recorder()
        cached_map(records('k1'), compute, cache_uri, None, 'run2', 'snap2')

        assert compute.computed == ['k1']

    def test_a_failure_does_not_block_its_neighbours(self, cache_uri):
        out = cached_map(records('k1', 'k2'), Recorder(fails={'k1'}), cache_uri, None, 'run1', 'snap1')

        assert out['cache_key'].to_list() == ['k2']
        assert read_cache(cache_uri, None)['cache_key'].to_list() == ['k2']

    def test_a_later_success_lands_in_the_cache(self, cache_uri):
        cached_map(records('k1'), Recorder(fails={'k1'}), cache_uri, None, 'run1', 'snap1')
        out = cached_map(records('k1'), Recorder(), cache_uri, None, 'run2', 'snap2')

        assert out['cache_key'].to_list() == ['k1']
        assert read_cache(cache_uri, None)['cache_key'].to_list() == ['k1']

    def test_a_success_is_never_recomputed(self, cache_uri):
        cached_map(records('k1'), Recorder(), cache_uri, None, 'run1', 'snap1')

        compute = Recorder()
        cached_map(records('k1'), compute, cache_uri, None, 'run2', 'snap2')

        assert compute.computed == []


class TestResume:
    def test_a_rerun_reuses_finished_shards(self, cache_uri):
        first = Recorder()
        # a shard that raises leaves the earlier shards staged
        boom = Recorder()

        def explode_on_second_shard(shard):
            if boom.calls:
                raise RuntimeError('shard failed')
            return boom(shard)

        with pytest.raises(RuntimeError):
            cached_map(
                records('k1', 'k2', 'k3', 'k4'),
                explode_on_second_shard,
                cache_uri,
                None,
                'run1',
                'snap1',
                shard_size=2,
            )

        # same run id, so the finished shard is picked back up rather than repaid
        cached_map(records('k1', 'k2', 'k3', 'k4'), first, cache_uri, None, 'run1', 'snap1', shard_size=2)

        assert sorted(first.computed) == ['k3', 'k4']

    def test_a_different_run_id_starts_clean(self, cache_uri):
        def explode(shard):
            raise RuntimeError('shard failed')

        with pytest.raises(RuntimeError):
            cached_map(records('k1'), explode, cache_uri, None, 'run1', 'snap1')

        compute = Recorder()
        cached_map(records('k1'), compute, cache_uri, None, 'run2', 'snap2')

        assert compute.computed == ['k1']
