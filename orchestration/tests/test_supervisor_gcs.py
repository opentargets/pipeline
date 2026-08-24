"""Tests for reading dataset statistics from a GCS-shaped bucket."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from orchestration.supervisor.gcs import DatasetStats, Footer, compare, is_data_file, read_stats


class FakeBlob:
    def __init__(self, name: str, size: int | None) -> None:
        self.name = name
        self.size = size


class FakeBucket:
    """Stands in for a GCS bucket.

    `list_blobs` is a plain unanchored string-prefix match, exactly like the real GCS
    API — it is what makes `read_stats`'s trailing-slash anchoring load-bearing rather
    than cosmetic.
    """

    def __init__(self, blobs: list[FakeBlob]) -> None:
        self.blobs = blobs

    def list_blobs(self, prefix: str) -> list[FakeBlob]:
        return [b for b in self.blobs if b.name.startswith(prefix)]


def _footer_reader(footers: dict[str, Footer]) -> Callable[[str], Footer]:
    """Build a `read_footer` callable backed by a name-to-footer dict, for tests."""

    def read(name: str) -> Footer:
        return footers[name]

    return read


class TestIsDataFile:
    def test_a_parquet_file_is_a_data_file(self) -> None:
        assert is_data_file('run/output/disease/part-0000.parquet') is True

    def test_a_success_marker_is_not_a_data_file(self) -> None:
        assert is_data_file('run/output/disease/_SUCCESS') is False

    def test_a_crc_sidecar_is_not_a_data_file(self) -> None:
        assert is_data_file('run/output/disease/.part-0000.parquet.crc') is False

    def test_a_directory_placeholder_is_not_a_data_file(self) -> None:
        """A prefix listing can return the directory object itself, whose basename is empty."""
        assert is_data_file('run/output/disease/') is False


class TestReadStatsAbsence:
    def test_a_dataset_with_no_objects_under_its_prefix_is_none(self) -> None:
        bucket = FakeBucket([])
        assert read_stats(bucket, 'run/output/disease') is None

    def test_a_dataset_with_files_but_zero_rows_is_not_none(self) -> None:
        """Distinguishes an absent dataset from a genuinely empty one.

        A file exists, but its footer reports zero rows — `read_stats` must not
        confuse "no data files at all" with "data files that happen to hold no rows".
        Breaking the `not blobs` check into `not blobs or total rows == 0` would make
        this return None too, which is exactly what this test rules out.
        """
        bucket = FakeBucket([FakeBlob('run/output/disease/part-0000.parquet', 100)])
        footers = {'run/output/disease/part-0000.parquet': Footer(rows=0)}
        stats = read_stats(bucket, 'run/output/disease', read_footer=_footer_reader(footers))
        assert stats is not None
        assert stats.rows == 0


class TestReadStatsParquet:
    def test_sums_bytes_and_files_across_every_footer(self) -> None:
        bucket = FakeBucket(
            [
                FakeBlob('run/output/disease/part-0000.parquet', 100),
                FakeBlob('run/output/disease/part-0001.parquet', 150),
            ]
        )
        footers = {
            'run/output/disease/part-0000.parquet': Footer(rows=10),
            'run/output/disease/part-0001.parquet': Footer(rows=20),
        }
        stats = read_stats(bucket, 'run/output/disease', read_footer=_footer_reader(footers))
        assert stats == DatasetStats(rows=30, total_bytes=250, files=2, column_types={}, countable=True)

    def test_sums_rows_across_every_footer_not_just_the_first(self) -> None:
        """Guards the accumulation loop specifically, apart from the totals above.

        Both footers carry a distinct, nonzero row count. Reading only the first
        footer (`rows = footer.rows` instead of `rows += footer.rows`) would still
        produce a plausible-looking positive total here (10, not 30), so this pins
        the exact sum rather than merely checking the total is nonzero.
        """
        bucket = FakeBucket(
            [
                FakeBlob('run/output/disease/part-0000.parquet', 100),
                FakeBlob('run/output/disease/part-0001.parquet', 150),
            ]
        )
        footers = {
            'run/output/disease/part-0000.parquet': Footer(rows=10),
            'run/output/disease/part-0001.parquet': Footer(rows=20),
        }
        stats = read_stats(bucket, 'run/output/disease', read_footer=_footer_reader(footers))
        assert stats is not None
        assert stats.rows == 30

    def test_the_schema_comes_from_one_footer_not_merged_across_files(self) -> None:
        """The schema is the first file's columns, never a union of every file's.

        The two footers here have disjoint column sets. Blobs are read in sorted
        name order, so `part-0000` is read first — its columns must be the whole
        result, with `part-0001`'s column absent, not merged in.
        """
        bucket = FakeBucket(
            [
                FakeBlob('run/output/disease/part-0000.parquet', 100),
                FakeBlob('run/output/disease/part-0001.parquet', 150),
            ]
        )
        footers = {
            'run/output/disease/part-0000.parquet': Footer(rows=10, column_types={'a': 'int64'}),
            'run/output/disease/part-0001.parquet': Footer(rows=20, column_types={'b': 'string'}),
        }
        stats = read_stats(bucket, 'run/output/disease', read_footer=_footer_reader(footers))
        assert stats is not None
        assert stats.column_types == {'a': 'int64'}

    def test_success_and_crc_sidecars_are_excluded_from_bytes_files_and_the_format_check(self) -> None:
        """A `_SUCCESS` marker beside parquet must not make the dataset uncountable.

        The sidecars carry large, distinctive sizes. If they were not filtered out,
        `total_bytes` and `files` would both be inflated, and — because neither
        sidecar name ends in `.parquet` — `countable` would flip to False purely
        from their presence, misreporting a clean parquet dataset as mixed-format.
        """
        bucket = FakeBucket(
            [
                FakeBlob('run/output/disease/part-0000.parquet', 100),
                FakeBlob('run/output/disease/part-0001.parquet', 150),
                FakeBlob('run/output/disease/_SUCCESS', 999),
                FakeBlob('run/output/disease/.part-0000.parquet.crc', 999),
            ]
        )
        footers = {
            'run/output/disease/part-0000.parquet': Footer(rows=10),
            'run/output/disease/part-0001.parquet': Footer(rows=20),
        }
        stats = read_stats(bucket, 'run/output/disease', read_footer=_footer_reader(footers))
        assert stats is not None
        assert stats.total_bytes == 250
        assert stats.files == 2
        assert stats.countable is True


class TestReadStatsNdjson:
    def test_an_ndjson_dataset_is_uncountable_with_no_rows_but_reports_bytes_and_files(self) -> None:
        bucket = FakeBucket(
            [
                FakeBlob('run/output/evidence/part-0000.json', 100),
                FakeBlob('run/output/evidence/part-0001.json', 150),
            ]
        )

        def _reader(name: str) -> Footer:
            raise AssertionError(f'read_footer must not be called for a non-parquet file: {name}')

        stats = read_stats(bucket, 'run/output/evidence', read_footer=_reader)
        assert stats == DatasetStats(rows=None, total_bytes=250, files=2, column_types={}, countable=False)


class TestReadStatsWithoutFooterReader:
    def test_no_reader_yields_sizes_only_and_never_calls_a_reader(self) -> None:
        """`read_footer=None` disables row counting even for an otherwise countable dataset.

        There is no reader to call in this case, so the absence of one is the proof
        it was never invoked.
        """
        bucket = FakeBucket(
            [
                FakeBlob('run/output/disease/part-0000.parquet', 100),
                FakeBlob('run/output/disease/part-0001.parquet', 150),
            ]
        )
        stats = read_stats(bucket, 'run/output/disease')
        assert stats == DatasetStats(rows=None, total_bytes=250, files=2, column_types={}, countable=True)


class TestReadStatsPrefixAnchoring:
    def test_a_sibling_dataset_sharing_a_name_prefix_is_not_folded_in(self) -> None:
        """`output/disease_hpo` must not be counted into `output/disease`.

        Both are real datasets in `pts/config.yaml`. `output/disease_hpo` is given
        three files at a distinctive size unrelated to `output/disease`'s own totals,
        so an unanchored prefix match (`str.startswith('run/output/disease')`, which
        `disease_hpo` also satisfies) would change both `total_bytes` and `files`
        here. The assertion is on `output/disease` alone — a test that merely checked
        both datasets were individually non-empty would pass with no anchoring at all.
        """
        bucket = FakeBucket(
            [
                FakeBlob('run/output/disease/part-0000.parquet', 100),
                FakeBlob('run/output/disease/part-0001.parquet', 150),
                FakeBlob('run/output/disease_hpo/part-0000.parquet', 50_000),
                FakeBlob('run/output/disease_hpo/part-0001.parquet', 50_000),
                FakeBlob('run/output/disease_hpo/part-0002.parquet', 50_000),
            ]
        )
        stats = read_stats(bucket, 'run/output/disease')
        assert stats is not None
        assert stats.total_bytes == 250
        assert stats.files == 2

    def test_the_trailing_slash_is_added_even_when_the_caller_omits_it(self) -> None:
        """The anchor is applied regardless of whether the caller's prefix already has one."""
        bucket = FakeBucket(
            [
                FakeBlob('run/output/disease/part-0000.parquet', 100),
                FakeBlob('run/output/disease_hpo/part-0000.parquet', 50_000),
            ]
        )
        with_slash = read_stats(bucket, 'run/output/disease/')
        without_slash = read_stats(bucket, 'run/output/disease')
        assert with_slash == without_slash
        assert with_slash is not None
        assert with_slash.total_bytes == 100


def _stats(**overrides: Any) -> DatasetStats:
    base: dict[str, Any] = {'rows': 1000, 'total_bytes': 1000, 'files': 2, 'column_types': {'a': 'int64'}}
    base.update(overrides)
    return DatasetStats(**base)


class TestCompare:
    def test_both_sides_present_compares_schemas(self) -> None:
        run = _stats(column_types={'a': 'int64', 'b': 'string'})
        reference = _stats(column_types={'a': 'int64'})
        diff = compare('output/disease', run, reference)
        assert diff.side == 'both'
        assert diff.run_rows == 1000
        assert diff.reference_rows == 1000
        assert [c.column for c in diff.columns] == ['b']

    def test_run_only_sets_side_and_compares_no_schema(self) -> None:
        run = _stats()
        diff = compare('output/disease', run, None)
        assert diff.side == 'run_only'
        assert diff.run_rows == 1000
        assert diff.reference_rows is None
        assert diff.run_bytes == 1000
        assert diff.reference_bytes == 0
        assert diff.columns == []

    def test_reference_only_sets_side_and_compares_no_schema(self) -> None:
        reference = _stats()
        diff = compare('output/disease', None, reference)
        assert diff.side == 'reference_only'
        assert diff.run_rows is None
        assert diff.reference_rows == 1000
        assert diff.run_bytes == 0
        assert diff.reference_bytes == 1000
        assert diff.columns == []

    def test_both_sides_absent_raises(self) -> None:
        with pytest.raises(ValueError, match='output/disease'):
            compare('output/disease', None, None)

    def test_an_uncountable_dataset_yields_none_rows_not_zero(self) -> None:
        """An NDJSON side's absent row count must surface as None, not the default 0.

        `countable=False` on the run side alone must also make the whole diff
        uncountable, since a row-based comparison is meaningless when one side has
        no footer to have counted from.
        """
        run = _stats(rows=None, column_types={}, countable=False)
        reference = _stats()
        diff = compare('output/evidence', run, reference)
        assert diff.run_rows is None
        assert diff.countable is False
