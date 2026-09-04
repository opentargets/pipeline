"""Tests for the shared dataset reader and writer."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.utils.dataset import scan_dataset, scan_datasets, write_dataset


def _write_parts(directory: Path, *frames: pl.DataFrame) -> str:
    """Write one parquet part per frame, the way an upstream step lays a dataset out."""
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        frame.write_parquet(directory / f'{index:08d}.parquet')
    return str(directory)


def test_read_parquet_from_a_single_file(tmp_path: Path) -> None:
    path = tmp_path / 'one.parquet'
    pl.DataFrame({'a': [1, 2]}).write_parquet(path)

    assert scan_dataset(str(path)).collect().height == 2


def test_read_parquet_from_a_directory_of_parts(tmp_path: Path) -> None:
    path = _write_parts(tmp_path / 'ds', pl.DataFrame({'a': [1]}), pl.DataFrame({'a': [2]}))

    assert sorted(scan_dataset(path).collect()['a'].to_list()) == [1, 2]


def test_read_parquet_does_not_skip_underscore_prefixed_files(tmp_path: Path) -> None:
    """Deliberate: `_`-prefixed files are NOT excluded, and this pins that.

    Spark's own markers (`_SUCCESS`, `_common_metadata`) carry no data extension, so the
    per-format glob already misses them, and we control how these datasets are produced. A
    `_`-prefixed parquet would therefore have to be put there on purpose. If someone reintroduces
    filtering, this test fails and they have to justify it rather than add it quietly.
    """
    directory = tmp_path / 'ds'
    path = _write_parts(directory, pl.DataFrame({'a': [1]}))
    pl.DataFrame({'a': [999]}).write_parquet(directory / '_hidden.parquet')

    assert sorted(scan_dataset(path).collect()['a'].to_list()) == [1, 999]


def test_read_parquet_ignores_files_of_another_format(tmp_path: Path) -> None:
    """A spark `_SUCCESS` marker, and anything else without the format's extension, is skipped."""
    directory = tmp_path / 'ds'
    path = _write_parts(directory, pl.DataFrame({'a': [1]}))
    (directory / '_SUCCESS').touch()
    (directory / 'notes.txt').write_text('ignore me')

    assert scan_dataset(path).collect()['a'].to_list() == [1]


def test_read_parquet_raises_when_a_directory_has_no_parts(tmp_path: Path) -> None:
    directory = tmp_path / 'ds'
    directory.mkdir()
    (directory / '_SUCCESS').touch()

    with pytest.raises(ValueError, match=r'no .* files found'):
        scan_dataset(str(directory))


def test_read_ndjson_from_a_gzipped_file(tmp_path: Path) -> None:
    path = tmp_path / 'x.json.gz'
    path.write_bytes(gzip.compress(json.dumps({'a': 'v'}).encode()))

    frame = scan_dataset(str(path), format='ndjson').collect()

    assert frame['a'].to_list() == ['v']


def test_read_ndjson_from_a_directory_of_parts(tmp_path: Path) -> None:
    directory = tmp_path / 'lit'
    directory.mkdir()
    for index, value in enumerate(['a', 'b']):
        (directory / f'part-{index}.json.gz').write_bytes(gzip.compress(json.dumps({'v': value}).encode()))

    frame = scan_dataset(str(directory), format='ndjson').collect()

    assert sorted(frame['v'].to_list()) == ['a', 'b']


def test_schema_pins_dtypes_and_column_order(tmp_path: Path) -> None:
    """Column order is load-bearing: it feeds de-duplication content hashes downstream."""
    path = tmp_path / 'x.json.gz'
    path.write_bytes(gzip.compress(json.dumps({'z': 1, 'a': 2}).encode()))

    schema = {'a': pl.String, 'z': pl.String}
    frame = scan_dataset(str(path), format='ndjson', schema=schema).collect()

    assert frame.columns == ['a', 'z']
    assert frame.dtypes == [pl.String, pl.String]


def test_schema_with_parquet_raises_rather_than_being_ignored(tmp_path: Path) -> None:
    """Silently ignoring a passed schema is the trap this guards.

    Polars would return the file's own dtypes with no error, so a caller pinning a schema
    would never learn it did nothing.
    """
    path = tmp_path / 'x.parquet'
    pl.DataFrame({'a': [1]}).write_parquet(path)

    with pytest.raises(ValueError, match='only applied to ndjson'):
        scan_dataset(str(path), schema={'a': pl.String})


def test_unrecognised_format_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='unrecognised format'):
        scan_dataset(str(tmp_path), format='avro')


def test_read_csv_and_tsv_differ_only_in_separator(tmp_path: Path) -> None:
    csv = tmp_path / 'x.csv'
    csv.write_text('a,b\n1,2\n')
    tsv = tmp_path / 'y.tsv'
    tsv.write_text('a\tb\n1\t2\n')

    assert scan_dataset(str(csv), format='csv').collect().to_dicts() == [{'a': 1, 'b': 2}]
    assert scan_dataset(str(tsv), format='tsv').collect().to_dicts() == [{'a': 1, 'b': 2}]


def test_read_csv_forwards_options(tmp_path: Path) -> None:
    """Delimited sources vary in ways parquet does not, so parsing options belong to the caller."""
    path = tmp_path / 'x.tsv'
    path.write_text('# a comment\n1\t2\n')

    frame = scan_dataset(
        str(path), format='tsv', has_header=False, comment_prefix='#', new_columns=['a', 'b']
    ).collect()

    assert frame.to_dicts() == [{'a': 1, 'b': 2}]


def test_read_options_rejected_for_parquet(tmp_path: Path) -> None:
    """Silently dropping them would leave a caller believing an option took effect."""
    path = tmp_path / 'x.parquet'
    pl.DataFrame({'a': [1]}).write_parquet(path)

    with pytest.raises(ValueError, match='not forwarded for parquet'):
        scan_dataset(str(path), has_header=False)


def test_read_ndjson_forwards_options(tmp_path: Path) -> None:
    """`infer_schema_length` is the reason ndjson takes options at all.

    Polars infers an ndjson schema from a bounded sample of leading rows by default, so a column
    that first appears after it is dropped SILENTLY -- no error, just a missing column. This pins
    both halves -- that the default loses the column, and that the option recovers it -- because a
    test asserting only the fix would still pass if the default were harmless.
    """
    path = tmp_path / 'x.json'
    rows = [{'a': i} for i in range(200)]
    rows.append({'a': 200, 'late': 1})
    path.write_text('\n'.join(json.dumps(row) for row in rows))

    assert 'late' not in scan_dataset(str(path), format='ndjson').collect().columns
    assert 'late' in scan_dataset(str(path), format='ndjson', infer_schema_length=None).collect().columns


def test_read_rejects_a_glob_path(tmp_path: Path) -> None:
    """A glob raises loudly rather than silently stripping to the containing directory.

    Stripping would work for `*.parquet` but silently WIDEN a selective pattern like
    `part-0*.parquet` to the whole directory -- a silent wrong-data bug, worse than a loud
    failure.
    """
    with pytest.raises(ValueError, match='looks like a glob'):
        scan_dataset(str(tmp_path / '*.parquet'))


def _codecs(path: Path) -> set[str]:
    """The compression codecs actually recorded in a parquet file's metadata."""
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(path).metadata
    return {metadata.row_group(0).column(i).compression for i in range(metadata.num_columns)}


def test_write_creates_a_directory_of_parts(tmp_path: Path) -> None:
    path = str(tmp_path / 'ds')

    write_dataset(pl.DataFrame({'a': [1, 2, 3]}), path)

    parts = sorted(Path(path).glob('*.parquet'))
    assert len(parts) == 1
    assert pl.read_parquet(parts).height == 3


def test_write_uses_zstd(tmp_path: Path) -> None:
    """Asserted from the file's own metadata, not from the argument we passed in."""
    path = str(tmp_path / 'ds')

    write_dataset(pl.DataFrame({'a': list(range(1000))}), path)

    assert _codecs(next(Path(path).glob('*.parquet'))) == {'ZSTD'}


def test_write_accepts_a_lazyframe(tmp_path: Path) -> None:
    path = str(tmp_path / 'ds')

    write_dataset(pl.LazyFrame({'a': [1, 2]}), path)

    assert pl.read_parquet(sorted(Path(path).glob('*.parquet'))).height == 2


def test_write_splits_when_over_the_threshold(tmp_path: Path) -> None:
    path = str(tmp_path / 'ds')
    frame = pl.DataFrame({'a': list(range(200_000))})

    write_dataset(frame, path, approximate_bytes_per_file=100_000)

    assert len(list(Path(path).glob('*.parquet'))) > 1


def test_write_refuses_an_occupied_destination(tmp_path: Path) -> None:
    """`PartitionBy` only ADDS numbered parts; it never clears.

    Writing into a populated directory would leave the previous run's files in place, and every
    consumer globbing `*.parquet` would read both and double-count. Refusing is chosen over
    clearing because otter has no remote delete, so a clear could only work locally and would be
    a silent no-op on the cloud destinations that matter.
    """
    directory = tmp_path / 'ds'
    directory.mkdir()
    pl.DataFrame({'a': [999]}).write_parquet(directory / 'stale.parquet')

    with pytest.raises(ValueError, match='already exists'):
        write_dataset(pl.DataFrame({'a': [1]}), str(directory))

    # the existing data is untouched -- refusing must not be destructive
    assert pl.read_parquet(sorted(directory.glob('*.parquet')))['a'].to_list() == [999]


def test_write_refuses_an_empty_but_existing_directory(tmp_path: Path) -> None:
    """Existence is the test, not emptiness: a bare directory may still be a live destination."""
    directory = tmp_path / 'ds'
    directory.mkdir()

    with pytest.raises(ValueError, match='already exists'):
        write_dataset(pl.DataFrame({'a': [1]}), str(directory))


def test_write_refuses_when_the_destination_is_a_file(tmp_path: Path) -> None:
    path = tmp_path / 'ds'
    path.write_text('not a directory')

    with pytest.raises(ValueError, match='already exists'):
        write_dataset(pl.DataFrame({'a': [1]}), str(path))


def test_round_trip(tmp_path: Path) -> None:
    path = str(tmp_path / 'ds')
    frame = pl.DataFrame({'a': [1, 2], 'b': ['x', 'y']})

    write_dataset(frame, path)

    assert scan_dataset(path).collect().sort('a').equals(frame.sort('a'))


def test_scan_datasets_reads_every_matching_dataset(tmp_path: Path) -> None:
    _write_parts(tmp_path / 'evidence_a', pl.DataFrame({'targetId': ['t1']}))
    _write_parts(tmp_path / 'evidence_b', pl.DataFrame({'targetId': ['t2']}))
    _write_parts(tmp_path / 'other', pl.DataFrame({'targetId': ['t3']}))

    frame = scan_datasets(str(tmp_path / 'evidence_*')).collect()

    assert sorted(frame['targetId'].to_list()) == ['t1', 't2']


def test_scan_datasets_unions_differing_schemas(tmp_path: Path) -> None:
    """Most evidence datasets have no `drugId`; spark's mergeSchema null-filled them."""
    _write_parts(tmp_path / 'evidence_a', pl.DataFrame({'targetId': ['t1'], 'drugId': ['d1']}))
    _write_parts(tmp_path / 'evidence_b', pl.DataFrame({'targetId': ['t2']}))

    frame = scan_datasets(str(tmp_path / 'evidence_*')).select('targetId', 'drugId').collect()

    assert sorted(zip(frame['targetId'], frame['drugId'], strict=True)) == [('t1', 'd1'), ('t2', None)]


def test_scan_datasets_raises_when_the_glob_matches_nothing(tmp_path: Path) -> None:
    """A mistyped pattern must fail loudly, not yield an empty frame.

    An empty evidence frame would silently strip every drug from the disease index rather
    than erroring, so this is the difference between a loud failure and a bad release.
    """
    with pytest.raises(ValueError, match='matched no datasets'):
        scan_datasets(str(tmp_path / 'nothing_*'))


def test_scan_datasets_requires_a_glob(tmp_path: Path) -> None:
    _write_parts(tmp_path / 'plain', pl.DataFrame({'a': [1]}))

    with pytest.raises(ValueError, match='is not a glob'):
        scan_datasets(str(tmp_path / 'plain'))
