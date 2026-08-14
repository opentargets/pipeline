"""Tests for the `evidence_postprocess` transformer entry point.

`Evidence` and the LUT builders it wires together are already covered end to end in
`test_evidence_polars.py`; what is new here is the reader's file-vs-directory handling
(`_read_evidence`/`_json_parts`) and the registry lookup failure -- the two behaviours
this module adds on top of the pieces it assembles.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.evidence_postprocess import _json_parts, _read_evidence, evidence_postprocess


def _write_parquet_dir(directory: Path, frame: pl.DataFrame) -> str:
    """Write a frame as a single-part parquet dataset directory, the way upstream steps do."""
    directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(directory / 'part-00000.parquet')
    return str(directory)


def _write_json_gz(path: Path, rows: list[dict]) -> str:
    """Write one gzipped newline delimited json file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = '\n'.join(json.dumps(row) for row in rows).encode()
    path.write_bytes(gzip.compress(payload))
    return str(path)


# --------------------------------------------------------------------------- parquet reading


def test_read_evidence_parquet_from_a_directory_of_parts(tmp_path: Path) -> None:
    path = _write_parquet_dir(tmp_path / 'evidence', pl.DataFrame({'targetFromSourceId': ['t1', 't2']}))

    assert _read_evidence(path, 'parquet').collect().height == 2


def test_read_evidence_parquet_from_a_single_file(tmp_path: Path) -> None:
    """`intermediate/evidence/*.parquet` is always a directory in config.yaml, but the reader
    checks rather than assumes -- `StorageHandle(path).open()` raises `IsADirectoryError` on the
    other shape, so a bare `pl.scan_parquet(path)` on a directory would fail the same way.
    """  # noqa: D205 -- explanatory continuation, not a second summary line
    file_path = tmp_path / 'evidence.parquet'
    pl.DataFrame({'targetFromSourceId': ['t1']}).write_parquet(file_path)

    assert _read_evidence(str(file_path), 'parquet').collect().height == 1


# --------------------------------------------------------------------------- json reading


def test_json_parts_of_a_single_file_is_the_file_itself(tmp_path: Path) -> None:
    path = _write_json_gz(tmp_path / 'evidence.json.gz', [{'targetFromSourceId': 't1'}])

    assert _json_parts(path) == [path]


def test_json_parts_of_a_directory_lists_every_part_sorted(tmp_path: Path) -> None:
    directory = tmp_path / 'evidence'
    _write_json_gz(directory / 'part-00001.json.gz', [{'targetFromSourceId': 't2'}])
    _write_json_gz(directory / 'part-00000.json.gz', [{'targetFromSourceId': 't1'}])

    parts = _json_parts(str(directory))

    assert [Path(p).name for p in parts] == ['part-00000.json.gz', 'part-00001.json.gz']


def test_json_parts_raises_on_an_empty_directory(tmp_path: Path) -> None:
    directory = tmp_path / 'empty'
    directory.mkdir()

    with pytest.raises(ValueError, match='no json files found'):
        _json_parts(str(directory))


def test_read_evidence_json_from_a_single_file(tmp_path: Path) -> None:
    path = _write_json_gz(tmp_path / 'evidence.json.gz', [{'targetFromSourceId': 't1'}])

    frame = _read_evidence(path, 'json').collect()

    assert frame.height == 1
    assert frame['targetFromSourceId'].to_list() == ['t1']


def test_read_evidence_json_from_a_directory_reads_every_part(tmp_path: Path) -> None:
    directory = tmp_path / 'evidence'
    _write_json_gz(directory / 'part-00000.json.gz', [{'targetFromSourceId': 't1'}])
    _write_json_gz(directory / 'part-00001.json.gz', [{'targetFromSourceId': 't2'}])

    frame = _read_evidence(str(directory), 'json').collect()

    assert sorted(frame['targetFromSourceId']) == ['t1', 't2']


def test_read_evidence_json_pins_the_full_evidence_schema(tmp_path: Path) -> None:
    """A column absent from a part's leading rows must not be inferred Null and dropped.

    Same defect class as `build_publication_lut` (see validation_lut.py): unlike that table,
    which only needs five columns, evidence needs the whole schema, so the pin here is the
    full `evidence.json` schema, not a handful of named fields.
    """
    rows = [{'targetFromSourceId': f't{i}'} for i in range(200)]
    rows.append({'targetFromSourceId': 't200', 'resourceScore': 0.5})
    path = _write_json_gz(tmp_path / 'evidence.json.gz', rows)

    frame = _read_evidence(path, 'json').collect()

    assert frame.filter(pl.col('targetFromSourceId') == 't200')['resourceScore'].item() == 0.5


# --------------------------------------------------------------------------- registry lookup


def test_evidence_postprocess_raises_clearly_for_an_unregistered_datasource(tmp_path: Path) -> None:
    """The registry lookup happens before any LUT is built or file is read, so this fails fast."""
    settings = {'datasource_id': 'not_a_real_datasource', 'evidence_format': 'parquet', 'unique_fields': []}
    missing = str(tmp_path / 'does_not_exist')
    source = {
        'evidence_path': missing,
        'target_path': missing,
        'disease_path': missing,
        'publication_date_lut': missing,
    }

    with pytest.raises(KeyError, match='not_a_real_datasource'):
        evidence_postprocess(source, {}, settings, None)
