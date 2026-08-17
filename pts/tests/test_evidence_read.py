"""Tests for reading raw evidence (`pts.transformers.evidence.read`).

Covers what this module adds on top of the shared reader: format dispatch, the json schema pin and
its column ordering, and the single-file requirement for json sources.

Directory-of-parts enumeration itself belongs to `scan_dataset` and is covered in
`test_dataset.py` -- including the deliberate decision NOT to skip `_`-prefixed files, which this
module used to do for parquet and no longer does.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl
import pytest

from pts.schemas.evidence import evidence_schema
from pts.transformers.evidence.core import QC_COLUMN, Evidence, EvidenceFlags
from pts.transformers.evidence.read import _json_schema, read_evidence


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

    assert read_evidence(path, 'parquet').collect().height == 2


def test_read_evidence_parquet_from_a_single_file(tmp_path: Path) -> None:
    """Every configured parquet source is a directory, but a single file reads too.

    The shared reader dispatches on what the path actually is rather than assuming, so this shape
    keeps working without a special case here.
    """
    file_path = tmp_path / 'evidence.parquet'
    pl.DataFrame({'targetFromSourceId': ['t1']}).write_parquet(file_path)

    assert read_evidence(str(file_path), 'parquet').collect().height == 1


def test_read_evidence_parquet_preserves_the_file_column_order(tmp_path: Path) -> None:
    """The json-reader ordering fix below must NOT be applied to parquet.

    Spark's parquet reader preserves the file's own column order and so does polars', so only the
    json path needs correcting; this pins that the fix did not leak across formats.
    """
    path = _write_parquet_dir(tmp_path / 'evidence', pl.DataFrame({'zebra': [1], 'alpha': [2], 'mango': [3]}))

    frame = read_evidence(path, 'parquet').collect()

    assert frame.columns == ['zebra', 'alpha', 'mango']


# --------------------------------------------------------------------------- json reading


def test_read_evidence_json_from_a_single_file(tmp_path: Path) -> None:
    path = _write_json_gz(tmp_path / 'evidence.json.gz', [{'targetFromSourceId': 't1'}])

    frame = read_evidence(path, 'json').collect()

    assert frame.height == 1
    assert frame['targetFromSourceId'].to_list() == ['t1']


def test_read_evidence_json_orders_columns_alphabetically_like_spark(tmp_path: Path) -> None:
    """Spark's json reader sorts its inferred columns alphabetically; polars' keeps file order.

    Measured on this exact fixture: spark yields `['alpha', 'mango', 'zebra']`, polars' own
    (unsorted) order is `['zebra', 'alpha', 'mango']`.

    Not cosmetic: harmonisation is order-preserving, so an unsorted reader's column order becomes
    the frame's column order all the way to `validate_uniqueness`, which hashes columns in whatever
    order the frame is actually in -- see the test below for the consequence downstream.
    """
    path = _write_json_gz(tmp_path / 'evidence.json.gz', [{'zebra': 1, 'alpha': 2, 'mango': 3}])

    frame = read_evidence(path, 'json').collect()

    assert frame.columns == ['alpha', 'mango', 'zebra']


def test_read_evidence_json_column_order_changes_the_uniqueness_survivor(tmp_path: Path) -> None:
    """Pins the actual consequence of the ordering defect, not just the reader's column order.

    Two rows share an `id` (same `keyField`) and differ only in `zCol`; `aCol` is constant so it
    cannot itself decide the ranking. With the reader's (alphabetical, spark-matching) column order
    the row with `zCol='a'` survives; forced back into FILE order -- simulating the pre-fix reader,
    `zCol`/`aCol` swapped -- the SAME two rows pick the OTHER row as the survivor. Same content,
    same id, different published row.

    The fixture was not hand-picked to "look plausible": a brute-force search over `zCol`/`aCol`
    values, run through the real `Evidence` pipeline (not a hand-rolled hash), found this as the
    first pair whose survivor actually flips between the two orderings.
    """
    rows = [
        {'keyField': 'same', 'zCol': 'a', 'aCol': 'a'},
        {'keyField': 'same', 'zCol': 'c', 'aCol': 'a'},
    ]
    path = _write_json_gz(tmp_path / 'evidence.json.gz', rows)
    lf = read_evidence(path, 'json')
    assert lf.collect_schema().names() == ['aCol', 'keyField', 'zCol']

    survivor = Evidence(lf).assign_evidence_identifier(['keyField']).validate_uniqueness().lf.collect()
    flagged = {row['zCol']: EvidenceFlags.DUPLICATED in row[QC_COLUMN] for row in survivor.to_dicts()}
    assert flagged == {'a': False, 'c': True}  # 'a' survives with spark's (alphabetical) column order

    # Force the columns back into FILE order -- what the pre-fix reader would have produced --
    # and show the survivor flips to the OTHER row, same two rows, same id.
    pre_fix_lf = lf.select('keyField', 'zCol', 'aCol')
    pre_fix = Evidence(pre_fix_lf).assign_evidence_identifier(['keyField']).validate_uniqueness().lf.collect()
    pre_fix_flagged = {row['zCol']: EvidenceFlags.DUPLICATED in row[QC_COLUMN] for row in pre_fix.to_dicts()}
    assert pre_fix_flagged == {'a': True, 'c': False}


def test_read_evidence_json_rejects_a_directory(tmp_path: Path) -> None:
    """Every configured `evidence_format: json` source is a single file, and the schema pin
    measures ONE file, so a directory must fail loudly with a clear message rather than being
    silently globbed or left to fail inside the schema scan.
    """  # noqa: D205 -- explanatory continuation, not a second summary line
    directory = tmp_path / 'evidence'
    _write_json_gz(directory / 'part-00000.json.gz', [{'targetFromSourceId': 't1'}])

    with pytest.raises(ValueError, match='json evidence must be a single file'):
        read_evidence(str(directory), 'json')


def test_json_schema_does_not_inflate_the_column_set(tmp_path: Path) -> None:
    """The pinned schema must match the SOURCE's columns, not evidence.json's full field set.

    A full `schema=evidence_schema` pin materialises every field the schema knows about, whether or
    not the source carries it -- measured on real data, `reactome.json.gz`'s 12 real columns became
    109 that way. Two columns in, two columns out.
    """
    rows = [{'targetFromSourceId': 't1', 'resourceScore': 0.5}]
    path = _write_json_gz(tmp_path / 'evidence.json.gz', rows)

    schema = _json_schema(path)

    assert set(schema) == {'targetFromSourceId', 'resourceScore'}
    assert len(schema) < len(evidence_schema)


def test_json_schema_prefers_the_evidence_json_dtype_for_a_known_column(tmp_path: Path) -> None:
    """A column that IS one of evidence.json's fields is typed from evidence.json, not inferred."""
    path = _write_json_gz(tmp_path / 'evidence.json.gz', [{'resourceScore': 1}])  # inferred as Int64

    schema = _json_schema(path)

    assert schema['resourceScore'] == evidence_schema['resourceScore']
    assert schema['resourceScore'] != pl.Int64


def test_json_schema_keeps_a_column_outside_evidence_json(tmp_path: Path) -> None:
    """A raw column with no evidence.json field is kept (typed by inference), not dropped."""
    path = _write_json_gz(tmp_path / 'evidence.json.gz', [{'targetFromSourceId': 't1', 'notInSchema': 'x'}])

    schema = _json_schema(path)

    assert 'notInSchema' in schema
    assert schema['notInSchema'] == pl.String


def test_read_evidence_json_finds_a_column_absent_from_a_bounded_sample(tmp_path: Path) -> None:
    """A column that only appears past a bounded sample must not be inferred Null and dropped.

    Same defect class `pts.schemas.literature` exists to avoid, here guarded by a full-file scan
    (`infer_schema_length=None`) rather than a named-column pin.
    """
    rows = [{'targetFromSourceId': f't{i}'} for i in range(200)]
    rows.append({'targetFromSourceId': 't200', 'resourceScore': 0.5})
    path = _write_json_gz(tmp_path / 'evidence.json.gz', rows)

    frame = read_evidence(path, 'json').collect()

    assert frame.filter(pl.col('targetFromSourceId') == 't200')['resourceScore'].item() == 0.5


def test_read_evidence_rejects_an_unrecognised_evidence_format(tmp_path: Path) -> None:
    """Only 'parquet' and 'json' occur in config.yaml; anything else must fail loudly rather than
    silently falling through to the json reader.
    """  # noqa: D205 -- explanatory continuation, not a second summary line
    with pytest.raises(ValueError, match="unrecognised evidence_format 'csv'"):
        read_evidence(str(tmp_path), 'csv')
