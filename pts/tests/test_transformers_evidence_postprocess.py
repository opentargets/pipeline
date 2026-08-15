"""Tests for the `evidence_postprocess` transformer entry point.

`Evidence` and the LUT builders it wires together are already covered end to end in
`test_evidence_polars.py`; what is new here is the reader -- file-vs-directory handling for both
formats, the `_`-prefix skip and unrecognised-format rejection, the json schema pin, and the
directory rejection for json sources -- and the registry lookup failure, the behaviours this
module adds on top of the pieces it assembles.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl
import pytest

from pts.schemas.evidence import evidence_schema
from pts.transformers.evidence_postprocess import _json_schema, _read_evidence, _write_partitioned, evidence_postprocess
from pts.transformers.utils.evidence import QC_COLUMN, Evidence, EvidenceFlags


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


def test_read_evidence_parquet_skips_underscore_prefixed_files(tmp_path: Path) -> None:
    """Spark's parquet reader treats a `_`-prefixed file as metadata (`_SUCCESS` etc.) and skips
    it; a bare `pl.scan_parquet(dir)` does not discriminate and would pick up its rows too, so
    this fails without the explicit filter in `_parquet_parts`.
    """  # noqa: D205 -- explanatory continuation, not a second summary line
    directory = tmp_path / 'evidence'
    directory.mkdir()
    pl.DataFrame({'targetFromSourceId': ['t1']}).write_parquet(directory / 'part-00000.parquet')
    pl.DataFrame({'targetFromSourceId': ['hidden']}).write_parquet(directory / '_hidden.parquet')

    frame = _read_evidence(str(directory), 'parquet').collect()

    assert frame['targetFromSourceId'].to_list() == ['t1']


def test_read_evidence_parquet_preserves_the_file_column_order(tmp_path: Path) -> None:
    """The json-reader ordering defect (below) does not apply to parquet: spark's parquet reader
    preserves the file's own column order, and so does polars' -- only the json path needs a fix,
    and this pins that the fix (`_json_schema`) was not accidentally applied here too.
    """  # noqa: D205 -- explanatory continuation, not a second summary line
    path = _write_parquet_dir(tmp_path / 'evidence', pl.DataFrame({'zebra': [1], 'alpha': [2], 'mango': [3]}))

    frame = _read_evidence(path, 'parquet').collect()

    assert frame.columns == ['zebra', 'alpha', 'mango']


def test_read_evidence_parquet_raises_on_a_directory_of_only_underscore_prefixed_files(tmp_path: Path) -> None:
    """Consistent with the legible `ValueError` `_read_evidence` raises when a json source is a
    directory at all.

    Left unguarded, `pl.scan_parquet([])` (`_read_evidence`, fed `_parquet_parts`'s now-filtered
    empty list) raises its own `ComputeError: empty input: paths: []` instead -- correct, but a
    less legible failure for the same underlying "no data files found" situation.
    """  # noqa: D205 -- explanatory continuation, not a second summary line
    directory = tmp_path / 'evidence'
    directory.mkdir()
    pl.DataFrame({'targetFromSourceId': ['hidden']}).write_parquet(directory / '_hidden.parquet')

    with pytest.raises(ValueError, match='no parquet files found'):
        _read_evidence(str(directory), 'parquet')


def test_read_evidence_parquet_from_a_single_file(tmp_path: Path) -> None:
    """`intermediate/evidence/*.parquet` is always a directory in config.yaml, but the reader
    checks rather than assumes: `StorageHandle(path).open()` raises `IsADirectoryError` on that
    other shape, elsewhere in this module (`_read_evidence`'s json branch), which is why
    `_parquet_parts` checks too -- a bare `pl.scan_parquet(path)` on a directory does not actually
    fail (measured), it just returns every part unfiltered, skipping the `_`-prefix skip covered
    separately above.
    """  # noqa: D205 -- explanatory continuation, not a second summary line
    file_path = tmp_path / 'evidence.parquet'
    pl.DataFrame({'targetFromSourceId': ['t1']}).write_parquet(file_path)

    assert _read_evidence(str(file_path), 'parquet').collect().height == 1


# --------------------------------------------------------------------------- json reading


def test_read_evidence_json_from_a_single_file(tmp_path: Path) -> None:
    path = _write_json_gz(tmp_path / 'evidence.json.gz', [{'targetFromSourceId': 't1'}])

    frame = _read_evidence(path, 'json').collect()

    assert frame.height == 1
    assert frame['targetFromSourceId'].to_list() == ['t1']


def test_read_evidence_json_orders_columns_alphabetically_like_spark(tmp_path: Path) -> None:
    """Spark's json reader sorts its inferred columns alphabetically; polars' keeps file order.

    Measured on this exact fixture: spark yields `['alpha', 'mango', 'zebra']`, polars' own
    (unsorted) order is `['zebra', 'alpha', 'mango']`.

    Not cosmetic: `_harmonise_to_schema` is order-preserving (`with_columns`), so an unsorted
    reader's column order becomes the frame's column order all the way to
    `validate_uniqueness`, which hashes columns in whatever order the frame is actually in --
    see `test_read_evidence_json_column_order_changes_the_uniqueness_survivor` below for the
    consequence this has downstream.
    """
    path = _write_json_gz(tmp_path / 'evidence.json.gz', [{'zebra': 1, 'alpha': 2, 'mango': 3}])

    frame = _read_evidence(path, 'json').collect()

    assert frame.columns == ['alpha', 'mango', 'zebra']


def test_read_evidence_json_column_order_changes_the_uniqueness_survivor(tmp_path: Path) -> None:
    """Pins the actual consequence of the ordering defect, not just the reader's column order.

    Two rows share an `id` (same `keyField`) and differ only in `zCol`; `aCol` is constant so it
    cannot itself decide the ranking. With the reader's (alphabetical, spark-matching) column
    order the row with `zCol='a'` survives; forced back into FILE order -- simulating the
    pre-fix reader, `zCol`/`aCol` swapped -- the SAME two rows pick the OTHER row as the
    survivor. Same content, same id, different published row: this is what "a different column
    order gives a different surviving row" (task-10-report.md) means concretely.

    The fixture was not hand-picked to "look plausible": a brute-force search over `zCol`/`aCol`
    values, run through the real `Evidence` pipeline (not a hand-rolled hash), found this as the
    first pair whose survivor actually flips between the two orderings.
    """
    rows = [
        {'keyField': 'same', 'zCol': 'a', 'aCol': 'a'},
        {'keyField': 'same', 'zCol': 'c', 'aCol': 'a'},
    ]
    path = _write_json_gz(tmp_path / 'evidence.json.gz', rows)
    lf = _read_evidence(path, 'json')
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
    """No configured `evidence_format: json` step points at a directory -- every one
    (`input/evidence/*.json.gz`) is a single file, unlike the parquet sources'
    directory-of-parts shape -- so this shape must fail loudly with a clear message
    rather than being silently globbed or left to fail inside polars with a confusing error.
    """  # noqa: D205 -- explanatory continuation, not a second summary line
    directory = tmp_path / 'evidence'
    _write_json_gz(directory / 'part-00000.json.gz', [{'targetFromSourceId': 't1'}])

    with pytest.raises(ValueError, match='json evidence must be a single file'):
        _read_evidence(str(directory), 'json')


def test_json_schema_does_not_inflate_the_column_set(tmp_path: Path) -> None:
    """The pinned schema must match the SOURCE's columns, not evidence.json's full 109 fields.

    A full `schema=evidence_schema` pin materialises every field the schema knows about, whether
    or not the source carries it -- measured on real data, `reactome.json.gz`'s 12 real columns
    became 109 that way. Two columns in, two columns out.
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

    Same defect class `validation_lut.LITERATURE_SCHEMA` exists to avoid, here guarded by a
    full-file scan (`infer_schema_length=None`) rather than a named-column pin.
    """
    rows = [{'targetFromSourceId': f't{i}'} for i in range(200)]
    rows.append({'targetFromSourceId': 't200', 'resourceScore': 0.5})
    path = _write_json_gz(tmp_path / 'evidence.json.gz', rows)

    frame = _read_evidence(path, 'json').collect()

    assert frame.filter(pl.col('targetFromSourceId') == 't200')['resourceScore'].item() == 0.5


def test_read_evidence_rejects_an_unrecognised_evidence_format(tmp_path: Path) -> None:
    """Only 'parquet' and 'json' occur in config.yaml; anything else must fail loudly rather than
    silently falling through to the json reader.
    """  # noqa: D205 -- explanatory continuation, not a second summary line
    with pytest.raises(ValueError, match="unrecognised evidence_format 'csv'"):
        _read_evidence(str(tmp_path), 'csv')


# --------------------------------------------------------------------------- registry lookup


def test_evidence_postprocess_raises_clearly_for_an_unregistered_datasource(tmp_path: Path) -> None:
    """The registry lookup happens before any LUT is built or file is read, so this fails fast.

    Asserts the CUSTOM message's own wording ('no score/direction expressions registered'), not
    just the datasource id: a bare `EXPRESSIONS[id]` KeyError also carries the id in its message
    (`KeyError: 'not_a_real_datasource'`), so matching on the id alone would pass even if the
    `try/except` that builds the clearer message were deleted.
    """
    settings = {'datasource_id': 'not_a_real_datasource', 'evidence_format': 'parquet', 'unique_fields': []}
    missing = str(tmp_path / 'does_not_exist')
    source = {
        'evidence_path': missing,
        'target_path': missing,
        'disease_path': missing,
        'publication_date_lut': missing,
    }

    with pytest.raises(KeyError, match='no score/direction expressions registered for datasource'):
        evidence_postprocess(source, {}, settings, None)


# --------------------------------------------------------------------------- partitioned writing


def test_write_partitioned_clears_a_stale_file_from_a_previous_layout(tmp_path: Path) -> None:
    """`pl.PartitionBy` never clears its destination, only ever adds numbered parts to it -- so a
    stale file left over from an earlier run (here: the single-file layout `evidence_postprocess`
    wrote before the switch to `pl.PartitionBy`) survives untouched alongside the new parts.
    Every consumer globs `*.parquet` in the destination directory, so the stale file is silently
    read and counted as current data. `otter`'s `check_destination(path, delete=True)` does not
    catch this: it only unlinks a `path.is_file()`, and does nothing for a directory.
    """  # noqa: D205 -- explanatory continuation, not a second summary line
    destination = tmp_path / 'evidence_gene_burden'
    destination.mkdir()
    stale = destination / 'evidence_gene_burden.parquet'
    pl.DataFrame({'targetFromSourceId': ['stale']}).write_parquet(stale)

    _write_partitioned(pl.LazyFrame({'targetFromSourceId': ['t1', 't2']}), str(destination))

    remaining = sorted(p.name for p in destination.glob('*.parquet'))
    assert stale.name not in remaining
    frame = pl.read_parquet(destination / '*.parquet')
    assert sorted(frame['targetFromSourceId']) == ['t1', 't2']


def test_write_partitioned_removes_orphaned_parts_from_a_larger_previous_run(tmp_path: Path) -> None:
    """Not only a stale-layout problem: if a re-run produces FEWER parts than the previous run
    (less data, or a different partition size), the extra old numbered parts must not remain
    either -- simulated here by planting parts a real earlier `pl.PartitionBy` run would have
    left, shaped exactly like its own output naming.
    """  # noqa: D205 -- explanatory continuation, not a second summary line
    destination = tmp_path / 'evidence_intogen'
    destination.mkdir()
    pl.DataFrame({'targetFromSourceId': ['old1']}).write_parquet(destination / '00000000.parquet')
    pl.DataFrame({'targetFromSourceId': ['old2']}).write_parquet(destination / '00000001.parquet')

    _write_partitioned(pl.LazyFrame({'targetFromSourceId': ['new1']}), str(destination))

    frame = pl.read_parquet(destination / '*.parquet')
    assert frame['targetFromSourceId'].to_list() == ['new1']


def test_write_partitioned_raises_if_the_destination_is_a_file(tmp_path: Path) -> None:
    """The configured destination is always a directory in config.yaml; a file there means the
    layout is not what this code expects, so it must refuse rather than silently delete it.
    """  # noqa: D205 -- explanatory continuation, not a second summary line
    destination = tmp_path / 'evidence_intogen'
    destination.write_text('not a directory')

    with pytest.raises(ValueError, match='directory'):
        _write_partitioned(pl.LazyFrame({'targetFromSourceId': ['t1']}), str(destination))
