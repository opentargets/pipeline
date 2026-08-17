"""Reading one datasource's raw evidence, before harmonisation.

Separate from `postprocess` on purpose: post-processing takes a frame and knows nothing about
storage, so a future per-datasource module that generates its evidence in memory never touches
this module at all. Only sources that arrive as files do.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.evidence import evidence_schema
from pts.transformers.utils.dataset import scan_dataset


def _json_schema(source: str) -> dict[str, Any]:
    """The schema to pin for one json evidence part: every column it actually carries, typed.

    A full `schema=evidence_schema` pin does not just constrain dtypes, it MATERIALISES every one
    of the schema's 109 fields whether or not the source carries it -- measured: `reactome.json.gz`'s
    12 real columns became 109 that way, filled with ~97 spurious all-null columns spark never
    produced. Spark's own json reader infers only the columns actually present, and
    `_harmonise_to_schema` only casts columns that are in both the frame and the target schema, so a
    wholesale materialise-every-field pin has no spark equivalent to justify it.

    The columns actually present are discovered with a FULL-file scan (`infer_schema_length=None`),
    not a bounded sample: a bounded 5,000-row sample measurably missed one of `cosmic.json.gz`'s
    real 9 columns in a real run here. Each discovered column is then typed with `evidence_schema`'s
    dtype where the column is one of its fields -- so casting downstream still lands on that type --
    or the inferred dtype otherwise, mapping an inferred Null (an all-null column with no
    `evidence_schema` field to borrow a type from) to String.

    Deliberately not `scan_dataset`: that reader forwards no parsing options to ndjson, and this
    needs `infer_schema_length=None`. The measurement above is the whole point of this function, so
    it reads directly rather than losing the full-file scan to a shared reader that cannot express
    it.

    Args:
        source: one json evidence part's path (plain or gzip -- polars decompresses gzip natively
            from a path, which is also what keeps the read lazy).

    Returns:
        `{column: dtype}` for exactly the columns `source` carries, in ALPHABETICAL order.
    """
    inferred = pl.scan_ndjson(source, infer_schema_length=None).collect_schema()
    # Sorted by name, not `inferred`'s own order: passing this dict as a schema also fixes the
    # resulting frame's COLUMN ORDER, not just its dtypes -- and that order must be alphabetical to
    # match spark. Spark's json reader sorts inferred columns alphabetically; polars' keeps the
    # order columns first appear in the file. Measured on a one-line
    # `{"zebra":..,"alpha":..,"mango":..}` fixture: spark yields `['alpha', 'mango', 'zebra']`,
    # polars (pre-sort) yields `['zebra', 'alpha', 'mango']`. Left unsorted, that source order
    # propagates through harmonisation all the way to `validate_uniqueness`'s content hash, which
    # iterates the frame's columns in whatever order they are in -- an unsorted reader silently
    # picks a different surviving row among same-`id` duplicates than spark did.
    return {
        name: evidence_schema.get(name, inferred[name] if inferred[name] != pl.Null else pl.String)
        for name in sorted(inferred)
    }


def read_evidence(path: str, evidence_format: str) -> pl.LazyFrame:
    """Read one datasource's raw evidence, before harmonisation to `evidence_schema`.

    Args:
        path: location of the evidence -- a directory of parquet parts, a single parquet file, or
            a single (possibly gzip-compressed) ndjson file.
        evidence_format: `'parquet'` or `'json'`.

    Returns:
        LazyFrame of the raw evidence, in its source columns/dtypes.

    Raises:
        ValueError: if `evidence_format` is neither `'parquet'` nor `'json'` -- only those two
            occur in config.yaml today, so an unrecognised value is a config error, not a case to
            fall through to the json reader silently. Also raised if `evidence_format` is `'json'`
            and `path` is a directory: every json evidence source in config.yaml is a single file,
            and the schema discovery above measures ONE file, so a directory here is a config error
            refused up front rather than left to fail confusingly inside the schema scan.
    """
    if evidence_format == 'parquet':
        return scan_dataset(path)
    if evidence_format == 'json':
        if StorageHandle(path).stat().is_dir:
            msg = f'json evidence must be a single file, got a directory: {path!r}'
            raise ValueError(msg)
        return scan_dataset(path, format='ndjson', schema=_json_schema(path))
    msg = f'unrecognised evidence_format {evidence_format!r} for {path!r}, expected "parquet" or "json"'
    raise ValueError(msg)
