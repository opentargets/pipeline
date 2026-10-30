"""Shared dataset reading and writing for transformers.

One place that knows how a dataset is laid out on storage: which files make up a dataset, which
compression it is written with, and how large a part may get.

`StorageHandle` is used for some things here and deliberately not others, which is worth stating
because the mix looks arbitrary otherwise:

* **Listing and existence go through it.** Deciding whether a location is one file or a directory
  of parts, enumerating those parts, and checking whether a destination is already occupied all
  have to work on every backend. That is what otter's abstraction is for.
* **Reads deliberately do not.** Polars receives URI STRINGS, never opened file objects. It reads
  cloud storage natively and pushes projections down to the file; handed a file object it can only
  materialise the whole thing, which on remote storage downloads every byte before a single column
  is read. Routing reads through the abstraction would cost more than it buys.
* **Deleting is not attempted at all**, because otter exposes no remote delete. Anything built on
  `pathlib` would work locally and be a silent no-op on the cloud destinations that matter, so
  `write_dataset` refuses to write into an occupied destination rather than clearing it. That
  behaves identically on every backend, which a clear could not.

So: use it where it adds reach, avoid it where it removes capability, and design around what it
cannot do rather than half-doing it.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Mapping
from typing import Any, Literal

import openpyxl
import polars as pl
from openpyxl.cell.cell import TYPE_ERROR, TYPE_NUMERIC
from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import NotFoundError

from pts.schemas.dataset import DatasetModel, to_pandera

#: Target size per written part. `pl.PartitionBy` sizes against the IN-MEMORY frame, not the
#: compressed output, so the resulting file size depends on how well a dataset compresses.
#: Calibrated against the worst compression ratio observed across the release rather than the
#: best, so that poorly-compressing datasets stay near the target instead of overshooting it --
#: which makes this a cap on typical data, not a guarantee. Aim is roughly 250 MB per file.
_DEFAULT_TARGET_BYTES = 1_400_000_000

#: `pandas.read_excel`'s default `na_values`.
_MISSING_CELL_VALUES = [
    '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA',
    'NULL', 'NaN', 'None', 'n/a', 'nan', 'null',
]  # fmt: skip

Format = Literal['parquet', 'ndjson', 'csv', 'tsv']

#: Glob used to enumerate a directory for each format.
_GLOBS: dict[str, str] = {
    'parquet': '*.parquet',
    # covers .json, .json.gz and .jsonl
    'ndjson': '*.json*',
    'csv': '*.csv*',
    'tsv': '*.tsv*',
}

#: The delimited formats, which differ only in separator and accept caller parsing options.
_SEPARATORS: dict[str, str] = {'csv': ',', 'tsv': '\t'}


def scan_dataset(
    path: str,
    *,
    format: Format = 'parquet',
    schema: Mapping[str, Any] | None = None,
    **options: Any,
) -> pl.LazyFrame:
    """Lazily read one dataset, whether it is a single file or a directory of parts.

    Always lazy: callers add `.collect()` where they need a `DataFrame`, so there is no second
    eager function to keep in sync.

    A directory is enumerated through `StorageHandle` rather than handed to polars as a glob
    string, so the part list is explicit and sorted and does not depend on a glob dialect. Files
    whose name starts with `_` are NOT excluded: spark writes `_SUCCESS` and `_common_metadata`
    without a data extension, so the per-format globs already miss them, and we control how these
    datasets are produced.

    `csv` and `tsv` differ only in separator. Both forward `**options` to `pl.scan_csv`, because
    delimited sources vary in ways parquet does not -- headers, quoting, comment prefixes,
    explicit column names -- and those belong to the caller, not here. `ndjson` forwards them to
    `pl.scan_ndjson` for the same reason; `infer_schema_length` in particular matters, since its
    default of 100 rows silently drops a column that first appears later in the file.

    Parquet takes no options: it carries its own schema, so there is nothing for a caller to
    describe.

    Args:
        path: the dataset location, already absolute.
        format: `'parquet'`, `'ndjson'`, `'csv'` or `'tsv'`.
        schema: when given, pins dtypes AND column order. This function never invents a schema --
            a caller that needs one discovered from the data computes it and passes it in. Not
            supported for parquet, which carries its own.
        **options: forwarded to the underlying polars reader for `ndjson`, `csv` and `tsv`.
            Rejected for parquet, where they would silently do nothing.

    Returns:
        LazyFrame of the dataset in its source columns and dtypes.

    Raises:
        ValueError: for an unrecognised `format`, a glob passed as `path`, a directory with no
            readable parts, a schema passed with parquet, or options passed to a format that
            cannot use them.
    """
    if format not in _GLOBS:
        msg = f'unrecognised format {format!r} for {path!r}, expected one of {sorted(_GLOBS)}'
        raise ValueError(msg)

    if any(char in path for char in '*?['):
        msg = (
            f'{path!r} looks like a glob; pass the containing directory instead. '
            'scan_dataset enumerates a directory itself.'
        )
        raise ValueError(msg)

    if schema is not None and format == 'parquet':
        msg = f'schema is only applied to ndjson, csv and tsv, but format is parquet for {path!r}'
        raise ValueError(msg)

    if options and format == 'parquet':
        msg = f'options {sorted(options)} are not forwarded for parquet, which carries its own schema'
        raise ValueError(msg)

    pattern = _GLOBS[format]
    handle = StorageHandle(path)
    if handle.stat().is_dir:
        parts: str | list[str] = sorted(handle.glob(pattern))
        if not parts:
            raise ValueError(f'no {pattern} files found in {path}')
    else:
        parts = path

    if format == 'parquet':
        return pl.scan_parquet(parts)
    if format == 'ndjson':
        return pl.scan_ndjson(parts, schema=schema, **options)
    return pl.scan_csv(parts, separator=_SEPARATORS[format], schema=schema, **options)


def scan_datasets(
    pattern: str | list[str],
    *,
    format: Format = 'parquet',
    **options: Any,
) -> pl.LazyFrame:
    """Lazily combine datasets selected by a directory glob or an explicit path list.

    `scan_dataset` refuses a glob because it enumerates a single directory itself. This is the
    dataset-level counterpart: the glob selects DIRECTORIES (`output/evidence_*`), and every
    matched directory is read and concatenated.

    Globs for the format's DATA FILES rather than the directories themselves, then derives each
    match's parent as the dataset directory. Globbing directory names directly would be simpler,
    but otter's GCS backend implements glob as `bucket.list_blobs(match_glob=...)`, which matches
    OBJECT names -- and a directory is not an object, so that approach returns zero hits on GCS,
    the only backend this runs against in production. Globbing the per-format data files
    (`_GLOBS[format]`) works identically on both backends, and, being the same suffix-anchored
    pattern `scan_dataset` itself uses, already excludes a spark `_SUCCESS` marker or a
    `.parquet.crc` sidecar rather than treating them as phantom datasets.

    Schemas are unioned rather than required to match, via `pl.concat(..., how='diagonal_relaxed')`:
    a column missing from some frames is null-filled and differing dtypes are relaxed to a common
    supertype. Most evidence datasets carry no `drugId` column at all, and the pyspark job relied
    on `mergeSchema=True` for the same effect.

    Raises rather than returning an empty frame when nothing matches: an empty evidence frame
    would silently strip every drug from the disease index instead of failing the step.

    Args:
        pattern: Absolute dataset paths, or a location with a glob in its LAST segment.
        format: `'parquet'`, `'ndjson'`, `'csv'` or `'tsv'`.
        **options: forwarded to `scan_dataset` for each matched dataset.

    Returns:
        LazyFrame of every matched dataset, vertically concatenated with a unioned schema.

    Raises:
        ValueError: for an unrecognised `format`, if `pattern` has no glob, if the glob is not
            in the last segment, if it matches no datasets, or if the path list is empty.
    """
    if format not in _GLOBS:
        msg = f'unrecognised format {format!r} for {pattern!r}, expected one of {sorted(_GLOBS)}'
        raise ValueError(msg)

    if isinstance(pattern, list):
        if not pattern:
            raise ValueError('At least one dataset path is required')
        directories = pattern
    else:
        parent, _, name_glob = pattern.rpartition('/')
        if not any(char in name_glob for char in '*?['):
            msg = f'{pattern!r} is not a glob; use scan_dataset for a single dataset'
            raise ValueError(msg)
        if any(char in parent for char in '*?['):
            msg = f'{pattern!r} globs above the dataset name; only the last segment may be a glob'
            raise ValueError(msg)

        files = StorageHandle(parent).glob(f'{name_glob}/{_GLOBS[format]}')
        directories = sorted({file.rsplit('/', 1)[0] for file in files})
        if not directories:
            msg = f'{pattern!r} matched no datasets'
            raise ValueError(msg)

    frames = [scan_dataset(directory, format=format, **options) for directory in directories]
    return pl.concat(frames, how='diagonal_relaxed')


def _pandas_cell(cell: Any) -> Any:
    """A cell's value as `pandas.read_excel` takes it: blank '', an error null, an integral number an int."""
    if cell.value is None:
        return ''
    if cell.data_type == TYPE_ERROR:
        return None
    if cell.data_type == TYPE_NUMERIC and int(cell.value) == cell.value:
        return int(cell.value)
    return cell.value


def _pandas_header(names: list[str]) -> list[str]:
    """`pandas.read_excel`'s column names: a blank one `Unnamed: <i>`, a repeat `<name>.<n>`, named ones first."""
    columns = [name or f'Unnamed: {i}' for i, name in enumerate(names)]
    counts: dict[str, int] = {}
    for i in [i for i, name in enumerate(names) if name] + [i for i, name in enumerate(names) if not name]:
        original = column = columns[i]
        count = counts.get(column, 0)
        while count > 0:
            counts[original] = count + 1
            column = f'{original}.{count}'
            count = count + 1 if column in columns else counts.get(column, 0)
        columns[i] = column
        counts[column] = count + 1
    return columns


def _sheet_as_pandas_text(data: bytes, sheet_name: str, *, has_header: bool) -> pl.DataFrame:
    """The sheet's cells as `pandas.read_excel(dtype=str)` renders them, before its missing values.

    With `has_header`, the first row is the header, blank or not; without it, the columns are
    `column_1`, `column_2`, ... A blank row is kept, unless no row with a value follows.
    """
    workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True, keep_links=False)
    try:
        sheet = workbook[sheet_name]
        sheet.reset_dimensions()
        rows = []
        for row in sheet.rows:
            values = [_pandas_cell(cell) for cell in row]
            while values and values[-1] == '':
                values.pop()
            rows.append(values)
    finally:
        workbook.close()
    while rows and not rows[-1]:
        rows.pop()
    if not rows:
        raise ValueError(f'sheet {sheet_name} is empty')
    width = max(len(row) for row in rows)
    text = [[None if v is None else str(v) for v in row + [''] * (width - len(row))] for row in rows]
    if not has_header:
        return pl.DataFrame(text, schema=[(f'column_{i + 1}', pl.String) for i in range(width)], orient='row')
    columns = _pandas_header([name or '' for name in text[0]])
    return pl.DataFrame(text[1:], schema=dict.fromkeys(columns, pl.String), orient='row')


def read_spreadsheet(
    path: str, sheet_name: str, *, infer_types: bool = True, has_header: bool = True
) -> pl.DataFrame:
    """Read one sheet of an Excel workbook with every cell as a string.

    Blanks and `pandas.read_excel`'s default missing-value strings are null, and whitespace is kept.
    With `infer_types`, each column is typed first, as `pandas.read_excel` does, and booleans are
    '1'/'0'; without it, each cell is rendered on its own, as `pandas.read_excel(dtype=str)` does:
    an integral number without a fraction, a boolean as 'True'/'False'. A sheet with only a header
    gives its columns and no rows. Without `has_header`, every row is data and the columns are
    `column_1`, `column_2`, ..., as `pandas.read_excel(header=None, dtype=str)` reads it; header
    rows the caller takes from those rows have had the missing-value strings nulled too. The
    workbook is read whole through `StorageHandle`.

    Args:
        path: location of the workbook.
        sheet_name: sheet to read.
        infer_types: type each column before rendering it as text.
        has_header: the first row names the columns; only without `infer_types`.

    Returns:
        DataFrame with one string column per sheet column.

    Raises:
        ValueError: if `has_header` is false and `infer_types` true.
    """
    if not has_header and infer_types:
        msg = 'a sheet without a header is read only with infer_types=False'
        raise ValueError(msg)
    with StorageHandle(path).open('rb') as f:
        data = f.read()
    if not infer_types:
        strings = _sheet_as_pandas_text(data, sheet_name, has_header=has_header)
        return strings.with_columns(pl.when(~pl.all().is_in(_MISSING_CELL_VALUES)).then(pl.all()).name.keep())
    sheet = pl.read_excel(
        io.BytesIO(data),
        sheet_name=sheet_name,
        engine='openpyxl',
        infer_schema_length=None,
        raise_if_empty=False,
    )
    if sheet.width == 0:
        header = pl.read_excel(
            io.BytesIO(data), sheet_name=sheet_name, engine='openpyxl', infer_schema_length=0, has_header=False
        )
        return pl.DataFrame(schema=dict.fromkeys(header.row(0), pl.String))
    strings = sheet.with_columns(pl.col(pl.Boolean).cast(pl.Int8)).cast(pl.String)
    return strings.with_columns(pl.when(~pl.all().is_in(_MISSING_CELL_VALUES)).then(pl.all()).name.keep())


def write_dataset(
    frame: pl.LazyFrame | pl.DataFrame,
    path: str,
    *,
    schema: type[DatasetModel] | None = None,
    approximate_bytes_per_file: int = _DEFAULT_TARGET_BYTES,
) -> None:
    """Write one dataset as a directory of size-capped zstd parquet parts.

    Always a directory, never a single named file, so no dataset can outgrow its layout as it
    grows.

    REFUSES to write into anything that already exists. `pl.PartitionBy` never clears a
    destination -- it only ever ADDS numbered parts -- so writing into a populated directory
    leaves the previous run's files in place, and every consumer globbing `*.parquet` reads them
    as current data. A re-run producing fewer or differently-sized parts silently duplicates rows
    that way, which is worse than not running at all.

    Clearing instead of refusing was considered and rejected: otter exposes no delete for remote
    storage, so a clear could only ever work locally and would be a silent no-op on the cloud
    destinations that matter. Refusing behaves identically everywhere. There is no overwrite mode
    yet; clear the destination deliberately, or write somewhere new.

    Note what this costs locally: re-running a step twice in a row now fails on the second run
    until its destination is removed. In production each release writes to its own URI, so this
    fires only on a genuine retry into a populated destination -- exactly the case that used to
    corrupt quietly.

    Args:
        frame: the data to write; a `DataFrame` is made lazy so there is one code path.
        path: destination directory, used as given -- never a parent or a derived path.
        schema: an optional dataset model to validate `frame` against before writing (see
            `validate_dataset`, which collects a lazy `frame`). Pass one only where the frame fits
            in memory.
        approximate_bytes_per_file: target part size, measured against the IN-MEMORY frame. See
            `_DEFAULT_TARGET_BYTES` for the calibration and its limits.

    Raises:
        ValueError: if anything already exists at `path`, file or directory.
        pandera.errors.SchemaErrors: if `schema` is given and `frame` violates it.
    """
    if schema is not None:
        # checked first, so an occupied destination fails before the frame is computed
        check_destinations_absent([path])
        frame = validate_dataset(frame, schema)
    sink_dataset(frame, path, approximate_bytes_per_file=approximate_bytes_per_file).collect()


def validate_dataset(frame: pl.LazyFrame | pl.DataFrame, schema: type[DatasetModel]) -> pl.DataFrame:
    """Validate a dataset against the pandera schema derived from its model.

    A lazy `frame` is COLLECTED: pandera's value-level checks (ranges, patterns, allowed values,
    nested-field rules, registered checks) need materialised data, unlike the dtype and column
    checks. All violations are collected before raising (`lazy=True`), rather than failing on the
    first one found.

    Args:
        frame: the data to validate.
        schema: the dataset model (`pts.schemas.dataset.DatasetModel`) to validate against.

    Returns:
        The validated, collected frame.

    Raises:
        pandera.errors.SchemaErrors: if `frame` violates `schema`.
    """
    df = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
    return to_pandera(schema).validate(df, lazy=True)


def check_destinations_absent(paths: Iterable[str]) -> None:
    """Fail fast if anything already exists at any of `paths`.

    Raises:
        ValueError: naming every existing path.
    """
    existing = []
    for path in paths:
        try:
            StorageHandle(path).stat()
        except NotFoundError:
            continue
        existing.append(path)
    if existing:
        msg = (
            f'{", ".join(map(repr, existing))} already exists; write_dataset never overwrites. '
            '`pl.PartitionBy` only adds parts, so writing there would leave the previous run behind. '
            'Remove it first.'
        )
        raise ValueError(msg)


def sink_dataset(
    frame: pl.LazyFrame | pl.DataFrame,
    path: str,
    *,
    approximate_bytes_per_file: int = _DEFAULT_TARGET_BYTES,
) -> pl.LazyFrame:
    """Plan the `write_dataset` write of one dataset without running it.

    Several sinks collected together with `pl.collect_all` compute their shared input once.

    Args:
        frame: the data to write; a `DataFrame` is made lazy so there is one code path.
        path: destination directory, used as given.
        approximate_bytes_per_file: target part size, as in `write_dataset`.

    Returns:
        The lazy sink; nothing is written until it is collected.

    Raises:
        ValueError: if anything already exists at `path`, file or directory.
    """
    check_destinations_absent([path])
    lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    return lf.sink_parquet(
        pl.PartitionBy(path, approximate_bytes_per_file=approximate_bytes_per_file),
        compression='zstd',
        lazy=True,
    )
