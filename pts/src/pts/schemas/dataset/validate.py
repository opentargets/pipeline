"""Validate a written dataset against its model, without holding the dataset in memory.

Datasets are validated after they are written, not on the way to disk, so steps keep streaming
their output, and datasets written by any step -- Polars or PySpark, this run or an old one -- can
be checked the same way. The dataset is read once, one parquet part at a time:

* **Strings cast to the model's Enums.** A part storing an Enum value as a string (as Spark writes it) is
  cast to the Enum first, so it is held to the same closed set; a value outside it is reported.
* **Every rule but the primary key, per part.** Each part is validated on its own against the
  model's pandera schema without its primary-key check. Every derived rule decides a row on that row
  alone, so validating the parts one by one finds the same violations as validating the whole.
  A model's `row_checks` are rules within a row too.
* **The primary key, over the whole dataset.** While a part is in memory, a 64-bit hash of each
  row's key is kept, in one array sized from the parquet footers. Once every part is read, the array
  is sorted in place and neighbours compared. No repeated hash proves the key unique; the rows behind
  a repeated hash are read again and compared on the key itself, so a hash collision is never
  reported as a duplicate.

A part's failures are logged as soon as it is validated, then reduced to a count and a few examples
per broken rule: however many values fail, only those are kept.

Memory is bounded by the largest part plus 8 bytes per row.

The result is a report with one row per broken rule -- the failures summed over the parts -- empty
for a valid dataset. Run it over a local
output directory, one folder per dataset::

    uv run python -m pts.schemas.dataset.validate ../work/output [dataset ...]
"""

import sys
from pathlib import Path

import numpy as np
import pandera.errors as pe
import polars as pl
from loguru import logger
from ot_datamodels.base import DatasetModel
from ot_datamodels.discovery import discover_datasets
from ot_datamodels.spec import dataset_spec
from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import NotFoundError

from pts.schemas.dataset.conform import strings_to_enums, targets, unknown_values
from pts.schemas.dataset.converter import to_pandera
from pts.transformers.utils.dataset import list_parts

#: Example values kept per broken rule.
EXAMPLES = 5

#: The report `validation_report` returns: one row per broken rule.
REPORT_SCHEMA = pl.Schema({
    'dataset': pl.String,
    'column': pl.String,
    'check': pl.String,
    'failures': pl.Int64,
    'parts': pl.Int64,
    'examples': pl.List(pl.String),
    'first_part': pl.String,
})

#: The `check` of a report row for a repeated primary key.
DUPLICATE_KEY = 'unique primary key'

#: The `check` of a report row for values a dataset stores as strings outside the model's Enum.
OUTSIDE_ENUM = 'one of its Enum categories'

#: A part's summary: `REPORT_SCHEMA` without the columns added over the parts.
_PART_SCHEMA = pl.Schema({
    name: REPORT_SCHEMA[name] for name in ('column', 'check', 'failures', 'examples', 'first_part')
})


def validate_dataset(model: type[DatasetModel], path: str) -> None:
    """Validate the dataset written at `path` against `model`.

    The call a transformer's output validation makes for each dataset it writes. Broken rules are
    logged as each part is validated (`validation_report`).

    Args:
        model: the dataset model to validate against.
        path: the dataset: a directory of parquet parts, or a single parquet file. Local or remote.

    Raises:
        ValueError: if nothing was written at `path`, or the dataset breaks the model's rules.
    """
    try:
        StorageHandle(path).stat()
    except NotFoundError as error:
        raise ValueError(f'{path} was not written, but must satisfy the model {model.__name__}') from error
    report = validation_report(path, model)
    if not report.is_empty():
        raise ValueError(f'{path} breaks its model {model.__name__}:\n{format_report(report)}')


def validation_report(path: str, model: type[DatasetModel]) -> pl.DataFrame:
    """Validate the dataset written at `path` against `model`.

    Args:
        path: the dataset: a directory of parquet parts, or a single parquet file. Local or remote.
        model: the dataset model to validate against.

    Returns:
        The violations, in `REPORT_SCHEMA`: per column and check, the number of failing values (or
        schema-level failures, such as a missing column), the number of parts they were found in,
        some example values and the first part with one. Empty if the dataset is valid.
    """
    parts = list_parts(path, '*.parquet')
    logger.info('validating {} ({} parts) against {}', path, len(parts), model.__name__)
    schema = to_pandera(model, unique=False)
    key = list(dataset_spec(model).primary_key)
    hashes = _KeyHashes(pl.scan_parquet(parts).select(pl.len()).collect().item()) if key else None

    summaries = []
    for part in parts:
        stored = pl.read_parquet(part)
        frame, summary = _as_enums(stored, model, part)
        try:
            schema.validate(frame, lazy=True)
        except pe.SchemaErrors as errors:
            summary = pl.concat([summary, _summarise_part(errors.failure_cases, part)])
        if not summary.is_empty():
            logger.warning(
                '{} part {} breaks {} rule(s):\n{}',
                model.dataset_name,
                part,
                summary.height,
                format_report(summary.with_columns(dataset=pl.lit(model.dataset_name), parts=pl.lit(1))),
            )
            summaries.append(summary)
        if hashes is not None:
            if set(key) <= set(stored.columns):  # as stored, as `_duplicate_keys` reads the key again
                hashes.add(stored.select(_key_hash(key)).to_series())
            else:
                hashes = None  # the missing key column is reported above; the key cannot be checked

    report = pl.concat([
        _summarise(summaries, model),
        _duplicate_keys(parts, key, hashes.repeated(), model) if hashes is not None else REPORT_SCHEMA.to_frame(),
    ])
    if report.is_empty():
        logger.info('{} is valid', model.dataset_name)
    else:
        logger.warning('{} breaks {} rule(s):\n{}', model.dataset_name, report.height, format_report(report))
    return report


def format_report(report: pl.DataFrame) -> str:
    """The report as text, one line per broken rule."""
    return '\n'.join(
        f'  {row["dataset"]}: {row["column"] or "<frame>"}: {row["check"]}: '
        f'{row["failures"]} failure(s) in {row["parts"]} part(s), e.g. {row["examples"]}'
        for row in report.iter_rows(named=True)
    )


class _KeyHashes:
    """The key hash of every row of a dataset, in one preallocated array."""

    def __init__(self, rows: int) -> None:
        self._hashes = np.empty(rows, dtype=np.uint64)
        self._filled = 0

    def add(self, hashes: pl.Series) -> None:
        end = self._filled + len(hashes)
        if end > len(self._hashes):  # more rows than the footers said; grow rather than fail
            self._hashes = np.concatenate([self._hashes[: self._filled], np.empty(end - self._filled, np.uint64)])
        self._hashes[self._filled : end] = hashes.to_numpy()
        self._filled = end

    def repeated(self) -> pl.Series:
        """The hashes held by more than one row."""
        hashes = self._hashes[: self._filled]
        hashes.sort()
        return pl.Series('hash', np.unique(hashes[1:][hashes[1:] == hashes[:-1]]), dtype=pl.UInt64)


def _key_hash(key: list[str]) -> pl.Expr:
    return pl.struct(key).hash(seed=0)


def _as_enums(frame: pl.DataFrame, model: type[DatasetModel], part: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The part with each string the model stores as an Enum cast to it, and the values outside an Enum.

    A Spark-written part stores Enums as strings. A column holding a value outside its Enum is left uncast, and
    the summary has a row per value path holding such values.
    """
    casts = targets(frame.schema, model, strings_to_enums)
    rows = []
    for name, dtype in casts.items():
        try:
            frame = frame.with_columns(pl.col(name).cast(dtype, strict=True))
        except pl.exceptions.InvalidOperationError:
            rows += [
                {'column': path, 'check': OUTSIDE_ENUM, 'failures': count, 'examples': values, 'first_part': part}
                for path, values, count in unknown_values(frame[name], dtype, name)
            ]
    return frame, pl.DataFrame(rows, schema=_PART_SCHEMA)


def _summarise_part(failure_cases: pl.DataFrame, part: str) -> pl.DataFrame:
    """One part's failures reduced to a count and a few examples per broken rule, so that only these are kept.

    pandera lists every failing value; a part where most rows fail would otherwise be held in full until the
    whole dataset is read.
    """
    return (
        failure_cases
        .group_by('column', 'check', maintain_order=True)
        .agg(
            pl.len().cast(pl.Int64).alias('failures'),
            pl.col('failure_case').cast(pl.String).unique(maintain_order=True).head(EXAMPLES).alias('examples'),
        )
        .with_columns(first_part=pl.lit(part))
    )


def _summarise(summaries: list[pl.DataFrame], model: type[DatasetModel]) -> pl.DataFrame:
    """The parts' summaries combined per broken rule: failures summed over the parts that had them."""
    if not summaries:
        return REPORT_SCHEMA.to_frame()
    return (
        pl
        .concat(summaries)
        .group_by('column', 'check', maintain_order=True)
        .agg(
            pl.col('failures').sum().cast(pl.Int64),
            pl.len().cast(pl.Int64).alias('parts'),
            pl
            .col('examples')
            .list.explode(keep_nulls=False, empty_as_null=False)
            .unique(maintain_order=True)
            .head(EXAMPLES),
            pl.col('first_part').first(),
        )
        .select(pl.lit(model.dataset_name).alias('dataset'), pl.all())
        .select(REPORT_SCHEMA.names())
    )


def _duplicate_keys(parts: list[str], key: list[str], repeated: pl.Series, model: type[DatasetModel]) -> pl.DataFrame:
    """Compare the rows behind repeated key hashes on the key itself, reading the parts again."""
    if repeated.is_empty():
        return REPORT_SCHEMA.to_frame()
    is_candidate = _key_hash(key).is_in(repeated.implode())
    candidates = pl.concat([
        pl.read_parquet(part, columns=key).filter(is_candidate).with_columns(part=pl.lit(part)) for part in parts
    ])
    duplicated = candidates.filter(pl.len().over(key) > 1)
    if duplicated.is_empty():
        return REPORT_SCHEMA.to_frame()  # only hash collisions
    keys = duplicated.select(key).unique().sort(key)
    example_key = pl.concat_str(key, separator=', ', ignore_nulls=False).fill_null('null')
    return pl.DataFrame(
        {
            'dataset': [model.dataset_name],
            'column': [', '.join(key)],
            'check': [DUPLICATE_KEY],
            # rows beyond the first of each key
            'failures': [duplicated.height - keys.height],
            'parts': [duplicated['part'].n_unique()],
            'examples': [keys.head(EXAMPLES).select(example_key).to_series().to_list()],
            'first_part': [duplicated['part'].min()],
        },
        schema=REPORT_SCHEMA,
    )


def main(argv: list[str]) -> int:
    """Validate every modelled dataset found under an output directory; exit 1 on any violation."""
    if not argv:
        print('usage: python -m pts.schemas.dataset.validate <output_dir> [dataset ...]', file=sys.stderr)  # noqa: T201
        return 2
    output_dir, names = Path(argv[0]), argv[1:]
    models = discover_datasets()
    unknown = sorted(set(names) - set(models))
    if unknown:
        print(f'no model for {unknown}; modelled datasets: {sorted(models)}', file=sys.stderr)  # noqa: T201
        return 2

    reports = []
    for name in names or sorted(models):
        location = output_dir / name
        if not location.exists():
            logger.warning('skipping {}: no {}', name, location)
            continue
        reports.append(validation_report(str(location), models[name]))

    if not reports:
        print(f'no modelled dataset found under {output_dir}', file=sys.stderr)  # noqa: T201
        return 2
    report = pl.concat([REPORT_SCHEMA.to_frame(), *reports])
    broken = report['dataset'].n_unique()
    print(f'{len(reports) - broken} of {len(reports)} dataset(s) valid; broken rules are logged above')  # noqa: T201
    return 1 if not report.is_empty() else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
