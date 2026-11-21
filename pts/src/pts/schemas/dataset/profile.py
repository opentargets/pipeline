"""Profile a run's outputs against the dataset models, to see what pts actually wrote.

For every modelled dataset under `<run>/output/<dataset_name>` two summaries are written:

* `validation.parquet`: the `validation_report` rows of every dataset, plus one `missing` row for a
  dataset whose folder is absent or holds no parquet parts.
* `profile.parquet`: per value of the model (`walk` paths, nested values included) the null count, the
  number of values, null lists and null list elements, min/max of numbers and the distinct values when
  there are few.

Datasets are read one at a time, lazily, so memory stays bounded whatever the dataset size::

    uv run python -m pts.schemas.dataset.profile gs://<bucket>/<run> <out_dir>
"""

import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from ot_datamodels.base import DatasetModel
from ot_datamodels.discovery import discover_datasets
from ot_datamodels.spec import Node, dataset_spec, walk
from otter.util.errors import NotFoundError

from pts.schemas.dataset.validate import REPORT_SCHEMA, validation_report
from pts.transformers.utils.dataset import list_parts

#: Most distinct values listed for a value.
MAX_DISTINCT = 50

#: The `check` of a validation row for a dataset that was not written.
MISSING = 'missing'

PROFILE_SCHEMA = pl.Schema({
    'dataset': pl.String,
    'path': pl.String,
    'nulls': pl.Int64,
    'rows': pl.Int64,
    'list_nulls': pl.Int64,
    'null_elements': pl.Int64,
    'min': pl.String,
    'max': pl.String,
    'distinct': pl.String,
})

_STEP = re.compile(r'\[\]|/[^/\[]+')


def _value(path: str) -> pl.Expr:
    """The values at a `walk` path: a list level explodes, a struct level takes the field."""
    column = re.match(r'[^/\[]+', path)
    assert column is not None
    expr = pl.col(column.group())
    for step in _STEP.findall(path[column.end() :]):
        expr = expr.list.explode(empty_as_null=False, keep_nulls=False) if step == '[]' else expr.struct.field(step[1:])
    return expr


def _distinct_applies(node: Node) -> bool:
    dtype = node.dtype
    return node.kind == 'primitive' and (
        dtype in {pl.String, pl.Boolean} or dtype.is_integer() or isinstance(dtype, pl.Enum)
    )


def _distinct(parts: list[str], expr: pl.Expr, node: Node) -> str | None:
    """The sorted distinct non-null values as JSON, or None above `MAX_DISTINCT`; one part at a time."""
    if isinstance(node.dtype, pl.Enum):
        expr = expr.cast(pl.String)
    seen: set[Any] = set()
    for part in parts:
        values = pl.scan_parquet(part).select(expr.drop_nulls().unique().head(MAX_DISTINCT + 1)).collect()
        seen.update(values.to_series().to_list())
        if len(seen) > MAX_DISTINCT:
            return None
    return json.dumps(sorted(seen))


def _paths(model: type[DatasetModel]) -> Iterator[tuple[str, Node]]:
    return walk(dataset_spec(model))


def field_profile(path: str, model: type[DatasetModel]) -> pl.DataFrame:
    """Profile the dataset written at `path` value by value.

    Args:
        path: the dataset: a directory of parquet parts, or a single parquet file. Local or remote.
        model: the dataset model whose values (nested ones too) are profiled.

    Returns:
        One row per `walk` path, in `PROFILE_SCHEMA`. `rows` is the number of values at that level (all
        rows for a column, the elements of the non-null lists for `column[]`). Values of a model column
        missing from the data, or of another shape, are skipped with a warning.
    """
    parts = list_parts(path, '*.parquet')
    lazy = pl.scan_parquet(parts)
    present = set(lazy.collect_schema().names())
    rows = []
    for value_path, node in _paths(model):
        if value_path.split('/')[0].split('[')[0] not in present:
            logger.warning('{}: {} is not in the data, not profiled', model.dataset_name, value_path)
            continue
        expr = _value(value_path)
        aggregates = [pl.len().alias('rows'), expr.null_count().alias('nulls')]
        if node.kind == 'list':
            aggregates.append(
                (expr.list.len() - expr.list.drop_nulls().list.len()).sum().fill_null(0).alias('null_elements')
            )
        if node.dtype.is_numeric():
            aggregates += [expr.min().cast(pl.String).alias('min'), expr.max().cast(pl.String).alias('max')]
        try:
            stats = lazy.select(aggregates).collect(engine='streaming').row(0, named=True)
            distinct = _distinct(parts, expr, node) if _distinct_applies(node) else None
        except pl.exceptions.PolarsError as error:
            logger.warning(
                '{}: {} does not have the modelled shape, not profiled: {}', model.dataset_name, value_path, error
            )
            continue
        rows.append({
            'dataset': model.dataset_name,
            'path': value_path,
            'nulls': stats['nulls'],
            'rows': stats['rows'],
            'list_nulls': stats['nulls'] if node.kind == 'list' else None,
            'null_elements': stats.get('null_elements'),
            'min': stats.get('min'),
            'max': stats.get('max'),
            'distinct': distinct,
        })
    return pl.DataFrame(rows, schema=PROFILE_SCHEMA)


def _missing(dataset: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            'dataset': [dataset],
            'column': [None],
            'check': [MISSING],
            'failures': [0],
            'parts': [0],
            'examples': [None],
            'first_part': [None],
        },
        schema=REPORT_SCHEMA,
    )


def profile_run(run_uri: str, out_dir: str) -> None:
    """Validate and profile every modelled dataset of a run, writing two parquet summaries.

    Args:
        run_uri: the run, local or `gs://`; dataset `x` is read from `<run_uri>/output/x`.
        out_dir: where `validation.parquet` and `profile.parquet` are written, local or `gs://`.
    """
    out = out_dir.rstrip('/')
    if '://' not in out:
        Path(out).mkdir(parents=True, exist_ok=True)
    validations = [REPORT_SCHEMA.to_frame()]
    profiles = [PROFILE_SCHEMA.to_frame()]
    for name, model in sorted(discover_datasets().items()):
        location = f'{run_uri.rstrip("/")}/output/{name}'
        try:
            list_parts(location, '*.parquet')
        except (NotFoundError, ValueError) as error:
            logger.warning('{}: nothing to profile at {}: {}', name, location, error)
            validations.append(_missing(name))
            continue
        validations.append(validation_report(location, model))
        profiles.append(field_profile(location, model))
    pl.concat(validations).write_parquet(f'{out}/validation.parquet')
    pl.concat(profiles).write_parquet(f'{out}/profile.parquet')


def main(argv: list[str]) -> int:
    """Profile a run's outputs: `<run_uri> <out_dir>`."""
    if len(argv) != 2:
        print('usage: python -m pts.schemas.dataset.profile <run_uri> <out_dir>', file=sys.stderr)  # noqa: T201
        return 2
    profile_run(*argv)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
