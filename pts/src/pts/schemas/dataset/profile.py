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
from typing import Any

import polars as pl
from loguru import logger
from ot_datamodels.base import DatasetModel
from ot_datamodels.discovery import discover_datasets
from ot_datamodels.spec import Node, dataset_spec, walk
from otter.storage.synchronous.handle import StorageHandle
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
    'present': pl.Boolean,
    'nulls': pl.Int64,
    'rows': pl.Int64,
    'list_nulls': pl.Int64,
    'null_elements': pl.Int64,
    'min': pl.String,
    'max': pl.String,
    'distinct': pl.String,
})

_STEP = re.compile(r'\[\]|/[^/\[]+')

#: The `check` of a validation row for a dataset that could not be validated or profiled.
ERROR = 'error'


def _value(path: str) -> pl.Expr:
    """The values at a `walk` path: a list level explodes, a struct level takes the field."""
    column = _column(path)
    expr = pl.col(column)
    for step in _STEP.findall(path[len(column) :]):
        expr = expr.list.explode(empty_as_null=False, keep_nulls=False) if step == '[]' else expr.struct.field(step[1:])
    return expr


def _column(path: str) -> str:
    """The top-level column of a `walk` path."""
    return re.split(r'[/\[]', path, maxsplit=1)[0]


def _lists_distinct(node: Node) -> bool:
    dtype = node.dtype
    return node.kind == 'primitive' and (
        dtype in {pl.String, pl.Boolean} or dtype.is_integer() or isinstance(dtype, pl.Enum)
    )


def _distinct_values(expr: pl.Expr, node: Node) -> pl.Expr:
    return (expr.cast(pl.String) if isinstance(node.dtype, pl.Enum) else expr).drop_nulls()


def _profile_column(lazy: pl.LazyFrame, dataset: str, paths: list[tuple[str, Node]]) -> list[dict[str, Any]]:
    """The profile rows of one top-level column and everything nested in it: two passes over the column.

    The first pass computes every statistic of every path in one select, the second lists the distinct
    values of the paths that have few.
    """
    aggregates = []
    for i, (path, node) in enumerate(paths):
        expr = _value(path)
        aggregates += [expr.len().alias(f'{i}/rows'), expr.null_count().alias(f'{i}/nulls')]
        if node.kind == 'list':
            aggregates.append(
                (expr.list.len() - expr.list.drop_nulls().list.len()).sum().fill_null(0).alias(f'{i}/null_elements')
            )
        if node.dtype.is_numeric():
            aggregates += [expr.min().cast(pl.String).alias(f'{i}/min'), expr.max().cast(pl.String).alias(f'{i}/max')]
        if _lists_distinct(node):
            aggregates.append(_distinct_values(expr, node).n_unique().alias(f'{i}/distinct'))
    stats = lazy.select(aggregates).collect(engine='streaming').row(0, named=True)

    few = [i for i, (_, node) in enumerate(paths) if _lists_distinct(node) and stats[f'{i}/distinct'] <= MAX_DISTINCT]
    values = (
        lazy
        .select([
            _distinct_values(_value(paths[i][0]), paths[i][1]).unique().sort().implode().alias(str(i)) for i in few
        ])
        .collect(engine='streaming')
        .row(0, named=True)
        if few
        else {}
    )
    return [
        {
            'dataset': dataset,
            'path': path,
            'present': True,
            'nulls': stats[f'{i}/nulls'],
            'rows': stats[f'{i}/rows'],
            'list_nulls': stats[f'{i}/nulls'] if node.kind == 'list' else None,
            'null_elements': stats.get(f'{i}/null_elements'),
            'min': stats.get(f'{i}/min'),
            'max': stats.get(f'{i}/max'),
            'distinct': json.dumps(values[str(i)]) if str(i) in values else None,
        }
        for i, (path, node) in enumerate(paths)
    ]


def field_profile(path: str, model: type[DatasetModel]) -> pl.DataFrame:
    """Profile the dataset written at `path` value by value.

    Args:
        path: the dataset: a directory of parquet parts, or a single parquet file. Local or remote.
        model: the dataset model whose values (nested ones too) are profiled.

    Returns:
        One row per `walk` path, in `PROFILE_SCHEMA`. `rows` is the number of values at that level (all
        rows for a column, the elements of the non-null lists for `column[]`). The values of a model
        column missing from the data get a row with `present` false and no statistics; a column of
        another shape than the model's is skipped with a warning.
    """
    lazy = pl.scan_parquet(list_parts(path, '*.parquet'))
    in_data = set(lazy.collect_schema().names())
    by_column: dict[str, list[tuple[str, Node]]] = {}
    for value_path, node in walk(dataset_spec(model)):
        by_column.setdefault(_column(value_path), []).append((value_path, node))

    rows: list[dict[str, Any]] = []
    for column, paths in by_column.items():
        if column not in in_data:
            logger.warning('{}: column {} is not in the data', model.dataset_name, column)
            rows += [{'dataset': model.dataset_name, 'path': p, 'present': False} for p, _ in paths]
            continue
        try:
            rows += _profile_column(lazy, model.dataset_name, paths)
        except pl.exceptions.PolarsError as error:
            logger.warning(
                '{}: column {} does not have the modelled shape, not profiled: {}', model.dataset_name, column, error
            )
    return pl.DataFrame(rows, schema=PROFILE_SCHEMA)


def _row(dataset: str, check: str, examples: list[str] | None = None) -> pl.DataFrame:
    """A validation row that is not a broken rule: the dataset was not written, or could not be checked."""
    return pl.DataFrame(
        {
            'dataset': [dataset],
            'column': [None],
            'check': [check],
            'failures': [0],
            'parts': [0],
            'examples': [examples],
            'first_part': [None],
        },
        schema=REPORT_SCHEMA,
    )


def _write(frame: pl.DataFrame, path: str) -> None:
    with StorageHandle(path).open('wb') as handle:
        frame.write_parquet(handle)


def profile_run(run_uri: str, out_dir: str) -> None:
    """Validate and profile every modelled dataset of a run, writing parquet summaries.

    Each dataset is written as soon as it is done, to `<out_dir>/validation/<dataset>.parquet` and
    `<out_dir>/profile/<dataset>.parquet`, so a crash keeps what came before it; a dataset that fails
    gets an `error` row and the run goes on. At the end `validation.parquet` and `profile.parquet`
    combine all datasets.

    Args:
        run_uri: the run, local or `gs://`; dataset `x` is read from `<run_uri>/output/x`.
        out_dir: where the summaries are written, local or `gs://`.
    """
    out = out_dir.rstrip('/')
    validations = [REPORT_SCHEMA.to_frame()]
    profiles = [PROFILE_SCHEMA.to_frame()]
    for name, model in sorted(discover_datasets().items()):
        location = f'{run_uri.rstrip("/")}/output/{name}'
        validation, profile = REPORT_SCHEMA.to_frame(), PROFILE_SCHEMA.to_frame()
        try:
            list_parts(location, '*.parquet')
        except (NotFoundError, ValueError) as error:
            logger.warning('{}: nothing to profile at {}: {}', name, location, error)
            validation = _row(name, MISSING)
        else:
            try:
                validation = validation_report(location, model)
                profile = field_profile(location, model)
            except Exception as error:  # one bad dataset must not lose the rest of the run
                logger.exception('{}: failed', name)
                validation = pl.concat([validation, _row(name, ERROR, [f'{type(error).__name__}: {error}'])])
        _write(validation, f'{out}/validation/{name}.parquet')
        _write(profile, f'{out}/profile/{name}.parquet')
        validations.append(validation)
        profiles.append(profile)
    _write(pl.concat(validations), f'{out}/validation.parquet')
    _write(pl.concat(profiles), f'{out}/profile.parquet')


def main(argv: list[str]) -> int:
    """Profile a run's outputs: `<run_uri> <out_dir>`."""
    if len(argv) != 2:
        print('usage: python -m pts.schemas.dataset.profile <run_uri> <out_dir>', file=sys.stderr)  # noqa: T201
        return 2
    profile_run(*argv)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
