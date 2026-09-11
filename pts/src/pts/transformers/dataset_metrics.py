"""Config-driven dataset profiler, emitting one JSON document for the whole release.

Sibling of `release_metrics`, and deliberately not part of it: `release_metrics` produces the
long-running HuggingFace series that tracks evidence and association counts release over release,
and its schema is fixed by that consumer. This answers a different question -- how big is every
dataset in this release, and how does each one break down -- for the metrics page.

Profiling is cheap by construction. `count` comes from parquet footers and `file_size` /
`number_of_partitions` from a single storage listing, so neither reads data. Only a dataset with
a configured grouping or filter is scanned at all, and then only the columns its expressions
name.

Datasets are assumed to be flat parquet directories, which is what `utils.dataset.write_dataset`
produces and what spark wrote before it. Hive-partitioned layouts (`key=value/part-*.parquet`)
have no top-level parquet files and would be skipped.
"""

from __future__ import annotations

import json
from collections import defaultdict
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import fsspec
import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.parquet_helpers import count_parquet_rows, discover_dataset_paths
from pts.transformers.utils.dataset import scan_dataset

NULL_KEY = 'null'

#: The scalars every dataset carries, in the order they appear in the document.
SCALAR_METRICS = ('count', 'file_size', 'number_of_partitions')


def _as_lazy(frame: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """Return a LazyFrame for either a DataFrame or a LazyFrame."""
    return frame if isinstance(frame, pl.LazyFrame) else frame.lazy()


def compute_breakdown(
    frame: pl.DataFrame | pl.LazyFrame,
    expression: str,
    schema: pl.Schema | None = None,
) -> dict[str, int]:
    """Group a frame by a SQL expression and count rows per group.

    The expression is evaluated with `pl.sql_expr`. A plain column, a derived expression, or a
    list column (auto-exploded) are all valid. Null group keys are coalesced to `"null"`. Returns
    `{value: count}` sorted by descending count then key. Pass a lazy frame for projection
    pushdown.

    Args:
        frame: dataset to group.
        expression: SQL expression producing the group key.
        schema: the frame's schema; when given and the expression is a bare column,
            list-detection avoids an extra metadata read (useful for remote storage).

    Returns:
        Count per group value.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"studyType": ["gwas", "eqtl", "gwas", None]})
        >>> compute_breakdown(df, "studyType")
        {'gwas': 2, 'eqtl': 1, 'null': 1}
        >>> df2 = pl.DataFrame({"rightStudyType": ["gwas", "eqtl", "eqtl"]})
        >>> compute_breakdown(df2, "concat('gwas-', rightStudyType)")
        {'gwas-eqtl': 2, 'gwas-gwas': 1}
        >>> df3 = pl.DataFrame({"tas": [["a", "b"], ["a"]]})
        >>> compute_breakdown(df3, "tas")
        {'a': 2, 'b': 1}
    """
    selected = _as_lazy(frame).select(pl.sql_expr(expression).alias('k'))
    if schema is not None and expression in schema:
        is_list = isinstance(schema[expression], pl.List)
    else:
        is_list = isinstance(selected.collect_schema()['k'], pl.List)
    if is_list:
        selected = selected.explode('k')
    counts = (
        selected
        .with_columns(pl.col('k').cast(pl.String).fill_null(NULL_KEY))
        .group_by('k')
        .agg(pl.len().alias('count'))
        .collect()
    )
    ordered = sorted(counts.iter_rows(), key=lambda kv: (-kv[1], kv[0]))
    return {key: int(count) for key, count in ordered}


def compute_filter_count(
    frame: pl.DataFrame | pl.LazyFrame,
    filter_expr: str,
    distinct: str | None = None,
) -> int:
    """Count rows (or distinct values) matching a SQL filter expression.

    Args:
        frame: dataset to filter.
        filter_expr: SQL boolean expression.
        distinct: when given, count distinct non-null values of this column. A null is not a
            distinct value.

    Returns:
        Matching row count, or distinct count of `distinct`.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"score": [0.9, 0.2, 0.7], "geneId": ["g1", "g2", "g1"]})
        >>> compute_filter_count(df, "score > 0.5")
        2
        >>> compute_filter_count(df, "score > 0.5", distinct="geneId")
        1
    """
    filtered = _as_lazy(frame).filter(pl.sql_expr(filter_expr))
    if distinct is not None:
        return int(filtered.select(pl.col(distinct).drop_nulls().n_unique()).collect().item())
    return int(filtered.select(pl.len()).collect().item())


def _dataset_file_stats(dataset_path: str) -> tuple[int, int]:
    """Return `(total bytes, number of parquet files)` for a dataset directory.

    Storage-agnostic via fsspec (local `LocalFileSystem` or `gs://` via gcsfs). Counts top-level
    `*.parquet` files, covering both spark `part-*.parquet` output and the numbered parts
    `write_dataset` produces; `_SUCCESS` and `.crc` markers are ignored.

    fsspec rather than otter's `StorageHandle`, which is used everywhere else in this module: one
    `ls` returns every name AND its size, where the handle would need a `stat` per part, and each
    of those is a `blob.reload()` round trip on GCS. Both authenticate through ambient ADC, so
    this is not a credential divergence.

    Args:
        dataset_path: absolute path to the dataset directory.

    Returns:
        Total bytes and number of parquet files.
    """
    fs, root = fsspec.core.url_to_fs(dataset_path)
    parts = [
        entry
        for entry in fs.ls(root, detail=True)
        if entry.get('type') != 'directory' and entry['name'].rstrip('/').endswith('.parquet')
    ]
    file_size = sum(int(entry.get('size') or 0) for entry in parts)
    return file_size, len(parts)


def _metric_row(
    run: str,
    dataset: str,
    kind: str,
    metric: str,
    value: int,
    expression: str | None = None,
    group_value: str | None = None,
) -> dict[str, Any]:
    """Build one tidy measurement row."""
    return {
        'run': run,
        'dataset': dataset,
        'kind': kind,
        'metric': metric,
        'expression': expression,
        'group_value': group_value,
        'value': int(value),
    }


def profile_dataset(
    dataset_path: str,
    name: str,
    dataset_config: dict[str, Any],
    run: str,
) -> list[dict[str, Any]] | None:
    """Profile one dataset into tidy rows, or `None` if it cannot be read.

    Emits scalar rows (`count`, `file_size`, `number_of_partitions`), one row per grouping bucket,
    and one row per filter.

    A dataset that cannot be read or listed is skipped with a warning, because a release root
    holds directories this step has no business failing over. A malformed grouping or filter
    expression raises instead: that is an author error in `config.yaml`, and silently dropping the
    metric would leave a gap nobody notices.

    Args:
        dataset_path: absolute path to the dataset directory.
        name: dataset id (basename).
        dataset_config: optional `{groupings, filter_counts}` overlay.
        run: run identifier stamped on every row.

    Returns:
        Tidy rows, or None when the dataset cannot be read.

    Raises:
        ValueError: if a configured grouping or filter expression fails to evaluate.
    """
    try:
        count = count_parquet_rows(dataset_path)
        file_size, n_partitions = _dataset_file_stats(dataset_path)
    except Exception:
        logger.opt(exception=True).warning(f'Skipping unreadable dataset `{name}` at {dataset_path}')
        return None

    rows: list[dict[str, Any]] = [
        _metric_row(run, name, 'scalar', 'count', count),
        _metric_row(run, name, 'scalar', 'file_size', file_size),
        _metric_row(run, name, 'scalar', 'number_of_partitions', n_partitions),
    ]

    groupings = dataset_config.get('groupings', {})
    filters = dataset_config.get('filter_counts', [])
    if not groupings and not filters:
        return rows

    lazy_frame = scan_dataset(dataset_path)
    schema = lazy_frame.collect_schema()

    for grouping_name, expression in groupings.items():
        try:
            counts = compute_breakdown(lazy_frame, expression, schema=schema)
        except Exception as e:
            msg = f"Failed to compute grouping '{grouping_name}' (expression '{expression}') for dataset '{name}'"
            raise ValueError(msg) from e
        rows.extend(
            _metric_row(run, name, 'grouping', grouping_name, count, expression=expression, group_value=bucket)
            for bucket, count in counts.items()
        )

    for spec in filters:
        filter_expr = spec['filter']
        distinct = spec.get('distinct')
        try:
            match_count = compute_filter_count(lazy_frame, filter_expr, distinct)
        except Exception as e:
            msg = f"Failed to compute filter '{spec['name']}' (filter '{filter_expr}') for dataset '{name}'"
            raise ValueError(msg) from e
        definition = f'distinct {distinct} where {filter_expr}' if distinct else filter_expr
        rows.append(_metric_row(run, name, 'filter', spec['name'], match_count, expression=definition))

    return rows


def _rows_to_document(rows: list[dict[str, Any]], run: str) -> dict[str, Any]:
    """Nest tidy measurement rows into the published document.

    Tidy rows are the right shape to compute and to test against; a page reading one dataset's
    numbers is not served by filtering a flat list, so it gets them keyed by dataset name. The
    expression behind each grouping and filter is NOT carried over -- `config.yaml` defines it and
    is the version-controlled answer to "what was counted"; duplicating it here would be a second
    copy to keep true.

    Datasets are ordered by name so two runs diff cleanly; groupings keep the descending-count
    order `compute_breakdown` produced.

    Args:
        rows: measurement rows from `profile_dataset`.
        run: run identifier for the document header.

    Returns:
        `{'run': ..., 'datasets': {name: {count, file_size, number_of_partitions, groupings,
        filter_counts}}}`.
    """
    datasets: dict[str, dict[str, Any]] = {}
    groupings: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))

    for row in rows:
        entry = datasets.setdefault(
            row['dataset'],
            {**dict.fromkeys(SCALAR_METRICS), 'groupings': {}, 'filter_counts': {}},
        )
        if row['kind'] == 'scalar':
            entry[row['metric']] = row['value']
        elif row['kind'] == 'grouping':
            groupings[row['dataset']][row['metric']][row['group_value']] = row['value']
        else:
            entry['filter_counts'][row['metric']] = row['value']

    for dataset, dataset_groupings in groupings.items():
        datasets[dataset]['groupings'] = {name: dict(buckets) for name, buckets in dataset_groupings.items()}

    return {'run': run, 'datasets': {name: datasets[name] for name in sorted(datasets)}}


def _config_for_dataset(name: str, datasets_config: dict[str, Any]) -> dict[str, Any]:
    """Return the config overlay for a dataset, supporting fnmatch pattern keys.

    Exact-name keys take precedence; otherwise the most specific matching glob pattern wins
    (longest pattern first, then alphabetical for determinism). Returns `{}` when nothing matches.

    Args:
        name: dataset basename.
        datasets_config: the `settings.datasets` overlay.

    Returns:
        The matched overlay, or an empty dict.
    """
    if name in datasets_config:
        return datasets_config[name]
    for pattern in sorted(datasets_config, key=lambda p: (-len(p), p)):
        if fnmatch(name, pattern):
            return datasets_config[pattern]
    return {}


def dataset_metrics(
    source: dict[str, Path],
    destination: dict[str, Path],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Profile every discovered dataset into one JSON document.

    Written through `StorageHandle`, which addresses `gs://` natively. A `pathlib` write would
    collapse the scheme, put the document on the container's local disk, and still report success.

    Args:
        source: unused.
        destination: `{"json": <file>}`, already resolved against the release root by otter.
        settings: `metric_scopes` (default `['/output/*']`) and a `datasets` overlay keyed by
            dataset basename or fnmatch pattern (`{groupings, filter_counts}`).
        config: injected; discovery reads `release_uri` or `work_path`, and the `run` identifier
            is the last path segment of that root.
    """
    del source

    data_root_uri = config.release_uri or str(config.work_path)
    run = data_root_uri.rstrip('/').rsplit('/', maxsplit=1)[-1]
    scope_globs = list(settings.get('metric_scopes', ['/output/*']))
    datasets_config = settings.get('datasets', {})

    discovered = discover_dataset_paths(data_root_uri, scope_globs, config)

    all_rows: list[dict[str, Any]] = []
    profiled = 0
    for rel_path in sorted(discovered):
        name = Path(rel_path).name
        rows = profile_dataset(discovered[rel_path], name, _config_for_dataset(name, datasets_config), run)
        if rows is None:
            continue
        all_rows.extend(rows)
        profiled += 1

    out_file = str(destination['json'])
    document = _rows_to_document(all_rows, run)
    StorageHandle(out_file).write_text(json.dumps(document, indent=2))
    logger.info(f'Profiled {profiled} datasets (of {len(discovered)} discovered) into {out_file}')
