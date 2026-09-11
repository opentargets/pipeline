"""Shared parquet discovery and counting helpers for transformer modules.

Used by both `release_metrics` and `dataset_metrics` to discover datasets under a release/work
root and to count rows without loading data.

Discovery globs for the DATA FILES rather than the directories themselves, because otter's GCS
backend implements glob as `bucket.list_blobs(match_glob=...)`, which matches OBJECT names -- and
a directory is not an object, so globbing directory names returns zero hits on GCS, the only
backend this runs against in production. This mirrors what `utils.dataset.scan_datasets` does for
the same reason.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.utils.dataset import scan_dataset


def to_parquet_glob(path: str | Path) -> str:
    """Normalize a dataset path to a parquet glob consumable by Polars."""
    path_str = str(path)
    if '.parquet' in path_str:
        return path_str
    return f'{path_str.rstrip("/")}/*.parquet'


def count_parquet_rows(path: str) -> int:
    """Count rows from a parquet dataset lazily without loading full data.

    Reads through `scan_dataset`, so a dataset written as a directory of size-capped parts counts
    every part. Takes a dataset PATH, never a glob -- `scan_dataset` enumerates the directory
    itself and refuses a glob.
    """
    return int(scan_dataset(path).select(pl.len()).collect().item())


def _to_release_relative_path(
    path: str,
    release_uri: str,
) -> str:
    """Convert an absolute dataset path into a release-relative path key."""
    release_root = release_uri.rstrip('/')
    if path.startswith(release_root):
        relative = path[len(release_root) :]
    else:
        relative = path

    relative = relative.rstrip('/')
    if not relative.startswith('/'):
        relative = f'/{relative}'
    return relative


def _build_absolute_scope_pattern(
    release_uri: str,
    scope: str,
) -> str:
    """Build an absolute storage glob pattern from release URI and scope."""
    scope_path = scope if scope.startswith('/') else f'/{scope}'
    return f'{release_uri.rstrip("/")}{scope_path}'


def _scope_to_parquet_file_glob(scope: str) -> str:
    """Convert a dataset scope into a parquet-file scope for robust discovery."""
    normalized = scope.rstrip('/')
    if normalized.endswith('.parquet'):
        return normalized
    return f'{normalized}/*.parquet'


def _dataset_path_from_parquet_file(path: str) -> str:
    """Return the dataset directory path for a parquet file URI/path."""
    return path.rsplit('/', maxsplit=1)[0]


def _has_glob_wildcards(path_pattern: str) -> bool:
    """Return whether a path pattern contains glob wildcards."""
    return any(char in path_pattern for char in '*?[')


def _expand_storage_glob(
    path_pattern: str,
    config: Config,
) -> list[str]:
    """Expand a storage glob pattern into concrete dataset paths.

    Example:
        path_pattern='gs://bucket/release/output/*/*.parquet'
        -> root='gs://bucket/release/output'
        -> glob pattern='*/*.parquet'
        -> returns e.g.
           [
               'gs://bucket/release/output/disease/part-00000.parquet',
               'gs://bucket/release/output/target/part-00000.parquet',
           ]
    If no wildcards are present, returns [path_pattern] unchanged.
    """
    wildcard_positions = [
        idx
        for idx in (
            path_pattern.find('*'),
            path_pattern.find('?'),
            path_pattern.find('['),
        )
        if idx != -1
    ]
    if not wildcard_positions:
        return [path_pattern]

    first_wildcard = min(wildcard_positions)
    slash_idx = path_pattern.rfind('/', 0, first_wildcard)
    if slash_idx == -1:
        msg = f'Invalid scope pattern: {path_pattern}'
        raise ValueError(msg)

    root = path_pattern[:slash_idx]
    pattern = path_pattern[slash_idx + 1 :]
    return sorted(StorageHandle(root, config=config).glob(pattern))


def discover_dataset_paths(
    release_uri: str,
    scope_globs: list[str],
    config: Config,
) -> dict[str, str]:
    """Discover datasets from configured scopes and key them by release-relative path."""
    discovered: dict[str, str] = {}
    for scope in scope_globs:
        abs_pattern = _build_absolute_scope_pattern(release_uri, scope)
        if not _has_glob_wildcards(scope):
            for match in _expand_storage_glob(abs_pattern, config):
                dataset_path = _dataset_path_from_parquet_file(match) if '.parquet' in match else match.rstrip('/')
                relative = _to_release_relative_path(dataset_path, release_uri)
                discovered[relative] = dataset_path

        abs_parquet_pattern = _build_absolute_scope_pattern(release_uri, _scope_to_parquet_file_glob(scope))
        for parquet_file in _expand_storage_glob(abs_parquet_pattern, config):
            dataset_path = _dataset_path_from_parquet_file(parquet_file)
            relative = _to_release_relative_path(dataset_path, release_uri)
            discovered[relative] = dataset_path
    return discovered
