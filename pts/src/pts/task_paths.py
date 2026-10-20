"""Source and destination path shapes shared by the transform and pyspark tasks."""

from collections.abc import Mapping
from typing import cast

from otter.config.model import Config
from otter.storage.util import make_absolute

DestinationPaths = str | dict[str, str]
SourcePaths = str | dict[str, str | list[str]]


def resolve_source_paths(source: SourcePaths, config: Config) -> SourcePaths:
    """Resolve paths with otter's `make_absolute`, also accepting a list inside a named source.

    `make_absolute` returns a string for a string, but is annotated `str | dict[str, str]`.
    """
    if isinstance(source, str):
        return cast(str, make_absolute(source, config))
    return {
        key: (
            [cast(str, make_absolute(path, config)) for path in value]
            if isinstance(value, list)
            else cast(str, make_absolute(value, config))
        )
        for key, value in source.items()
    }


def flatten_paths(paths: str | Mapping[str, str | list[str]]) -> str | list[str]:
    """All paths as a flat list (or the single path), for recording task artifacts."""
    if isinstance(paths, str):
        return paths
    return [path for value in paths.values() for path in (value if isinstance(value, list) else [value])]
