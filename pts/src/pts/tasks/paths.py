"""Source and destination path shapes shared by the transform and pyspark tasks."""

from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

DestinationPaths = str | dict[str, str]
SourcePaths = str | dict[str, str | list[str]]


def resolve_source_paths(source: SourcePaths, config: Config) -> SourcePaths:
    """Resolve source paths against the config, preserving named groups and lists."""
    if isinstance(source, str):
        return StorageHandle(source, config=config).absolute
    return {
        key: (
            [StorageHandle(path, config=config).absolute for path in value]
            if isinstance(value, list)
            else StorageHandle(value, config=config).absolute
        )
        for key, value in source.items()
    }


def flatten_paths(paths: SourcePaths) -> str | list[str]:
    """All paths as a flat list (or the single path), for recording task artifacts."""
    if isinstance(paths, str):
        return paths
    return [path for value in paths.values() for path in (value if isinstance(value, list) else [value])]
