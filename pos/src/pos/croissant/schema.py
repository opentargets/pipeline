"""The release's dataset folders, their models and their hive partitions."""

from __future__ import annotations

from loguru import logger
from ot_datamodels.base import DatasetModel
from ot_datamodels.discovery import discover_datasets
from ot_datamodels.spec import DatasetSpec, dataset_spec, walk
from pyarrow import fs

NO_MODEL_WARNING = '[{dataset}]: no dataset model, left out of the croissant descriptor'

MISSING_REFERENCE_ERROR = 'a referenced dataset is missing from the release: {details}'


def _filesystem(path: str) -> tuple[fs.FileSystem, str]:
    return fs.FileSystem.from_uri(path if '://' in path else f'file://{path}')


def dataset_specs(
    paths: list[str], models: dict[str, type[DatasetModel]] | None = None
) -> list[tuple[str, DatasetSpec]]:
    """Each dataset folder in `paths` that has an output model, with the model's spec.

    A folder is named after its dataset; one with no model is left out with a warning.
    """
    models = discover_datasets() if models is None else models
    out: list[tuple[str, DatasetSpec]] = []
    for path in paths:
        dataset = path.rstrip('/').split('/')[-1]
        if (model := models.get(dataset)) is None:
            logger.warning(NO_MODEL_WARNING.format(dataset=dataset))
            continue
        out.append((path, dataset_spec(model)))
    _check_references(out)
    return out


def _check_references(specs: list[tuple[str, DatasetSpec]]) -> None:
    """Raise if a dataset references one that is not among `specs`."""
    present = {spec.name for _, spec in specs}
    missing: list[str] = []
    for _, spec in specs:
        for column, node in walk(spec):
            if node.reference is not None and node.reference.dataset not in present:
                missing.append(f'{spec.name}/{column} references {node.reference.dataset}')
    if missing:
        raise ValueError(MISSING_REFERENCE_ERROR.format(details='; '.join(missing)))


def partition_columns(path: str) -> list[str]:
    """The hive partition columns of the dataset at `path`: its `<column>=<value>` folders."""
    filesystem, root = _filesystem(path)
    infos = filesystem.get_file_info(fs.FileSelector(root))
    return sorted({
        info.base_name.split('=')[0]
        for info in infos
        if info.type == fs.FileType.Directory and '=' in info.base_name
    })
