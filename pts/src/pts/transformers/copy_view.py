"""Copy a served output into its view: the columns of the view model, in model order, selected from the output.

The view is the model whose `dataset_name` is the destination's basename, as pos names a view's `input_dir`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from ot_datamodels.base import DatasetModel
from ot_datamodels.discovery import VIEWS_PACKAGE, discover_models
from otter.config.model import Config

from pts.schemas.dataset.validate import validate_dataset
from pts.transformers.utils.dataset import scan_dataset, write_dataset


def view_model(destination: str) -> type[DatasetModel]:
    """The view model whose dataset name is the basename of `destination`.

    Raises:
        ValueError: if the basename names no view model.
    """
    name = Path(str(destination)).name
    models = discover_models([VIEWS_PACKAGE])
    if name not in models:
        raise ValueError(f'{name!r} (from {destination}) names no view model')
    return models[name]


def copy_view(
    source: str,
    destination: str,
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Write the view's columns of the source, in model order, streaming.

    Args:
        source: the output dataset the view copies.
        destination: the view directory; its basename is the view's dataset name.
        settings: unused.
        config: unused.

    Raises:
        ValueError: if the destination names no view model, or the source lacks a view column.
    """
    model = view_model(destination)
    columns = list(model.model_fields)
    frame = scan_dataset(str(source))
    missing = [column for column in columns if column not in frame.collect_schema().names()]
    if missing:
        raise ValueError(f'{source} lacks the columns {missing} of {model.__name__}')
    logger.info(f'copying {source} into {destination} as {model.__name__}')
    write_dataset(frame.select(columns), str(destination))


def validate(
    source: str,
    destination: str,
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Validate the written view against its model."""
    validate_dataset(view_model(destination), str(destination))
