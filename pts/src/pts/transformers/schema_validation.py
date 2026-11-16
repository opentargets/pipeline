"""Validate written datasets against their models, as a task of its own.

A step writes its datasets without validating them; a `schema_validation` task configured after it
reads them back and checks them against their models (`pts.schemas.dataset.validate`). The task
writes a report of the broken rules, then fails in its validation phase if the report is not
empty, so the report is kept for a failed run::

    - name: transform schema_validation biosample
      requires:
        - transform biosample
      transformer: schema_validation
      source:
        biosample: output/biosample
      destination: intermediate/schema_validation/biosample

Each source is named after the dataset model to validate it against (`dataset_name`).
"""

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.schemas.dataset.discovery import discover_datasets
from pts.schemas.dataset.validate import REPORT_SCHEMA, format_report, validate_dataset
from pts.transformers.utils.dataset import scan_dataset, write_dataset


def schema_validation(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Validate each source dataset against the model of the same name, and write the report.

    Args:
        source: dataset paths, keyed by the `dataset_name` of their models.
        destination: where the report is written; empty if every dataset is valid.
        settings: unused.
        config: unused.

    Raises:
        ValueError: if a source is not a single dataset, or no model has its name.
    """
    if not isinstance(source, dict) or not all(isinstance(path, str) for path in source.values()):
        raise ValueError(f'schema_validation takes one path per dataset, keyed by dataset name; got {source!r}')
    models = discover_datasets()
    unknown = sorted(set(source) - set(models))
    if unknown:
        raise ValueError(f'no dataset model named {unknown}; modelled datasets: {sorted(models)}')

    reports = [validate_dataset(path, models[name]) for name, path in source.items()]
    report = pl.concat([REPORT_SCHEMA.to_frame(), *reports])
    logger.info(f'writing the schema validation report ({report.height} broken rules) to {destination}')
    write_dataset(report, destination)


def schema_validation_check(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Fail if the report `schema_validation` wrote lists any broken rule.

    Raises:
        ValueError: listing the broken rules.
    """
    report = scan_dataset(destination).collect()
    if not report.is_empty():
        raise ValueError(f'{report["dataset"].n_unique()} dataset(s) break their models:\n{format_report(report)}')
