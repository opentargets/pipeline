"""The Polars schema of a dataset."""

import polars as pl

from ot_datamodels.base import DatasetModel
from ot_datamodels.spec import dataset_spec


def to_polars_schema(model: type[DatasetModel]) -> pl.Schema:
    """The Polars schema of a dataset: its columns and their dtypes, in declaration order."""
    return pl.Schema({f.name: f.node.dtype for f in dataset_spec(model).fields})
