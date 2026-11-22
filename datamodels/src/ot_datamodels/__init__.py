"""Declare the datasets of the release once, as Pydantic models, and derive their Polars schema and documentation.

* `base` -- `DatasetModel`/`StructModel` and the `Annotated` markers (`PrimaryKey`, `Reference`, ...).
* `spec` -- the field tree every other module reads, and the supported subset of Pydantic.
* `polars` -- the Polars schema of a dataset (`to_polars_schema`).
* `docs` -- documentation JSON.
* `discovery` -- every dataset model under `ot_datamodels.outputs`, without a registry.
* `derive` -- view columns copied from output columns (`from_output`, `row_of`).
"""

from ot_datamodels.base import (
    Bioregistry,
    ClickHouseType,
    DatasetModel,
    Note,
    PolarsDtype,
    PrimaryKey,
    Reference,
    RowCheck,
    StructModel,
)
from ot_datamodels.derive import from_output, row_of
from ot_datamodels.polars import to_polars_schema
from ot_datamodels.spec import SchemaDefinitionError

__all__ = [
    'Bioregistry',
    'ClickHouseType',
    'DatasetModel',
    'Note',
    'PolarsDtype',
    'PrimaryKey',
    'Reference',
    'RowCheck',
    'SchemaDefinitionError',
    'StructModel',
    'from_output',
    'row_of',
    'to_polars_schema',
]
