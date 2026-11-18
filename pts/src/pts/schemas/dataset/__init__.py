"""Declare PTS datasets once, as Pydantic models, and derive their validation and documentation.

* `base` -- `DatasetModel`/`StructModel` and the `Annotated` markers (`PrimaryKey`, `Reference`, ...).
* `spec` -- the field tree every other module reads, and the supported subset of Pydantic.
* `converter` -- the pandera schema a dataset is validated against (`to_pandera`).
* `validate` -- validate a written dataset, one part at a time (`validation_report`).
* `docs` -- documentation JSON.
* `discovery` -- every dataset model under `pts.schemas.outputs`, without a registry.
"""

from pts.schemas.dataset.base import (
    Bioregistry,
    DatasetModel,
    Note,
    PolarsDtype,
    PrimaryKey,
    Reference,
    RowCheck,
    StructModel,
)
from pts.schemas.dataset.converter import to_pandera, to_polars_schema
from pts.schemas.dataset.spec import SchemaDefinitionError

__all__ = [
    'Bioregistry',
    'DatasetModel',
    'Note',
    'PolarsDtype',
    'PrimaryKey',
    'Reference',
    'RowCheck',
    'SchemaDefinitionError',
    'StructModel',
    'to_pandera',
    'to_polars_schema',
]
