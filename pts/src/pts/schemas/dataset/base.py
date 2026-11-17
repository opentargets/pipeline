r"""Base classes and annotation markers for declaring PTS datasets as Pydantic models.

A dataset is declared once, as a `DatasetModel` subclass, and everything else is derived from it:
the pandera schema written datasets are validated against (`pts.schemas.dataset.to_pandera`) and the
documentation JSON (`pts.schemas.dataset.docs`). The models are a specification: Pydantic never
validates data with them. `DatasetModel.validate_dataset` validates a written dataset through the
derived pandera schema, one parquet part at a time (`pts.schemas.dataset.validate`).

Facts a Pydantic `Field` has no slot for are attached as `Annotated` markers, so they can sit on a
top-level column, a nested struct field or a list element alike::

    targetId: Annotated[str, Reference('target', 'id'), Bioregistry('ensembl')] = Field(
        pattern=r'^ENSG\d{11}$', description='Open Targets target identifier'
    )
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl
from pydantic import BaseModel, ConfigDict


class SchemaModel(BaseModel):
    """Shared configuration of every model making up a dataset schema.

    `strict` keeps Pydantic from coercing types the way pandera never would (`'5'` into an `int`),
    and `extra='forbid'` is what makes the derived pandera schema reject undeclared columns.
    """

    model_config = ConfigDict(strict=True, extra='forbid')


class StructModel(SchemaModel):
    """A struct nested inside a dataset column, or inside another struct.

    Field order is significant: it becomes the order of the Polars struct's fields, and Polars
    struct dtypes only compare equal with their fields in the same order.
    """


@dataclass(frozen=True, eq=False)
class RowCheck:
    """A rule each row must satisfy, across its columns, that no field constraint states.

    Attributes:
        name: the rule's name, e.g. `h3_and_h4_are_exclusive`.
        description: what the rule requires, shown in failures and the documentation.
        passes: a boolean expression over the row's columns, true where the row passes.
    """

    name: str
    description: str
    passes: pl.Expr


class DatasetModel(SchemaModel):
    """One output dataset.

    A concrete dataset sets `dataset_name`, `dataset_title`, `dataset_description` and `example`
    in its own class body, and `dataset_tags` where it has any. A subclass that does not set
    `dataset_name` is an abstract base sharing fields between datasets -- never discovered,
    converted or documented on its own.

    Attributes:
        dataset_name: name of the dataset, matching the output name its step writes.
        dataset_title: human-readable name of the dataset, e.g. `Biosample core annotation`.
        dataset_description: description of the dataset.
        dataset_tags: categories the dataset belongs to, e.g. `Target-Disease`.
        example: one valid row, used by the test suite and shown in the documentation. Every list
            in it should hold at least one element, so every nested field is exercised.
        row_checks: rules across a row's columns that no field constraint states. Inherited, so a
            dataset adds to its base's with `Base.row_checks + (...)`.
    """

    dataset_name: ClassVar[str]
    dataset_title: ClassVar[str]
    dataset_description: ClassVar[str]
    dataset_tags: ClassVar[tuple[str, ...]] = ()
    example: ClassVar[dict[str, Any]]
    row_checks: ClassVar[tuple[RowCheck, ...]] = ()

    @classmethod
    def is_concrete(cls) -> bool:
        """Whether this class is a dataset in its own right, rather than a shared base."""
        return 'dataset_name' in cls.__dict__

    @classmethod
    def validate_dataset(cls, path: str) -> None:
        """Validate the dataset written at `path` against this model.

        The call a transformer module's `validate` makes for each dataset it writes, e.g.
        `EvidenceEncore.validate_dataset(destination['evidence'])`. Broken rules are logged as each
        part is validated (`pts.schemas.dataset.validate.validation_report`).

        Args:
            path: the dataset: a directory of parquet parts, or a single parquet file. Local or remote.

        Raises:
            ValueError: if nothing was written at `path`, or the dataset breaks this model's rules.
        """
        # imported here: the validator and the storage layer import this module
        from otter.storage.synchronous.handle import StorageHandle
        from otter.util.errors import NotFoundError

        from pts.schemas.dataset.validate import format_report, validation_report

        try:
            StorageHandle(path).stat()
        except NotFoundError as error:
            raise ValueError(f'{path} was not written, but must satisfy the model {cls.__name__}') from error
        report = validation_report(path, cls)
        if not report.is_empty():
            raise ValueError(f'{path} breaks its model {cls.__name__}:\n{format_report(report)}')


@dataclass(frozen=True)
class PrimaryKey:
    """Marks a top-level column as (part of) the dataset's primary key.

    A single primary-key column is validated as unique; several are validated as unique together.
    """


@dataclass(frozen=True)
class Reference:
    """Marks values that resolve against another dataset's primary key.

    Documentation only: never enforced on a single dataset, since checking it needs the other one.
    May sit on a top-level column, a nested struct field or a list element.

    Attributes:
        dataset: `dataset_name` of the referenced dataset.
        column: the referenced dataset's primary-key column.
    """

    dataset: str
    column: str


@dataclass(frozen=True)
class Bioregistry:
    """Names the identifier registry (https://bioregistry.io) values belong to. Documentation only.

    Attributes:
        prefix: the bioregistry prefix, e.g. `ensembl`.
    """

    prefix: str


@dataclass(frozen=True)
class Note:
    """A free-text remark about a column or struct field. Documentation only.

    Attributes:
        text: the remark.
    """

    text: str


@dataclass(frozen=True)
class PolarsDtype:
    """Overrides the Polars dtype a primitive maps to, where precision matters.

    Must stay within the annotated type's kind: an integer dtype for `int`, a float dtype for
    `float`.

    Attributes:
        dtype: the Polars dtype, e.g. `pl.Int32`.
    """

    dtype: pl.DataType | type[pl.DataType]
