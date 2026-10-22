"""Sequence Ontology: the `so` output.

Generated from the 26.09 release output (2,620 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey


class So(DatasetModel):
    """Sequence Ontology."""

    dataset_name: ClassVar[str] = 'so'
    dataset_title: ClassVar[str] = 'Sequence Ontology'
    dataset_description: ClassVar[str] = 'Sequence ontology lookup table'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Ontology',)
    example: ClassVar[dict] = {'id': 'SO:0005854', 'label': 'gene_cassette_array'}

    id: Annotated[str, Bioregistry('so'), PrimaryKey()] = Field(
        min_length=1, description='Sequence ontology term identifier'
    )
    label: str = Field(min_length=1, description='Sequence ontology term label')
