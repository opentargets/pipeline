"""Biosample core annotation: the `biosample` output.

Generated from the 26.09 release output (35,477 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey


class Biosample(DatasetModel):
    """Biosample core annotation."""

    dataset_name: ClassVar[str] = 'biosample'
    dataset_title: ClassVar[str] = 'Biosample core annotation'
    dataset_description: ClassVar[str] = (
        'Integration of metadata related to tissues, cell types or cell lines '
        '(biosamples) derived from the UBERON, CL and EFO ontologies'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Ontology',)
    example: ClassVar[dict] = {
        'biosampleId': 'BFO_0000017',
        'biosampleName': 'realizable entity',
        'description': 'a core relation that holds between a part and its whole',
        'xrefs': ['BFO:0000050'],
        'synonyms': ['anterior to (developmentally)'],
        'parents': ['BFO_0000020'],
        'ancestors': ['BFO_0000001'],
        'children': ['BFO_0000016'],
        'descendants': ['BFO_0000016'],
    }

    biosampleId: Annotated[str, PrimaryKey()] = Field(min_length=1, description='Unique identifier for the biosample')
    biosampleName: str = Field(min_length=1, description='Name of the biosample')
    description: str | None = Field(min_length=1, description='Description of the biosample')
    xrefs: list[Annotated[str, Field(min_length=1)]] = Field(description='Cross-reference IDs from other ontologies')
    synonyms: list[Annotated[str, Field(min_length=1)]] = Field(description='List of synonymous names for the term')
    parents: list[Annotated[str, Field(min_length=1)]] | None = Field(
        min_length=1, description='Direct parent biosample IDs in the ontology'
    )
    ancestors: list[Annotated[str, Field(min_length=1)]] = Field(
        description='List of ancestor biosample IDs in the ontology'
    )
    children: list[Annotated[str, Field(min_length=1)]] | None = Field(
        min_length=1, description='Direct child biosample IDs in the ontology'
    )
    descendants: list[Annotated[str, Field(min_length=1)]] = Field(
        description='List of descendant biosample IDs in the ontology'
    )
