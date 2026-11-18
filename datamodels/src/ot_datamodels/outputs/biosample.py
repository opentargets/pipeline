"""Biosample core annotation: the `biosample` output.

Generated from the 26.12 release output (35,594 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar

from pydantic import Field

from ot_datamodels import DatasetModel, PrimaryKey


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
        'biosampleId': 'UBERON_0007958',
        'biosampleName': 'central carpal bone',
        'description': (
            'A small bone occasionally found at the dorsal aspect of the wrist '
            'between the scaphoid, capitate, and trapezoid; it is developed as an '
            'independent cartilage in early foetal life but usually becomes fused '
            'with the scaphoid; it occurs normally in most monkeys.'
        ),
        'xrefs': ['FMA:321692'],
        'synonyms': ['central bone'],
        'parents': ['UBERON_0001435'],
        'ancestors': ['UBERON_0012131'],
        'descendants': ['UBERON_2002043'],
        'children': ['UBERON_2002043'],
    }

    biosampleId: Annotated[str, PrimaryKey()] = Field(min_length=1, description='Unique identifier for the biosample')
    biosampleName: str = Field(min_length=1, description='Name of the biosample')
    description: str | None = Field(min_length=1, description='Description of the biosample')
    xrefs: list[Annotated[str, Field(min_length=1)]] = Field(description='Cross-reference IDs from other ontologies')
    synonyms: list[Annotated[str, Field(min_length=1)]] = Field(description='List of synonymous names for the term')
    parents: list[Annotated[str, Field(min_length=1)]] = Field(
        description='Direct parent biosample IDs in the ontology'
    )
    ancestors: list[Annotated[str, Field(min_length=1)]] = Field(
        description='List of ancestor biosample IDs in the ontology'
    )
    descendants: list[Annotated[str, Field(min_length=1)]] = Field(
        description='List of descendant biosample IDs in the ontology'
    )
    children: list[Annotated[str, Field(min_length=1)]] = Field(
        description='Direct child biosample IDs in the ontology'
    )
