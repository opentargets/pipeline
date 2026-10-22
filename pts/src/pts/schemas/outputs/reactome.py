"""Target - Reactome pathway information: the `reactome` output.

Generated from the 26.09 release output (2,883 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey


class Reactome(DatasetModel):
    """Target - Reactome pathway information."""

    dataset_name: ClassVar[str] = 'reactome'
    dataset_title: ClassVar[str] = 'Target - Reactome pathway information'
    dataset_description: ClassVar[str] = 'Pathway metadata from Reactome pathway database'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Ontology',)
    example: ClassVar[dict] = {
        'id': 'R-HSA-164843',
        'label': '2-LTR circle formation',
        'ancestors': ['R-HSA-162587'],
        'descendants': ['R-HSA-5678420'],
        'children': ['R-HSA-5678420'],
        'parents': ['R-HSA-162592'],
        'path': [['R-HSA-1643685']],
    }

    id: Annotated[str, Bioregistry('reactome'), PrimaryKey()] = Field(
        min_length=1, description='Unique identifier for the Reactome pathway'
    )
    label: str = Field(min_length=1, description='Name or label of the pathway')
    ancestors: list[Annotated[str, Field(min_length=1)]] = Field(
        description='List of ancestor pathways in the ontology'
    )
    descendants: list[Annotated[str, Field(min_length=1)]] = Field(
        description='List of descendant pathways in the ontology'
    )
    children: list[Annotated[str, Field(min_length=1)]] = Field(
        description='Direct child pathways of the current pathway'
    )
    parents: list[Annotated[str, Field(min_length=1)]] = Field(
        description='Direct parent pathways of the current pathway'
    )
    path: list[Annotated[list[Annotated[str, Field(min_length=1)]], Field(min_length=1)]] = Field(
        min_length=1, description='Hierarchical path representing the position of the pathway in Reactome'
    )
