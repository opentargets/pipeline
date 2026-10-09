"""The biosample index: the output of `pts_biosample`.

Descriptions are copied verbatim from croissant's `biosample.json`, as is `biosampleId`'s
`isPrimaryKey` (as `PrimaryKey`); croissant tags no references or registries here.

`biosampleId` has no pattern: a real output (35,477 rows) spans dozens of ontology prefixes
(`UBERON`, `CL`, `EFO`, `GO`, `PR`, plus ~35 rarer ones pulled in as imported terms), so no single
regex is true of it. All six relationship lists are nullable: gentropy's reference output has null
`parents`/`children`, which its merge logic does not explain, so nullability is asserted uniformly
rather than guaranteed per column. Their elements are never null.
"""

from typing import Annotated, ClassVar

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey


class BiosampleIndex(DatasetModel):
    """Biosample index merged from Cell Ontology, Uberon and EFO."""

    dataset_name: ClassVar[str] = 'biosample'
    dataset_description: ClassVar[str] = 'Biosample index merged from Cell Ontology, Uberon and EFO'
    example: ClassVar[dict] = {
        'biosampleId': 'UBERON_0002107',
        'biosampleName': 'liver',
        'description': 'An exocrine gland which secretes bile and functions in metabolism.',
        'xrefs': ['FMA:7197'],
        'synonyms': ['iecur'],
        'parents': ['UBERON_0002423'],
        'ancestors': ['UBERON_0002423', 'UBERON_0000062'],
        'children': ['UBERON_0001114'],
        'descendants': ['UBERON_0001114'],
    }

    biosampleId: Annotated[str, PrimaryKey()] = Field(description='Unique identifier for the biosample')
    biosampleName: str = Field(description='Name of the biosample')
    description: str | None = Field(description='Description of the biosample')
    xrefs: list[str] | None = Field(description='Cross-reference IDs from other ontologies')
    synonyms: list[str] | None = Field(description='List of synonymous names for the term')
    parents: list[str] | None = Field(description='Direct parent biosample IDs in the ontology')
    ancestors: list[str] | None = Field(description='List of ancestor biosample IDs in the ontology')
    children: list[str] | None = Field(description='Direct child biosample IDs in the ontology')
    descendants: list[str] | None = Field(description='List of descendant biosample IDs in the ontology')
