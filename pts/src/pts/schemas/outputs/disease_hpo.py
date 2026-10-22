"""Human Phenotype Ontology: the `disease_hpo` output.

Generated from the 26.09 release output (32,082 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey
from pts.schemas.outputs.patterns import ONTOLOGY_ID


class DiseaseHpo(DatasetModel):
    """Human Phenotype Ontology."""

    dataset_name: ClassVar[str] = 'disease_hpo'
    dataset_title: ClassVar[str] = 'Human Phenotype Ontology'
    dataset_description: ClassVar[str] = 'Human Phenotype Ontology subset of information included in the Platform'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Ontology',)
    example: ClassVar[dict] = {
        'id': 'CHEBI_10545',
        'name': 'electron',
        'description': (
            'Elementary particle not affected by the strong force having a spin \xbd, '
            'a negative elementary charge and a rest mass of 0.000548579903(13) u, '
            'or 0.51099906(15) MeV.'
        ),
        'dbXRefs': ['CALOHA:TS-2035'],
        'parents': ['CHEBI_36338'],
        'obsoleteTerms': ['CHEBI_22318'],
    }

    id: Annotated[str, PrimaryKey()] = Field(
        pattern=ONTOLOGY_ID, description=('Unique identifier for the disease in the Human Phenotype Ontology (HPO)')
    )
    name: str = Field(min_length=1, description='Name of the disease (in HPO)')
    description: str | None = Field(min_length=1, description='Description of the disease')
    dbXRefs: list[Annotated[str, Field(min_length=1)]] | None = Field(
        min_length=1, description='Database cross-references for the HPO disease'
    )
    parents: list[Annotated[str, Field(pattern=ONTOLOGY_ID)]] | None = Field(
        min_length=1, description='List of parent terms for the disease'
    )
    obsoleteTerms: list[Annotated[str, Field(pattern=ONTOLOGY_ID)]] | None = Field(
        min_length=1, description='List of obsolete terms for the disease'
    )
