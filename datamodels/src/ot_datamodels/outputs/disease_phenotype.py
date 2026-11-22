"""Disease/Phenotype - clinical signs and symptoms: the `disease_phenotype` output.

Generated from the 26.12 release output (168,445 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar

import polars as pl
from pydantic import Field

from ot_datamodels import Bioregistry, DatasetModel, PolarsDtype, PrimaryKey, Reference, StructModel
from ot_datamodels.outputs.patterns import ONTOLOGY_ID
from ot_datamodels.vocabularies import HpoAspect, HpoEvidenceCode, HpoResource, HpoSex


class Evidence(StructModel):
    """`evidence[]`."""

    aspect: Annotated[HpoAspect, PolarsDtype(pl.Enum)] = Field(
        description='The category of biological information being provided (e.g., clinical, genetic, etc.)'
    )
    bioCuration: str = Field(min_length=1, description='Indicates whether the evidence has been manually curated')
    diseaseFromSourceId: str = Field(
        min_length=1, description='Unique identifier for the disease from the original source'
    )
    diseaseFromSource: str = Field(min_length=1, description='The disease name as recorded in the original source')
    diseaseName: str = Field(min_length=1, description='Standardised disease name')
    evidenceType: Annotated[HpoEvidenceCode, PolarsDtype(pl.Enum)] = Field(
        description='The type of evidence supporting the disease-phenotype relationship'
    )
    frequency: str | None = Field(
        min_length=1, description=('The observed frequency of the phenotype in individuals with the disease')
    )
    modifiers: list[Annotated[str, Field(min_length=1)]] = Field(
        description='Additional characteristics or modifiers related to the phenotype'
    )
    onset: list[Annotated[str, Field(min_length=1)]] = Field(
        description='Age or stage of disease onset for the phenotype'
    )
    qualifier: str | None = Field(description='Additional qualifiers specifying the phenotype-disease relationship')
    qualifierNot: bool = Field(description='Specifies if a phenotype is not observed in a given disease')
    references: list[Annotated[Annotated[str, Bioregistry('pubmed')], Field(min_length=1)]] = Field(
        description='References or citations supporting the evidence'
    )
    sex: Annotated[HpoSex, PolarsDtype(pl.Enum)] | None = Field(
        description='Indicates if the association is specific to a biological sex'
    )
    resource: Annotated[HpoResource, PolarsDtype(pl.Enum)] = Field(
        description='The data source from which the evidence was obtained'
    )


class DiseasePhenotype(DatasetModel):
    """Disease/Phenotype - clinical signs and symptoms."""

    dataset_name: ClassVar[str] = 'disease_phenotype'
    dataset_title: ClassVar[str] = 'Disease/Phenotype - clinical signs and symptoms'
    dataset_description: ClassVar[str] = (
        'Clinical signs and symptoms observed in diseases or phenotypes. Signs '
        'and symptoms are integrated from multiple sources including EFO, '
        'MONDO and HPO'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Disease',)
    example: ClassVar[dict] = {
        'disease': 'MONDO_0014076',
        'phenotype': 'HP_0001263',
        'evidence': [
            {
                'aspect': 'P',
                'bioCuration': 'HPO:skoehler[2013-06-05]',
                'diseaseFromSourceId': 'OMIM:615190',
                'diseaseFromSource': 'Dyskeratosis congenita, autosomal recessive 5',
                'diseaseName': 'dyskeratosis congenita, autosomal recessive 5',
                'evidenceType': 'TAS',
                'frequency': 'HP_0040281',
                'modifiers': ['HP:0012840'],
                'onset': ['HP:0011461'],
                'qualifier': None,
                'qualifierNot': False,
                'references': ['OMIM:615190'],
                'sex': 'MALE',
                'resource': 'HPO',
            }
        ],
    }

    disease: Annotated[str, Reference('disease', 'id'), PrimaryKey()] = Field(
        pattern=ONTOLOGY_ID, description='Disease identifier'
    )
    phenotype: Annotated[str, PrimaryKey()] = Field(
        pattern=ONTOLOGY_ID, description='The phenotype linked to the disease.'
    )
    evidence: list[Evidence] = Field(
        description=('A container for all evidence-related attributes supporting the disease-phenotype association.')
    )
