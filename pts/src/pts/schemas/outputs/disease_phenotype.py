"""Disease/Phenotype - clinical signs and symptoms: the `disease_phenotype` output.

Generated from the 26.09 release output (163,973 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import ONTOLOGY_ID


class Evidence(StructModel):
    """`evidence[]`."""

    aspect: Literal['C', 'H', 'I', 'M', 'P'] = Field(
        description=('The category of biological information being provided (e.g., clinical, genetic, etc.)')
    )
    bioCuration: str = Field(min_length=1, description='Indicates whether the evidence has been manually curated')
    diseaseFromSourceId: str = Field(
        min_length=1, description='Unique identifier for the disease from the original source'
    )
    diseaseFromSource: str = Field(min_length=1, description='The disease name as recorded in the original source')
    diseaseName: str = Field(min_length=1, description='Standardised disease name')
    evidenceType: Literal['IEA', 'PCS', 'TAS'] = Field(
        description='The type of evidence supporting the disease-phenotype relationship'
    )
    frequency: str | None = Field(
        min_length=1, description=('The observed frequency of the phenotype in individuals with the disease')
    )
    modifiers: list[Annotated[str, Field(min_length=1)]] = Field(
        description='Additional characteristics or modifiers related to the phenotype'
    )
    onset: list[
        Literal[
            'HP:0003577',
            'HP:0003581',
            'HP:0003584',
            'HP:0003593',
            'HP:0003596',
            'HP:0003621',
            'HP:0003623',
            'HP:0011461',
            'HP:0011462',
            'HP:0011463',
            'HP:0025708',
            'HP:0025709',
            'HP:0025710',
            'HP:0030674',
            'HP:0034197',
            'HP:0034198',
            'HP:0034199',
        ]
    ] = Field(description='Age or stage of disease onset for the phenotype')
    qualifier: str | None = Field(description='Additional qualifiers specifying the phenotype-disease relationship')
    qualifierNot: bool = Field(description='Specifies if a phenotype is not observed in a given disease')
    references: list[Annotated[Annotated[str, Bioregistry('pubmed')], Field(min_length=1)]] = Field(
        min_length=1, description='References or citations supporting the evidence'
    )
    sex: Literal['FEMALE', 'MALE'] | None = Field(
        description='Indicates if the association is specific to a biological sex'
    )
    resource: Literal['HPO'] = Field(description='The data source from which the evidence was obtained')


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
        'disease': 'MONDO_0008791',
        'phenotype': 'HP_0002414',
        'evidence': [
            {
                'aspect': 'P',
                'bioCuration': 'HPO:iea[2009-02-17];HPO:probinson[2021-07-14]',
                'diseaseFromSourceId': 'OMIM:206500',
                'diseaseFromSource': 'Anencephaly',
                'diseaseName': 'anencephaly 1',
                'evidenceType': 'PCS',
                'frequency': '1/1',
                'modifiers': ['HP:0012828'],
                'onset': ['HP:0011461'],
                'qualifier': None,
                'qualifierNot': False,
                'references': ['PMID:28087737'],
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
        min_length=1,
        description=('A container for all evidence-related attributes supporting the disease-phenotype association.'),
    )
