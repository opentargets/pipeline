"""Cancer Gene Census evidence: the `evidence_cancer_gene_census` output.

Generated from the 26.12 release output (74,341 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from ot_datamodels import ClickHouseType, DatasetModel, PrimaryKey, Reference, StructModel
from ot_datamodels.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX


class MutatedSamples(StructModel):
    """`mutatedSamples[]`."""

    functionalConsequenceId: Annotated[str, Reference('so', 'id')] = Field(
        pattern=ONTOLOGY_ID, description=('Sequence ontology (SO) identifier of the mutation type the counts refer to')
    )
    numberMutatedSamples: Annotated[int, ClickHouseType('UInt32')] = Field(
        ge=0,
        le=4294967295,
        description=('Number of COSMIC samples of the disease with a mutation of any type in the target'),
    )
    numberSamplesTested: Annotated[int, ClickHouseType('UInt32')] = Field(
        ge=0,
        le=4294967295,
        description=('Number of COSMIC samples of the disease screened for mutations in the target'),
    )
    numberSamplesWithMutationType: Annotated[int, ClickHouseType('UInt32')] = Field(
        ge=0,
        le=4294967295,
        description=('Number of COSMIC samples of the disease with a mutation of the given type in the target'),
    )


class EvidenceCancerGeneCensus(DatasetModel):
    """Cancer Gene Census evidence."""

    dataset_name: ClassVar[str] = 'evidence_cancer_gene_census'
    dataset_title: ClassVar[str] = 'Cancer Gene Census evidence'
    dataset_description: ClassVar[str] = 'Disease-target evidence based on Cancer Gene Census provided by COSMIC'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000148584',
        'id': 'f9e904c32d71b627a7b349946e82d9c9e0986789',
        'targetFromSourceId': 'ENSG00000148584',
        'diseaseFromSourceMappedId': 'MONDO_0000640',
        'datatypeId': 'somatic_mutation',
        'studyId': '379',
        'mutatedSamples': [
            {
                'functionalConsequenceId': 'SO_0001059',
                'numberMutatedSamples': 6,
                'numberSamplesTested': 449,
                'numberSamplesWithMutationType': 6,
            }
        ],
        'datasourceId': 'cancer_gene_census',
        'resourceScore': 0.5,
        'literature': ['24662767'],
        'diseaseFromSourceId': None,
        'qualityControls': [],
        'diseaseId': 'MONDO_0000640',
        'publicationDate': '2011-06-29',
        'evidenceDate': '2011-06-29',
        'score': 0.5,
        'directionOnTrait': 'risk',
        'directionOnTarget': 'GoF',
    }

    targetId: Annotated[str, Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        pattern=ENSEMBL_GENE_ID, description='Ensembl gene identifier of the target as reported by COSMIC'
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    datatypeId: str = Field(pattern=ONTOLOGY_ID, description='Type of the evidence')
    studyId: str | None = Field(pattern=DIGITS, description='COSMIC identifier of the study reporting the mutations')
    mutatedSamples: list[MutatedSamples] = Field(
        description=('Counts of COSMIC samples with mutations in the target, by mutation type')
    )
    datasourceId: Literal['cancer_gene_census'] = Field(description='Identifer of the evidence source')
    resourceScore: float = Field(
        ge=0.0,
        le=1.0,
        description=('Score of the target-disease association provided by the COSMIC Cancer Gene Census'),
    )
    literature: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        description=('PubMed identifiers of the publications reporting the mutations in COSMIC')
    )
    diseaseFromSourceId: str | None = Field(description='Disease identifier as reported by COSMIC')
    qualityControls: list[str] = Field(max_length=0, description='Evidence quality flags')
    diseaseId: Annotated[str, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    publicationDate: str | None = Field(
        pattern=ISO_DATE, description='Date of the earliest publication supporting the evidence'
    )
    evidenceDate: str | None = Field(pattern=ISO_DATE, description='Earliest data for the evidence')
    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            'Score of the evidence reflecting the strength of the disease/target '
            'relationship; equal to the COSMIC resource score'
        ),
    )
    directionOnTrait: Literal['risk', 'protect'] = Field(description='Predicted direction of effect on the trait')
    directionOnTarget: Literal['GoF', 'LoF'] | None = Field(
        description=(
            'Gain of function for oncogenes and loss of function for tumour '
            'suppressor genes, as classified in the Cancer Gene Census'
        )
    )
