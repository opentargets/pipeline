"""Cancer Gene Census evidence: the `evidence_cancer_gene_census` output.

Generated from the 26.09 release output (91,572 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX


class MutatedSamples(StructModel):
    """`mutatedSamples[]`."""

    functionalConsequenceId: Annotated[str, Bioregistry('so'), Reference('so', 'id')] = Field(
        pattern=ONTOLOGY_ID, description='Functional consequene identifier of the pooled mutations'
    )
    numberMutatedSamples: int = Field(
        ge=1, le=24226, description=('Number of cohort samples in which target is mutated with a mutation of any type')
    )
    numberSamplesTested: int = Field(ge=1, le=71128, description='Number of cohort samples tested')
    numberSamplesWithMutationType: int = Field(
        ge=1,
        le=23978,
        description=('Number of cohort samples in which target is mutated with a specific mutation type'),
    )


class EvidenceCancerGeneCensus(DatasetModel):
    """Cancer Gene Census evidence."""

    dataset_name: ClassVar[str] = 'evidence_cancer_gene_census'
    dataset_title: ClassVar[str] = 'Cancer Gene Census evidence'
    dataset_description: ClassVar[str] = 'Disease-target evidence based on Cancer Gene Census provided by COSMIC'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000147548',
        'id': '01e1d217b34b617d35c7e270e435664a8bd7fbfd',
        'targetFromSourceId': 'ENSG00000147548',
        'diseaseFromSourceMappedId': 'EFO_1000141',
        'datasourceId': 'cancer_gene_census',
        'datatypeId': 'somatic_mutation',
        'diseaseFromSourceId': None,
        'literature': ['24705251'],
        'mutatedSamples': [
            {
                'functionalConsequenceId': 'SO_0001059',
                'numberMutatedSamples': 1,
                'numberSamplesTested': 56,
                'numberSamplesWithMutationType': 1,
            }
        ],
        'resourceScore': 0.5,
        'studyId': '322',
        'qualityControls': [],
        'diseaseId': 'EFO_1000141',
        'publicationDate': '2014-04-06',
        'evidenceDate': '2014-04-06',
        'score': 0.5,
        'directionOnTrait': 'risk',
        'directionOnTarget': 'GoF',
    }

    targetId: Annotated[str, Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        pattern=ENSEMBL_GENE_ID,
        description=(
            'Target ID in resource of origin (accepted sources include Ensembl '
            'gene ID, Uniprot ID, gene symbol), only capital letters are accepted'
        ),
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    datasourceId: Literal['cancer_gene_census'] = Field(description='Identifer of the evidence source')
    datatypeId: Literal['somatic_mutation'] = Field(description='Type of the evidence')
    diseaseFromSourceId: str | None = Field(description='Disease identifier from the original source')
    literature: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        min_length=1, description='List of PubMed or preprint reference identifiers'
    )
    mutatedSamples: list[MutatedSamples] = Field(min_length=1, description='Samples with a given mutation tested')
    resourceScore: float = Field(
        ge=0.25, le=1.0, description=('Score provided by datasource indicating strength of target-disease association')
    )
    studyId: str | None = Field(pattern=DIGITS, description='Identifier of the study generating the data')
    qualityControls: list[str] = Field(max_length=0, description='Evidence quality flags')
    diseaseId: Annotated[str, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    publicationDate: str | None = Field(
        pattern=ISO_DATE, description='Date of the earliest publication supporting the evidence'
    )
    evidenceDate: str | None = Field(pattern=ISO_DATE, description='Earliest data for the evidence')
    score: float = Field(
        ge=0.25,
        le=1.0,
        description=('Score of the evidence reflecting the strength of the disease/target relationship'),
    )
    directionOnTrait: Literal['risk'] = Field(description='Predicted direction of effect on the trait')
    directionOnTarget: Literal['GoF', 'LoF'] | None = Field(
        description=(
            'Gain or loss of function effect of the evidence on the target '
            'resulting from genetic variants, pharmacological modulation, or other '
            'perturbations'
        )
    )
