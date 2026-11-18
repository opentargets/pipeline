"""IntOGen evidence: the `evidence_intogen` output.

Generated from the 26.12 release output (4,223 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX


class MutatedSamples(StructModel):
    """`mutatedSamples[]`."""

    functionalConsequenceId: Annotated[str, Reference('so', 'id')] | None = Field(
        pattern=ONTOLOGY_ID, description=('Sequence ontology (SO) identifier of the mutation type the counts refer to')
    )
    numberMutatedSamples: int = Field(
        ge=0, description='Number of samples of the IntOGen cohort in which the target is mutated'
    )
    numberSamplesTested: int = Field(ge=0, description='Number of samples of the IntOGen cohort tested')
    numberSamplesWithMutationType: int | None = Field(
        description=('Number of samples of the IntOGen cohort with a mutation of the given type in the target')
    )


class EvidenceIntogen(DatasetModel):
    """IntOGen evidence."""

    dataset_name: ClassVar[str] = 'evidence_intogen'
    dataset_title: ClassVar[str] = 'IntOGen evidence'
    dataset_description: ClassVar[str] = (
        'Cancer-driver gene evidence based on the analysis of large-scale mutational data from IntOGen'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000125257',
        'id': 'f6748716e2b24b03ac649028fdc56feb5006d737',
        'targetFromSourceId': 'ABCC4',
        'diseaseFromSourceMappedId': 'MONDO_0016982',
        'datatypeId': 'somatic_mutation',
        'datasourceId': 'intogen',
        'cohortId': 'CBIOP_WXS_ANGS_UNTREAT_2020',
        'cohortShortName': 'CBIOP_WXS_ANGS_UNTREAT_2020',
        'cohortDescription': 'Angiosarcoma from ANGioSarcoma Project',
        'significantDriverMethods': ['oncodriveclustl'],
        'mutatedSamples': [
            {
                'functionalConsequenceId': 'SO_0002053',
                'numberMutatedSamples': 4,
                'numberSamplesTested': 4,
                'numberSamplesWithMutationType': None,
            }
        ],
        'diseaseFromSource': 'Angiosarcoma',
        'resourceScore': 1.833927821088493e-05,
        'literature': ['24071849'],
        'qualityControls': [],
        'diseaseId': 'MONDO_0016982',
        'publicationDate': '2013-10-01',
        'evidenceDate': '2013-10-01',
        'score': 0.5613848134248127,
        'directionOnTrait': 'risk',
        'directionOnTarget': 'GoF',
    }

    targetId: Annotated[str, Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(min_length=1, description='Gene symbol of the target as reported by IntOGen')
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    datatypeId: str = Field(pattern=ONTOLOGY_ID, description='Type of the evidence')
    datasourceId: Literal['intogen'] = Field(description='Identifer of the evidence source')
    cohortId: str = Field(min_length=1, description='IntOGen identifier of the studied cohort')
    cohortShortName: str = Field(min_length=1, description='IntOGen short name of the studied cohort')
    cohortDescription: str = Field(
        min_length=1, description='Description of the IntOGen cohort (tumour type and data source)'
    )
    significantDriverMethods: list[Annotated[str, Field(min_length=1)]] = Field(
        description=(
            'IntOGen driver detection methods reporting the target as a '
            "significant cancer driver (e.g. 'dndscv', 'mutpanning')"
        )
    )
    mutatedSamples: list[MutatedSamples] = Field(
        description='Counts of samples of the IntOGen cohort with mutations in the target'
    )
    diseaseFromSource: str = Field(min_length=1, description='Cancer type of the IntOGen cohort')
    resourceScore: float = Field(
        ge=0.0, le=1.0, description='Combined q-value of the IntOGen cancer driver detection methods'
    )
    literature: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        description='PubMed identifiers of the publications describing the cohort'
    )
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
            'relationship, derived from the log10 of the IntOGen combined q-value'
        ),
    )
    directionOnTrait: Literal['risk', 'protect'] = Field(description='Predicted direction of effect on the trait')
    directionOnTarget: Literal['GoF', 'LoF'] | None = Field(
        description=(
            'Gain or loss of function effect of the evidence on the target '
            'resulting from genetic variants, pharmacological modulation, or other '
            'perturbations'
        )
    )
