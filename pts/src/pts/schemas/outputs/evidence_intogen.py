"""IntOGen evidence: the `evidence_intogen` output.

Generated from the 26.09 release output (4,223 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX


class MutatedSamples(StructModel):
    """`mutatedSamples[]`."""

    functionalConsequenceId: Annotated[str, Bioregistry('so'), Reference('so', 'id')] | None = Field(
        pattern=ONTOLOGY_ID, description='Functional consequene identifier of the pooled mutations'
    )
    numberMutatedSamples: int = Field(
        ge=2, le=395, description=('Number of cohort samples in which target is mutated with a mutation of any type')
    )
    numberSamplesTested: int = Field(ge=2, le=395, description='Number of cohort samples tested')
    numberSamplesWithMutationType: int | None = Field(
        description=('Number of cohort samples in which target is mutated with a specific mutation type')
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
        'targetId': 'ENSG00000105568',
        'id': '38ab91171f4572bc2aac710a43895ed7f9477b25',
        'targetFromSourceId': 'PPP2R1A',
        'diseaseFromSourceMappedId': 'MONDO_0011962',
        'datatypeId': 'somatic_mutation',
        'datasourceId': 'intogen',
        'cohortId': 'CPTAC_WXS_UCEC_2020',
        'cohortShortName': 'CPTAC_WXS_UCEC_2020',
        'cohortDescription': 'Endometrial Carcinoma from CPTAC',
        'significantDriverMethods': ['dndscv'],
        'mutatedSamples': [
            {
                'functionalConsequenceId': 'SO_0002053',
                'numberMutatedSamples': 9,
                'numberSamplesTested': 9,
                'numberSamplesWithMutationType': None,
            }
        ],
        'diseaseFromSource': 'Endometrial Carcinoma',
        'resourceScore': 6.2022461487574244e-18,
        'literature': ['32025007'],
        'qualityControls': [],
        'diseaseId': 'MONDO_0011962',
        'publicationDate': '2020-02-05',
        'evidenceDate': '2020-02-05',
        'score': 1.0,
        'directionOnTrait': 'risk',
        'directionOnTarget': 'GoF',
    }

    targetId: Annotated[str, Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        min_length=1,
        description=(
            'Target ID in resource of origin (accepted sources include Ensembl '
            'gene ID, Uniprot ID, gene symbol), only capital letters are accepted'
        ),
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    datatypeId: Literal['somatic_mutation'] = Field(description='Type of the evidence')
    datasourceId: Literal['intogen'] = Field(description='Identifer of the evidence source')
    cohortId: str = Field(min_length=1, description='Identifier of the studied cohort')
    cohortShortName: str = Field(min_length=1, description='Short name of the studied cohort')
    cohortDescription: str = Field(min_length=1, description='Description of the studied cohort')
    significantDriverMethods: list[
        Literal[
            'cbase', 'combination', 'dndscv', 'hotmaps', 'mutpanning', 'oncodriveclustl', 'oncodrivefml', 'smregions'
        ]
    ] = Field(min_length=1, description='Methods to detect cancer driver genes producing significant results')
    mutatedSamples: list[MutatedSamples] = Field(min_length=1, description='Samples with a given mutation tested')
    diseaseFromSource: str = Field(min_length=1, description='Disease label from the original source')
    resourceScore: float = Field(
        ge=6.17183985323648e-71,
        le=0.4140553710715359,
        description=('Score provided by datasource indicating strength of target-disease association'),
    )
    literature: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        min_length=1, description='List of PubMed or preprint reference identifiers'
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
