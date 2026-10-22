"""Clinical Precedence evidence: the `evidence_clinical_precedence` output.

Generated from the 26.09 release output (884,666 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey, Reference
from pts.schemas.outputs.patterns import CHEMBL_ID, DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX


class EvidenceClinicalPrecedence(DatasetModel):
    """Clinical Precedence evidence."""

    dataset_name: ClassVar[str] = 'evidence_clinical_precedence'
    dataset_title: ClassVar[str] = 'Clinical Precedence evidence'
    dataset_description: ClassVar[str] = (
        'Disease-target evidence based on clinical trials, drug indications, and mechanism of action'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease', 'Drug')
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000101162',
        'id': '003c319d6577e8d841c1bcfee224e122d90d995d',
        'targetFromSourceId': 'ENSG00000101162',
        'diseaseFromSourceMappedId': 'MONDO_0007254',
        'clinicalReportId': 'nct00785291',
        'clinicalStage': 'PHASE_3',
        'trialWhyStopped': ('Phase 2 portion not initiated, no longer pursuing ovarian cancer indication'),
        'trialStopReasonCategories': ['Regulatory'],
        'studyStartDate': '2008-10-13',
        'evidenceDate': '2008-10-13',
        'literature': ['26056183'],
        'diseaseFromSource': 'breast cancer',
        'drugFromSource': 'ixabepilone',
        'drugId': 'CHEMBL1201752',
        'datasourceId': 'clinical_precedence',
        'datatypeId': 'clinical',
        'qualityControls': [],
        'diseaseId': 'MONDO_0007254',
        'publicationDate': '2015-06-08',
        'score': 0.7,
        'directionOnTrait': 'protect',
        'directionOnTarget': 'LoF',
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
    clinicalReportId: Annotated[str, Reference('clinical_report', 'id')] = Field(
        min_length=1, description='Identifier of the clinical report'
    )
    clinicalStage: Literal[
        'APPROVAL',
        'EARLY_PHASE_1',
        'IND',
        'PHASE_1',
        'PHASE_1_2',
        'PHASE_2',
        'PHASE_2_3',
        'PHASE_3',
        'PHASE_4',
        'PREAPPROVAL',
        'PRECLINICAL',
        'UNKNOWN',
        'WITHDRAWAL',
    ] = Field(description='Clinical stage of the drug-disease pair')
    trialWhyStopped: str | None = Field(min_length=1, description='Reason why the trial was stopped, as reported')
    trialStopReasonCategories: (
        list[
            Literal[
                'Another_Study',
                'Business_Administrative',
                'Covid19',
                'Insufficient_Enrollment',
                'Invalid_Reason',
                'Logistics_Resources',
                'Negative',
                'No_Context',
                'Regulatory',
                'Safety_Sideeffects',
                'Study_Design',
                'Study_Staff_Moved',
                'Uncategorised',
            ]
        ]
        | None
    ) = Field(min_length=1, description='Categorised reason(s) why the trial was stopped')
    studyStartDate: str | None = Field(pattern=ISO_DATE, description='Start date of study in a YYYY-MM-DD format')
    evidenceDate: str | None = Field(min_length=1, description='Earliest data for the evidence')
    literature: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        min_length=1, description='PubMed or preprint reference identifiers'
    )
    diseaseFromSource: str = Field(min_length=1, description='Disease label from the original source')
    drugFromSource: str = Field(min_length=1, description='Drug name/family in resource of origin')
    drugId: Annotated[str, Bioregistry('chembl'), Reference('drug_molecule', 'id')] = Field(
        pattern=CHEMBL_ID, description='Open Targets drug identifier'
    )
    datasourceId: Literal['clinical_precedence'] = Field(description='Identifer of the evidence source')
    datatypeId: Literal['clinical'] = Field(description='Type of the evidence')
    qualityControls: list[str] = Field(max_length=0, description='Evidence quality flags')
    diseaseId: Annotated[str, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    publicationDate: str | None = Field(pattern=ISO_DATE, description='Publication date of the literature reference')
    score: float = Field(
        ge=0.005,
        le=1.0,
        description=('Score of the evidence reflecting the strength of the disease/target relationship'),
    )
    directionOnTrait: Literal['protect'] = Field(description='Predicted direction of effect on the trait')
    directionOnTarget: Literal['GoF', 'LoF'] | None = Field(
        description=(
            'Gain or loss of function effect of the evidence on the target based '
            'on the type of drug mechanism of action'
        )
    )
