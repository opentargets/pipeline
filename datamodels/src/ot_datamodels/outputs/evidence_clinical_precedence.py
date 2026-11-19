"""Clinical Precedence evidence: the `evidence_clinical_precedence` output.

Generated from the 26.12 release output (909,884 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from ot_datamodels import Bioregistry, DatasetModel, PrimaryKey, Reference
from ot_datamodels.outputs.patterns import CHEMBL_ID, DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX


class EvidenceClinicalPrecedence(DatasetModel):
    """Clinical Precedence evidence."""

    dataset_name: ClassVar[str] = 'evidence_clinical_precedence'
    dataset_title: ClassVar[str] = 'Clinical Precedence evidence'
    dataset_description: ClassVar[str] = (
        'Disease-target evidence based on clinical trials, drug indications, and mechanism of action'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease', 'Drug')
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000126934',
        'id': '1eee40974d61d76db8e8a94d562063a7ce53b6da',
        'targetFromSourceId': 'ENSG00000126934',
        'diseaseFromSourceMappedId': 'MONDO_0008903',
        'clinicalReportId': '0002ad27-779d-42ab-83b5-bc65453412a1',
        'clinicalStage': 'APPROVAL',
        'trialWhyStopped': 'Unable to recruit adequate number of subjects',
        'trialStopReasonCategories': ['Insufficient_Enrollment'],
        'studyStartDate': '1990-12-31',
        'evidenceDate': '2013',
        'literature': ['21843876'],
        'diseaseFromSource': 'lung cancer',
        'drugFromSource': 'TRAMETINIB DIMETHYL SULFOXIDE',
        'drugId': 'CHEMBL2105741',
        'datasourceId': 'clinical_precedence',
        'datatypeId': 'clinical',
        'qualityControls': [],
        'diseaseId': 'MONDO_0008903',
        'publicationDate': '2011-08-16',
        'score': 1.0,
        'directionOnTrait': 'protect',
        'directionOnTarget': 'LoF',
    }

    targetId: Annotated[str, Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        pattern=ENSEMBL_GENE_ID,
        description=('Ensembl gene identifier of the drug target, from the drug mechanism of action'),
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    clinicalReportId: Annotated[str, Reference('clinical_report', 'id')] = Field(
        min_length=1, description='Identifier of the clinical report'
    )
    clinicalStage: Literal[
        'WITHDRAWAL',
        'APPROVAL',
        'PHASE_4',
        'PREAPPROVAL',
        'PHASE_3',
        'PHASE_2_3',
        'PHASE_2',
        'PHASE_1_2',
        'PHASE_1',
        'EARLY_PHASE_1',
        'IND',
        'PRECLINICAL',
        'UNKNOWN',
    ] = Field(
        description=("Clinical stage of the drug-disease pair in the clinical report (e.g. 'PHASE_1', 'PHASE_3')")
    )
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
    ) = Field(description='Categorised reason(s) why the trial was stopped')
    studyStartDate: str | None = Field(
        pattern=ISO_DATE, description='Start date of the clinical trial in a YYYY-MM-DD format'
    )
    evidenceDate: str | None = Field(min_length=1, description='Earliest data for the evidence')
    literature: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        description='PubMed identifiers of the publications linked to the clinical report'
    )
    diseaseFromSource: str = Field(min_length=1, description='Indication as reported in the clinical report')
    drugFromSource: str = Field(min_length=1, description='Drug name as reported in the clinical report')
    drugId: Annotated[str, Bioregistry('chembl'), Reference('drug_molecule', 'id')] = Field(
        pattern=CHEMBL_ID, description='Open Targets drug identifier'
    )
    datasourceId: str = Field(pattern=ONTOLOGY_ID, description='Identifer of the evidence source')
    datatypeId: Literal['clinical'] = Field(description='Type of the evidence')
    qualityControls: list[str] = Field(max_length=0, description='Evidence quality flags')
    diseaseId: Annotated[str, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    publicationDate: str | None = Field(pattern=ISO_DATE, description='Publication date of the literature reference')
    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            'Score of the evidence reflecting the strength of the disease/target '
            'relationship, derived from the clinical stage and down-weighted by '
            'the reason the trial was stopped'
        ),
    )
    directionOnTrait: Literal['risk', 'protect'] = Field(description='Predicted direction of effect on the trait')
    directionOnTarget: Literal['GoF', 'LoF'] | None = Field(
        description=(
            'Gain or loss of function effect of the evidence on the target based '
            'on the type of drug mechanism of action'
        )
    )
