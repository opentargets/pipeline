"""Clinical report: the `clinical_report` output.

Generated from the 26.09 release output (294,797 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

import datetime
from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from pts.schemas.dataset import DatasetModel, PolarsDtype, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import CHEMBL_ID, DIGITS, ONTOLOGY_ID


class Drugs(StructModel):
    """`drugs[]`."""

    drugFromSource: str = Field(min_length=1, description='Drug name as reported in the source')
    drugId: Annotated[str, Reference('drug_molecule', 'id')] | None = Field(
        pattern=CHEMBL_ID, description='Open Targets molecule identifier'
    )


class TrialSponsor(StructModel):
    """`trialSponsor`."""

    agencyClass: Literal['FED', 'INDIV', 'INDUSTRY', 'NETWORK', 'NIH', 'OTHER', 'OTHER_GOV', 'UNKNOWN'] = Field(
        description='Classification of the lead trial sponsor (e.g. INDUSTRY, NIH, OTHER)'
    )
    name: str = Field(min_length=1, description='Name of the lead sponsor associated with the clinical trial')


class Diseases(StructModel):
    """`diseases[]`."""

    diseaseFromSource: str = Field(min_length=1, description='Disease name as reported in the source')
    diseaseId: Annotated[str, Reference('disease', 'id')] | None = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )


class TrialLiterature(StructModel):
    """`trialLiterature[]`."""

    id: str | None = Field(
        pattern=DIGITS, description='PubMed identifier of the reference, when the source records one'
    )
    type: Literal['BACKGROUND', 'DERIVED', 'RESULT'] = Field(
        description=(
            'How the reference relates to the trial: RESULT and DERIVED report its '
            'outcome, BACKGROUND is literature its authors cited'
        )
    )


class SideEffects(StructModel):
    """`sideEffects[]`."""

    diseaseId: Annotated[str, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier for the side effect'
    )
    diseaseFromSource: str = Field(min_length=1, description='Side effect disease name as reported in the source')


class ClinicalReport(DatasetModel):
    """Clinical report."""

    dataset_name: ClassVar[str] = 'clinical_report'
    dataset_title: ClassVar[str] = 'Clinical report'
    dataset_description: ClassVar[str] = (
        'Clinical trial and regulatory reports from multiple sources, '
        'capturing trial metadata, associated drugs, diseases, side effects, '
        'and trial outcomes'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Drug', 'Disease')
    example: ClassVar[dict] = {
        'id': '00011703-bc55-4c0c-858c-149dc674bc3c',
        'clinicalStage': 'APPROVAL',
        'origin': 'DRUG_LABEL',
        'type': 'INDICATION',
        'source': 'DailyMed',
        'provider': 'ChEMBL',
        'drugs': [{'drugFromSource': 'OFLOXACIN', 'drugId': 'CHEMBL4'}],
        'countries': ['United States'],
        'trialDescription': (
            'This study will test the ability of extended release nifedipine '
            '(Procardia XL), a blood pressure medication, to permit a decrease in '
            'the dose of glucocorticoid medication children take to treat '
            'congenital adrenal hyperplasia (CAH).'
        ),
        'phaseFromSource': 'APPROVAL',
        'trialOfficialTitle': (
            'Vaccination With Tetanus Toxoid and Keyhole Limpet Hemocyanin (KLH) '
            'to Assess Antigen-Specific Immune Responses'
        ),
        'trialSponsor': {'agencyClass': 'NIH', 'name': 'National Center for Research Resources (NCRR)'},
        'diseases': [{'diseaseFromSource': 'infection', 'diseaseId': 'EFO_0000544'}],
        'trialLiterature': [{'id': '37261916', 'type': 'DERIVED'}],
        'sideEffects': [{'diseaseId': 'EFO_0009658', 'diseaseFromSource': 'adverse effect'}],
        'trialPhase': 'PHASE1/PHASE2',
        'trialWhyStopped': 'Replaced by another study.',
        'trialStudyType': 'INTERVENTIONAL',
        'qualityControls': ['UNVALIDATED_INDICATION'],
        'trialPrimaryPurpose': 'TREATMENT',
        'title': 'Report in Approval stage for Ofloxacin and Infection',
        'trialNumberOfArms': 2,
        'trialOverallStatus': 'COMPLETED',
        'year': 2013,
        'trialStopReasonCategories': ['Another_Study'],
        'url': ('https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=00011703-bc55-4c0c-858c-149dc674bc3c'),
        'trialStartDate': datetime.date(2002, 7, 31),
    }

    id: Annotated[str, PrimaryKey()] = Field(min_length=1, description='Unique identifier for the clinical report')
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
    ] = Field(description='Clinical stage of the report')
    origin: Literal['CLINICAL_TRIAL', 'CURATED_RESOURCE', 'DRUG_LABEL', 'REGULATORY_AGENCY'] = Field(
        description=(
            'Nature of the record the report originates from (e.g. CLINICAL_TRIAL, '
            'DRUG_LABEL, REGULATORY_AGENCY, CURATED_RESOURCE)'
        )
    )
    type: Literal['INDICATION', 'SAFETY'] = Field(
        description=(
            'Kind of evidence the report describes: INDICATION for a drug/disease claim, SAFETY for a drug warning'
        )
    )
    source: str = Field(
        min_length=1, description=('Source database or registry from which the clinical report was obtained')
    )
    provider: Literal['AACT', 'ChEMBL', 'EMA', 'PMDA', 'TTD'] = Field(
        description=(
            'Resource or organisation that distributes the data fetched from the '
            'primary source (e.g. AACT, ChEMBL, EMA, PMDA, TTD)'
        )
    )
    drugs: list[Drugs] = Field(min_length=1, description='Drugs associated with the clinical report')
    countries: list[Annotated[str, Field(min_length=1)]] | None = Field(
        min_length=1, description='Countries associated with the clinical report'
    )
    trialDescription: str | None = Field(min_length=1, description='Brief description of the clinical trial')
    phaseFromSource: str | None = Field(min_length=1, description='Clinical phase as reported in the source')
    trialOfficialTitle: str | None = Field(
        min_length=1, description='Official title of the clinical trial as registered'
    )
    trialSponsor: TrialSponsor | None = Field(description='Lead sponsor associated with the clinical trial')
    diseases: list[Diseases] | None = Field(min_length=1, description='Diseases associated with the clinical report')
    trialLiterature: list[TrialLiterature] | None = Field(
        min_length=1, description='Literature references associated with the clinical trial'
    )
    sideEffects: list[SideEffects] | None = Field(
        min_length=1, description='Side effects or adverse events associated with the clinical report'
    )
    trialPhase: (
        Literal['EARLY_PHASE1', 'PHASE1', 'PHASE1/PHASE2', 'PHASE2', 'PHASE2/PHASE3', 'PHASE3', 'PHASE4'] | None
    ) = Field(description='Phase of the clinical trial')
    trialWhyStopped: str | None = Field(min_length=1, description='Reason provided for stopping the clinical trial')
    trialStudyType: Literal['EXPANDED_ACCESS', 'INTERVENTIONAL', 'OBSERVATIONAL'] | None = Field(
        description='Type of clinical study (e.g. Interventional, Observational)'
    )
    qualityControls: (
        list[
            Literal[
                'INDIRECT_PRIMARY_PURPOSE',
                'NO_DISEASE',
                'PHASE_IV_NOT_APPROVED',
                'SAFETY_REPORT',
                'UNVALIDATED_INDICATION',
            ]
        ]
        | None
    ) = Field(min_length=1, description='Quality control flags or notes for the clinical report')
    trialPrimaryPurpose: (
        Literal[
            'BASIC_SCIENCE',
            'DEVICE_FEASIBILITY',
            'DIAGNOSTIC',
            'ECT',
            'HEALTH_SERVICES_RESEARCH',
            'OTHER',
            'PREVENTION',
            'SCREENING',
            'SUPPORTIVE_CARE',
            'TREATMENT',
        ]
        | None
    ) = Field(description='Primary purpose of the clinical trial (e.g. Treatment, Prevention)')
    title: str = Field(min_length=1, description='Title of the clinical report')
    trialNumberOfArms: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=1, le=63, description='Number of arms in the clinical trial'
    )
    trialOverallStatus: (
        Literal[
            'ACTIVE_NOT_RECRUITING',
            'APPROVED_FOR_MARKETING',
            'AVAILABLE',
            'COMPLETED',
            'ENROLLING_BY_INVITATION',
            'NOT_YET_RECRUITING',
            'NO_LONGER_AVAILABLE',
            'RECRUITING',
            'SUSPENDED',
            'TEMPORARILY_NOT_AVAILABLE',
            'TERMINATED',
            'UNKNOWN',
            'WITHDRAWN',
        ]
        | None
    ) = Field(description='Overall status of the clinical trial (e.g. Completed, Terminated)')
    year: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=1958, le=2026, description='Year the report was registered or published'
    )
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
    ) = Field(min_length=1, description='Categories describing reasons why the trial was stopped')
    url: str = Field(min_length=1, description='URL linking to the source record')
    trialStartDate: datetime.date | None = Field(
        ge=datetime.date(1900, 1, 31), le=datetime.date(2099, 1, 1), description='Start date of the clinical trial'
    )
