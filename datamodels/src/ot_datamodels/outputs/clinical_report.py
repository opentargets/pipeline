"""Clinical report: the `clinical_report` output.

Generated from the 26.12 release output (296,494 rows) and its croissant metadata.
"""

import datetime
from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from ot_datamodels import DatasetModel, PolarsDtype, PrimaryKey, Reference, StructModel
from ot_datamodels.outputs.common import MappedDisease
from ot_datamodels.outputs.patterns import CHEMBL_ID, DIGITS, ONTOLOGY_ID
from ot_datamodels.vocabularies import (
    ClinicalReportType,
    ClinicalSource,
    ClinicalStage,
    Origin,
    Provider,
    TrialAgencyClass,
    TrialOverallStatus,
    TrialPhase,
    TrialPrimaryPurpose,
    TrialReferenceType,
    TrialStopReasonCategory,
    TrialStudyType,
)


class Drugs(StructModel):
    """`drugs[]`."""

    drugFromSource: str = Field(min_length=1, description='Drug name as reported in the source')
    drugId: Annotated[str, Reference('drug_molecule', 'id')] | None = Field(
        pattern=CHEMBL_ID, description='Open Targets molecule identifier'
    )


class TrialSponsor(StructModel):
    """`trialSponsor`."""

    agencyClass: Annotated[TrialAgencyClass, PolarsDtype(pl.Enum)] = Field(
        description='Classification of the lead trial sponsor (e.g. INDUSTRY, NIH, OTHER)'
    )
    name: str = Field(min_length=1, description='Name of the lead sponsor associated with the clinical trial')




class TrialLiterature(StructModel):
    """`trialLiterature[]`."""

    id: str | None = Field(
        pattern=DIGITS, description='PubMed identifier of the reference, when the source records one'
    )
    type: Annotated[TrialReferenceType, PolarsDtype(pl.Enum)] = Field(
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
        'year': 2025,
        'trialStopReasonCategories': ['Another_Study'],
        'url': ('https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=00011703-bc55-4c0c-858c-149dc674bc3c'),
        'trialStartDate': datetime.date(2002, 7, 31),
    }

    id: Annotated[str, PrimaryKey()] = Field(min_length=1, description='Unique identifier for the clinical report')
    clinicalStage: Annotated[ClinicalStage, PolarsDtype(pl.Enum)] = Field(description='Clinical stage of the report')
    origin: Annotated[Origin, PolarsDtype(pl.Enum)] = Field(
        description=(
            'Nature of the record the report originates from (e.g. CLINICAL_TRIAL, '
            'DRUG_LABEL, REGULATORY_AGENCY, CURATED_RESOURCE)'
        )
    )
    type: Annotated[ClinicalReportType, PolarsDtype(pl.Enum)] = Field(
        description=(
            'Kind of evidence the report describes: INDICATION for a drug/disease claim, SAFETY for a drug warning'
        )
    )
    source: Annotated[ClinicalSource, PolarsDtype(pl.Enum)] = Field(
        description=('Source database or registry from which the clinical report was obtained')
    )
    provider: Annotated[Provider, PolarsDtype(pl.Enum)] = Field(
        description=(
            'Resource or organisation that distributes the data fetched from the '
            'primary source (e.g. AACT, ChEMBL, EMA, FDA, PMDA, TTD)'
        )
    )
    drugs: list[Drugs] | None = Field(
        description='Drugs associated with the clinical report; null for a report flagged NO_DRUGS'
    )
    countries: list[Annotated[str, Field(min_length=1)]] | None = Field(
        description='Countries associated with the clinical report'
    )
    trialDescription: str | None = Field(min_length=1, description='Brief description of the clinical trial')
    phaseFromSource: str | None = Field(min_length=1, description='Clinical phase as reported in the source')
    trialOfficialTitle: str | None = Field(
        min_length=1, description='Official title of the clinical trial as registered'
    )
    trialSponsor: TrialSponsor | None = Field(description='Lead sponsor associated with the clinical trial')
    diseases: list[MappedDisease] | None = Field(description='Diseases associated with the clinical report')
    trialLiterature: list[TrialLiterature] | None = Field(
        description='Literature references associated with the clinical trial'
    )
    sideEffects: list[SideEffects] | None = Field(
        description='Side effects or adverse events associated with the clinical report'
    )
    trialPhase: Annotated[TrialPhase, PolarsDtype(pl.Enum)] | None = Field(description='Phase of the clinical trial')
    trialWhyStopped: str | None = Field(min_length=1, description='Reason provided for stopping the clinical trial')
    trialStudyType: Annotated[TrialStudyType, PolarsDtype(pl.Enum)] | None = Field(
        description='Type of clinical study (e.g. Interventional, Observational)'
    )
    qualityControls: (
        list[
            Literal[
                'PHASE_IV_NOT_APPROVED',
                'UNVALIDATED_INDICATION',
                'INDIRECT_PRIMARY_PURPOSE',
                'OBSERVATIONAL_STUDY',
                'NO_DISEASE',
                'NO_DRUGS',
                'SAFETY_REPORT',
            ]
        ]
        | None
    ) = Field(description='Quality control flags or notes for the clinical report')
    trialPrimaryPurpose: Annotated[TrialPrimaryPurpose, PolarsDtype(pl.Enum)] | None = Field(
        description='Primary purpose of the clinical trial (e.g. Treatment, Prevention)'
    )
    title: str = Field(min_length=1, description='Title of the clinical report')
    trialNumberOfArms: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, description='Number of arms in the clinical trial'
    )
    trialOverallStatus: Annotated[TrialOverallStatus, PolarsDtype(pl.Enum)] | None = Field(
        description='Overall status of the clinical trial (e.g. Completed, Terminated)'
    )
    year: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, description='Year the report was registered or published'
    )
    trialStopReasonCategories: list[Annotated[TrialStopReasonCategory, PolarsDtype(pl.Enum)]] | None = Field(
        description='Categories describing reasons why the trial was stopped'
    )
    url: str = Field(min_length=1, description='URL linking to the source record')
    trialStartDate: datetime.date | None = Field(description='Start date of the clinical trial')
