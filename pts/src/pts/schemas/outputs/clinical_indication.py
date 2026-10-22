"""Clinical indication: the `clinical_indication` output.

Generated from the 26.09 release output (87,182 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey, Reference
from pts.schemas.outputs.patterns import CHEMBL_ID, ONTOLOGY_ID


class ClinicalIndication(DatasetModel):
    """Clinical indication."""

    dataset_name: ClassVar[str] = 'clinical_indication'
    dataset_title: ClassVar[str] = 'Clinical indication'
    dataset_description: ClassVar[str] = (
        'Approved and investigational indications for drug molecules, '
        'capturing the maximum clinical stage reached for each drug-disease '
        'pair'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Drug', 'Disease')
    example: ClassVar[dict] = {
        'id': 'bcae49c2072f6bce9286dcced5374756568bbc17c3c389e188a091351dc06607',
        'maxClinicalStage': 'APPROVAL',
        'clinicalReportIds': ['00011703-bc55-4c0c-858c-149dc674bc3c'],
        'drugId': 'CHEMBL4',
        'diseaseId': 'EFO_0000544',
    }

    id: Annotated[str, PrimaryKey()] = Field(min_length=1, description='Unique identifier for the clinical indication')
    maxClinicalStage: Literal[
        'APPROVAL',
        'EARLY_PHASE_1',
        'IND',
        'PHASE_1',
        'PHASE_1_2',
        'PHASE_2',
        'PHASE_2_3',
        'PHASE_3',
        'PREAPPROVAL',
        'PRECLINICAL',
        'UNKNOWN',
    ] = Field(description='Maximum clinical stage reached for this drug-disease indication')
    clinicalReportIds: list[Annotated[Annotated[str, Reference('clinical_report', 'id')], Field(min_length=1)]] = Field(
        min_length=1, description='List of clinical report identifiers associated with this indication'
    )
    drugId: Annotated[str, Reference('drug_molecule', 'id')] = Field(
        pattern=CHEMBL_ID, description='Open Targets molecule identifier'
    )
    diseaseId: Annotated[str, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
