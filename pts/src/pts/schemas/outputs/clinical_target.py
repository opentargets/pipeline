"""Clinical target: the `clinical_target` output.

Generated from the 26.09 release output (13,262 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import CHEMBL_ID, ENSEMBL_GENE_ID, ONTOLOGY_ID


class Diseases(StructModel):
    """`diseases[]`."""

    diseaseFromSource: str = Field(min_length=1, description='Disease name as reported in the source')
    diseaseId: Annotated[str, Reference('disease', 'id')] | None = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )


class ClinicalTarget(DatasetModel):
    """Clinical target."""

    dataset_name: ClassVar[str] = 'clinical_target'
    dataset_title: ClassVar[str] = 'Clinical target'
    dataset_description: ClassVar[str] = (
        'Target-drug associations derived from clinical reports, capturing the '
        'maximum clinical stage and associated diseases for each target-drug '
        'pair'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Drug', 'Target')
    example: ClassVar[dict] = {
        'id': '3b7ad4b95a794fe4e04bea03d9823ceb5645f49f67058f769685e3093f427e72',
        'drugId': 'CHEMBL515914',
        'targetId': 'ENSG00000114491',
        'diseases': [{'diseaseFromSource': 'psoriasis vulgaris', 'diseaseId': 'EFO_1001494'}],
        'clinicalReportIds': ['0031dba951c368a84dccf46b2a9f977a8e3f2b03fee1a13147117311e0b41cde'],
        'maxClinicalStage': 'WITHDRAWAL',
    }

    id: Annotated[str, PrimaryKey()] = Field(
        min_length=1, description='Unique identifier for the clinical target-drug association'
    )
    drugId: Annotated[str, Reference('drug_molecule', 'id')] = Field(
        pattern=CHEMBL_ID, description='Open Targets molecule identifier'
    )
    targetId: Annotated[str, Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    diseases: list[Diseases | None] = Field(
        min_length=1, description='Diseases associated with this target-drug clinical pair'
    )
    clinicalReportIds: list[Annotated[Annotated[str, Reference('clinical_report', 'id')], Field(min_length=1)]] = Field(
        min_length=1, description=('List of clinical report identifiers associated with this target-drug pair')
    )
    maxClinicalStage: Literal[
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
    ] = Field(description='Maximum clinical stage reached for this target-drug pair')
