"""Clinical target: the `clinical_target` output.

Generated from the 26.12 release output (16,853 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar

import polars as pl
from pydantic import Field

from ot_datamodels import ClickHouseTable, ClickHouseType, DatasetModel, PolarsDtype, PrimaryKey, Reference
from ot_datamodels.outputs.common import MappedDisease
from ot_datamodels.outputs.patterns import CHEMBL_ID, ENSEMBL_GENE_ID
from ot_datamodels.vocabularies import ClinicalStage


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
    clickhouse: ClassVar[tuple[ClickHouseTable, ...]] = (
        ClickHouseTable(
            'clinical_target',
            'mergetree',
            order_by=('targetId', 'id'),
            columns=('id', 'drugId', 'targetId', 'diseases', 'maxClinicalStage', 'clinicalReportIds'),
            where='targetId IS NOT NULL',
        ),
    )
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
    drugId: Annotated[str, Reference('drug_molecule', 'id'), ClickHouseType('LowCardinality(String)')] = Field(
        pattern=CHEMBL_ID, description='Open Targets molecule identifier'
    )
    targetId: Annotated[str, Reference('target', 'id'), ClickHouseType('LowCardinality(String)')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    diseases: list[MappedDisease | None] = Field(description='Diseases associated with this target-drug clinical pair')
    clinicalReportIds: list[Annotated[Annotated[str, Reference('clinical_report', 'id')], Field(min_length=1)]] = Field(
        description=('List of clinical report identifiers associated with this target-drug pair')
    )
    maxClinicalStage: Annotated[ClinicalStage, PolarsDtype(pl.Enum)] = Field(
        description='Maximum clinical stage reached for this target-drug pair'
    )
