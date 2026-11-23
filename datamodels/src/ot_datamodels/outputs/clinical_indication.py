"""Clinical indication: the `clinical_indication` output.

Generated from the 26.12 release output (88,300 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar

import polars as pl
from pydantic import Field

from ot_datamodels import ClickHouseTable, ClickHouseType, DatasetModel, PolarsDtype, PrimaryKey, Reference
from ot_datamodels.outputs.patterns import CHEMBL_ID, ONTOLOGY_ID
from ot_datamodels.vocabularies import ClinicalStage


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
    clickhouse: ClassVar[tuple[ClickHouseTable, ...]] = (
        ClickHouseTable(
            'clinical_indication_drug',
            'mergetree',
            order_by=('drugId', 'id'),
            primary_key=('drugId',),
            columns=('id', 'drugId', 'diseaseId', 'maxClinicalStage', 'clinicalReportIds'),
            where='drugId IS NOT NULL',
            optimize=True,
        ),
        ClickHouseTable(
            'clinical_indication_disease',
            'mergetree',
            order_by=('diseaseId', 'id'),
            primary_key=('diseaseId',),
            columns=('id', 'drugId', 'diseaseId', 'maxClinicalStage', 'clinicalReportIds'),
            where='diseaseId IS NOT NULL',
            optimize=True,
        ),
    )
    example: ClassVar[dict] = {
        'id': 'bcae49c2072f6bce9286dcced5374756568bbc17c3c389e188a091351dc06607',
        'maxClinicalStage': 'APPROVAL',
        'clinicalReportIds': ['00011703-bc55-4c0c-858c-149dc674bc3c'],
        'drugId': 'CHEMBL4',
        'diseaseId': 'EFO_0000544',
    }

    id: Annotated[str, PrimaryKey()] = Field(min_length=1, description='Unique identifier for the clinical indication')
    maxClinicalStage: Annotated[ClinicalStage, PolarsDtype(pl.Enum)] = Field(
        description='Maximum clinical stage reached for this drug-disease indication'
    )
    clinicalReportIds: list[Annotated[Annotated[str, Reference('clinical_report', 'id')], Field(min_length=1)]] = Field(
        description='List of clinical report identifiers associated with this indication'
    )
    drugId: Annotated[str, Reference('drug_molecule', 'id'), ClickHouseType('LowCardinality(String)')] = Field(
        pattern=CHEMBL_ID, description='Open Targets molecule identifier'
    )
    diseaseId: Annotated[str, Reference('disease', 'id'), ClickHouseType('LowCardinality(String)')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
