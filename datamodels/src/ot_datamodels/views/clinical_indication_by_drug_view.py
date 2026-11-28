"""Clinical indications by drug: the `clinical_indication_by_drug_view` view (pts `copy_view`).

The `clinical_indication` rows with a drug, served by drug.
"""

from typing import ClassVar

from ot_datamodels import DatasetModel, from_output
from ot_datamodels.outputs.clinical_indication import ClinicalIndication
from ot_datamodels.serving import Serving

Id = from_output(ClinicalIndication, 'id', primary_key=True)
DrugId = from_output(ClinicalIndication, 'drugId', nullable=False)
DiseaseId = from_output(ClinicalIndication, 'diseaseId')
MaxClinicalStage = from_output(ClinicalIndication, 'maxClinicalStage')
ClinicalReportIds = from_output(ClinicalIndication, 'clinicalReportIds')


class ClinicalIndicationByDrugView(DatasetModel):
    """Clinical indications by drug."""

    dataset_name: ClassVar[str] = 'clinical_indication_by_drug_view'
    dataset_title: ClassVar[str] = 'Clinical indications by drug'
    dataset_description: ClassVar[str] = 'The clinical indications of each drug molecule, as served by molecule'
    dataset_tags: ClassVar[tuple[str, ...]] = ClinicalIndication.dataset_tags
    clickhouse: ClassVar[Serving] = Serving(
        engine='mergetree',
        order_by=('drugId', 'id'),
        column_types={'drugId': 'LowCardinality(String)', 'diseaseId': 'LowCardinality(String)'},
    )
    example: ClassVar[dict] = {
        c: ClinicalIndication.example[c] for c in ('id', 'drugId', 'diseaseId', 'maxClinicalStage', 'clinicalReportIds')
    }

    id: Id
    drugId: DrugId
    diseaseId: DiseaseId
    maxClinicalStage: MaxClinicalStage
    clinicalReportIds: ClinicalReportIds
