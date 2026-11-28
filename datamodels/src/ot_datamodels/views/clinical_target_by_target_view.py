"""Clinical targets by target: the `clinical_target_by_target_view` view (pts `copy_view`).

The `clinical_target` rows with a target, served by target.
"""

from typing import ClassVar

from ot_datamodels import DatasetModel, from_output
from ot_datamodels.outputs.clinical_target import ClinicalTarget
from ot_datamodels.serving import Serving

Id = from_output(ClinicalTarget, 'id', primary_key=True)
DrugId = from_output(ClinicalTarget, 'drugId')
TargetId = from_output(ClinicalTarget, 'targetId', nullable=False)
Diseases = from_output(ClinicalTarget, 'diseases')
MaxClinicalStage = from_output(ClinicalTarget, 'maxClinicalStage')
ClinicalReportIds = from_output(ClinicalTarget, 'clinicalReportIds')


class ClinicalTargetByTargetView(DatasetModel):
    """Clinical targets by target."""

    dataset_name: ClassVar[str] = 'clinical_target_by_target_view'
    dataset_title: ClassVar[str] = 'Clinical targets by target'
    dataset_description: ClassVar[str] = 'The drugs of each target in clinical reports, as served by target'
    dataset_tags: ClassVar[tuple[str, ...]] = ClinicalTarget.dataset_tags
    clickhouse: ClassVar[Serving] = Serving(
        engine='mergetree',
        order_by=('targetId', 'id'),
        column_types={'drugId': 'LowCardinality(String)', 'targetId': 'LowCardinality(String)'},
    )
    example: ClassVar[dict] = {
        c: ClinicalTarget.example[c]
        for c in ('id', 'drugId', 'targetId', 'diseases', 'maxClinicalStage', 'clinicalReportIds')
    }

    id: Id
    drugId: DrugId
    targetId: TargetId
    diseases: Diseases
    maxClinicalStage: MaxClinicalStage
    clinicalReportIds: ClinicalReportIds
