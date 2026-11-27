"""Clinical report view: the `clinical_report_view` view (pts `clinical_report_view`).

A copy of `clinical_report`, served from ClickHouse.
"""

from typing import ClassVar

from ot_datamodels.outputs.clinical_report import ClinicalReport
from ot_datamodels.serving import Serving


class ClinicalReportView(ClinicalReport):
    """Clinical report view."""

    dataset_name: ClassVar[str] = 'clinical_report_view'
    dataset_title: ClassVar[str] = 'Clinical report view'
    dataset_description: ClassVar[str] = ClinicalReport.dataset_description
    clickhouse: ClassVar[Serving] = Serving(engine='rocksdb')
