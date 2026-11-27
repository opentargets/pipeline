"""Drug/Target - pharmacogenetics view: the `pharmacogenomics_view` view (pts `copy_view`).

A copy of `pharmacogenomics`, served from ClickHouse.
"""

from typing import ClassVar

from ot_datamodels.outputs.pharmacogenomics import Pharmacogenomics
from ot_datamodels.serving import Serving


class PharmacogenomicsView(Pharmacogenomics):
    """Drug/Target - pharmacogenetics view."""

    dataset_name: ClassVar[str] = 'pharmacogenomics_view'
    dataset_title: ClassVar[str] = 'Drug/Target - pharmacogenetics view'
    dataset_description: ClassVar[str] = Pharmacogenomics.dataset_description
    clickhouse: ClassVar[Serving] = Serving(engine='rocksdb')
