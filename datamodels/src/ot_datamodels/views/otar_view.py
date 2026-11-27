"""Open Targets Projects view: the `otar_view` view (pts `otar_view`).

A copy of `otar`, served from ClickHouse.
"""

from typing import ClassVar

from ot_datamodels.outputs.otar import Otar
from ot_datamodels.serving import Serving


class OtarView(Otar):
    """Open Targets Projects view."""

    dataset_name: ClassVar[str] = 'otar_view'
    dataset_title: ClassVar[str] = 'Open Targets Projects view'
    dataset_description: ClassVar[str] = Otar.dataset_description
    clickhouse: ClassVar[Serving] = Serving(engine='rocksdb')
