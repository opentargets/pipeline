"""Target - gene ontology view: the `go_view` view (pts `go_view`).

A copy of `go`, served from ClickHouse.
"""

from typing import ClassVar

from ot_datamodels.outputs.go import Go
from ot_datamodels.serving import Serving


class GoView(Go):
    """Target - gene ontology view."""

    dataset_name: ClassVar[str] = 'go_view'
    dataset_title: ClassVar[str] = 'Target - gene ontology view'
    dataset_description: ClassVar[str] = Go.dataset_description
    clickhouse: ClassVar[Serving] = Serving(engine='rocksdb')
