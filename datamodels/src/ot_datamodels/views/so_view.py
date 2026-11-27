"""Sequence Ontology view: the `so_view` view (pts `copy_view`).

A copy of `so`, served from ClickHouse.
"""

from typing import ClassVar

from ot_datamodels.outputs.so import So
from ot_datamodels.serving import ClickHouseDictionary, Serving


class SoView(So):
    """Sequence Ontology view."""

    dataset_name: ClassVar[str] = 'so_view'
    dataset_title: ClassVar[str] = 'Sequence Ontology view'
    dataset_description: ClassVar[str] = So.dataset_description
    clickhouse: ClassVar[Serving] = Serving(
        engine='rocksdb',
        dictionaries=(ClickHouseDictionary('sequence_ontology', key='id', attributes=('label',)),),
    )
