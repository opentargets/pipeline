"""Biosample core annotation view: the `biosample_view` view (pts `biosample_view`).

A copy of `biosample`, served from ClickHouse.
"""

from typing import ClassVar

from ot_datamodels.outputs.biosample import Biosample
from ot_datamodels.serving import ClickHouseDictionary, Serving


class BiosampleView(Biosample):
    """Biosample core annotation view."""

    dataset_name: ClassVar[str] = 'biosample_view'
    dataset_title: ClassVar[str] = 'Biosample core annotation view'
    dataset_description: ClassVar[str] = Biosample.dataset_description
    clickhouse: ClassVar[Serving] = Serving(
        engine='rocksdb',
        dictionaries=(
            ClickHouseDictionary(
                'biosample_names',
                key='id',
                attributes=('name',),
                source_columns={'id': 'biosampleId', 'name': 'biosampleName'},
            ),
        ),
    )
