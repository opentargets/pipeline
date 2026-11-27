"""Target - DepMap essentiality view: the `target_essentiality_view` view (pts `copy_view`).

A copy of `target_essentiality`, served from ClickHouse.
"""

from typing import ClassVar

from ot_datamodels.outputs.target_essentiality import TargetEssentiality
from ot_datamodels.serving import Serving


class TargetEssentialityView(TargetEssentiality):
    """Target - DepMap essentiality view."""

    dataset_name: ClassVar[str] = 'target_essentiality_view'
    dataset_title: ClassVar[str] = 'Target - DepMap essentiality view'
    dataset_description: ClassVar[str] = TargetEssentiality.dataset_description
    clickhouse: ClassVar[Serving] = Serving(
        engine='rocksdb',
        column_types={
            'depMapEssentiality[]/screens[]/geneEffect': 'Float64',
            'depMapEssentiality[]/screens[]/expression': 'Float64',
        },
    )
