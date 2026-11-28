"""Molecular interactions evidence view: the `interaction_evidence_view` view (pts `copy_view`).

A copy of `interaction_evidence`, served from ClickHouse.
"""

from typing import ClassVar

from ot_datamodels.outputs.interaction_evidence import InteractionEvidence
from ot_datamodels.serving import Serving


class InteractionEvidenceView(InteractionEvidence):
    """Molecular interactions evidence view."""

    dataset_name: ClassVar[str] = 'interaction_evidence_view'
    dataset_title: ClassVar[str] = 'Molecular interactions evidence view'
    dataset_description: ClassVar[str] = InteractionEvidence.dataset_description
    clickhouse: ClassVar[Serving] = Serving(
        engine='mergetree',
        order_by=('interactionId',),
        column_types={'interactionResources/databaseVersion': 'LowCardinality(String)', 'interactionId': 'UInt64'},
    )
