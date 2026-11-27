"""Enhancer to gene view: the `enhancer_to_gene_view` view (pts `copy_view`).

A copy of `enhancer_to_gene`, served from ClickHouse.
"""

from typing import ClassVar

from ot_datamodels.outputs.enhancer_to_gene import EnhancerToGene
from ot_datamodels.serving import Serving


class EnhancerToGeneView(EnhancerToGene):
    """Enhancer to gene view."""

    dataset_name: ClassVar[str] = 'enhancer_to_gene_view'
    dataset_title: ClassVar[str] = 'Enhancer to gene view'
    dataset_description: ClassVar[str] = EnhancerToGene.dataset_description
    clickhouse: ClassVar[Serving] = Serving(engine='mergetree', order_by=('chromosome', 'start', 'end'))
