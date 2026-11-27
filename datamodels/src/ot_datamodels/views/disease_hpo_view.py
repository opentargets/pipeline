"""Human Phenotype Ontology view: the `disease_hpo_view` view (pts `copy_view`).

A copy of `disease_hpo`, served from ClickHouse.
"""

from typing import ClassVar

from ot_datamodels.outputs.disease_hpo import DiseaseHpo
from ot_datamodels.serving import Serving


class DiseaseHpoView(DiseaseHpo):
    """Human Phenotype Ontology view."""

    dataset_name: ClassVar[str] = 'disease_hpo_view'
    dataset_title: ClassVar[str] = 'Human Phenotype Ontology view'
    dataset_description: ClassVar[str] = DiseaseHpo.dataset_description
    clickhouse: ClassVar[Serving] = Serving(engine='rocksdb')
