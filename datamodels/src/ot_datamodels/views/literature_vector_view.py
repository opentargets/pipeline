"""Literature vector view: the `literature_vector_view` view (pts `literature_vector_view`).

A copy of `literature_vector`, served from ClickHouse.
"""

from typing import ClassVar

from ot_datamodels.outputs.literature_vector import LiteratureVector
from ot_datamodels.serving import Serving


class LiteratureVectorView(LiteratureVector):
    """Literature vector view."""

    dataset_name: ClassVar[str] = 'literature_vector_view'
    dataset_title: ClassVar[str] = 'Literature vector view'
    dataset_description: ClassVar[str] = LiteratureVector.dataset_description
    clickhouse: ClassVar[Serving] = Serving(engine='mergetree', order_by=('word',))
