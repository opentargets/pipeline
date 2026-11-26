"""Study view: the `study_view` view (pts `study_view`).

Every study row with the ids of its credible sets.
"""

from typing import Annotated, ClassVar

from pydantic import Field

from ot_datamodels.outputs.patterns import MD5_HEX
from ot_datamodels.outputs.study import Study
from ot_datamodels.serving import Serving

# a list element takes no description, so not `from_output`; tests/test_views.py checks it against credible_set.id
_CredibleSetId = Annotated[str, Field(pattern=MD5_HEX)]


class StudyView(Study):
    """Study view."""

    dataset_name: ClassVar[str] = 'study_view'
    dataset_title: ClassVar[str] = 'Study view'
    dataset_description: ClassVar[str] = 'Studies with the identifiers of their credible sets nested in'
    clickhouse: ClassVar[Serving] = Serving(engine='rocksdb')
    example: ClassVar[dict] = Study.example | {'credibleSetIds': ['0002c80bb07a1c4ab11e5b6d2c9e0d6c']}

    credibleSetIds: list[_CredibleSetId] = Field(description='Credible sets of the study, sorted')
