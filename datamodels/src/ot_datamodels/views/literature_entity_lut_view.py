"""Literature entity lookup view: the `literature_entity_lut_view` view (pts `copy_view`).

The columns of `literature_entity_lut` the API reads (all but `day` and `keywordType`), in the order they are served.
"""

from typing import ClassVar

from ot_datamodels import DatasetModel, from_output
from ot_datamodels.outputs.literature_entity_lut import LiteratureEntityLut
from ot_datamodels.serving import Serving

Pmid = from_output(LiteratureEntityLut, 'pmid')
Pmcid = from_output(LiteratureEntityLut, 'pmcid')
KeywordId = from_output(LiteratureEntityLut, 'keywordId')
Relevance = from_output(LiteratureEntityLut, 'relevance')
Date = from_output(LiteratureEntityLut, 'date')
Year = from_output(LiteratureEntityLut, 'year')
Month = from_output(LiteratureEntityLut, 'month')


class LiteratureEntityLutView(DatasetModel):
    """Literature entity lookup view."""

    dataset_name: ClassVar[str] = 'literature_entity_lut_view'
    dataset_title: ClassVar[str] = 'Literature view'
    dataset_description: ClassVar[str] = (
        'Unique list of entities identified for each literature entry, without the day and the keyword type'
    )
    clickhouse: ClassVar[Serving] = Serving(engine='mergetree', order_by=('keywordId', 'year', 'month', 'relevance'))
    example: ClassVar[dict] = {k: v for k, v in LiteratureEntityLut.example.items() if k not in ('day', 'keywordType')}

    pmid: Pmid
    pmcid: Pmcid
    keywordId: KeywordId
    relevance: Relevance
    date: Date
    year: Year
    month: Month
