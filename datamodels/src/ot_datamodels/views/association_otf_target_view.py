"""Association on-the-fly, target side: the `association_otf_target_view` view (pts `association_otf`).

One row per evidence: the target is `A`, the disease `B`, and the
disease's ancestors are the indirect keys.
"""

from typing import Annotated, ClassVar

import polars as pl
from pydantic import Field

from ot_datamodels import DatasetModel, PolarsDtype, from_output
from ot_datamodels.outputs.disease import Disease
from ot_datamodels.serving import Serving
from ot_datamodels.views.association_otf_disease_view import (
    DiseaseId,
    NoveltyDirect,
    NoveltyIndirect,
    RowId,
    RowScore,
    TargetId,
)
from ot_datamodels.vocabularies import Datasource, Datatype

Ancestors = from_output(Disease, 'ancestors')


class AssociationOtfTargetView(DatasetModel):
    """Association on-the-fly, target side."""

    dataset_name: ClassVar[str] = 'association_otf_target_view'
    dataset_title: ClassVar[str] = 'Association on-the-fly, target side'
    dataset_description: ClassVar[str] = (
        'Every evidence keyed by its target (A) and disease (B), with lower-cased labels for search, the '
        'disease ancestors as indirect keys and the novelty of its direct and indirect association, as served '
        'for associations computed on the fly'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    clickhouse: ClassVar[Serving] = Serving(engine='mergetree', order_by=('A', 'B', 'datasourceId'))
    example: ClassVar[dict] = {
        'rowId': '0000155ec8bd9ff7b4e5e3b4e1c9c5f0f1e2d3a4',
        'A': 'ENSG00000100031',
        'B': 'EFO_0000270',
        'datatypeId': 'genetic_association',
        'datasourceId': 'eva',
        'rowScore': 0.7,
        'searchA': 'ensg00000100031 gamma-glutamyltransferase 1 ggt1',
        'searchB': 'efo_0000270 asthma',
        'isMeasurement': False,
        'noveltyDirect': 0.01,
        'noveltyIndirect': None,
        'indirect': ['EFO_0000540'],
    }

    rowId: RowId
    A: TargetId
    B: DiseaseId
    datatypeId: Annotated[Datatype, PolarsDtype(pl.Enum)] = Field(description='Type of the evidence')
    datasourceId: Annotated[Datasource, PolarsDtype(pl.Enum)] = Field(description='Identifier of the evidence source')
    rowScore: RowScore
    searchA: str | None = Field(
        description='Lower-cased target identifier, approved name and approved symbol, separated by spaces'
    )
    searchB: str | None = Field(description='Lower-cased disease identifier and name, separated by a space')
    isMeasurement: bool = Field(description='Whether the disease is a measurement (a therapeutic area of EFO_0001444)')
    noveltyDirect: NoveltyDirect
    noveltyIndirect: NoveltyIndirect
    indirect: Ancestors
