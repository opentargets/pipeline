"""Association on-the-fly, disease side: the `association_otf_disease_view` view (pts `association_otf`).

One row per evidence as served in `associations_otf_disease`: the disease is `A`, the target `B`, and the
labels are lower-cased for search.
"""

from typing import Annotated, ClassVar

import polars as pl
from pydantic import Field

from ot_datamodels import DatasetModel, PolarsDtype, from_output
from ot_datamodels.outputs.association_overall_direct import AssociationOverallDirect
from ot_datamodels.outputs.association_overall_indirect import AssociationOverallIndirect
from ot_datamodels.outputs.evidence_eva import EvidenceEva
from ot_datamodels.vocabularies import Datasource, Datatype

# every evidence model has the same id, diseaseId, targetId and score (tests/test_views.py)
_Evidence = EvidenceEva

RowId = from_output(_Evidence, 'id', primary_key=True)
DiseaseId = from_output(_Evidence, 'diseaseId')
TargetId = from_output(_Evidence, 'targetId')
RowScore = from_output(_Evidence, 'score')
NoveltyDirect = from_output(AssociationOverallDirect, 'currentNovelty', nullable=True)
NoveltyIndirect = from_output(AssociationOverallIndirect, 'currentNovelty', nullable=True)


class AssociationOtfDiseaseView(DatasetModel):
    """Association on-the-fly, disease side."""

    dataset_name: ClassVar[str] = 'association_otf_disease_view'
    dataset_title: ClassVar[str] = 'Association on-the-fly, disease side'
    dataset_description: ClassVar[str] = (
        'Every evidence keyed by its disease (A) and target (B), with lower-cased labels for search and the '
        'novelty of its direct and indirect association, as served for associations computed on the fly'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'rowId': '0000155ec8bd9ff7b4e5e3b4e1c9c5f0f1e2d3a4',
        'A': 'EFO_0000270',
        'B': 'ENSG00000100031',
        'datatypeId': 'genetic_association',
        'datasourceId': 'eva',
        'rowScore': 0.7,
        'searchA': 'efo_0000270 asthma',
        'searchB': 'ensg00000100031 gamma-glutamyltransferase 1 ggt1',
        'noveltyDirect': 0.01,
        'noveltyIndirect': None,
        'indirect': [],
    }

    rowId: RowId
    A: DiseaseId
    B: TargetId
    datatypeId: Annotated[Datatype, PolarsDtype(pl.Enum)] = Field(description='Type of the evidence')
    datasourceId: Annotated[Datasource, PolarsDtype(pl.Enum)] = Field(description='Identifier of the evidence source')
    rowScore: RowScore
    searchA: str | None = Field(description='Lower-cased disease identifier and name, separated by a space')
    searchB: str | None = Field(
        description='Lower-cased target identifier, approved name and approved symbol, separated by spaces'
    )
    noveltyDirect: NoveltyDirect
    noveltyIndirect: NoveltyIndirect
    indirect: list[str] = Field(description='Always empty: the disease side has no indirect keys')
