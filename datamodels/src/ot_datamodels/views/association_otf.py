"""Association on-the-fly: the `association_otf` view (pts `association_otf`).

One row per evidence, from every evidence dataset, with its disease and target labels and the direct and
indirect novelty of its association.
"""

from typing import Annotated, ClassVar

import polars as pl
from pydantic import Field

from ot_datamodels import DatasetModel, PolarsDtype, from_output
from ot_datamodels.outputs.association_overall_direct import AssociationOverallDirect
from ot_datamodels.outputs.association_overall_indirect import AssociationOverallIndirect
from ot_datamodels.outputs.evidence_ot_crispr import EvidenceOtCrispr
from ot_datamodels.vocabularies import Datasource, Datatype

# every evidence model has the same id, diseaseId, targetId and score (tests/test_views.py)
_Evidence = EvidenceOtCrispr

RowId = from_output(_Evidence, 'id', primary_key=True)
DiseaseId = from_output(_Evidence, 'diseaseId')
TargetId = from_output(_Evidence, 'targetId')
RowScore = from_output(_Evidence, 'score')
NoveltyDirect = from_output(AssociationOverallDirect, 'currentNovelty', nullable=True)
NoveltyIndirect = from_output(AssociationOverallIndirect, 'currentNovelty', nullable=True)


class AssociationOtf(DatasetModel):
    """Association on-the-fly."""

    dataset_name: ClassVar[str] = 'association_otf'
    dataset_title: ClassVar[str] = 'Association on-the-fly'
    dataset_description: ClassVar[str] = (
        'Every evidence with its disease and target labels and the novelty of its direct and indirect '
        'association, for computing associations on the fly'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'rowId': '0000155ec8bd9ff7b4e5e3b4e1c9c5f0f1e2d3a4',
        'diseaseId': 'EFO_0000270',
        'targetId': 'ENSG00000100031',
        'datasourceId': 'eva',
        'datatypeId': 'genetic_association',
        'rowScore': 0.7,
        'diseaseData': 'EFO_0000270 asthma',
        'targetData': 'ENSG00000100031 gamma-glutamyltransferase 1 GGT1',
        'noveltyDirect': 0.01,
        'noveltyIndirect': None,
    }

    rowId: RowId
    diseaseId: DiseaseId
    targetId: TargetId
    datasourceId: Annotated[Datasource, PolarsDtype(pl.Enum)] = Field(description='Identifier of the evidence source')
    datatypeId: Annotated[Datatype, PolarsDtype(pl.Enum)] = Field(description='Type of the evidence')
    rowScore: RowScore
    diseaseData: str | None = Field(description='Disease identifier and name, separated by a space')
    targetData: str | None = Field(
        description='Target identifier, approved name and approved symbol, separated by spaces'
    )
    noveltyDirect: NoveltyDirect
    noveltyIndirect: NoveltyIndirect
