"""Associations - direct (overall score): the `association_overall_direct` output.

Generated from the 26.09 release output (7,253,664 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PolarsDtype, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID, ONTOLOGY_ID


class Timeseries(StructModel):
    """`timeseries[]`."""

    year: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=1891, le=2026, description='Year, until evidence is considered'
    )
    associationScore: float = Field(
        ge=0.0, le=0.9139355764745518, description='Association score based on evidence up until a given year'
    )
    novelty: float | None = Field(
        ge=0.0, le=0.8297674161107118, description='Novelty of the association at the given year'
    )
    yearlyEvidenceCount: int | None = Field(ge=1, le=11798, description='Number of evidence reported at a given year')


class AssociationOverallDirect(DatasetModel):
    """Associations - direct (overall score)."""

    dataset_name: ClassVar[str] = 'association_overall_direct'
    dataset_title: ClassVar[str] = 'Associations - direct (overall score)'
    dataset_description: ClassVar[str] = (
        'Overall metrics for direct target-disease associations and its '
        'development over time implying association novelty. Dataset contains '
        'the overall association score for the aggregation of all evidence '
        'supporting any given target-disease. This dataset only contains '
        'direct associations, therefore no evidence is propagated across the '
        'disease or phenotype space using the Experimental Factor Ontology'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'diseaseId': 'EFO_0000180',
        'targetId': 'ENSG00000009413',
        'aggregationType': 'overall',
        'aggregationValue': 'None',
        'associationScore': 0.0014800995840213117,
        'evidenceCount': 1,
        'timeseries': [{'year': 2018, 'associationScore': 0.0, 'novelty': 0.0, 'yearlyEvidenceCount': 1}],
        'currentNovelty': 0.00017643219529906104,
    }

    diseaseId: Annotated[str, Reference('disease', 'id'), PrimaryKey()] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    targetId: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id'), PrimaryKey()] = Field(
        pattern=ENSEMBL_GENE_ID, description='Unique identifier for the target'
    )
    aggregationType: Literal['overall'] = Field(description='How evidence data was aggregated: overall')
    aggregationValue: Literal['None'] = Field(description='For overall association this column is empty')
    associationScore: float = Field(
        ge=0.0,
        le=0.9139355764745518,
        description=('Current disease/target association score of evidence from any datasource'),
    )
    evidenceCount: int | None = Field(
        ge=1,
        le=82242,
        description=('Total number of evidence supporting the disease/target association for any datasource'),
    )
    timeseries: list[Timeseries] = Field(min_length=1, description='Timeseries describing the evolution of association')
    currentNovelty: float | None = Field(
        ge=0.0, le=0.6482235207574601, description='Novelty of the disease/target association'
    )
