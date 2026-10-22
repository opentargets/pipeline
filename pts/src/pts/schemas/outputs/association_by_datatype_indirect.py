"""Associations - indirect (by data type): the `association_by_datatype_indirect` output.

Generated from the 26.09 release output (20,699,404 rows) and its croissant metadata. Rules are
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
        ge=0.0, le=0.988562906674573, description='Association score based on evidence up until a given year'
    )
    novelty: float | None = Field(
        ge=0.0, le=0.9861185631397393, description='Novelty of the association at the given year'
    )
    yearlyEvidenceCount: int | None = Field(ge=1, le=39591, description='Number of evidence reported at a given year')


class AssociationByDatatypeIndirect(DatasetModel):
    """Associations - indirect (by data type)."""

    dataset_name: ClassVar[str] = 'association_by_datatype_indirect'
    dataset_title: ClassVar[str] = 'Associations - indirect (by data type)'
    dataset_description: ClassVar[str] = (
        'Data type-specific metrics for indirect target-disease associations '
        'and its development over time implying association novelty. Dataset '
        'contains association scores calculated for each data type '
        'indepedently for the aggregation of all evidence supporting any given '
        'target-disease. This dataset contains indirect associations, '
        'therefore evidence is propagated across the disease or phenotype '
        'space using the Experimental Factor Ontology'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'diseaseId': 'EFO_0000180',
        'targetId': 'ENSG00000159216',
        'aggregationType': 'datatypeId',
        'aggregationValue': 'literature',
        'associationScore': 0.15511566285435666,
        'evidenceCount': 2,
        'timeseries': [
            {
                'year': 2023,
                'associationScore': 0.15511566285435666,
                'novelty': 0.0030339631426358005,
                'yearlyEvidenceCount': 1,
            }
        ],
        'currentNovelty': 0.001520741792689767,
    }

    diseaseId: Annotated[str, Reference('disease', 'id'), PrimaryKey()] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    targetId: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id'), PrimaryKey()] = Field(
        pattern=ENSEMBL_GENE_ID, description='Unique identifier for the target'
    )
    aggregationType: Literal['datatypeId'] = Field(description='Evidence column used to aggregate data: datatypeId')
    aggregationValue: Annotated[str, PrimaryKey()] = Field(
        min_length=1, description='Datasource identifier of the group: eg. genetics_literature'
    )
    associationScore: float = Field(
        ge=0.0,
        le=0.988562906674573,
        description=('Current disease/target association score of evidence from the given datatype'),
    )
    evidenceCount: int | None = Field(
        ge=1,
        le=444021,
        description=('Total number of evidence supporting the disease/target association for a given datatype'),
    )
    timeseries: list[Timeseries] = Field(min_length=1, description='Timeseries describing the evolution of association')
    currentNovelty: float | None = Field(
        ge=0.0, le=0.9815264903564463, description='Novelty of the disease/target association'
    )
