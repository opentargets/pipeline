"""Associations - indirect (by data source): the `association_by_datasource_indirect` output.

Generated from the 26.12 release output (18,872,584 rows) and its croissant metadata by
`scripts/output_schemas/generate.py`, whose docstring lists which observations become rules.
"""

from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PolarsDtype, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID, ONTOLOGY_ID


class Timeseries(StructModel):
    """`timeseries[]`."""

    year: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(ge=0, description='Year, until evidence is considered')
    associationScore: float = Field(
        ge=0.0, le=1.0, description='Association score based on evidence up until a given year'
    )
    novelty: float | None = Field(ge=0.0, le=1.0, description='Novelty of the association at the given year')
    yearlyEvidenceCount: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, description='Number of evidence reported at a given year'
    )


class AssociationByDatasourceIndirect(DatasetModel):
    """Associations - indirect (by data source)."""

    dataset_name: ClassVar[str] = 'association_by_datasource_indirect'
    dataset_title: ClassVar[str] = 'Associations - indirect (by data source)'
    dataset_description: ClassVar[str] = (
        'Data source-specific metrics for indirect target-disease associations '
        'and its development over time implying association novelty. Dataset '
        'contains association scores calculated for each data source '
        'indepedently for the aggregation of all evidence supporting any given '
        'target-disease. This dataset contains indirect associations, '
        'therefore evidence is propagated across the disease or phenotype '
        'space using the Experimental Factor Ontology'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'diseaseId': 'EFO_0000180',
        'targetId': 'ENSG00000020633',
        'aggregationType': 'datasourceId',
        'aggregationValue': 'europepmc',
        'associationScore': 0.030414835853795424,
        'evidenceCount': 1,
        'timeseries': [
            {
                'year': 2023,
                'associationScore': 0.030414835853795424,
                'novelty': 1.8687406198341946e-07,
                'yearlyEvidenceCount': 1,
            }
        ],
        'currentNovelty': 0.0,
    }

    diseaseId: Annotated[str, Reference('disease', 'id'), PrimaryKey()] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    targetId: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id'), PrimaryKey()] = Field(
        pattern=ENSEMBL_GENE_ID, description='Unique identifier for the target'
    )
    aggregationType: Literal['datasourceId'] = Field(description='Evidence column used to aggregate data: datasourceId')
    aggregationValue: Annotated[str, PrimaryKey()] = Field(
        min_length=1, description='Datasource identifier of the group: eg. gwas_credible_set'
    )
    associationScore: float = Field(
        ge=0.0, le=1.0, description=('Current disease/target association score of evidence from the given datasource')
    )
    evidenceCount: int | None = Field(
        ge=0, description=('Total number of evidence supporting the disease/target association for a given datasource')
    )
    timeseries: list[Timeseries] = Field(description='Timeseries describing the evolution of association')
    currentNovelty: float | None = Field(ge=0.0, le=1.0, description='Novelty of the disease/target association')
