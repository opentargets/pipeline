"""Literature: the `literature_entity_lut` output.

Generated from the 26.09 release output (173,379,617 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

import datetime
from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from pts.schemas.dataset import DatasetModel, PolarsDtype


class LiteratureEntityLut(DatasetModel):
    """Literature."""

    dataset_name: ClassVar[str] = 'literature_entity_lut'
    dataset_title: ClassVar[str] = 'Literature'
    dataset_description: ClassVar[str] = 'Unique list of entities identified for each literature entry'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Literature',)
    example: ClassVar[dict] = {
        'pmid': '1000336',
        'pmcid': 'PMC2132931',
        'date': datetime.date(1976, 12, 1),
        'year': 1976,
        'month': 12,
        'day': 1,
        'keywordId': 'ENSG00000152266',
        'relevance': 0.8,
        'keywordType': 'GP',
    }

    pmid: str = Field(min_length=1, description='Europe PubMed Central identifier (EPMCID) of the literature entry')
    pmcid: str | None = Field(min_length=1, description='PubMed Central ID (PMCID) of the literature entry')
    date: datetime.date | None = Field(
        ge=datetime.date(1781, 6, 1),
        le=datetime.date(2124, 2, 1),
        description='Date of publication in YYYY-MM-DD format',
    )
    year: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(ge=1781, le=2124, description='Year of publication')
    month: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(ge=1, le=12, description='Month of publication')
    day: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(ge=1, le=31, description='Day of publication')
    keywordId: str = Field(
        min_length=1, description='Unique identifier for the keyword associated with the literature entry'
    )
    relevance: float = Field(
        ge=0.01, le=1.510077108820886, description='Relevance score of the keyword within the literature entry'
    )
    keywordType: Literal['CD', 'DS', 'GP'] = Field(
        description=(
            'Type of keyword identified in the literature entry. Values: DS: '
            'disease/syndrome, GP: gene/protein, CD: chemical/drug'
        )
    )
