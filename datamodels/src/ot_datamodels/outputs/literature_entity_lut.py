"""Literature: the `literature_entity_lut` output.

Generated from the 26.12 release output (193,220,592 rows) and its croissant metadata.
"""

import datetime
from typing import Annotated, ClassVar

import polars as pl
from pydantic import Field

from ot_datamodels import ClickHouseType, DatasetModel, PolarsDtype
from ot_datamodels.vocabularies import LiteratureKeywordType


class LiteratureEntityLut(DatasetModel):
    """Literature."""

    dataset_name: ClassVar[str] = 'literature_entity_lut'
    dataset_title: ClassVar[str] = 'Literature'
    dataset_description: ClassVar[str] = 'Unique list of entities identified for each literature entry'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Literature',)
    example: ClassVar[dict] = {
        'pmid': '1000336',
        'pmcid': 'PMC2132941',
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
    date: datetime.date | None = Field(description='Date of publication in YYYY-MM-DD format')
    year: Annotated[int, PolarsDtype(pl.Int32), ClickHouseType('UInt16')] | None = Field(
        ge=0, le=65535, description='Year of publication'
    )
    month: Annotated[int, PolarsDtype(pl.Int32), ClickHouseType('UInt8')] | None = Field(
        ge=0, le=255, description='Month of publication'
    )
    day: Annotated[int, PolarsDtype(pl.Int32), ClickHouseType('UInt8')] | None = Field(
        ge=0, le=255, description='Day of publication'
    )
    keywordId: str = Field(
        min_length=1, description='Unique identifier for the keyword associated with the literature entry'
    )
    relevance: float = Field(ge=0.0, description='Relevance score of the keyword within the literature entry')
    keywordType: Annotated[LiteratureKeywordType, PolarsDtype(pl.Enum)] = Field(
        description=(
            'Type of keyword identified in the literature entry. Values: DS: '
            'disease/syndrome, GP: gene/protein, CD: chemical/drug'
        )
    )
