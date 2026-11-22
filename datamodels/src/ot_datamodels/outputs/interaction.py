"""Molecular interactions: the `interaction` output.

Generated from the 26.12 release output (12,067,375 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar

import polars as pl
from pydantic import Field

from ot_datamodels import Bioregistry, ClickHouseType, DatasetModel, PolarsDtype, PrimaryKey, Reference, StructModel
from ot_datamodels.vocabularies import InteractionSource, SpeciesMnemonic, SpeciesScientificName


class SpeciesA(StructModel):
    """`speciesA`."""

    mnemonic: Annotated[SpeciesMnemonic, PolarsDtype(pl.Enum)] = Field(
        description='Short mnemonic name of the species associated with target A'
    )
    scientificName: Annotated[SpeciesScientificName, PolarsDtype(pl.Enum)] = Field(
        description='Scientific name of the species associated with target A'
    )
    taxonId: Annotated[int, ClickHouseType('UInt32')] = Field(
        ge=0, le=4294967295, description='NCBI taxon ID of the species associated with target A'
    )


class SpeciesB(StructModel):
    """`speciesB`."""

    mnemonic: Annotated[str, ClickHouseType('LowCardinality(String)')] = Field(
        min_length=1, description='Short mnemonic name of the species associated with target B'
    )
    scientificName: Annotated[str, ClickHouseType('LowCardinality(String)')] | None = Field(
        min_length=1, description='Scientific name of the species associated with target B'
    )
    taxonId: int = Field(description='NCBI taxon ID of the species associated with target B')


class Interaction(DatasetModel):
    """Molecular interactions."""

    dataset_name: ClassVar[str] = 'interaction'
    dataset_title: ClassVar[str] = 'Molecular interactions'
    dataset_description: ClassVar[str] = (
        'Integration of molecular interactions reporting experimental or '
        'functional interactions between molecules represented as Platform '
        'targets. This dataset contains pairwise interactions deposited in '
        'several databases capturing: physical interactions (e.g. IntAct), '
        'directional interactions (e.g. Signor), pathway relationships (e.g. '
        'Reactome) or functional interactions (e.g. STRINGdb)'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target',)
    example: ClassVar[dict] = {
        'sourceDatabase': 'intact',
        'targetA': 'ENSANAG00000000634.1',
        'intA': 'URS0000561D92_9606',
        'intABiologicalRole': 'regulator',
        'targetB': 'ENSG00000009307',
        'intB': 'NM_001317122.1',
        'intBBiologicalRole': 'regulator target',
        'speciesA': {'mnemonic': 'human', 'scientificName': 'Homo sapiens', 'taxonId': 9606},
        'speciesB': {'mnemonic': 'human', 'scientificName': 'Homo sapiens', 'taxonId': 9606},
        'count': 1,
        'scoring': 0.28,
        'interactionId': 1561581066758384050,
    }

    sourceDatabase: Annotated[InteractionSource, PolarsDtype(pl.Enum)] = Field(
        description='Source database reporting the molecular interaction'
    )
    targetA: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id')] = Field(
        min_length=1, description=('Open Targets target identifier of the first molecule (target A) in the interaction')
    )
    intA: str = Field(min_length=1, description='Identifier for target A in source')
    intABiologicalRole: Annotated[str, ClickHouseType('LowCardinality(String)')] = Field(
        min_length=1, description='Biological role of target A in the interaction'
    )
    targetB: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id')] | None = Field(
        min_length=1,
        description=('Open Targets target identifier of the second molecule (target B) in the interaction'),
    )
    intB: str = Field(min_length=1, description='Identifier for target B in source')
    intBBiologicalRole: Annotated[str, ClickHouseType('LowCardinality(String)')] = Field(
        min_length=1, description='Biological role of target B in the interaction'
    )
    speciesA: SpeciesA = Field(description='Taxonomic annotation of target A')
    speciesB: SpeciesB = Field(description='Taxonomic annotation of target B')
    count: Annotated[int, ClickHouseType('UInt8')] = Field(
        ge=0, le=255, description='Number of interaction occurrences reported in the source databases'
    )
    scoring: float | None = Field(ge=0.0, le=1.0, description='Scoring or confidence value assigned to the interaction')
    interactionId: Annotated[int, PrimaryKey(), ClickHouseType('UInt64')] = Field(
        ge=0, description='Unique identifier of the molecular interaction'
    )
