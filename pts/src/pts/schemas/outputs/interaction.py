"""Molecular interactions: the `interaction` output.

Generated from the 26.09 release output (14,612,199 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey, Reference, StructModel


class SpeciesA(StructModel):
    """`speciesA`."""

    mnemonic: Literal['human'] = Field(description='Short mnemonic name of the species associated with target A')
    scientificName: Literal['Homo sapiens'] = Field(
        description='Scientific name of the species associated with target A'
    )
    taxonId: int = Field(ge=9606, le=9606, description='NCBI taxon ID of the species associated with target A')


class SpeciesB(StructModel):
    """`speciesB`."""

    mnemonic: str = Field(min_length=1, description='Short mnemonic name of the species associated with target B')
    scientificName: str | None = Field(
        min_length=1, description='Scientific name of the species associated with target B'
    )
    taxonId: int = Field(ge=-2, le=3052342, description='NCBI taxon ID of the species associated with target B')


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
        'targetA': 'ENSG00000087302',
        'intA': 'Q9Y224',
        'intABiologicalRole': 'unspecified role',
        'targetB': 'ENSG00000077514',
        'intB': 'B4URF7',
        'intBBiologicalRole': 'unspecified role',
        'speciesA': {'mnemonic': 'human', 'scientificName': 'Homo sapiens', 'taxonId': 9606},
        'speciesB': {
            'mnemonic': 'influenza a virus (a/wsn/1933(h1n1))',
            'scientificName': 'Influenza A virus (A/WSN/1933(H1N1))',
            'taxonId': 382835,
        },
        'count': 2,
        'scoring': 0.43,
        'interactionId': -7262777738973363773,
    }

    sourceDatabase: Literal['intact', 'reactome', 'signor', 'string'] = Field(
        description='Source database reporting the molecular interaction'
    )
    targetA: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id')] = Field(
        min_length=1, description=('Open Targets target identifier of the first molecule (target A) in the interaction')
    )
    intA: str = Field(min_length=1, description='Identifier for target A in source')
    intABiologicalRole: str = Field(min_length=1, description='Biological role of target A in the interaction')
    targetB: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id')] | None = Field(
        min_length=1,
        description=('Open Targets target identifier of the second molecule (target B) in the interaction'),
    )
    intB: str = Field(min_length=1, description='Identifier for target B in source')
    intBBiologicalRole: str = Field(min_length=1, description='Biological role of target B in the interaction')
    speciesA: SpeciesA = Field(description='Taxonomic annotation of target A')
    speciesB: SpeciesB = Field(description='Taxonomic annotation of target B')
    count: int = Field(ge=1, le=227, description='Number of interaction occurrences reported in the source databases')
    scoring: float | None = Field(
        ge=0.15, le=1.0, description='Scoring or confidence value assigned to the interaction'
    )
    interactionId: Annotated[int, PrimaryKey()] = Field(
        ge=-9223371397784454700, le=9223371751993628076, description='Unique identifier of the molecular interaction'
    )
