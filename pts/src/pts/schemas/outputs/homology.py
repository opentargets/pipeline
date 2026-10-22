"""Target - homologues: the `homology` output.

Generated from the 26.09 release output (3,817,240 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PolarsDtype, PrimaryKey, Reference
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID


class Homology(DatasetModel):
    """Target - homologues."""

    dataset_name: ClassVar[str] = 'homology'
    dataset_title: ClassVar[str] = 'Target - homologues'
    dataset_description: ClassVar[str] = (
        'Ensembl compara homologue/ortholog pairs for each target, resolved to Ensembl target identifiers'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target',)
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000231192',
        'speciesId': '9606',
        'speciesName': 'Human',
        'homologyType': 'other_paralog',
        'targetGeneId': 'ENSG00000279395',
        'isHighConfidence': '1',
        'targetGeneSymbol': 'OR5L1',
        'queryPercentageIdentity': 41.5335,
        'targetPercentageIdentity': 41.8006,
        'priority': 0,
    }

    targetId: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id'), PrimaryKey()] = Field(
        pattern=ENSEMBL_GENE_ID, description='Ensembl gene identifier of the query target'
    )
    speciesId: Literal[
        '10090', '10116', '10141', '6239', '7227', '7955', '8364', '9544', '9598', '9606', '9615', '9823', '9986'
    ] = Field(description='Species ID for the homologue')
    speciesName: Literal[
        'Caenorhabditis elegans (Nematode, N2)',
        'Chimpanzee',
        'Dog',
        'Drosophila melanogaster - (Fruit fly)',
        'Guinea Pig',
        'Human',
        'Macaque',
        'Mouse',
        'Norway rat - BN/NHsdMcwi',
        'Pig',
        'Rabbit',
        'Tropical clawed frog',
        'Zebrafish',
    ] = Field(description='Species name for the homologue')
    homologyType: Literal[
        'gene_split',
        'ortholog_many2many',
        'ortholog_one2many',
        'ortholog_one2one',
        'other_paralog',
        'within_species_paralog',
    ] = Field(description='Type of homology relationship')
    targetGeneId: Annotated[str, PrimaryKey()] = Field(min_length=1, description='Target gene ID of the homologue')
    isHighConfidence: Literal['0', '1'] | None = Field(description='Indicates if the homology is high confidence')
    targetGeneSymbol: str = Field(min_length=1, description='Gene symbol of the homologous target')
    queryPercentageIdentity: float = Field(
        ge=0.0, le=100.0, description='Percentage identity of the query gene in the homologue'
    )
    targetPercentageIdentity: float = Field(
        ge=0.0, le=100.0, description='Percentage identity of the homologue in the query gene'
    )
    priority: Annotated[int, PolarsDtype(pl.Int32)] = Field(ge=0, le=12, description='Priority score for the homologue')
