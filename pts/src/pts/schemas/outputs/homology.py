"""Target - homologues: the `homology` output.

Generated from the 26.12 release output (3,817,240 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar

import polars as pl
from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PolarsDtype, PrimaryKey, Reference
from pts.schemas.outputs.patterns import DIGITS, ENSEMBL_GENE_ID


class Homology(DatasetModel):
    """Target - homologues."""

    dataset_name: ClassVar[str] = 'homology'
    dataset_title: ClassVar[str] = 'Target - homologues'
    dataset_description: ClassVar[str] = (
        'Ensembl compara homologue/ortholog pairs for each target, resolved to Ensembl target identifiers'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target',)
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000000003',
        'speciesId': '9615',
        'speciesName': 'Dog',
        'homologyType': 'ortholog_one2one',
        'targetGeneId': 'ENSCAFG00845015955',
        'isHighConfidence': '1',
        'targetGeneSymbol': 'TSPAN6',
        'queryPercentageIdentity': 94.6939,
        'targetPercentageIdentity': 94.6939,
        'priority': 7,
    }

    targetId: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id'), PrimaryKey()] = Field(
        pattern=ENSEMBL_GENE_ID, description='Ensembl gene identifier of the query target'
    )
    speciesId: str = Field(pattern=DIGITS, description='Species ID for the homologue')
    speciesName: str = Field(min_length=1, description='Species name for the homologue')
    homologyType: str = Field(min_length=1, description='Type of homology relationship')
    targetGeneId: Annotated[str, PrimaryKey()] = Field(min_length=1, description='Target gene ID of the homologue')
    isHighConfidence: str | None = Field(pattern=DIGITS, description='Indicates if the homology is high confidence')
    targetGeneSymbol: str = Field(min_length=1, description='Gene symbol of the homologous target')
    queryPercentageIdentity: float = Field(ge=0.0, description='Percentage identity of the query gene in the homologue')
    targetPercentageIdentity: float = Field(
        ge=0.0, description='Percentage identity of the homologue in the query gene'
    )
    priority: Annotated[int, PolarsDtype(pl.Int32)] = Field(ge=0, description='Priority score for the homologue')
