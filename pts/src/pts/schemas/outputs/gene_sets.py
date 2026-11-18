"""Target - gene sets: the `gene_sets` output.

Generated from the 26.12 release output (24,439 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey, Reference
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID


class GeneSets(DatasetModel):
    """Target - gene sets."""

    dataset_name: ClassVar[str] = 'gene_sets'
    dataset_title: ClassVar[str] = 'Target - gene sets'
    dataset_description: ClassVar[str] = (
        'Gene sets for GO terms, Reactome pathways and ChEMBL target classes. '
        'Each set holds the targets annotated to the term or to any of its '
        'descendants'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target',)
    example: ClassVar[dict] = {
        'id': '1',
        'label': 'Enzyme',
        'source': 'chembl_target_class',
        'category': 'ChEMBL Target Class',
        'parentIds': ['1'],
        'entityIds': ['ENSG00000000419'],
    }

    id: Annotated[str, PrimaryKey()] = Field(
        min_length=1, description='Term identifier: GO term, Reactome pathway or ChEMBL protein class'
    )
    label: str = Field(min_length=1, description='Term name')
    source: Annotated[str, PrimaryKey()] = Field(
        min_length=1, description='Hierarchy the term belongs to: go, reactome or chembl_target_class'
    )
    category: Literal['ChEMBL Target Class', 'GO:BP', 'GO:CC', 'GO:MF', 'Reactome'] = Field(
        description=('Gene set category: GO:BP, GO:MF, GO:CC, Reactome or ChEMBL Target Class')
    )
    parentIds: list[Annotated[str, Field(min_length=1)]] = Field(
        description='Direct parent terms in the same hierarchy'
    )
    entityIds: list[Annotated[Annotated[str, Reference('target', 'id')], Field(pattern=ENSEMBL_GENE_ID)]] = Field(
        description='Targets annotated to the term or to any of its descendants'
    )
