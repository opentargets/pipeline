"""Target - tractability: the `target_tractability` output.

Generated from the 26.09 release output (528,332 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey, Reference
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID


class TargetTractability(DatasetModel):
    """Target - tractability."""

    dataset_name: ClassVar[str] = 'target_tractability'
    dataset_title: ClassVar[str] = 'Target - tractability'
    dataset_description: ClassVar[str] = (
        'Per-target tractability bucket assessments across small molecule, antibody, and other drug modalities'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target',)
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000146648',
        'modality': 'SM',
        'category': 'Approved Drug',
        'value': True,
    }

    targetId: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id'), PrimaryKey()] = Field(
        pattern=ENSEMBL_GENE_ID, description='Ensembl gene identifier of the assessed target'
    )
    modality: Annotated[str, PrimaryKey()] = Field(
        min_length=1,
        description=('Drug modality the tractability assessment applies to (e.g. Small molecule, Antibody)'),
    )
    category: Annotated[str, PrimaryKey()] = Field(
        min_length=1,
        description=(
            'Identifier of the tractability bucket assessed (e.g. clinical phase, structural evidence source)'
        ),
    )
    value: bool = Field(description='Boolean outcome of the tractability bucket assessment')
