"""Target Prioritisation: the `target_prioritisation` output.

Generated from the 26.12 release output (78,733 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar

import polars as pl
from pydantic import Field

from pts.schemas.dataset import DatasetModel, PolarsDtype, PrimaryKey, Reference
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID


class TargetPrioritisation(DatasetModel):
    """Target Prioritisation."""

    dataset_name: ClassVar[str] = 'target_prioritisation'
    dataset_title: ClassVar[str] = 'Target Prioritisation'
    dataset_description: ClassVar[str] = 'List of target-specific attributes from the Target Engine project'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target',)
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000000003',
        'isInMembrane': 0,
        'isSecreted': 0,
        'hasSafetyEvent': -1,
        'hasPocket': 0,
        'hasLigand': 0,
        'hasSmallMoleculeBinder': 0,
        'geneticConstraint': 0.42952615721720067,
        'paralogMaxIdentityPercentage': 0.0,
        'mouseOrthologMaxIdentityPercentage': 0.65306,
        'isCancerDriverGene': -1,
        'mouseKOScore': 0.0,
        'hasHighQualityChemicalProbes': 1,
        'maxClinicalStage': 1.0,
        'tissueSpecificity': -0.3191645030410264,
        'tissueDistribution': 0.03703703703703709,
        'celltypeSpecificity': -0.14944849048259712,
        'celltypeDistribution': 0.2588235294117647,
    }

    targetId: Annotated[str, Reference('target', 'id'), PrimaryKey()] = Field(
        pattern=ENSEMBL_GENE_ID, description='Unique identifier for the target, typically an Ensembl gene ID'
    )
    isInMembrane: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, description=('Indicates if the target protein is located in the cell or plasma membrane')
    )
    isSecreted: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, description=('Indicates if the target protein is secreted or predicted to be secreted')
    )
    hasSafetyEvent: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        description='Indicates if there are known safety events associated with the target'
    )
    hasPocket: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, description=('Indicates if the target has predicted binding pockets suitable for small molecule binding')
    )
    hasLigand: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, description='Indicates if the target binds at least one high-quality ligand'
    )
    hasSmallMoleculeBinder: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, description='Indicates if the target has at least one small molecule binder'
    )
    geneticConstraint: float | None = Field(
        description=(
            'Represents the genetic constraint of the target, indicating how intolerant it is to genetic variation'
        )
    )
    paralogMaxIdentityPercentage: float | None = Field(
        description="Maximum percentage identity among the target's paralogues"
    )
    mouseOrthologMaxIdentityPercentage: float | None = Field(
        ge=0.0, le=1.0, description='Maximum percentage identity between the target and its mouse ortholog'
    )
    isCancerDriverGene: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        description='Indicates if the target is identified as a cancer driver gene'
    )
    mouseKOScore: float | None = Field(
        description=('Represents the phenotypic score from mouse knockout models for the target')
    )
    hasHighQualityChemicalProbes: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, description=('Indicates if there are high-quality chemical probes available for the target')
    )
    maxClinicalStage: float | None = Field(
        ge=0.0,
        le=1.0,
        description=('Score derived from the highest clinical stage that the target has reached for any indication'),
    )
    tissueSpecificity: float | None = Field(
        description=("Describes the specificity of the target's expression across different tissues")
    )
    tissueDistribution: float | None = Field(
        description=("Describes the distribution pattern of the target's expression in various tissues")
    )
    celltypeSpecificity: float | None = Field(
        description=("Describes the specificity of the target's expression across different cell types")
    )
    celltypeDistribution: float | None = Field(
        description=("Describes the distribution pattern of the target's expression in various cell types")
    )
