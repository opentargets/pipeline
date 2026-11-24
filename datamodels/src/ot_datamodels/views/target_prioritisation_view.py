"""Target prioritisation view: the `target_prioritisation_view` view, served as `target_prioritisation`.

Written by pts `target_prioritisation_view`: one row per `target_prioritisation` row, its non-null values as
(name, value) strings in ClickHouse `toString` form, then the target's DepMap essentiality.
"""

from typing import ClassVar, Literal

from pydantic import Field

from ot_datamodels import DatasetModel, StructModel, from_output
from ot_datamodels.outputs.target_prioritisation import TargetPrioritisation

# the pair names, in served order; a plain Literal, served as String
PrioritisationKey = Literal[
    'geneticConstraint',
    'hasHighQualityChemicalProbes',
    'hasLigand',
    'hasPocket',
    'hasSafetyEvent',
    'hasSmallMoleculeBinder',
    'isCancerDriverGene',
    'isInMembrane',
    'isSecreted',
    'maxClinicalStage',
    'mouseKOScore',
    'mouseOrthologMaxIdentityPercentage',
    'paralogMaxIdentityPercentage',
    'tissueDistribution',
    'tissueSpecificity',
    'celltypeSpecificity',
    'celltypeDistribution',
    'geneEssentiality',
]

TargetId = from_output(TargetPrioritisation, 'targetId', primary_key=True)


class PrioritisationItem(StructModel):
    """`items[]`: one prioritisation value of the target."""

    key: PrioritisationKey = Field(description='Name of the target_prioritisation column, or geneEssentiality')
    value: str = Field(
        min_length=1,
        description=(
            'The value as ClickHouse toString writes it; geneEssentiality is -1 for a DepMap common essential '
            'gene and 0 for another gene DepMap screened'
        ),
    )


class TargetPrioritisationView(DatasetModel):
    """Target prioritisation view."""

    dataset_name: ClassVar[str] = 'target_prioritisation_view'
    dataset_title: ClassVar[str] = 'Target prioritisation view'
    dataset_description: ClassVar[str] = 'The target prioritisation values of each target as name/value strings'
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000000003',
        'items': [
            {'key': 'geneticConstraint', 'value': '0.42952615721720067'},
            {'key': 'hasHighQualityChemicalProbes', 'value': '1'},
            {'key': 'hasSafetyEvent', 'value': '-1'},
            {'key': 'maxClinicalStage', 'value': '1'},
            {'key': 'geneEssentiality', 'value': '0'},
        ],
    }

    targetId: TargetId
    items: list[PrioritisationItem] = Field(
        description=(
            'Non-null values in a fixed order, then geneEssentiality for a target DepMap screened; '
            'empty when all are null'
        )
    )
