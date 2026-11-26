"""Mouse phenotypes by target: the `mouse_phenotype_by_target_view` view (pts `mouse_phenotype_by_target`).

One row per human target of `mouse_phenotype`, as served in `mouse_phenotypes`: its mouse phenotypes ordered by
`modelPhenotypeId, targetInModelMgiId`, each the output row.
"""

from typing import ClassVar

from pydantic import Field

from ot_datamodels import DatasetModel, from_output, row_of
from ot_datamodels.outputs.mouse_phenotype import MousePhenotype

TargetFromSourceId = from_output(MousePhenotype, 'targetFromSourceId', primary_key=True)
MousePhenotypeItem = row_of(MousePhenotype, list(MousePhenotype.model_fields), name='MousePhenotypeItem')


class MousePhenotypeByTargetView(DatasetModel):
    """Mouse phenotypes by target."""

    dataset_name: ClassVar[str] = 'mouse_phenotype_by_target_view'
    dataset_title: ClassVar[str] = 'Mouse phenotypes by target'
    dataset_description: ClassVar[str] = 'Phenotypes of mouse models mapped to each human target, as served by target'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target',)
    example: ClassVar[dict] = {
        'targetFromSourceId': MousePhenotype.example['targetFromSourceId'],
        'mouse_phenotypes': [MousePhenotype.example],
    }

    targetFromSourceId: TargetFromSourceId
    mouse_phenotypes: list[MousePhenotypeItem] = Field(
        min_length=1, description='Mouse phenotypes of the target, by modelPhenotypeId, targetInModelMgiId'
    )
