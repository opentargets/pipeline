"""Drug warnings by drug: the `drug_warning_by_drug_view` view (pts `drug_warning_by_drug`).

One row per molecule named in a `drug_warning` row's `chemblIds`, as served in `drug_warnings`: the warnings naming
it, ordered by `id`, each the output row. A warning naming no molecule is in no row.
"""

from typing import Annotated, ClassVar

from pydantic import Field

from ot_datamodels import DatasetModel, PrimaryKey, Reference, row_of
from ot_datamodels.outputs.drug_warning import DrugWarning
from ot_datamodels.outputs.patterns import CHEMBL_ID
from ot_datamodels.serving import Serving

DrugWarningItem = row_of(DrugWarning, list(DrugWarning.model_fields), name='DrugWarningItem')


class DrugWarningByDrugView(DatasetModel):
    """Drug warnings by drug."""

    dataset_name: ClassVar[str] = 'drug_warning_by_drug_view'
    dataset_title: ClassVar[str] = 'Drug warnings by drug'
    dataset_description: ClassVar[str] = 'Blackbox and withdrawn warnings of each drug molecule, as served by molecule'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Drug',)
    clickhouse: ClassVar[Serving] = Serving(engine='rocksdb')
    example: ClassVar[dict] = {'chemblId': DrugWarning.example['chemblIds'][0], 'drugWarnings': [DrugWarning.example]}

    chemblId: Annotated[str, Reference('drug_molecule', 'id'), PrimaryKey()] = Field(
        pattern=CHEMBL_ID, description='Open Targets molecule identifier'
    )
    drugWarnings: list[DrugWarningItem] = Field(min_length=1, description='Warnings naming the molecule, by id')
