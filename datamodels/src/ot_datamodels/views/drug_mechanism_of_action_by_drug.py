"""Mechanisms of action by drug: the `drug_mechanism_of_action_by_drug` view (pts `drug_mechanism_of_action_by_drug`).

One row per molecule named in a `drug_mechanism_of_action` row's `chemblIds`, as served in `mechanism_of_action`:
the mechanisms naming it and the distinct target and action types among them, in vocabulary order. A mechanism
naming no molecule is in no row.
"""

from typing import Annotated, ClassVar

import polars as pl
from pydantic import Field

from ot_datamodels import DatasetModel, PolarsDtype, PrimaryKey, Reference, row_of
from ot_datamodels.outputs.drug_mechanism_of_action import DrugMechanismOfAction
from ot_datamodels.outputs.patterns import CHEMBL_ID
from ot_datamodels.vocabularies import ChemblActionType, ChemblTargetType

#: the mechanism fields served, in output order
MECHANISM_FIELDS = ['mechanismOfAction', 'actionType', 'references', 'targetName', 'targets', 'curation']
MechanismItem = row_of(DrugMechanismOfAction, MECHANISM_FIELDS, name='MechanismItem')

_EXAMPLE = DrugMechanismOfAction.example


class DrugMechanismOfActionByDrug(DatasetModel):
    """Mechanisms of action by drug."""

    dataset_name: ClassVar[str] = 'drug_mechanism_of_action_by_drug'
    dataset_title: ClassVar[str] = 'Mechanisms of action by drug'
    dataset_description: ClassVar[str] = 'Mechanisms of action of each drug molecule, as served by molecule'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Drug',)
    example: ClassVar[dict] = {
        'chemblId': _EXAMPLE['chemblIds'][0],
        'rows': [{f: _EXAMPLE[f] for f in MECHANISM_FIELDS}],
        'uniqueTargetTypes': [_EXAMPLE['targetType']],
        'uniqueActionTypes': [_EXAMPLE['actionType']],
    }

    chemblId: Annotated[str, Reference('drug_molecule', 'id'), PrimaryKey()] = Field(
        pattern=CHEMBL_ID, description='Open Targets molecule identifier'
    )
    rows: list[MechanismItem] = Field(
        min_length=1,
        description='Mechanisms naming the molecule, by mechanismOfAction, actionType, targetName, curation',
    )
    uniqueTargetTypes: list[Annotated[ChemblTargetType, PolarsDtype(pl.Enum)]] = Field(
        min_length=1, description='Distinct target types of the mechanisms, in vocabulary order'
    )
    uniqueActionTypes: list[Annotated[ChemblActionType, PolarsDtype(pl.Enum)]] = Field(
        min_length=1, description='Distinct action types of the mechanisms, in vocabulary order'
    )
