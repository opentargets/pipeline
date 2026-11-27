"""Pharmacovigilance by drug: the `pharmacovigilance_by_drug_view` view (pts `pharmacovigilance_by_drug`).

One row per drug of `pharmacovigilance`: its adverse events by `logLR` descending,
then `name`, and the drug's critical value.
"""

from typing import ClassVar, Literal

from pydantic import Field

from ot_datamodels import DatasetModel, StructModel, from_output
from ot_datamodels.outputs.pharmacovigilance import Pharmacovigilance
from ot_datamodels.serving import Serving

ChemblId = from_output(Pharmacovigilance, 'chembl_id', primary_key=True)
Count = from_output(Pharmacovigilance, 'count')
Llr = from_output(Pharmacovigilance, 'llr')
Critval = from_output(Pharmacovigilance, 'critval')

_EXAMPLE = Pharmacovigilance.example


def _description(field: str) -> str:
    description = Pharmacovigilance.model_fields[field].description
    assert description is not None
    return description


class AdverseEvent(StructModel):
    """`adverse_events[]`: the output row but the drug, `event` as `name` and `llr` as `logLR`."""

    name: str = Field(min_length=1, description=_description('event'))
    count: Count
    logLR: Llr
    critval: Critval
    meddraCode: Literal[''] = Field(description=_description('meddraCode'))


class PharmacovigilanceByDrugView(DatasetModel):
    """Pharmacovigilance by drug."""

    dataset_name: ClassVar[str] = 'pharmacovigilance_by_drug_view'
    dataset_title: ClassVar[str] = 'Pharmacovigilance by drug'
    dataset_description: ClassVar[str] = 'Significant adverse drug reactions of each drug, as served by drug'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Drug',)
    clickhouse: ClassVar[Serving] = Serving(engine='rocksdb')
    example: ClassVar[dict] = {
        'chembl_id': _EXAMPLE['chembl_id'],
        'adverse_events': [
            {
                'name': _EXAMPLE['event'],
                'count': _EXAMPLE['count'],
                'logLR': _EXAMPLE['llr'],
                'critval': _EXAMPLE['critval'],
                'meddraCode': _EXAMPLE['meddraCode'],
            }
        ],
        'criticalValue': _EXAMPLE['critval'],
    }

    chembl_id: ChemblId
    adverse_events: list[AdverseEvent] = Field(
        min_length=1, description='Adverse events of the drug, by logLR descending, then name'
    )
    criticalValue: Critval
