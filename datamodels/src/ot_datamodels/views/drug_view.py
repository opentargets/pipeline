"""Drug view: the `drug_view` view (pts `drug_view`).

Every drug molecule row with the ids of the pharmacogenomics records that name it.
"""

from typing import Annotated, ClassVar

from pydantic import Field

from ot_datamodels import ClickHouseType
from ot_datamodels.outputs.drug_molecule import DrugMolecule

# a list element takes no description, so not `from_output`; tests/test_views.py checks it equals pharmacogenomicsId
_PharmacogenomicsId = Annotated[int, ClickHouseType('UInt64'), Field(ge=0)]


class DrugView(DrugMolecule):
    """Drug view."""

    dataset_name: ClassVar[str] = 'drug_view'
    dataset_title: ClassVar[str] = 'Drug view'
    dataset_description: ClassVar[str] = (
        'Drug molecules with the identifiers of the pharmacogenomics records naming them nested in'
    )
    example: ClassVar[dict] = DrugMolecule.example | {'pharmacogenomicsIds': [1]}

    pharmacogenomicsIds: list[_PharmacogenomicsId] = Field(
        description='pharmacogenomicsId of the pharmacogenomics records naming the drug, ascending'
    )
