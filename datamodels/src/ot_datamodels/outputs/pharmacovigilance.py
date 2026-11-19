"""Pharmacovigilance (by drug): the `pharmacovigilance` output.

Generated from the 26.12 release output (118,060 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from ot_datamodels import DatasetModel, PrimaryKey, Reference
from ot_datamodels.outputs.patterns import CHEMBL_ID


class Pharmacovigilance(DatasetModel):
    """Pharmacovigilance (by drug)."""

    dataset_name: ClassVar[str] = 'pharmacovigilance'
    dataset_title: ClassVar[str] = 'Pharmacovigilance (by drug)'
    dataset_description: ClassVar[str] = (
        'Significant adverse drug reactions (ADRs) associated with drugs. This '
        "dataset is based on the FDA's Adverse Event Reporting System (FAERS) "
        "reporting post-marketing surveillance data and it's filtered to "
        'include only reports submitted by health professionals. The '
        'significance of a given drug-ADR is estimated using a Likelihood '
        'Ratio Test (LRT) using all reports associated with the given drug'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Drug',)
    example: ClassVar[dict] = {
        'chembl_id': 'CHEMBL1000',
        'event': 'abdominal discomfort',
        'count': 2007,
        'llr': 5959.957148894588,
        'critval': 70.78422221082292,
        'meddraCode': '',
    }

    chembl_id: Annotated[str, Reference('drug_molecule', 'id'), PrimaryKey()] = Field(
        pattern=CHEMBL_ID, description='Unique identifier for the drug or clinical candidate'
    )
    event: Annotated[str, PrimaryKey()] = Field(
        min_length=1, description='Reported adverse drug reaction associated with the drug'
    )
    count: int = Field(ge=0, description='Number of occurrences of the adverse event in the dataset')
    llr: float = Field(
        ge=0.0,
        description=(
            'Log-likelihood ratio (LLR) measuring the strength of the association '
            'between the drug and the adverse event'
        ),
    )
    critval: float = Field(
        ge=0.0, description=('Critical value used to determine statistical significance of the association')
    )
    meddraCode: Literal[''] = Field(
        description=('Medical Dictionary for Regulatory Activities (MedDRA) code for the adverse drug reaction')
    )
