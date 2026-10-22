"""Pharmacovigilance (by drug): the `openfda_significant_adverse_drug_reactions` output.

Generated from the 26.09 release output (118,452 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey, Reference
from pts.schemas.outputs.patterns import CHEMBL_ID


class OpenfdaSignificantAdverseDrugReactions(DatasetModel):
    """Pharmacovigilance (by drug)."""

    dataset_name: ClassVar[str] = 'openfda_significant_adverse_drug_reactions'
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
        'chembl_id': 'CHEMBL1240705',
        'event': 'acinetobacter infection',
        'count': 25,
        'llr': 89.30229723540833,
        'critval': 45.78399690483275,
        'meddraCode': '',
    }

    chembl_id: Annotated[str, Reference('drug_molecule', 'id'), PrimaryKey()] = Field(
        pattern=CHEMBL_ID, description='Unique identifier for the drug or clinical candidate'
    )
    event: Annotated[str, PrimaryKey()] = Field(
        min_length=1, description='Reported adverse drug reaction associated with the drug'
    )
    count: int = Field(ge=1, le=36582, description='Number of occurrences of the adverse event in the dataset')
    llr: float = Field(
        ge=5.4139244850725845,
        le=42515.4722000036,
        description=(
            'Log-likelihood ratio (LLR) measuring the strength of the association '
            'between the drug and the adverse event'
        ),
    )
    critval: float = Field(
        ge=5.277737117370862,
        le=6407.026716470718,
        description=('Critical value used to determine statistical significance of the association'),
    )
    meddraCode: Literal[''] = Field(
        description=('Medical Dictionary for Regulatory Activities (MedDRA) code for the adverse drug reaction')
    )
