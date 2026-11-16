"""Drug - mechanism of action: the `drug_mechanism_of_action` output.

Generated from the 26.12 release output (8,375 rows) and its croissant metadata by
`scripts/output_schemas/generate.py`, whose docstring lists which observations become rules.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import DatasetModel, Reference, StructModel
from pts.schemas.outputs.patterns import CHEMBL_ID, ENSEMBL_GENE_ID


class References(StructModel):
    """`references[]`."""

    source: str = Field(min_length=1, description='List of source names documenting the mechanism information')
    ids: list[Annotated[str, Field(min_length=1)]] = Field(
        description='List of identification codes for the reference sources'
    )
    urls: list[Annotated[str, Field(min_length=1)]] = Field(
        description='List of web addresses referencing the mechanism information'
    )


class DrugMechanismOfAction(DatasetModel):
    """Drug - mechanism of action."""

    dataset_name: ClassVar[str] = 'drug_mechanism_of_action'
    dataset_title: ClassVar[str] = 'Drug - mechanism of action'
    dataset_description: ClassVar[str] = (
        'Mechanisms of action for drug or clinical candidate molecules '
        'included in ChEMBL database. The mechanism of action includes '
        'information about the pharmacological target of the drug, the mode of '
        'action and the drug modality'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Drug',)
    example: ClassVar[dict] = {
        'mechanismOfAction': 'Carbonic anhydrase VII inhibitor',
        'actionType': 'INHIBITOR',
        'chemblIds': ['CHEMBL19'],
        'references': [
            {'source': 'PubMed', 'ids': ['18336310'], 'urls': ['http://europepmc.org/abstract/MED/18336310']}
        ],
        'targetName': 'Carbonic anhydrase 7',
        'targetType': 'single protein',
        'targets': ['ENSG00000168748'],
        'curation': 'manual',
    }

    mechanismOfAction: str = Field(min_length=1, description='Drug pharmacological action')
    actionType: str = Field(
        min_length=1, description=('Classification of how the drug interacts with its target (e.g., ACTIVATOR)')
    )
    chemblIds: list[Annotated[Annotated[str, Reference('drug_molecule', 'id')], Field(pattern=CHEMBL_ID)]] = Field(
        description='List of Open Targets molecule identifiers'
    )
    references: list[References] = Field(description='List of sources documenting the mechanism information')
    targetName: str = Field(min_length=1, description='Specific biological entity on which the drug has an effect')
    targetType: str = Field(
        min_length=1, description=("Classification of the target's biological category (e.g., protein complex)")
    )
    targets: list[Annotated[Annotated[str, Reference('target', 'id')], Field(pattern=ENSEMBL_GENE_ID)]] = Field(
        description='List of Open Targets target identifiers'
    )
    curation: Literal['electronic', 'manual'] = Field(
        description=(
            'How the mechanism was curated: manual (ChEMBL) or electronic '
            '(inferred by a language model from ClinicalTrials.gov trial text or '
            'Europe PMC literature sentences)'
        )
    )
