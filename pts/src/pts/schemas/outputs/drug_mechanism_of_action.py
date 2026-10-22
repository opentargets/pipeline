"""Drug - mechanism of action: the `drug_mechanism_of_action` output.

Generated from the 26.09 release output (6,092 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import DatasetModel, Reference, StructModel
from pts.schemas.outputs.patterns import CHEMBL_ID, ENSEMBL_GENE_ID


class References(StructModel):
    """`references[]`."""

    source: Literal[
        'BNF',
        'ClinicalTrials',
        'DOI',
        'DailyMed',
        'EMA',
        'Expert',
        'FDA',
        'HMA',
        'ISBN',
        'IUPHAR',
        'InterPro',
        'KEGG',
        'Other',
        'PMC',
        'PMDA',
        'Patent',
        'PubChem',
        'PubMed',
        'UniProt',
        'Wikipedia',
    ] = Field(description='List of source names documenting the mechanism information')
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
    }

    mechanismOfAction: str = Field(min_length=1, description='Drug pharmacological action')
    actionType: str = Field(
        min_length=1, description=('Classification of how the drug interacts with its target (e.g., ACTIVATOR)')
    )
    chemblIds: list[Annotated[Annotated[str, Reference('drug_molecule', 'id')], Field(pattern=CHEMBL_ID)]] = Field(
        min_length=1, description='List of Open Targets molecule identifiers'
    )
    references: list[References] = Field(description='List of sources documenting the mechanism information')
    targetName: str = Field(min_length=1, description='Specific biological entity on which the drug has an effect')
    targetType: Literal[
        'chimeric protein',
        'nucleic-acid',
        'protein complex',
        'protein complex group',
        'protein family',
        'protein nucleic-acid complex',
        'protein-protein interaction',
        'selectivity group',
        'single protein',
    ] = Field(description=("Classification of the target's biological category (e.g., protein complex)"))
    targets: list[Annotated[Annotated[str, Reference('target', 'id')], Field(pattern=ENSEMBL_GENE_ID)]] = Field(
        description='List of Open Targets target identifiers'
    )
