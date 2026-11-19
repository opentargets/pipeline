"""Target - safety events: the `target_safety_event` output.

Generated from the 26.12 release output (4,387 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from ot_datamodels import Bioregistry, DatasetModel, Reference, StructModel
from ot_datamodels.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ONTOLOGY_ID


class Effects(StructModel):
    """`effects[]`."""

    direction: Literal['Activation/Increase/Upregulation', 'Inhibition/Decrease/Downregulation'] = Field(
        description='Direction of the reported effect (e.g. increase or decrease)'
    )
    dosing: str | None = Field(min_length=1, description='Dosing conditions related to the reported effect')


class Biosamples(StructModel):
    """`biosamples[]`."""

    tissueLabel: str | None = Field(min_length=1, description='Label of the biosample tissue')
    tissueId: str | None = Field(min_length=1, description='Tissue ID for the biosample')
    cellLabel: str | None = Field(min_length=1, description='Label of the biosample cell')
    cellFormat: str | None = Field(min_length=1, description='Format of the biosample cells')
    cellId: str | None = Field(description='Cell identifier')


class Studies(StructModel):
    """`studies[]`."""

    description: str | None = Field(min_length=1, description='Description of the safety study')
    name: str | None = Field(min_length=1, description='Name of the safety study')
    type: Literal['biochemical', 'cell-based', 'clinical', 'preclinical'] | None = Field(
        description='Type of safety study'
    )


class TargetSafetyEvent(DatasetModel):
    """Target - safety events."""

    dataset_name: ClassVar[str] = 'target_safety_event'
    dataset_title: ClassVar[str] = 'Target - safety events'
    dataset_description: ClassVar[str] = (
        'Safety-related events associated with biological targets, including '
        'assay observations, affected biosamples, supporting studies, and '
        'evidence sources'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target',)
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000001084',
        'event': 'a drug hypersensitivity reaction',
        'eventId': 'HP_0031860',
        'effects': [{'direction': 'Activation/Increase/Upregulation', 'dosing': 'acute'}],
        'biosamples': [
            {
                'tissueLabel': 'cardiovascular',
                'tissueId': 'UBERON_0004535',
                'cellLabel': 'HepaRG',
                'cellFormat': 'cell line',
                'cellId': None,
            }
        ],
        'datasource': 'ClinPGx',
        'literature': '28216264',
        'url': 'https://www.clinpgx.org/search?query=ENSG00000001084',
        'studies': [
            {
                'description': (
                    'Genetic variation linked to this safety liability: 6_53527310_A_A,A, '
                    '6_53527310_A_A,C, 6_53527310_A_C,C'
                ),
                'name': 'sulfamethoxazole induced effect',
                'type': 'clinical',
            }
        ],
    }

    targetId: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Ensembl gene identifier of the target the safety event was reported on'
    )
    event: str = Field(min_length=1, description='Safety event associated with the target')
    eventId: str | None = Field(pattern=ONTOLOGY_ID, description='Unique identifier for the safety event')
    effects: list[Effects] | None = Field(description='Effects reported for the safety event')
    biosamples: list[Biosamples] | None = Field(description='Biosamples used in safety assessments')
    datasource: str = Field(min_length=1, description='Data source reporting the safety event')
    literature: str | None = Field(pattern=DIGITS, description='Literature references for the safety event')
    url: str | None = Field(min_length=1, description='URL linking to more details on the safety event')
    studies: list[Studies] = Field(description='Studies related to safety assessments')
