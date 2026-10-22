"""Target - safety events: the `target_safety_event` output.

Generated from the 26.09 release output (4,387 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, Reference, StructModel
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID, ONTOLOGY_ID


class Effects(StructModel):
    """`effects[]`."""

    direction: Literal['Activation/Increase/Upregulation', 'Inhibition/Decrease/Downregulation'] = Field(
        description='Direction of the reported effect (e.g. increase or decrease)'
    )
    dosing: Literal['acute', 'chronic', 'developmental toxicity', 'general'] | None = Field(
        description='Dosing conditions related to the reported effect'
    )


class Biosamples(StructModel):
    """`biosamples[]`."""

    tissueLabel: str | None = Field(min_length=1, description='Label of the biosample tissue')
    tissueId: str | None = Field(min_length=1, description='Tissue ID for the biosample')
    cellLabel: str | None = Field(min_length=1, description='Label of the biosample cell')
    cellFormat: (
        Literal['cell line', 'cell-free', 'primary cell', 'primary cell co-culture', 'tissue-based cell-free'] | None
    ) = Field(description='Format of the biosample cells')
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
        'targetId': 'ENSG00000085563',
        'event': 'regulation of transcription factor activity',
        'eventId': 'HP_0031860',
        'effects': [{'direction': 'Activation/Increase/Upregulation', 'dosing': 'acute'}],
        'biosamples': [
            {
                'tissueLabel': 'brain',
                'tissueId': 'UBERON_0001264',
                'cellLabel': 'HepaRG',
                'cellFormat': 'cell line',
                'cellId': None,
            }
        ],
        'datasource': 'ToxCast',
        'literature': '28216264',
        'url': ('https://www.epa.gov/chemical-research/exploring-toxcast-data-downloadable-data'),
        'studies': [
            {
                'description': (
                    'LTEA_HepaRG_ABCB1is one of 100 assay component(s) measured in the '
                    'LTEA HepaRG cell culture assay. It is a marker of Steatosis/NR '
                    'mediated transport.'
                ),
                'name': 'LTEA_HepaRG_ABCB1_up',
                'type': 'cell-based',
            }
        ],
    }

    targetId: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Ensembl gene identifier of the target the safety event was reported on'
    )
    event: str = Field(min_length=1, description='Safety event associated with the target')
    eventId: str | None = Field(pattern=ONTOLOGY_ID, description='Unique identifier for the safety event')
    effects: list[Effects] | None = Field(min_length=1, description='Effects reported for the safety event')
    biosamples: list[Biosamples] | None = Field(min_length=1, description='Biosamples used in safety assessments')
    datasource: Literal[
        'AOP-Wiki',
        'Bowes et al. (2012)',
        'Brennan et al. (2024)',
        'ClinPGx',
        'Force et al. (2011)',
        'Lamore et al. (2017)',
        'Lynch et al. (2017)',
        'ToxCast',
        'Urban et al. (2012)',
    ] = Field(description='Data source reporting the safety event')
    literature: Literal['21283106', '23197038', '28216264', '28453775', '38773351'] | None = Field(
        description='Literature references for the safety event'
    )
    url: str | None = Field(min_length=1, description='URL linking to more details on the safety event')
    studies: list[Studies] = Field(min_length=1, description='Studies related to safety assessments')
