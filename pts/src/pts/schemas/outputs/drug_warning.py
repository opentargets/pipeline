"""Drug - withdrawn and black box warnings: the `drug_warning` output.

Generated from the 26.09 release output (2,009 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from pts.schemas.dataset import DatasetModel, PolarsDtype, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import CHEMBL_ID


class References(StructModel):
    """`references[]`."""

    id: str = Field(min_length=1, description='List of identifiers for the reference sources')
    source: Literal[
        'DOI',
        'DailyMed',
        'EMA',
        'FDA',
        'HIS',
        'HPRA',
        'Health Canada',
        'MEDSAFE',
        'MHRA',
        'NICE',
        'NTP',
        'Other',
        'PubMed',
        'TGA',
        'USGPO',
        'WHO',
    ] = Field(description='List of bodies reporting the warning')
    url: str = Field(min_length=1, description='List of web addresses referencing the information')


class DrugWarning(DatasetModel):
    """Drug - withdrawn and black box warnings."""

    dataset_name: ClassVar[str] = 'drug_warning'
    dataset_title: ClassVar[str] = 'Drug - withdrawn and black box warnings'
    dataset_description: ClassVar[str] = (
        'Blackbox and withdrawn information for drug molecules included in ChEMBL database'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Drug',)
    example: ClassVar[dict] = {
        'chemblIds': ['CHEMBL4303288'],
        'toxicityClass': 'hepatotoxicity',
        'country': 'United States',
        'description': (
            'FDA\u2019s determination that carbetapentane citrate has not been shown to '
            'be effective at the over-the-counter (OTC) doses stated in the '
            'exempting regulation.'
        ),
        'id': 1,
        'references': [
            {
                'id': 'de109a2b-e36c-40d0-85fc-a67a9e7f1ae8',
                'source': 'DailyMed',
                'url': (
                    'https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=de109a2b-e36c-40d0-85fc-a67a9e7f1ae8'
                ),
            }
        ],
        'warningType': 'Black Box Warning',
        'year': 2007,
        'efoTerm': 'lack of efficacy',
        'efoId': 'EFO:0020002',
        'efoIdForWarningClass': 'EFO:0011052',
    }

    chemblIds: list[Annotated[Annotated[str, Reference('drug_molecule', 'id')], Field(pattern=CHEMBL_ID)]] = Field(
        min_length=1, description='List of Open Targets molecule identifiers'
    )
    toxicityClass: (
        Literal[
            'carcinogenicity',
            'cardiotoxicity',
            'dermatological toxicity',
            'drug misuse',
            'gastrointestinal toxicity',
            'hematological toxicity',
            'hepatotoxicity',
            'immune system toxicity',
            'infectious disease',
            'metabolic toxicity',
            'misuse',
            'musculoskeletal toxicity',
            'nephrotoxicity',
            'neurotoxicity',
            'occular toxicity',
            'ototoxicity',
            'psychiatric toxicity',
            'respiratory toxicity',
            'teratogenicity',
            'vascular toxicity',
        ]
        | None
    ) = Field(description='Classification of toxicity type associated with the drug')
    country: str = Field(min_length=1, description='Country where the warning was issued')
    description: str | None = Field(min_length=1, description='Description of the drug adverse effect')
    id: Annotated[int, PrimaryKey()] = Field(
        ge=1, le=5221, description='Internal identifier for the drug warning record'
    )
    references: list[References] = Field(description='List of sources supporting the warning information')
    warningType: Literal['Black Box Warning', 'Withdrawn'] = Field(
        description=('Classification of action taken (drug is withdrawn or has a black box warning)')
    )
    year: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=1958, le=2024, description='Year when the warning was issued'
    )
    efoTerm: str | None = Field(min_length=1, description='List of disease labels')
    efoId: Annotated[str, Reference('disease', 'id')] | None = Field(
        min_length=1, description='List of disease identifiers'
    )
    efoIdForWarningClass: Annotated[str, Reference('disease', 'id')] | None = Field(
        min_length=1, description='Disease identifier categorising the type of warning'
    )
