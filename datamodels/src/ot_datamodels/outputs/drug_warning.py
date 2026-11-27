"""Drug - withdrawn and black box warnings: the `drug_warning` output.

Generated from the 26.12 release output (2,009 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar

import polars as pl
from pydantic import Field

from ot_datamodels import DatasetModel, PolarsDtype, PrimaryKey, Reference, StructModel
from ot_datamodels.outputs.patterns import CHEMBL_ID
from ot_datamodels.vocabularies import ChemblWarningType


class References(StructModel):
    """`references[]`."""

    id: str = Field(min_length=1, description='List of identifiers for the reference sources')
    source: str = Field(min_length=1, description='List of bodies reporting the warning')
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
        description='List of Open Targets molecule identifiers'
    )
    toxicityClass: str | None = Field(
        min_length=1, description='Classification of toxicity type associated with the drug'
    )
    country: str = Field(min_length=1, description='Country where the warning was issued')
    description: str | None = Field(min_length=1, description='Description of the drug adverse effect')
    id: Annotated[int, PrimaryKey()] = Field(
        ge=0, le=4294967295, description='Internal identifier for the drug warning record'
    )
    references: list[References] = Field(description='List of sources supporting the warning information')
    warningType: Annotated[ChemblWarningType, PolarsDtype(pl.Enum)] = Field(
        description='Classification of action taken (drug is withdrawn or has a black box warning)'
    )
    year: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, le=65535, description='Year when the warning was issued'
    )
    efoTerm: str | None = Field(min_length=1, description='List of disease labels')
    efoId: Annotated[str, Reference('disease', 'id')] | None = Field(
        min_length=1, description='List of disease identifiers'
    )
    efoIdForWarningClass: Annotated[str, Reference('disease', 'id')] | None = Field(
        min_length=1, description='Disease identifier categorising the type of warning'
    )
