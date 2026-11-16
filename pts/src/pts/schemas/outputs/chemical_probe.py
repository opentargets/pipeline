"""Target - chemical probes: the `chemical_probe` output.

Generated from the 26.12 release output (1,265 rows) and its croissant metadata by
`scripts/output_schemas/generate.py`, whose docstring lists which observations become rules.
"""

from typing import Annotated, ClassVar

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import CHEMBL_ID, ENSEMBL_GENE_ID


class Urls(StructModel):
    """`urls[]`."""

    niceName: str = Field(min_length=1, description='Nice name for the linked URL')
    url: str | None = Field(min_length=1, description='URL providing details about the chemical probe')


class ChemicalProbe(DatasetModel):
    """Target - chemical probes."""

    dataset_name: ClassVar[str] = 'chemical_probe'
    dataset_title: ClassVar[str] = 'Target - chemical probes'
    dataset_description: ClassVar[str] = (
        'Chemical probes and their molecular characteristics, resolved to Ensembl target identifiers'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target',)
    example: ClassVar[dict] = {
        'targetFromSourceId': 'B2RXH2',
        'drugFromSource': 'ML001',
        'drugFromSourceId': 'PD016117',
        'drugId': 'CHEMBL1612647',
        'mechanismOfAction': ['inhibitor'],
        'origin': ['experimental'],
        'control': 'cisSGK3-PROTAC1',
        'isHighQuality': False,
        'probesDrugsScore': 10.0,
        'scoreInCells': 50.0,
        'scoreInOrganisms': 0.0,
        'urls': [{'niceName': 'MLP Probes', 'url': 'http://www.ncbi.nlm.nih.gov/books/NBK47352/'}],
        'targetId': 'ENSG00000235268',
    }

    targetFromSourceId: str = Field(
        min_length=1,
        description=(
            'Gene symbol or protein accession identifying the target as reported '
            'by the probe source, prior to ENSG resolution'
        ),
    )
    drugFromSource: str = Field(min_length=1, description='Compound name of the chemical probe')
    drugFromSourceId: Annotated[str, PrimaryKey()] = Field(
        min_length=1, description=('Probe identifier from the source, used to cross-reference drug information')
    )
    drugId: Annotated[str, Reference('drug_molecule', 'id')] | None = Field(
        pattern=CHEMBL_ID,
        description=('Open Targets molecule identifier of the drug associated with the chemical probe'),
    )
    mechanismOfAction: list[Annotated[str, Field(min_length=1)]] | None = Field(
        description='Mechanism of action of the chemical probe'
    )
    origin: list[Annotated[str, Field(min_length=1)]] = Field(
        description='Origin of the chemical probe (e.g. experimental, calculated)'
    )
    control: str | None = Field(min_length=1, description='Whether the chemical probe serves as a control')
    isHighQuality: bool = Field(description='Indicates if the chemical probe is high quality')
    probesDrugsScore: float = Field(ge=0.0, description='Score for chemical probes related to druggability')
    scoreInCells: float | None = Field(ge=0.0, description='Score indicating chemical probe activity in cells')
    scoreInOrganisms: float | None = Field(ge=0.0, description='Score indicating chemical probe activity in organisms')
    urls: list[Urls] = Field(description='URLs linking to more information about the chemical probe')
    targetId: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id'), PrimaryKey()] = Field(
        pattern=ENSEMBL_GENE_ID,
        description=('Ensembl gene identifier of the target, resolved from targetFromSourceId via output/target'),
    )
