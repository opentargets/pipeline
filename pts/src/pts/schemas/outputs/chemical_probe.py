"""Target - chemical probes: the `chemical_probe` output.

Generated from the 26.09 release output (1,265 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import CHEMBL_ID, ENSEMBL_GENE_ID


class Urls(StructModel):
    """`urls[]`."""

    niceName: Literal[
        ('A Collection of Useful Nuisance Compounds (CONS) for Interrogation of Bioassay Integrity'),
        'Bromodomains chemical toolbox',
        'Chemical Probes for Understudied Kinases',
        'Chemical Probes.org',
        'Concise Guide to Pharmacology 2025/26',
        'Gray Laboratory Probes',
        'Kinase Chemogenomic Set (KCGS)',
        'Kinase Inhibitors (best-in-class)',
        'MLP Probes',
        'Nature Chemical Biology Probes',
        'Novartis Chemogenetic Library (NIBR MoA Box)',
        'Nuisance compounds in cellular assays',
        'Open Science Probes',
        'Protein methyltransferases chemical toolbox',
        'SGC Probes',
        'opnMe Portal',
    ] = Field(description='Nice name for the linked URL')
    url: (
        Literal[
            'http://www.chemicalprobes.org/',
            'http://www.nature.com/nchembio/chemical_probes.html',
            'http://www.ncbi.nlm.nih.gov/books/NBK47352/',
            'http://www.sgc-ffm.uni-frankfurt.de/#!start',
            'http://www.thesgc.org/chemical-probes',
            'https://graylab.stanford.edu/probe-resources/',
            'https://opnme.com/',
            'https://pubs.acs.org/doi/10.1021/acs.jmedchem.1c00980',
            'https://www.nature.com/articles/s41467-018-07905-4',
            'https://www.nature.com/articles/s41467-019-09672-2#article-info',
        ]
        | None
    ) = Field(description='URL providing details about the chemical probe')


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
    mechanismOfAction: (
        list[
            Literal[
                'activator',
                'agonist',
                'allosteric modulator',
                'antagonist',
                'biased agonist',
                'binding agent',
                'blocker',
                'competitive inhibitor',
                'degrader',
                'inhibitor',
                'inverse agonist',
                'modulator',
                'negative allosteric modulator',
                'partial agonist',
                'potentiator',
            ]
        ]
        | None
    ) = Field(min_length=1, description='Mechanism of action of the chemical probe')
    origin: list[Literal['calculated', 'experimental']] = Field(
        min_length=1, description='Origin of the chemical probe (e.g. experimental, calculated)'
    )
    control: str | None = Field(min_length=1, description='Whether the chemical probe serves as a control')
    isHighQuality: bool = Field(description='Indicates if the chemical probe is high quality')
    probesDrugsScore: float = Field(ge=0.0, le=100.0, description='Score for chemical probes related to druggability')
    scoreInCells: float | None = Field(
        ge=0.0, le=100.0, description='Score indicating chemical probe activity in cells'
    )
    scoreInOrganisms: float | None = Field(
        ge=0.0, le=100.0, description='Score indicating chemical probe activity in organisms'
    )
    urls: list[Urls] = Field(min_length=1, description='URLs linking to more information about the chemical probe')
    targetId: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id'), PrimaryKey()] = Field(
        pattern=ENSEMBL_GENE_ID,
        description=('Ensembl gene identifier of the target, resolved from targetFromSourceId via output/target'),
    )
