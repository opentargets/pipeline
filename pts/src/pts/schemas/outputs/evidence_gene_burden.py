"""Gene burden evidence: the `evidence_gene_burden` output.

Generated from the 26.09 release output (55,016 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID, ONTOLOGY_ID, SHA1_HEX


class Urls(StructModel):
    """`urls[]`."""

    niceName: Literal['AstraZeneca PheWAS Portal', 'Broad CVDI Human Disease Portal'] = Field(
        description='Human readable label of the remote reference'
    )
    url: str | None = Field(min_length=1, description='URL to remote resource')


class EvidenceGeneBurden(DatasetModel):
    """Gene burden evidence."""

    dataset_name: ClassVar[str] = 'evidence_gene_burden'
    dataset_title: ClassVar[str] = 'Gene burden evidence'
    dataset_description: ClassVar[str] = (
        'Disease-target evidence based on testing aggregated effect of '
        'variants, curated from the literature by the Open Targets team'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000118137',
        'id': '0bdbe6c6679238e3e2663beaeba8873d1ad1bf88',
        'targetFromSourceId': 'APOA1',
        'diseaseFromSourceMappedId': 'EFO_0022282',
        'datasourceId': 'gene_burden',
        'datatypeId': 'genetic_association',
        'allelicRequirements': ['dominant'],
        'diseaseFromSource': '23642#Free Cholesterol to Total Lipids in Medium HDL percentage',
        'pValueMantissa': 1.677,
        'pValueExponent': -15,
        'beta': -0.37999990582466125,
        'betaConfidenceIntervalLower': -0.4735148549079895,
        'betaConfidenceIntervalUpper': -0.286484956741333,
        'oddsRatio': 9.985600471496582,
        'oddsRatioConfidenceIntervalLower': 6.044000148773193,
        'oddsRatioConfidenceIntervalUpper': 16.497699737548828,
        'resourceScore': 1.67672106996612e-15,
        'ancestry': 'EUR',
        'ancestryId': 'HANCESTRO_0005',
        'literature': ['34375979'],
        'projectId': 'AstraZeneca PheWAS Portal',
        'cohortId': 'UK Biobank 470k',
        'releaseVersion': 'ba08a93f-501e-44e6-a332-98ce2f852279',
        'studySampleSize': 101137,
        'studyCases': 101137,
        'studyCasesWithQualifyingVariants': 353,
        'statisticalMethod': 'flexdmg',
        'statisticalMethodOverview': ('Burden test carried out with damaging variants with a MAF smaller than 0.01%.'),
        'urls': [
            {
                'niceName': 'AstraZeneca PheWAS Portal',
                'url': (
                    'https://azphewas.com/phenotypeView/ba08a93f-501e-44e6-a332-98ce2f852279/bf7e678f-5f8c-45e1-aa29-475ea8a742da/'
                ),
            }
        ],
        'sex': ['women'],
        'diseaseFromSourceId': '30280',
        'qualityControls': [],
        'diseaseId': 'EFO_0022282',
        'publicationDate': '2021-08-10',
        'evidenceDate': '2021-08-10',
        'score': 0.8331654383658091,
        'directionOnTrait': 'protect',
        'directionOnTarget': 'LoF',
    }

    targetId: Annotated[str, Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        min_length=1,
        description=(
            'Target ID in resource of origin (accepted sources include Ensembl '
            'gene ID, Uniprot ID, gene symbol), only capital letters are accepted'
        ),
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    datasourceId: Literal['gene_burden'] = Field(description='Identifer of the evidence source')
    datatypeId: Literal['genetic_association'] = Field(description='Type of the evidence')
    allelicRequirements: (
        list[
            Literal[
                'Autosomal dominant inheritance',
                'Autosomal recessive inheritance',
                'Dominant/multiple',
                'X-linked dominant/multiple',
                'dominant',
                'recessive',
            ]
        ]
        | None
    ) = Field(min_length=1, description='Inheritance patterns')
    diseaseFromSource: str = Field(min_length=1, description='Disease label from the original source')
    pValueMantissa: float = Field(ge=1.0, le=10.0, description='Mantissa of the p-value')
    pValueExponent: int = Field(ge=-320, le=-3, description='Exponent of the p-value')
    beta: float | None = Field(ge=-5.029595851898193, le=5.30117, description='Effect size of numeric traits')
    betaConfidenceIntervalLower: float | None = Field(
        ge=-6.695067882537842, le=4.970186999999999, description='Lower value of the confidence interval'
    )
    betaConfidenceIntervalUpper: float | None = Field(
        ge=-3.5512733459472656, le=6.64535665512085, description='Upper value of the confidence interval'
    )
    oddsRatio: float | None = Field(ge=0.0, le=23838.220703125, description='Size of effect captured as odds ratio')
    oddsRatioConfidenceIntervalLower: float | None = Field(
        ge=0.012400000356137753,
        le=10453.8115234375,
        description='Lower value of the confidence interval for odds ratio',
    )
    oddsRatioConfidenceIntervalUpper: float | None = Field(
        ge=0.22849999368190765, le=76310.421875, description='Upper value of the confidence interval for odds ratio'
    )
    resourceScore: float = Field(
        ge=5.498e-320,
        le=0.00251,
        description=('Score provided by datasource indicating strength of target-disease association'),
    )
    ancestry: (
        Literal[
            'AFR',
            'Admixed American',
            'African',
            'Central and South Asian',
            'EAS',
            'EUR',
            'East Asian',
            'European',
            'Pakistani and Bangladeshi',
            'SAS',
            'finnish',
            'non-European',
        ]
        | None
    ) = Field(description='Genetic origin of a population')
    ancestryId: (
        Annotated[
            Literal[
                'HANCESTRO_0005',
                'HANCESTRO_0006',
                'HANCESTRO_0009',
                'HANCESTRO_0010',
                'HANCESTRO_0014',
                'HANCESTRO_0321',
            ],
            Bioregistry('hancestro'),
        ]
        | None
    ) = Field(description='Identifier of the ancestry in the HANCESTRO ontology')
    literature: list[
        Literal[
            '31085678',
            '31327507',
            '31981491',
            '32150548',
            '34375979',
            '34662886',
            '35396579',
            '35568032',
            '35982159',
            '35999217',
            '36653562',
            '36778668',
            '37348876',
            '38052806',
            '39210047',
            '41896352',
            '42238450',
        ]
    ] = Field(min_length=1, description='List of PubMed or preprint reference identifiers')
    projectId: (
        Literal[
            'AMP-PD',
            'AstraZeneca PheWAS Portal',
            'Autism Sequencing Consortium',
            'BRaVa Consortium',
            'CVDI Human Disease Portal',
            'Epi25 collaborative',
            'FinnGen',
            'Genebass',
            'Genes & Health',
            'OTAR022',
            'REGENERON',
            'SCHEMA consortium',
        ]
        | None
    ) = Field(description='The identifer of the project that generated the data')
    cohortId: (
        Literal[
            'BRaVa Consortium',
            'CUMC, AURORA',
            'FinnGen R12',
            'Genes & Health',
            'Genes & Health + UK Biobank',
            'INTERVAL',
            'PAGES, DBS',
            'SPARK, ASC, MSSNG, SSC',
            'UK Biobank 200k',
            'UK Biobank 450k',
            'UK Biobank 450k/All of Us/MGB',
            'UK Biobank 470k',
            'UK Biobank, MDCS,  MCPS',
            'gnomAD and SCHEMA consortia',
        ]
        | None
    ) = Field(description='Identifier of the studied cohort')
    releaseVersion: Literal['R12', 'ba08a93f-501e-44e6-a332-98ce2f852279'] | None = Field(
        description='Open Targets data release version'
    )
    studySampleSize: int = Field(ge=798, le=1029916, description='Sample size of study')
    studyCases: int | None = Field(ge=30, le=418636, description='Number of cases in case-control study')
    studyCasesWithQualifyingVariants: int | None = Field(
        ge=0,
        le=59487,
        description=('Number of cases in case-control study that carry at least one allele of the qualifying variant'),
    )
    statisticalMethod: str = Field(min_length=1, description='Statistical method used to calculate the association')
    statisticalMethodOverview: str = Field(
        min_length=1, description='Overview of the statistical method used to calculate the association'
    )
    urls: list[Urls] | None = Field(
        min_length=1,
        description=(
            'Reference to linked external resource (e.g. clinical trials, studies, package inserts, reports, etc.)'
        ),
    )
    sex: list[Literal['men', 'women']] | None = Field(
        min_length=1, description='Sex of the samples/population included in the analysis'
    )
    diseaseFromSourceId: str | None = Field(min_length=1, description='Disease identifier from the original source')
    qualityControls: list[str] = Field(max_length=0, description='Evidence quality flags')
    diseaseId: Annotated[str, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    publicationDate: Literal[
        '2019-05-13',
        '2019-07-18',
        '2020-01-23',
        '2020-03-09',
        '2021-08-10',
        '2021-10-18',
        '2022-04-08',
        '2022-05-13',
        '2022-08-15',
        '2022-08-18',
        '2022-08-23',
        '2023-01-18',
        '2023-11-01',
        '2023-12-05',
        '2024-08-29',
        '2026-03-27',
        '2026-05-24',
    ] = Field(description='Date of the earliest publication supporting the evidence')
    evidenceDate: Literal[
        '2019-05-13',
        '2019-07-18',
        '2020-01-23',
        '2020-03-09',
        '2021-08-10',
        '2021-10-18',
        '2022-04-08',
        '2022-05-13',
        '2022-08-15',
        '2022-08-18',
        '2022-08-23',
        '2023-01-18',
        '2023-11-01',
        '2023-12-05',
        '2024-08-29',
        '2026-03-27',
        '2026-05-24',
    ] = Field(description='Earliest data for the evidence')
    score: float = Field(
        ge=0.25,
        le=1.0,
        description=('Score of the evidence reflecting the strength of the disease/target relationship'),
    )
    directionOnTrait: Literal['protect', 'risk'] | None = Field(
        description='Predicted direction of effect on the trait'
    )
    directionOnTarget: Literal['LoF'] = Field(
        description=(
            'Gain or loss of function effect of the evidence on the target '
            'resulting from genetic variants, pharmacological modulation, or other '
            'perturbations'
        )
    )
