"""Gene burden evidence: the `evidence_gene_burden` output.

Generated from the 26.12 release output (55,611 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from ot_datamodels import ClickHouseType, DatasetModel, PrimaryKey, Reference, StructModel
from ot_datamodels.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX


class Urls(StructModel):
    """`urls[]`."""

    niceName: Literal[
        'AstraZeneca PheWAS Portal', 'BipEx browser', 'Broad CVDI Human Disease Portal', 'SCHEMA browser'
    ] = Field(description='Human readable label of the remote reference')
    url: str = Field(min_length=1, description='URL to remote resource')


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
        'targetId': 'ENSG00000213390',
        'id': '2208f7b3ab51b9ee9b8c86cab834bcc9eebda633',
        'targetFromSourceId': 'ARHGAP19',
        'diseaseFromSourceMappedId': 'MONDO_0007018',
        'datasourceId': 'gene_burden',
        'datatypeId': 'genetic_association',
        'allelicRequirements': ['dominant'],
        'diseaseFromSource': '41202#N762#Acute vulvitis',
        'pValueMantissa': 5.657,
        'pValueExponent': -8,
        'beta': 1.032819151878357,
        'betaConfidenceIntervalLower': 0.699565052986145,
        'betaConfidenceIntervalUpper': 1.3660731315612793,
        'oddsRatio': 55.94160079956055,
        'oddsRatioConfidenceIntervalLower': 22.432199478149414,
        'oddsRatioConfidenceIntervalUpper': 139.5072021484375,
        'resourceScore': 5.657e-08,
        'ancestry': 'EUR',
        'ancestryId': 'HANCESTRO_0005',
        'literature': ['34375979'],
        'projectId': 'AstraZeneca PheWAS Portal',
        'cohortId': 'UK Biobank 470k',
        'releaseVersion': 'ba08a93f-501e-44e6-a332-98ce2f852279',
        'studySampleSize': 158767,
        'studyCases': 159,
        'studyCasesWithQualifyingVariants': 5,
        'statisticalMethod': 'ptv',
        'statisticalMethodOverview': 'Burden test carried out with PTVs with a MAF smaller than 0.1%.',
        'urls': [
            {
                'niceName': 'AstraZeneca PheWAS Portal',
                'url': (
                    'https://azphewas.com/phenotypeView/ba08a93f-501e-44e6-a332-98ce2f852279/cc28e264-b223-43ca-aafc-79abc2592483/'
                ),
            }
        ],
        'sex': ['women'],
        'diseaseFromSourceId': '3581',
        'qualityControls': [],
        'diseaseId': 'MONDO_0007018',
        'publicationDate': '2021-08-10',
        'evidenceDate': '2021-08-10',
        'score': 0.2685560365944693,
        'directionOnTrait': 'risk',
        'directionOnTarget': 'LoF',
    }

    targetId: Annotated[str, Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        min_length=1,
        description=('Ensembl gene identifier or gene symbol of the target as reported by the burden study'),
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    datasourceId: str = Field(pattern=ONTOLOGY_ID, description='Identifer of the evidence source')
    datatypeId: str = Field(pattern=ONTOLOGY_ID, description='Type of the evidence')
    allelicRequirements: list[Annotated[str, Field(min_length=1)]] | None = Field(
        description="Genetic model of the burden test (e.g. 'dominant', 'recessive')"
    )
    diseaseFromSource: str = Field(min_length=1, description='Trait or phenotype label as reported by the burden study')
    pValueMantissa: float = Field(ge=0.0, description='Mantissa of the p-value of the burden test')
    pValueExponent: Annotated[int, ClickHouseType('Int32')] = Field(
        description='Exponent of the p-value of the burden test'
    )
    beta: float | None = Field(description='Effect size of the gene burden on a quantitative trait')
    betaConfidenceIntervalLower: float | None = Field(
        description='Lower value of the confidence interval of the effect size'
    )
    betaConfidenceIntervalUpper: float | None = Field(
        description='Upper value of the confidence interval of the effect size'
    )
    oddsRatio: float | None = Field(
        ge=0.0, description=('Effect size of the gene burden on a binary trait captured as odds ratio')
    )
    oddsRatioConfidenceIntervalLower: float | None = Field(
        ge=0.0, description='Lower value of the confidence interval for odds ratio'
    )
    oddsRatioConfidenceIntervalUpper: float | None = Field(
        ge=0.0, description='Upper value of the confidence interval for odds ratio'
    )
    resourceScore: float = Field(ge=0.0, le=1.0, description='P-value of the burden test')
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
            'Middle Eastern',
            'Pakistani and Bangladeshi',
            'SAS',
            'finnish',
            'non-European',
        ]
        | None
    ) = Field(description='Ancestry of the population included in the burden analysis')
    ancestryId: str | None = Field(
        pattern=ONTOLOGY_ID,
        description=(
            'Identifier of the ancestry of the population included in the burden analysis in the HANCESTRO ontology'
        ),
    )
    literature: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        description='PubMed identifiers of the publications describing the burden study'
    )
    projectId: str | None = Field(
        min_length=1,
        description=(
            'Project or consortium that generated the burden results (e.g. '
            "'Genebass', 'FinnGen', 'Epi25 collaborative')"
        ),
    )
    cohortId: str | None = Field(
        min_length=1,
        description=("Cohort(s) the burden analysis was run on (e.g. 'UK Biobank 450k', 'Genes & Health')"),
    )
    releaseVersion: str | None = Field(
        min_length=1, description="Release version of the source data (e.g. FinnGen 'R12')"
    )
    studySampleSize: Annotated[int, ClickHouseType('UInt32')] = Field(
        ge=0, description='Total number of samples in the burden analysis'
    )
    studyCases: Annotated[int, ClickHouseType('UInt32')] | None = Field(
        ge=0, description='Number of cases in the burden analysis of a binary trait'
    )
    studyCasesWithQualifyingVariants: Annotated[int, ClickHouseType('UInt32')] | None = Field(
        ge=0, description='Number of cases carrying at least one qualifying variant in the target'
    )
    statisticalMethod: str = Field(
        min_length=1,
        description=(
            'Identifier of the burden test model, typically encoding the test, the '
            'variant mask and the allele frequency threshold (e.g. '
            "'ADD-SKATO_D.0.001')"
        ),
    )
    statisticalMethodOverview: str = Field(
        min_length=1,
        description=('Description of the burden test, the variant mask and the allele frequency threshold'),
    )
    urls: list[Urls] | None = Field(
        description=("Links to the burden result in the source portal (e.g. 'AstraZeneca PheWAS Portal')")
    )
    sex: list[Annotated[str, Field(min_length=1)]] | None = Field(
        description='Sex of the samples included in the burden analysis'
    )
    diseaseFromSourceId: str | None = Field(
        min_length=1,
        description=(
            'Trait or phenotype code as reported by the burden study (e.g. FinnGen endpoint, UK Biobank field)'
        ),
    )
    qualityControls: list[str] = Field(max_length=0, description='Evidence quality flags')
    diseaseId: Annotated[str, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    publicationDate: str | None = Field(
        pattern=ISO_DATE, description='Date of the earliest publication supporting the evidence'
    )
    evidenceDate: str | None = Field(pattern=ISO_DATE, description='Earliest data for the evidence')
    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            'Score of the evidence reflecting the strength of the disease/target '
            'relationship, derived from the log10 of the burden test p-value'
        ),
    )
    directionOnTrait: Literal['risk', 'protect'] | None = Field(
        description='Predicted direction of effect on the trait'
    )
    directionOnTarget: Literal['GoF', 'LoF'] = Field(
        description=(
            'Gain or loss of function effect of the evidence on the target '
            'resulting from genetic variants, pharmacological modulation, or other '
            'perturbations'
        )
    )
