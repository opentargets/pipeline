"""GWAS/QTL Study: the `study` output.

Generated from the 26.09 release output (3,905,750 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PolarsDtype, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID, ONTOLOGY_ID


class LdPopulationStructure(StructModel):
    """`ldPopulationStructure[]`."""

    ldPopulation: Literal['afr', 'amr', 'eas', 'fin', 'nfe'] = Field(description='Population identifier')
    relativeSampleSize: float | None = Field(
        ge=0.0, le=1.0, description='Fraction of the total sample represented by the population'
    )


class DiscoverySamples(StructModel):
    """`discoverySamples[]`."""

    sampleSize: Annotated[int, PolarsDtype(pl.Int32)] = Field(ge=0, le=5519491, description='Discovery sample size')
    ancestry: str = Field(min_length=1, description='Discovery sample ancestry name')


class ReplicationSamples(StructModel):
    """`replicationSamples[]`."""

    sampleSize: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, le=3322939, description='Replication sample size'
    )
    ancestry: str = Field(min_length=1, description='Replication sample ancestry name')


class SumstatQCValues(StructModel):
    """`sumstatQCValues[]`."""

    QCCheckName: Literal[
        'LDSC_SNP-h2',
        'LDSC_SNP-h2_se',
        'LDSC_gc_lambda',
        'LDSC_intercept',
        'LDSC_intercept_se',
        'LDSC_mean_chisq',
        'gc_lambda',
        'mean_beta',
        'mean_diff_pz',
        'n_variants',
        'n_variants_sig',
        'se_diff_pz',
    ] = Field(description='Quality control metric identifier')
    QCCheckValue: Annotated[float, PolarsDtype(pl.Float32)] = Field(
        ge=-420.74603271484375, le=84398360.0, description='Quality control metric value'
    )


class Study(DatasetModel):
    """GWAS/QTL Study."""

    dataset_name: ClassVar[str] = 'study'
    dataset_title: ClassVar[str] = 'GWAS/QTL Study'
    dataset_description: ClassVar[str] = (
        'Metadata for all complex trait GWAS and molecular QTL studies in the '
        'Platform. The dataset includes study metadata, phenotype information, '
        'sample sizes, publication information and more. Molecular QTL studies '
        'are split by their target trait (e.g. gene, splice junction, etc), '
        'biosample (tissue, cell type or cell line) and condition (e.g. '
        'stimulation, time period, etc), potentially leading to tens of '
        'thousands of studies derived from the same publication'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Genetics',)
    example: ClassVar[dict] = {
        'studyId': 'FINNGEN_R12_AUTOIMMUNE_HYPERTHYROIDISM',
        'geneId': 'ENSG00000006282',
        'projectId': 'FINNGEN_R12',
        'studyType': 'gwas',
        'traitFromSource': 'Autoimmune hyperthyroidism',
        'traitFromSourceMappedIds': ['EFO_0004237'],
        'biosampleFromSourceId': 'CL_0000129',
        'pubmedId': '36653562',
        'publicationTitle': (
            'Genome-wide association studies in an isolated founder population from the Pacific Island of Kosrae.'
        ),
        'publicationFirstAuthor': 'Lowe JK',
        'publicationDate': '2024-11-01',
        'publicationJournal': 'PLoS Genet',
        'backgroundTraitFromSourceMappedIds': ['MONDO_0004979'],
        'initialSampleSize': '500,348 (282,064 females and 218,284 males)',
        'nCases': 2469,
        'nControls': 370637,
        'nSamples': 373106,
        'cohorts': ['FinnGen'],
        'ldPopulationStructure': [{'ldPopulation': 'fin', 'relativeSampleSize': 1.0}],
        'discoverySamples': [{'sampleSize': 500348, 'ancestry': 'Finnish'}],
        'replicationSamples': [{'sampleSize': 979, 'ancestry': 'European'}],
        'qualityControls': [('Study has non empty or zero nCases, nControls & nSamples with matching sum')],
        'analysisFlags': ['Metabolite'],
        'summarystatsLocation': (
            'gs://finngen-public-data-r12/summary_stats/finngen_R12_AUTOIMMUNE_HYPERTHYROIDISM.gz'
        ),
        'hasSumstats': True,
        'condition': 'IFN',
        'sumstatQCValues': [{'QCCheckName': 'mean_beta', 'QCCheckValue': -0.0007608377491123974}],
        'diseaseIds': ['MONDO_0005364'],
        'backgroundDiseaseIds': ['MONDO_0004979'],
        'biosampleId': 'CL_0000129',
    }

    studyId: Annotated[str, PrimaryKey()] = Field(
        min_length=1, description='Unique identifier of the GWAS or molQTL study.'
    )
    geneId: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id')] | None = Field(
        pattern=ENSEMBL_GENE_ID,
        description=(
            'Ensembl identifier of the gene representing a specific gene whose '
            'molecular is being analysed in molQTL study'
        ),
    )
    projectId: str = Field(
        min_length=1,
        description=('Identifier of the source project collection that the study information is derived from'),
    )
    studyType: Literal['eqtl', 'gwas', 'pqtl', 'sceqtl', 'sqtl', 'tuqtl'] = Field(
        description=('Field specifying if study contains phenotype/disease or molecular genetic associations')
    )
    traitFromSource: str = Field(
        min_length=1, description=('Molecular or phenotypic trait, derived from source, analysed in the study')
    )
    traitFromSourceMappedIds: list[Annotated[str, Field(min_length=1)]] | None = Field(
        description='Phenotypic trait IDs that map to the analysed trait reported by study'
    )
    biosampleFromSourceId: str | None = Field(
        pattern=ONTOLOGY_ID, description='Identifier of the referenced biological material'
    )
    pubmedId: Annotated[str, Bioregistry('pubmed'), Reference('literature_entity_lut', 'pmid')] | None = Field(
        min_length=1, description='PubMed identifier of the publication that references the study'
    )
    publicationTitle: str | None = Field(min_length=1, description='Title of the publication that references the study')
    publicationFirstAuthor: str | None = Field(
        min_length=1, description=('Last name and initials of the author of the publication that references the study')
    )
    publicationDate: str | None = Field(min_length=1, description='Date of the publication that references the study')
    publicationJournal: str | None = Field(
        min_length=1, description=('Abbreviated journal name where the publication referencing study was published')
    )
    backgroundTraitFromSourceMappedIds: list[Annotated[str, Field(pattern=ONTOLOGY_ID)]] | None = Field(
        description='Any background trait(s) shared by all individuals in the study'
    )
    initialSampleSize: str | None = Field(min_length=1, description='Study initial sample size')
    nCases: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, le=906935, description='The number of cases in this broad ancestry group'
    )
    nControls: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, le=14013085, description='The number of controls in this broad ancestry group'
    )
    nSamples: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=6, le=16938258, description='The number of samples tested in GWAS analysis'
    )
    cohorts: list[str] | None = Field(description='List of cohort(s) represented in the discovery sample')
    ldPopulationStructure: list[LdPopulationStructure] | None = Field(
        description='Collection of populations referenced by the study'
    )
    discoverySamples: list[DiscoverySamples] | None = Field(
        description='Collection of ancestries reported by the study discovery phase'
    )
    replicationSamples: list[ReplicationSamples] | None = Field(
        description='Collection of ancestries reported by the study replication phase'
    )
    qualityControls: list[
        Literal[
            'Harmonized summary statistics are not available or empty',
            'Sample size is null or equal to zero',
            ('Study has non empty or zero nCases, nControls & nSamples with matching sum'),
            ('Study has non empty or zero nCases, nControls & nSamples with non matching sum'),
            'Study has non empty or zero nSamples with empty nCases and nControls',
            'Study has only cases or only controls',
            'The number of SNPs in the study is below the expected threshold',
        ]
    ] = Field(min_length=1, description='Control metrics refining study validation')
    analysisFlags: list[
        Literal[
            'Case-case study, wgsGWAS',
            'ExWAS',
            'GxE',
            'GxE, Case-case study',
            'GxG',
            'Metabolite',
            'Multivariate analysis',
            'Non-additive model',
            'metabolite, wgsGWAS',
            'multivariate',
            'wgsGWAS',
        ]
    ] = Field(
        description=('Collection of flags indicating the type of the analysis conducted in the association study')
    )
    summarystatsLocation: str | None = Field(
        min_length=1, description='Path to the source study summary statistics (if exists at the source)'
    )
    hasSumstats: bool | None = Field(description='Indication whether the summary statistics exist in the source')
    condition: str | None = Field(min_length=1, description='Reported sample conditions')
    sumstatQCValues: list[SumstatQCValues] | None = Field(min_length=1, description='Mapping of quality control flags')
    diseaseIds: list[Annotated[Annotated[str, Reference('disease', 'id')], Field(pattern=ONTOLOGY_ID)]] = Field(
        description='Disease identifier associated with a studied trait'
    )
    backgroundDiseaseIds: list[Annotated[str, Field(pattern=ONTOLOGY_ID)]] = Field(
        description='Any background disease(s) shared by all individuals in the study'
    )
    biosampleId: Annotated[str, Bioregistry('uberon'), Reference('biosample', 'biosampleId')] | None = Field(
        pattern=ONTOLOGY_ID, description='Biosample identifier that maps to the biosample reported by study'
    )
