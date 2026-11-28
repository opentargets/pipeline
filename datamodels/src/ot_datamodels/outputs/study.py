"""GWAS/QTL Study: the `study` output.

Generated from the 26.12 release output (3,969,781 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from ot_datamodels import Bioregistry, DatasetModel, PolarsDtype, PrimaryKey, Reference, StructModel
from ot_datamodels.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID
from ot_datamodels.vocabularies import LdPopulation, StudyType


class LdPopulationStructure(StructModel):
    """`ldPopulationStructure[]`."""

    ldPopulation: Annotated[LdPopulation, PolarsDtype(pl.Enum)] = Field(description='Population identifier')
    relativeSampleSize: float | None = Field(
        ge=0.0, le=1.0, description='Fraction of the total sample represented by the population'
    )


class DiscoverySamples(StructModel):
    """`discoverySamples[]`."""

    sampleSize: Annotated[int, PolarsDtype(pl.Int32)] = Field(ge=0, description='Discovery sample size')
    ancestry: str = Field(min_length=1, description='Discovery sample ancestry name')


class ReplicationSamples(StructModel):
    """`replicationSamples[]`."""

    sampleSize: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(ge=0, description='Replication sample size')
    ancestry: str = Field(min_length=1, description='Replication sample ancestry name')


class SumstatQCValues(StructModel):
    """`sumstatQCValues[]`."""

    QCCheckName: str = Field(min_length=1, description='Quality control metric identifier')
    QCCheckValue: Annotated[float, PolarsDtype(pl.Float32)] = Field(description='Quality control metric value')


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
        'studyId': 'GCST90201095',
        'projectId': 'GCST',
        'studyType': 'gwas',
        'traitFromSource': 'Isobutyrylcarnitine (C4) levels',
        'traitFromSourceMappedIds': ['EFO_0021023'],
        'diseaseIds': ['EFO_0021023'],
        'geneId': 'ENSG00000198931',
        'biosampleFromSourceId': 'UBERON_0001969',
        'biosampleId': 'UBERON_0001969',
        'pubmedId': '36635386',
        'publicationTitle': (
            'Genomic atlas of the plasma metabolome prioritizes metabolites implicated in human diseases.'
        ),
        'publicationFirstAuthor': 'Chen Y',
        'publicationDate': '2023-01-12',
        'publicationJournal': 'Nat Genet',
        'backgroundTraitFromSourceMappedIds': ['MONDO_0005375'],
        'backgroundDiseaseIds': ['MONDO_0005375'],
        'initialSampleSize': '60 African ancestry individuals',
        'nCases': 0,
        'nControls': 0,
        'nSamples': 60,
        'cohorts': ['CLSA'],
        'ldPopulationStructure': [{'ldPopulation': 'afr', 'relativeSampleSize': 1.0}],
        'discoverySamples': [{'sampleSize': 60, 'ancestry': 'African unspecified'}],
        'replicationSamples': [{'sampleSize': 259, 'ancestry': 'African American or Afro-Caribbean'}],
        'qualityControls': ['Study has non empty or zero nSamples with empty nCases and nControls'],
        'analysisFlags': ['Metabolite'],
        'summarystatsLocation': (
            'gs://gentropy-vault/ukb-ppp/UKB-PPP pGWAS summary statistics/European '
            '(discovery)/APRT_P07741_OID20927_v1_Neurology.tar'
        ),
        'hasSumstats': True,
        'condition': 'naive',
        'sumstatQCValues': [{'QCCheckName': 'gc_lambda', 'QCCheckValue': 1.0019054412841797}],
    }

    studyId: Annotated[str, PrimaryKey()] = Field(
        min_length=1, description='Unique identifier of the GWAS or molQTL study.'
    )
    projectId: str = Field(
        min_length=1,
        description=('Identifier of the source project collection that the study information is derived from'),
    )
    studyType: Annotated[StudyType, PolarsDtype(pl.Enum)] = Field(
        description='Field specifying if study contains phenotype/disease or molecular genetic associations'
    )
    traitFromSource: str = Field(
        min_length=1, description=('Molecular or phenotypic trait, derived from source, analysed in the study')
    )
    traitFromSourceMappedIds: list[Annotated[str, Field(min_length=1)]] | None = Field(
        description='Phenotypic trait IDs that map to the analysed trait reported by study'
    )
    diseaseIds: list[Annotated[Annotated[str, Reference('disease', 'id')], Field(pattern=ONTOLOGY_ID)]] = Field(
        description='Disease identifier associated with a studied trait'
    )
    geneId: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id')] | None = Field(
        pattern=ENSEMBL_GENE_ID,
        description=(
            'Ensembl identifier of the gene representing a specific gene whose '
            'molecular is being analysed in molQTL study'
        ),
    )
    biosampleFromSourceId: str | None = Field(
        pattern=ONTOLOGY_ID, description='Identifier of the referenced biological material'
    )
    biosampleId: Annotated[str, Bioregistry('uberon'), Reference('biosample', 'biosampleId')] | None = Field(
        pattern=ONTOLOGY_ID, description='Biosample identifier that maps to the biosample reported by study'
    )
    pubmedId: Annotated[str, Bioregistry('pubmed'), Reference('literature_entity_lut', 'pmid')] | None = Field(
        pattern=DIGITS, description='PubMed identifier of the publication that references the study'
    )
    publicationTitle: str | None = Field(min_length=1, description='Title of the publication that references the study')
    publicationFirstAuthor: str | None = Field(
        min_length=1, description=('Last name and initials of the author of the publication that references the study')
    )
    publicationDate: str | None = Field(
        pattern=ISO_DATE, description='Date of the publication that references the study'
    )
    publicationJournal: str | None = Field(
        min_length=1, description=('Abbreviated journal name where the publication referencing study was published')
    )
    backgroundTraitFromSourceMappedIds: list[Annotated[str, Field(pattern=ONTOLOGY_ID)]] | None = Field(
        description='Any background trait(s) shared by all individuals in the study'
    )
    backgroundDiseaseIds: list[Annotated[str, Field(pattern=ONTOLOGY_ID)]] = Field(
        description='Any background disease(s) shared by all individuals in the study'
    )
    initialSampleSize: str | None = Field(min_length=1, description='Study initial sample size')
    nCases: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, description='The number of cases in this broad ancestry group'
    )
    nControls: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, description='The number of controls in this broad ancestry group'
    )
    nSamples: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, description='The number of samples tested in GWAS analysis'
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
            'Target/gene identifier could not match to reference',
            'No valid disease identifier found',
            'This type of study is not supported',
            'Biosample identifier was not found in the reference',
            'The identifier of this study is not unique',
            'Harmonized summary statistics are not available or empty',
            'GWAS Catalog study has not been curated by Open Targets',
            'The mean beta QC check value is not within the expected range',
            'The PZ QC check values are not within the expected range',
            'The GC lambda value is not within the expected range',
            'The number of SNPs in the study is below the expected threshold',
            'Case-case study design',
            'Deprecated project',
            'Sample size is null or equal to zero',
            'Study has non empty or zero nCases, nControls & nSamples with matching sum',
            'Study has non empty or zero nCases, nControls & nSamples with non matching sum',
            'Study has only cases or only controls',
            'Study has non empty or zero nSamples with empty nCases and nControls',
        ]
    ] = Field(description='Control metrics refining study validation')
    analysisFlags: list[Annotated[str, Field(min_length=1)]] = Field(
        description=('Collection of flags indicating the type of the analysis conducted in the association study')
    )
    summarystatsLocation: str | None = Field(
        min_length=1, description='Path to the source study summary statistics (if exists at the source)'
    )
    hasSumstats: bool | None = Field(description='Indication whether the summary statistics exist in the source')
    condition: str | None = Field(min_length=1, description='Reported sample conditions')
    sumstatQCValues: list[SumstatQCValues] | None = Field(description='Mapping of quality control flags')
