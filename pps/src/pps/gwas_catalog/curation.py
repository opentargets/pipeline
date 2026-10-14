"""Module to handle the curation of GWAS Catalog data."""

from __future__ import annotations

from enum import StrEnum

import polars as pl
from google.cloud.storage import Client
from loguru import logger

from pps.errors import PpsError, PpsErrorMessage


class CurationSchema(StrEnum):
    """Enum to define the schema for curation tasks."""

    STUDY_ID = 'studyId'
    """The unique identifier for a study."""
    STUDY_TYPE = 'studyType'
    """The type of study (e.g., gwas)."""
    ANALYSIS_FLAG = 'analysisFlag'
    """Flag indicating the type of analysis performed."""
    QUALITY_CONTROL = 'qualityControl'
    """Quality control status of the study."""
    IS_CURATED = 'isCurated'
    """Flag indicating whether the study has been curated."""
    PUBMED_ID = 'pubmedId'
    """The PubMed identifier for the study."""
    PUBLICATION_TITLE = 'publicationTitle'
    """The title of the publication associated with the study."""
    TRAIT_FROM_SOURCE = 'traitFromSource'
    """The trait as reported in the source data."""

    @classmethod
    def columns(cls) -> list[str]:
        """Get the list of columns defined in the schema."""
        return [member.value for member in cls]

    @classmethod
    def extended_columns(cls) -> list[str]:
        """Get the list of columns defined in the schema, including additional metadata."""
        return [*cls.columns(), 'status']


class DownloadStudiesSchema(StrEnum):
    """Enum to define the columns for the download studies task."""

    STUDY_ID = 'studyId'
    """The unique identifier for a study."""
    TRAIT_FROM_SOURCE = 'traitFromSource'
    """The trait as reported in the source data."""
    PUBMED_ID = 'pubmedId'
    """The PubMed identifier for the study."""
    PUBLICATION_TITLE = 'publicationTitle'
    """The title of the publication associated with the study."""

    @classmethod
    def mapping(cls) -> dict[str, str]:
        """Get the mapping of columns to their respective names."""
        return {
            'STUDY ACCESSION': cls.STUDY_ID,
            'STUDY': cls.PUBLICATION_TITLE,
            'PUBMED ID': cls.PUBMED_ID,
            'DISEASE/TRAIT': cls.TRAIT_FROM_SOURCE,
        }

    @classmethod
    def columns(cls) -> list[str]:
        """Get the list of columns defined in the schema."""
        return [member.value for member in cls]


class SyncedSummaryStatisticsSchema(StrEnum):
    """Enum to define the columns for synced summary statistics."""

    FILE_PATH = 'filePath'
    """The GCS file path of the summary statistics file."""
    SYNCED = 'isSynced'
    """Flag indicating whether the file has been synced."""
    STUDY_ID = 'studyId'
    """The unique identifier for a study."""

    @classmethod
    def columns(cls) -> list[str]:
        """Get the list of columns defined in the schema."""
        return [member.value for member in cls]


class CuratedStudyStatus(StrEnum):
    """Enum to define the status of a curated study."""

    REMOVED = 'removed'
    """The study has been removed from the GWAS Catalog."""
    TO_CURATE = 'to_curate'
    """The study is new and needs to be curated."""
    CURATED = 'curated'
    """The study has been curated and is still in the GWAS Catalog."""
    NO_SUMSTATS = 'no_summary_statistics'
    """The study has no associated summary statistics."""


class GCSSummaryStatisticsFileCrawler:
    """Class to crawl GCS for summary statistics files."""

    def __init__(self, gcs_glob: str):
        """Initialize the GCSSummaryStatisticsFileCrawler with a GCS glob pattern."""
        self.gcs_glob = gcs_glob
        logger.debug('Initialized GCSSummaryStatisticsFileCrawler with glob: {}', gcs_glob)

    def _fetch_paths(self) -> list[str]:
        """Fetch file paths from GCS based on the glob pattern."""
        # Implementation to fetch file paths from GCS
        c = Client()
        bucket_name = self.gcs_glob.split('/')[2]
        prefix = '/'.join(self.gcs_glob.split('/')[3:-1])
        suffix = self.gcs_glob.split('/')[-1].replace('*', '')
        logger.debug('Crawling GCS bucket: {}, prefix: {}, suffix: {}', bucket_name, prefix, suffix)
        bucket = c.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        return [f'gs://{bucket_name}/{blob.name}' for blob in blobs if blob.name.endswith(suffix)]

    def crawl(self) -> pl.DataFrame:
        """Crawl GCS and return a DataFrame of summary statistics files."""
        # Implementation to crawl GCS and return a DataFrame
        file_paths = self._fetch_paths()
        logger.debug('Found {} summary statistics files.', len(file_paths))
        data = pl.DataFrame({
            SyncedSummaryStatisticsSchema.FILE_PATH: file_paths,
            SyncedSummaryStatisticsSchema.SYNCED: [True] * len(file_paths),
        }).with_columns(
            pl
            .col(SyncedSummaryStatisticsSchema.FILE_PATH)
            .str.extract(r'\/(GCST\d+)\/', 1)
            .alias(SyncedSummaryStatisticsSchema.STUDY_ID)
        )
        # Post check to find if there are any studies with multiple files.
        multi_files = data.group_by(SyncedSummaryStatisticsSchema.STUDY_ID).len().filter(pl.col('len') > 1)
        if not multi_files.is_empty():
            logger.warning('Studies with multiple summary statistics files found: {}', multi_files)
            logger.warning('DataFrame shape before deduplication: {}', data.shape)
            logger.warning('Synced data preview:\n{}', data.head())
            data = data.unique(subset=SyncedSummaryStatisticsSchema.STUDY_ID)
            logger.warning('Synced data after deduplication:\n{}', data.shape)
        return data


class GWASCatalogCuration:
    """Class to handle the curation of GWAS Catalog data."""

    def __init__(self, previous_curation: pl.DataFrame, studies: pl.DataFrame, synced: pl.DataFrame):
        """Initialize the GWASCatalogCuration with previous curation and studies data."""
        logger.debug('Initializing GWASCatalogCuration with previous curation and studies data.')
        self.previous_curation = previous_curation
        logger.debug('Previous curation data loaded with shape: {}', previous_curation.shape)
        self.studies = studies
        logger.debug('Studies data loaded with shape: {}', studies.shape)
        self.synced = synced
        logger.debug('Synced summary statistics data loaded with shape: {}', synced.shape)

    @classmethod
    def from_prev_curation(
        cls,
        previous_curation_path: str,
        download_studies_path: str,
        summary_statistics_glob: str,
    ) -> GWASCatalogCuration:
        """Create a GWASCatalogCuration instance from previous curation and studies."""
        crawled_summary_statistics = GCSSummaryStatisticsFileCrawler(summary_statistics_glob).crawl()

        previous_curation_df = pl.read_csv(
            previous_curation_path,
            separator='\t',
            has_header=True,
            columns=CurationSchema.columns(),
        )
        if previous_curation_df.is_empty():
            raise PpsError(PpsErrorMessage.PREVIOUS_CURATION_EMPTY, path=previous_curation_path)
        studies_df = pl.read_csv(
            download_studies_path,
            separator='\t',
            quote_char='`',
            has_header=True,
            columns=list(DownloadStudiesSchema.mapping().keys()),
        )
        if studies_df.is_empty():
            raise PpsError(PpsErrorMessage.DOWNLOAD_STUDIES_EMPTY, path=download_studies_path)
        studies_df = studies_df.rename(mapping=DownloadStudiesSchema.mapping())
        return cls(previous_curation_df, studies_df, crawled_summary_statistics)

    @property
    def result(self) -> pl.DataFrame:
        """Curate the GWAS Catalog data."""
        # Studies that are curated but were removed from the GWAS Catalog
        removed_studies = self.previous_curation.join(self.studies, on=CurationSchema.STUDY_ID, how='anti').select(
            CurationSchema.STUDY_ID, pl.lit(CuratedStudyStatus.REMOVED).alias('status')
        )
        logger.debug('Removed studies identified: {}', removed_studies.shape[0])

        # studies that are curated and still in the GWAS Catalog
        curated_studies = self.previous_curation.join(self.studies, on=CurationSchema.STUDY_ID, how='inner').select(
            CurationSchema.STUDY_ID, pl.lit(CuratedStudyStatus.CURATED).alias('status')
        )
        logger.debug('Curated studies identified: {}', curated_studies.shape[0])

        # Combine all previous studies with updated status information.
        prev_studies = pl.concat([removed_studies, curated_studies], how='vertical')
        logger.debug('Previous studies after combining removed and curated: {}', prev_studies.shape[0])

        # Bring back the information from the previous curation
        prev_studies = (
            prev_studies
            .join(self.previous_curation, on=CurationSchema.STUDY_ID, how='full', coalesce=True)
            .with_columns(pl.coalesce(CurationSchema.IS_CURATED, pl.lit(False)).alias(CurationSchema.IS_CURATED))
            .select(*CurationSchema.extended_columns())
        )
        logger.debug('Previous studies after bringing previous curation data: {}', prev_studies.shape[0])

        assert all(prev_studies.select(CurationSchema.STUDY_ID).is_unique()), 'Study IDs must be unique after merging.'

        # Studies that are new in the GWAS Catalog
        new_studies = self.studies.join(self.previous_curation, on=CurationSchema.STUDY_ID, how='anti')
        # Annotate new studies with info if they have summary statistics synced to the GCS bucket
        new_studies_annotated = new_studies.join(self.synced, on=CurationSchema.STUDY_ID, how='left')
        # Assign status NO_SUMSTATS to new studies without synced summary statistics
        # (left join to drop info about already curated studies)
        new_studies_annotated = new_studies_annotated.select(
            CurationSchema.STUDY_ID,
            pl.lit(None).alias(CurationSchema.STUDY_TYPE),
            pl.lit(None).alias(CurationSchema.ANALYSIS_FLAG),
            pl.lit(None).alias(CurationSchema.QUALITY_CONTROL),
            pl.lit(False).alias(CurationSchema.IS_CURATED),
            CurationSchema.PUBMED_ID,
            CurationSchema.PUBLICATION_TITLE,
            CurationSchema.TRAIT_FROM_SOURCE,
            pl
            .when(pl.col(SyncedSummaryStatisticsSchema.SYNCED).is_null())
            .then(pl.lit(CuratedStudyStatus.NO_SUMSTATS))
            .otherwise(pl.lit(CuratedStudyStatus.TO_CURATE))
            .alias('status'),
        )
        logger.debug('New studies identified: {}', new_studies.shape[0])

        # Union of new studies and previously curated studies
        all_studies = pl.concat([prev_studies, new_studies_annotated], how='vertical')

        logger.debug('All studies after combining new and previous: {}', all_studies.shape[0])

        # Ensure the contract on the output dataframe
        assert all(all_studies.select(CurationSchema.STUDY_ID).is_unique()), 'Study IDs must be unique after merging.'
        assert all_studies.shape[0] == all_studies.shape[0], 'The number of studies must match after merging.'

        return all_studies.select(CurationSchema.extended_columns())
