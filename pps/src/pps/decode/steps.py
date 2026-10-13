"""deCODE proteomics steps."""

from __future__ import annotations

from pydantic import BaseModel

from pps.session import Session


class ManifestGenerationSettings(BaseModel):
    """Settings for `manifest_generation`."""

    bucket_name: str
    """Name of the S3 bucket to list (without any URI prefix)."""
    prefix: str
    """Key prefix to restrict the listing (e.g. ``"Proteomics/"``)."""
    output_path: str
    """Destination path for the manifest Parquet dataset."""


def manifest_generation(session: Session, settings: ManifestGenerationSettings) -> None:
    """Build the deCODE manifest Parquet dataset by listing the S3 bucket directly.

    Ported from `legacy_gentropy.decode_ingestion.deCODEManifestGenerationStep`.
    """
    raise NotImplementedError


class SummaryStatisticsIngestionSettings(BaseModel):
    """Settings for `summary_statistics_ingestion`."""

    manifest_path: str
    """Path to the manifest Parquet dataset produced by `deCODEManifestGenerationStep`."""
    raw_summary_statistics_path: str
    """Destination path for the raw summary statistics Parquet dataset, partitioned by ``studyId``."""


def summary_statistics_ingestion(session: Session, settings: SummaryStatisticsIngestionSettings) -> None:
    """Ingest deCODE summary statistics from gzipped TSV files into Parquet format.

    Ported from `legacy_gentropy.decode_ingestion.deCODESummaryStatisticsIngestionStep`.
    """
    raise NotImplementedError


class SummaryStatisticsHarmonisationSettings(BaseModel):
    """Settings for `summary_statistics_harmonisation`."""

    raw_summary_statistics_path: str
    """Path to the raw summary-statistics Parquet dataset produced by `deCODESummaryStatisticsIngestionStep`."""
    manifest_path: str
    """Path to the manifest Parquet dataset produced by `deCODEManifestGenerationStep`."""
    aptamer_metadata_path: str
    """Path to the aptamer metadata TSV file."""
    variant_direction_path: str
    """Path to the gnomAD `VariantDirection` Parquet dataset used for allele flipping and EAF inference."""
    molecular_complex_path: str
    """Path to the molecular-complex Parquet dataset produced by `MolecularComplexIngestionStep`."""
    harmonised_summary_statistics_path: str
    """Destination path for the harmonised summary-statistics Parquet dataset, partitioned by ``studyId``."""
    protein_qtl_study_index_path: str
    """Destination path for the pQTL study index Parquet dataset annotated with QC results."""
    perform_min_allele_count_filter: bool = True
    """Whether to apply the MAC threshold filter."""
    min_mac_threshold: int = 50
    """Minimum minor allele count (MAC) required to retain a variant."""
    perform_sample_size_filter: bool = True
    """Whether to apply the sample-size threshold filter."""
    min_sample_size_threshold: int = 30000
    """Minimum sample size required to retain a variant."""
    flipping_window_size: int = 10000000
    """Genomic window size (bp) used to partition the VariantDirection dataset for the allele-flipping join.  Must
    match the value used when building the VariantDirection dataset."""
    remove_equal_alleles: bool = True
    """Whether to remove variants with equal effect and other alleles during harmonisation."""
    remove_ambiguous_alleles: bool = False
    """Whether to exclude strand-ambiguous variants (A/T or C/G) from the gnomAD reference slice used for allele
    flipping. Ambiguous sumstat variants that have no gnomAD match are still retained in the output with no flip
    applied."""
    verify_atgc: bool = True
    """Whether to verify that all alleles are A/T/G/C during harmonisation. Strict ATGC validation also removes `*`
    (star) and `!` (multiallelic) alleles."""


def summary_statistics_harmonisation(session: Session, settings: SummaryStatisticsHarmonisationSettings) -> None:
    """Harmonise ingested deCODE summary statistics and generate the pQTL study index.

    Ported from `legacy_gentropy.decode_ingestion.deCODESummaryStatisticsHarmonisationStep`.
    """
    raise NotImplementedError


class SummaryStatisticsQcSettings(BaseModel):
    """Settings for `summary_statistics_qc`."""

    harmonised_summary_statistics_path: str
    """Path to the harmonised summary-statistics Parquet dataset produced by
    `deCODESummaryStatisticsHarmonisationStep`."""
    protein_qtl_study_index_path: str
    """Path to the pQTL study index Parquet dataset produced by `deCODESummaryStatisticsHarmonisationStep`."""
    qc_summary_statistics_path: str
    """Destination path for the QC summary-statistics Parquet dataset, partitioned by ``studyId``."""
    protein_qtl_study_index_qc_annotated_path: str
    """Destination path for the pQTL study index Parquet dataset annotated with QC results."""
    pval_threshold: float = 5e-08
    """P-value threshold used to define significant variants for QC purposes."""


def summary_statistics_qc(session: Session, settings: SummaryStatisticsQcSettings) -> None:
    """Step to calculate quality control metrics on the harmonised deCODE summary statistics.

    Ported from `legacy_gentropy.decode_ingestion.deCODESummaryStatisticsQCStep`.
    """
    raise NotImplementedError
