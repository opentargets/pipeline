"""FinnGen, UKB and MVP meta-analysis steps."""

from __future__ import annotations

from pydantic import BaseModel

from pps.session import Session


class IngestionSettings(BaseModel):
    """Settings for `ingestion`."""

    source_manifest_path: str
    """Path to the manifest file."""
    efo_curation_path: str
    """Path to the EFO curation file."""
    gnomad_variant_index_path: str
    """Path to the gnomAD variant index file."""
    study_index_output_path: str
    """Output path for the study index."""
    raw_summary_statistics_output_path: str
    """Output path for raw summary statistics."""
    harmonised_summary_statistics_output_path: str
    """Output path for harmonised summary statistics."""
    harmonised_summary_statistics_qc_output_path: str
    """Output path for harmonised summary statistics QC results."""
    perform_meta_analysis_filter: bool = True
    """Whether to filter non-meta analyzed variants."""
    imputation_score_threshold: float = 0.8
    """Imputation score threshold."""
    perform_imputation_score_filter: bool = True
    """Whether to filter low imputation scores."""
    min_allele_count_threshold: int = 20
    """Minimum allele count threshold."""
    perform_min_allele_count_filter: bool = True
    """Whether to filter low allele counts."""
    min_allele_frequency_threshold: float = 0.0001
    """Minimum allele frequency threshold."""
    perform_min_allele_frequency_filter: bool = False
    """Whether to filter low allele frequencies."""
    filter_out_ambiguous_variants: bool = False
    """Whether to filter out ambiguous variants."""


def ingestion(session: Session, settings: IngestionSettings) -> None:
    """FinnGen UK Biobank and Million Veteran Program meta-analysis summary statistics ingestion step.

    Ported from `legacy_gentropy.finngen_ukb_mvp_meta.FinngenUkbMvpMetaSummaryStatisticsIngestionStep`.
    """
    raise NotImplementedError
