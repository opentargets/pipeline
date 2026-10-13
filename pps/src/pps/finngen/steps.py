"""FinnGen steps."""

from __future__ import annotations

from pydantic import BaseModel

from pps.session import Session


class StudiesSettings(BaseModel):
    """Settings for `studies`."""

    finngen_study_index_out: str
    """Output FinnGen study index path."""
    finngen_phenotype_table_url: str
    """URL to the FinnGen phenotype table."""
    efo_curation_mapping_url: str
    """URL to the EFO curation mapping file."""
    finngen_release_prefix: str = 'FINNGEN_R11_'
    """FinnGen release prefix."""
    finngen_summary_stats_url_prefix: str = 'gs://finngen-public-data-r11/summary_stats/finngen_R11_'
    """FinnGen summary stats URL prefix."""
    finngen_summary_stats_url_suffix: str = '.gz'
    """FinnGen summary stats URL suffix."""
    sample_size: int = 453733
    """Number of individuals that participated in sample collection, derived from finngen release metadata."""


def studies(session: Session, settings: StudiesSettings) -> None:
    """FinnGen study index generation step.

    Ported from `legacy_gentropy.finngen_studies.FinnGenStudiesStep`.
    """
    raise NotImplementedError


class FinemappingIngestionSettings(BaseModel):
    """Settings for `finemapping_ingestion`."""

    finngen_finemapping_out: str
    """Output path for the finemapping results in StudyLocus format."""
    finngen_susie_finemapping_snp_files: str = 'gs://finngen-public-data-r11/finemap/full/susie/*.snp.bgz'
    finngen_susie_finemapping_cs_summary_files: str = (
        'gs://finngen-public-data-r11/finemap/summary/*SUSIE.cred.summary.tsv'
    )
    """FinnGen SuSIE summaries for CS filters(LBF>2)."""
    finngen_finemapping_lead_pvalue_threshold: float = 1e-05
    """Lead p-value threshold."""
    finngen_release_prefix: str = 'FINNGEN_R11'
    """Finngen project release prefix. Should look like FINNGEN_R*."""


def finemapping_ingestion(session: Session, settings: FinemappingIngestionSettings) -> None:
    """FinnGen study table ingestion step.

    Ported from `legacy_gentropy.finngen_finemapping_ingestion.FinnGenFinemappingIngestionStep`.
    """
    raise NotImplementedError
