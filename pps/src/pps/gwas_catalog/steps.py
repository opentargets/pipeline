"""GWAS Catalog steps."""

from __future__ import annotations

from pydantic import BaseModel

from pps.session import Session


class TopHitIngestionSettings(BaseModel):
    """Settings for `top_hit_ingestion`."""

    catalog_study_files: list[str]
    """Raw GWAS Catalog study files."""
    catalog_ancestry_files: list[str]
    """Raw GWAS Catalog ancestry annotation files."""
    catalog_associations_file: str
    """Raw GWAS Catalog associations file."""
    variant_annotation_path: str
    """Path to the gnomAD variant index."""
    catalog_studies_out: str
    """Output path for the GWAS Catalog study index."""
    catalog_associations_out: str
    """Output path for the window-based clumped GWAS Catalog study loci."""
    distance: int = 500_000
    """Distance within which tagging variants are collected around the semi-index."""


def top_hit_ingestion(session: Session, settings: TopHitIngestionSettings) -> None:
    """Extract GWAS Catalog top hits into a study index and window-based clumped study loci.

    Reads the GWAS Catalog studies, ancestries and curated associations, splits the
    studies by association traits, flags studies without summary statistics and
    clumps the associations within `distance` of each lead variant.

    Ported from `legacy_gentropy.gwas_catalog_top_hits.GWASCatalogTopHitIngestionStep`.
    """
    raise NotImplementedError


class StudyIndexSettings(BaseModel):
    """Settings for `study_index`."""

    catalog_study_files: list[str]
    """List of raw GWAS catalog studies file."""
    catalog_ancestry_files: list[str]
    """List of raw ancestry annotations files from GWAS Catalog."""
    study_index_path: str
    """Output GWAS catalog studies path."""
    gwas_catalog_study_curation_file: str | None = None
    """Csv file or URL containing the curation table."""
    sumstats_qc_path: str | None = None
    """Path to the summary statistics QC table."""


def study_index(session: Session, settings: StudyIndexSettings) -> None:
    """GWAS Catalog study index generation.

    Ported from `legacy_gentropy.gwas_catalog_study_index.GWASCatalogStudyIndexGenerationStep`.
    """
    raise NotImplementedError
