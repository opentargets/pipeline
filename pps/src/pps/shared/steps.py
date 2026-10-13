"""Steps shared across datasources."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from pps.session import Session


class LdClumpingSettings(BaseModel):
    """Settings for `ld_clumping`."""

    study_locus_input_path: str
    """Path to the input study locus."""
    study_index_path: str
    """Path to the study index."""
    ld_index_path: str
    """Path to the LD index."""
    clumped_study_locus_output_path: str
    """Output path for the LD-clumped study locus."""


def ld_clumping(session: Session, settings: LdClumpingSettings) -> None:
    """Perform LD-based clumping on a study locus dataset.

    Annotates the study loci with population-specific linked variants from the LD
    index (weighted by the ancestries in the study index), then flags study loci that
    are explained by a more significant association from the same study.

    Ported from `legacy_gentropy.ld_based_clumping.LDBasedClumpingStep`.
    """
    raise NotImplementedError


class PicsSettings(BaseModel):
    """Settings for `pics`."""

    study_locus_ld_annotated_in: str
    """Path to the LD-annotated study locus."""
    picsed_study_locus_out: str
    """Output path for the PICS credible sets."""
    credible_interval: Literal['IS95', 'IS99'] = 'IS99'
    """Credible interval the credible sets are filtered to."""
    pvalue_cutoff: float = 1e-8
    """Lead p-value below which loci are kept unflagged."""


def pics(session: Session, settings: PicsSettings) -> None:
    """Fine-map an LD-annotated study locus dataset with PICS.

    Builds credible sets with PICS, filters them to `credible_interval` and flags
    loci whose lead p-value is above `pvalue_cutoff`.

    Ported from `legacy_gentropy.pics.PICSStep`.
    """
    raise NotImplementedError


class WindowClumpingSettings(BaseModel):
    """Settings for `window_clumping`."""

    summary_statistics_input_path: str
    """Path to the harmonized summary statistics dataset."""
    study_locus_output_path: str
    """Output path for the resulting study locus dataset."""
    gwas_significance: float = 1e-08
    """GWAS significance threshold."""
    distance: int = 500000
    """Distance, within which tagging variants are collected around the semi-index."""
    collect_locus: bool = False
    """Whether to collect locus around semi-indices."""
    collect_locus_distance: int = 500000
    """Distance, within which tagging variants are collected around the semi-index."""
    inclusion_list_path: str | None = None
    """Path to the inclusion list (list of white-listed study identifier)."""
    recursive_file_lookup: bool = True
    """Whether to recursively look for summary statistics files in the input path. Note that if an inclusion list is
    provided, this flag is set to `True` always."""


def window_clumping(session: Session, settings: WindowClumpingSettings) -> None:
    """Apply window based clumping on summary statistics datasets.

    Ported from `legacy_gentropy.window_based_clumping.WindowBasedClumpingStep`.
    """
    raise NotImplementedError


class LocusBreakerClumpingSettings(BaseModel):
    """Settings for `locus_breaker_clumping`."""

    summary_statistics_input_path: str
    """Path to the input study locus."""
    clumped_study_locus_output_path: str
    """Path of the resulting, clumped study-locus dataset."""
    wbc_pvalue_threshold: float
    """P-value threshold for window breaker clumping."""
    lbc_baseline_pvalue: float = 1e-05
    """Baseline p-value for locus breaker clumping."""
    lbc_distance_cutoff: int = 250000
    """Distance cutoff for locus breaker clumping."""
    lbc_pvalue_threshold: float = 1e-08
    """P-value threshold for locus breaker clumping."""
    lbc_flanking_distance: int = 100000
    """Flanking distance for locus breaker clumping."""
    large_loci_size: int = 1500000
    """Threshold distance to define large loci for window-based clumping."""
    wbc_clump_distance: int = 500000
    """Clump distance for window breaker clumping."""
    collect_locus: bool = False
    """Whether to collect locus."""
    remove_mhc: bool = True
    """If true will use exclude_region() to remove the MHC region."""


def locus_breaker_clumping(session: Session, settings: LocusBreakerClumpingSettings) -> None:
    """Step to perform locus-breaker clumping on a study.

    Ported from `legacy_gentropy.locus_breaker_clumping.LocusBreakerClumpingStep`.
    """
    raise NotImplementedError


class CredibleSetQcSettings(BaseModel):
    """Settings for `credible_set_qc`."""

    credible_sets_path: str
    """Path to credible sets file."""
    output_path: str
    """Path to write the output file."""
    p_value_threshold: float = 1e-05
    """P-value threshold for credible set quality control."""
    purity_min_r2: float = 0.01
    """Minimum R2 for purity estimation."""
    clump: bool = False
    """Whether to clump the credible sets by LD."""
    ld_index_path: str | None = None
    """Path to LD index file."""
    study_index_path: str | None = None
    """Path to study index file."""
    ld_min_r2: float | None = 0.8
    """Minimum R2 for LD estimation."""
    n_partitions: int | None = 200
    """Number of partitions to coalesce the dataset after reading."""


def credible_set_qc(session: Session, settings: CredibleSetQcSettings) -> None:
    """Credible set quality control step for fine mapped StudyLoci.

    Ported from `legacy_gentropy.credible_set_qc.CredibleSetQCStep`.
    """
    raise NotImplementedError


class PqtlStudyIndexSettings(BaseModel):
    """Settings for `pqtl_study_index`."""

    protein_study_index_path: str
    """Path to the `ProteinQuantitativeTraitLocusStudyIndex`."""
    study_index_path: str
    """Destination path for the resolved `StudyIndex` Parquet dataset."""
    target_index_path: str
    """Path to the `TargetIndex` Parquet dataset used to map gene symbols to Ensembl gene IDs."""


def pqtl_study_index(session: Session, settings: PqtlStudyIndexSettings) -> None:
    """Transform a `ProteinQuantitativeTraitLocusStudyIndex` into a standard `StudyIndex`.

    Ported from `legacy_gentropy.pqtl_study.pQTLStudyIndexTransformationStep`.
    """
    raise NotImplementedError
