"""The sources, destinations and settings of the genetic correlation tasks, as `config.yaml` gives them."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pts.transformers.genetic_correlation.previous import PreviousOutput
from pts.transformers.study import QualityFlag

LD_ANCESTRIES = ('afr', 'amr', 'eas', 'fin', 'nfe')
"""The LD populations with a score panel, as `ldPopulationStructure` names them."""


class RepresentativesSource(BaseModel):
    """The study index, heritability estimates and credible sets to choose representatives from."""

    model_config = ConfigDict(extra='forbid')

    study: str
    heritability: str
    """LDSC heritability estimates, one row per run, with `runStatus`, `h2`, `h2_se` and `lambda_gc`."""
    credible_set: str
    """The release's credible sets; a study's genome-wide significant ones rank it."""
    excluded_study: str | None = None
    """The studies the release excluded, read only with `include_excluded_studies`."""
    previous_output: PreviousOutput = None
    """The previous run's output folder, to compare the representatives with."""


class RepresentativesDestination(BaseModel):
    """The representatives, and how they changed since the previous run."""

    model_config = ConfigDict(extra='forbid')

    representatives: str
    changes: str


class RepresentativesSettings(BaseModel):
    """Which studies are eligible, and how the best one per disease and ancestry is chosen."""

    model_config = ConfigDict(extra='forbid')

    ancestries: list[str] = Field(default=['nfe'], min_length=1)
    """LD ancestries to keep studies of; each needs a score panel."""
    min_ancestry_share: float = Field(default=0.9, ge=0, le=1)
    """Lowest share of a study's samples, by `relativeSampleSize`, from its LD ancestry: the LD of a
    mixed-ancestry meta-analysis matches no single panel."""
    disease_column: Literal['diseaseIds', 'traitFromSourceMappedIds'] = 'diseaseIds'
    """The study column whose identifiers define a disease."""
    min_effective_sample_size: float = Field(default=10_000, gt=0)
    min_h2: float = 0.0
    max_h2: float = 1.0
    min_h2_z: float = Field(default=7.0, ge=0)
    """Lowest `h2 / h2_se`, the power threshold of stratified LDSC (Finucane et al. 2015)."""
    min_lambda_gc: float = 0.8
    max_lambda_gc: float = 2.5
    min_significant_variants: int = Field(default=1, ge=0)
    """Lowest `n_variants_sig` of the study's summary statistics QC: genome-wide significant variants."""
    max_lead_p_value: float = Field(default=5e-8, gt=0, le=1)
    """A credible set counts as a significant locus when its lead variant's p-value is at most this."""
    allowed_analysis_flags: list[str] = ['exwas', 'wgsgwas', 'metabolite']
    """A study's `analysisFlags`, lower-cased, must all be among these."""
    disqualifying_quality_controls: list[str] = [
        QualityFlag.SUMSTATS_NOT_AVAILABLE,
        QualityFlag.UNRESOLVED_TARGET,
        QualityFlag.UNRESOLVED_DISEASE,
        QualityFlag.UNKNOWN_STUDY_TYPE,
        QualityFlag.DUPLICATED_STUDY,
        QualityFlag.UNKNOWN_BIOSAMPLE,
        QualityFlag.FAILED_MEAN_BETA_CHECK,
        QualityFlag.FAILED_PZ_CHECK,
        QualityFlag.FAILED_GC_LAMBDA_CHECK,
        QualityFlag.NO_OT_CURATION,
        QualityFlag.CASE_CASE_STUDY_DESIGN,
        QualityFlag.DEPRECATED_PROJECT,
    ]
    """The study step's invalid reasons, and missing summary statistics."""
    include_excluded_studies: bool = False
    """Fall back on studies the release excluded only for `excluded_tolerated_quality_controls`; one
    is chosen only for a disease and ancestry with no eligible release study."""
    excluded_tolerated_quality_controls: list[str] = [
        QualityFlag.FAILED_PZ_CHECK,
        QualityFlag.FAILED_MEAN_BETA_CHECK,
    ]


class MungeSource(BaseModel):
    """The representatives, the LD score panels and where every study's summary statistics are."""

    model_config = ConfigDict(extra='forbid')

    representatives: str
    ldscore: str
    """Directory of one panel per ancestry, named by `MungeSettings.ldscore_template`."""
    sumstats: str
    """Directory with one parquet dataset per study, named by its `studyId`."""
    previous_output: PreviousOutput = None
    """The previous run's output folder, whose munged studies are reused where nothing changed."""


class MungeDestination(BaseModel):
    """One parquet file per study under `munged/<ancestry>/`, each study's SNP count, and each panel's M."""

    model_config = ConfigDict(extra='forbid')

    munged: str
    studies: str
    panels: str


class MungeSettings(BaseModel):
    """How the panels are named and read, and how the studies are fetched."""

    model_config = ConfigDict(extra='forbid')

    ldscore_template: str = 'gnomad_r2.1.1_{ancestry}_hg38.csv.gz'
    """Panel file (or directory of parts) name, with `{ancestry}` filled in. Tab separated."""
    maf_threshold: float = Field(default=0.05, ge=0, lt=0.5)
    """Panel SNPs with a frequency inside `[threshold, 1 - threshold]` count towards M, the number
    of SNPs the LD scores stand for; LDSC's `M_5_50` convention."""
    billing_project: str | None = None
    """Project billed for reading the summary statistics, when their bucket is requester pays."""
    min_snps: int = Field(default=200_000, gt=0)
    """Fewest panel SNPs a munged study needs to be correlated; LDSC warns below 200,000."""


class PairsSource(BaseModel):
    """The munged studies, their index from the munge step, and each panel's M."""

    model_config = ConfigDict(extra='forbid')

    studies: str
    munged: str
    panels: str
    previous_output: PreviousOutput = None
    """The previous run's output folder, whose pairs are reused where nothing changed and kept,
    flagged as deprecated, where a study is no longer correlated."""


class PairsDestination(BaseModel):
    """The genetic correlation of every pair, and the parameters it was computed with."""

    model_config = ConfigDict(extra='forbid')

    genetic_correlation: str
    parameters: str


class PairsSettings(BaseModel):
    """The regression's parameters and the parallelism of the run."""

    model_config = ConfigDict(extra='forbid')

    n_blocks: int = Field(default=200, gt=1)
    """Jackknife blocks; a pair with fewer shared SNPs is skipped."""
    twostep: float | None = 30.0
    """Chi-square cut-off of the two-step estimator; None fits every regression in one step."""
    intercept: float | None = None
    """Fix the genetic covariance intercept; None estimates it."""
    min_overlap_snps: int = Field(default=50, gt=0)
    tile_studies: int = Field(default=20, gt=0)
    """Studies on each side of a tile: a Spark task loads up to twice this many studies and
    correlates every pair between them."""
    min_tiles: int = Field(default=500, gt=0)
    """Fewest tiles a run is cut into, shrinking `tile_studies` for a small run so that it still
    spreads over the cluster; the dynamic allocation of executors hides the cluster's size."""
