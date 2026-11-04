"""The sources, destinations and settings of the genetic correlation tasks, as `config.yaml` gives them."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pts.transformers.study import QualityFlag

LD_ANCESTRIES = ('afr', 'amr', 'eas', 'fin', 'nfe')
"""The LD populations with a score panel, as `ldPopulationStructure` names them."""


class RepresentativesSource(BaseModel):
    """The study index and the heritability estimates to choose representatives from."""

    model_config = ConfigDict(extra='forbid')

    study: str
    heritability: str
    """LDSC heritability estimates, one row per run, with `runStatus`, `h2`, `h2_se` and `lambda_gc`."""


class RepresentativesSettings(BaseModel):
    """Which studies are eligible, and how the best one per disease and ancestry is chosen."""

    model_config = ConfigDict(extra='forbid')

    ancestries: list[str] = Field(default=['nfe'], min_length=1)
    """LD ancestries to keep studies of; each needs a score panel."""
    disease_column: Literal['diseaseIds', 'traitFromSourceMappedIds'] = 'diseaseIds'
    """The study column whose identifiers define a disease."""
    min_effective_sample_size: float = Field(default=10_000, gt=0)
    min_h2: float = 0.0
    max_h2: float = 1.0
    min_h2_z_squared: float = Field(default=4.0, ge=0)
    """Lowest `(h2 / h2_se)^2`: the study needs detectable polygenic signal."""
    min_lambda_gc: float = 0.8
    max_lambda_gc: float = 2.5
    allowed_analysis_flags: list[str] = ['exwas', 'wgsgwas', 'metabolite']
    """A study's `analysisFlags`, lower-cased, must all be among these."""
    disqualifying_quality_controls: list[str] = [
        QualityFlag.SUMSTATS_NOT_AVAILABLE,
        QualityFlag.FAILED_PZ_CHECK,
        QualityFlag.FAILED_MEAN_BETA_CHECK,
        QualityFlag.FAILED_GC_LAMBDA_CHECK,
        QualityFlag.CASE_CASE_STUDY_DESIGN,
        QualityFlag.DUPLICATED_STUDY,
    ]


class MungeSource(BaseModel):
    """The representatives, the LD score panels and where every study's summary statistics are."""

    model_config = ConfigDict(extra='forbid')

    representatives: str
    ldscore: str
    """Directory of one panel per ancestry, named by `MungeSettings.ldscore_template`."""
    sumstats: str
    """Directory with one parquet dataset per study, named by its `studyId`."""


class MungeDestination(BaseModel):
    """One parquet file per study under `munged/<ancestry>/`, and each panel's SNP count."""

    model_config = ConfigDict(extra='forbid')

    munged: str
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
    parallel_studies: int = Field(default=8, gt=0)
    """Studies fetched and munged at the same time."""


class PairsSource(BaseModel):
    """The representatives and the munged studies, with each panel's SNP count."""

    model_config = ConfigDict(extra='forbid')

    representatives: str
    munged: str
    panels: str


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
    parallel_pairs: int = Field(default=8, gt=0)
    """Worker processes."""
    worker_threads: int = Field(default=2, gt=0)
    """Polars threads of each worker."""
    pairs_per_chunk: int = Field(default=200, gt=0)
    """Pairs handed to a worker at a time."""
