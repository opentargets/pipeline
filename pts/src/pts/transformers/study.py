"""Study validation and annotation."""

from enum import StrEnum
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from pydantic import BaseModel, ConfigDict, Field, field_validator

from pts.schemas.study import study_schema
from pts.transformers.utils.dataset import scan_dataset, scan_datasets, write_dataset
from pts.transformers.utils.quality_flags import valid_rows


class QualityFlag(StrEnum):
    """Quality control flags recorded on studies."""

    UNRESOLVED_TARGET = 'Target/gene identifier could not match to reference'
    UNRESOLVED_DISEASE = 'No valid disease identifier found'
    UNKNOWN_STUDY_TYPE = 'This type of study is not supported'
    UNKNOWN_BIOSAMPLE = 'Biosample identifier was not found in the reference'
    DUPLICATED_STUDY = 'The identifier of this study is not unique'
    SUMSTATS_NOT_AVAILABLE = 'Harmonized summary statistics are not available or empty'
    NO_OT_CURATION = 'GWAS Catalog study has not been curated by Open Targets'
    FAILED_MEAN_BETA_CHECK = 'The mean beta QC check value is not within the expected range'
    FAILED_PZ_CHECK = 'The PZ QC check values are not within the expected range'
    FAILED_GC_LAMBDA_CHECK = 'The GC lambda value is not within the expected range'
    SMALL_NUMBER_OF_SNPS = 'The number of SNPs in the study is below the expected threshold'
    CASE_CASE_STUDY_DESIGN = 'Case-case study design'
    DEPRECATED_PROJECT = 'Deprecated project'
    INVALID_SAMPLE_SIZE = 'Sample size is null or equal to zero'
    CASE_CONTROL_STUDY_DESIGN = 'Study has non empty or zero nCases, nControls & nSamples with matching sum'
    CASE_CONTROL_SUM_NEQ_SAMPLE_SIZE = 'Study has non empty or zero nCases, nControls & nSamples with non matching sum'
    ONE_ONLY_CASE_OR_CONTROL = 'Study has only cases or only controls'
    MEASUREMENT_STUDY_DESIGN = 'Study has non empty or zero nSamples with empty nCases and nControls'


class StudySource(BaseModel):
    """Input dataset paths for study validation."""

    model_config = ConfigDict(extra='forbid')

    study: list[str]
    target: str
    disease: str
    biosample: str
    heritability: str


class StudyDestination(BaseModel):
    """Output dataset paths for accepted and excluded studies."""

    model_config = ConfigDict(extra='forbid')

    output: str
    excluded: str


class StudySettings(BaseModel):
    """Settings for study validation."""

    model_config = ConfigDict(extra='forbid')

    invalid_qc_reasons: list[QualityFlag] = Field(default_factory=list)
    """QC reason names that cause a study to be excluded."""
    deprecated_project_ids: list[str] = Field(default_factory=list)
    """Project IDs that should be flagged as deprecated."""

    @field_validator('invalid_qc_reasons', mode='before')
    @classmethod
    def resolve_qc_names(cls, value: Any) -> Any:
        """Resolve configured enum names before validating the QC flags."""
        if isinstance(value, list):
            return [QualityFlag.__members__.get(item, item) if isinstance(item, str) else item for item in value]
        return value


def disease_lut(disease: pl.LazyFrame) -> pl.LazyFrame:
    """Map canonical and obsolete disease identifiers to the canonical disease ID."""
    return (
        disease
        .select(
            pl.col('id').alias('diseaseId'),
            pl
            .concat_list(pl.col('id'), pl.col('obsoleteTerms').fill_null([]))
            .list.unique(maintain_order=True)
            .alias('efo'),
        )
        .explode('efo')
        .with_columns(pl.coalesce('efo', 'diseaseId').alias('efo'))
    )


def _flag(studies: pl.LazyFrame, condition: pl.Expr, flag: QualityFlag) -> pl.LazyFrame:
    """Append a QC flag, preserving existing flags and normalizing null lists."""
    qc = pl.col('qualityControls').fill_null([])
    return studies.with_columns(
        pl
        .when(condition)
        .then(pl.concat_list(qc, pl.lit([flag.value])).list.unique().list.sort())
        .otherwise(qc)
        .alias('qualityControls')
    )


def deconvolute_studies(studies: pl.LazyFrame) -> pl.LazyFrame:
    """Merge repeated study IDs and propagate their QC and most granular traits.

    For conflicting unaggregated fields, retain the first input record. Trait ties
    retain the last non-null value, making the source order the tie breaker.
    """
    logger.info('Deconvoluting study records')
    has_summary = pl.col('hasSumstats').fill_null(False).any().over('studyId')
    qc = pl.col('qualityControls').fill_null([])
    prepared = studies.with_columns(
        pl
        .when(~pl.col('hasSumstats'))
        .then(qc.list.eval(pl.element().filter(pl.element() != QualityFlag.NO_OT_CURATION.value)))
        .otherwise(qc)
        .alias('qualityControls'),
        pl.col('traitFromSourceMappedIds').list.len().fill_null(-1).min().over('studyId').alias('_minimum_traits'),
        has_summary.alias('_has_summary'),
    ).with_columns(
        pl
        .when(pl.col('_has_summary'))
        .then(
            pl.col('qualityControls').list.eval(
                pl.element().filter(pl.element() != QualityFlag.SUMSTATS_NOT_AVAILABLE.value)
            )
        )
        .otherwise(pl.col('qualityControls'))
        .alias('qualityControls')
    )
    granular = pl.col('traitFromSourceMappedIds').list.len().fill_null(-1) == pl.col('_minimum_traits')
    aggregated = {
        'qualityControls': pl
        .col('qualityControls')
        .filter(pl.col('qualityControls').list.len() > 0)
        .explode()
        .unique(maintain_order=True),
        'sumstatQCValues': pl.col('sumstatQCValues').drop_nulls().first(),
        'analysisFlags': pl.col('analysisFlags').filter(pl.col('analysisFlags').list.len() > 0).explode(),
        'hasSumstats': pl
        .when(pl.col('_has_summary').first())
        .then(True)
        .when((~pl.col('hasSumstats')).any())
        .then(False)
        .otherwise(None),
        'traitFromSourceMappedIds': pl.col('traitFromSourceMappedIds').filter(granular).drop_nulls().last(),
        'traitFromSource': pl.col('traitFromSource').filter(granular).drop_nulls().last(),
        'studyType': pl.col('studyType').drop_nulls().unique(maintain_order=True).str.join(','),
    }
    columns = studies.collect_schema().names()
    return (
        prepared
        .group_by('studyId', maintain_order=True)
        .agg([
            aggregated.get(column, pl.col(column).first()).alias(column) for column in columns if column != 'studyId'
        ])
        .select(columns)
    )


def validate_study_type(studies: pl.LazyFrame) -> pl.LazyFrame:
    """Flag study types other than GWAS or types ending in qtl."""
    logger.info('Validating study types')
    supported = (pl.col('studyType') == 'gwas') | pl.col('studyType').str.ends_with('qtl')
    return _flag(studies, ~supported.fill_null(False), QualityFlag.UNKNOWN_STUDY_TYPE)


def validate_project_id(studies: pl.LazyFrame, deprecated_project_ids: list[str]) -> pl.LazyFrame:
    """Flag studies from deprecated projects."""
    logger.info('Validating deprecated project IDs')
    if not deprecated_project_ids:
        return studies
    return _flag(studies, pl.col('projectId').is_in(deprecated_project_ids), QualityFlag.DEPRECATED_PROJECT)


def validate_target(studies: pl.LazyFrame, target: pl.LazyFrame) -> pl.LazyFrame:
    """Flag non-GWAS studies whose gene identifier is absent from the target index."""
    logger.info('Validating study targets')
    lookup = target.select(pl.col('id').alias('geneId'), pl.lit(True).alias('_target_found'))
    return _flag(
        studies.join(lookup, on='geneId', how='left'),
        (pl.col('studyType') != 'gwas') & pl.col('_target_found').is_null(),
        QualityFlag.UNRESOLVED_TARGET,
    ).drop('_target_found')


def validate_disease(studies: pl.LazyFrame, disease_map: pl.LazyFrame) -> pl.LazyFrame:
    """Resolve foreground and background diseases and flag unresolved GWAS studies."""
    logger.info('Normalizing and validating study diseases')
    result = studies.drop('diseaseIds', 'backgroundDiseaseIds')
    for source, destination in [
        ('traitFromSourceMappedIds', 'diseaseIds'),
        ('backgroundTraitFromSourceMappedIds', 'backgroundDiseaseIds'),
    ]:
        mapped = (
            studies
            .filter(pl.col(source).list.len() > 0)
            .select('studyId', pl.col(source).alias('efo'))
            .explode('efo')
            .join(disease_map, on='efo', how='left')
            .group_by('studyId')
            .agg(pl.col('diseaseId').drop_nulls().unique(maintain_order=True).alias(destination))
        )
        result = result.join(mapped, on='studyId', how='left').with_columns(pl.col(destination).fill_null([]))
    return _flag(
        result,
        (pl.col('studyType') == 'gwas') & (pl.col('diseaseIds').list.len() == 0),
        QualityFlag.UNRESOLVED_DISEASE,
    )


def validate_biosample(studies: pl.LazyFrame, biosample: pl.LazyFrame) -> pl.LazyFrame:
    """Resolve source biosample IDs and flag unmatched non-GWAS studies."""
    logger.info('Validating study biosamples')
    lookup = biosample.select(
        pl.col('biosampleId').alias('biosampleFromSourceId'),
        'biosampleId',
        pl.lit(True).alias('_biosample_found'),
    )
    return _flag(
        studies.drop('biosampleId').join(lookup, on='biosampleFromSourceId', how='left'),
        (pl.col('studyType') != 'gwas') & pl.col('_biosample_found').is_null(),
        QualityFlag.UNKNOWN_BIOSAMPLE,
    ).drop('_biosample_found')


def validate_analysis_flags(studies: pl.LazyFrame) -> pl.LazyFrame:
    """Flag studies curated as case-case designs."""
    logger.info('Validating study analysis flags')
    return _flag(studies, pl.col('analysisFlags').list.contains('Case-case study'), QualityFlag.CASE_CASE_STUDY_DESIGN)


def validate_ccs(studies: pl.LazyFrame) -> pl.LazyFrame:
    """Annotate sample validity and case-control or measurement study designs."""
    logger.info('Validating case, control and sample counts')
    cases, controls, samples = pl.col('nCases'), pl.col('nControls'), pl.col('nSamples')
    positive_cases = cases > 0
    positive_controls = controls > 0
    positive_samples = samples > 0
    empty_cases = cases.is_null() | (cases == 0)
    empty_controls = controls.is_null() | (controls == 0)
    both = positive_cases & positive_controls & positive_samples
    matching = cases + controls == samples
    for condition, flag in [
        (both & matching, QualityFlag.CASE_CONTROL_STUDY_DESIGN),
        (empty_cases & empty_controls & positive_samples, QualityFlag.MEASUREMENT_STUDY_DESIGN),
        (samples.is_null() | (samples == 0), QualityFlag.INVALID_SAMPLE_SIZE),
        (both & ~matching, QualityFlag.CASE_CONTROL_SUM_NEQ_SAMPLE_SIZE),
        (
            ((positive_cases & empty_controls) | (empty_cases & positive_controls)) & positive_samples,
            QualityFlag.ONE_ONLY_CASE_OR_CONTROL,
        ),
    ]:
        studies = _flag(studies, condition, flag)
    return studies


def collect_heritability(studies: pl.LazyFrame, heritability: pl.LazyFrame) -> pl.LazyFrame:
    """Append successful LDSC estimates to existing summary-statistic QC values."""
    logger.info('Annotating studies with successful LDSC estimates')
    fields = {
        'h2': 'LDSC_SNP-h2',
        'h2_se': 'LDSC_SNP-h2_se',
        'intercept': 'LDSC_intercept',
        'intercept_se': 'LDSC_intercept_se',
        'mean_chisq': 'LDSC_mean_chisq',
        'lambda_gc': 'LDSC_gc_lambda',
    }
    annotations = heritability.filter(pl.col('runStatus') == 'success').select(
        'studyId',
        pl.concat_list([
            pl.struct(
                pl.lit(label).alias('QCCheckName'),
                pl.col(field).cast(pl.Float32, strict=False).alias('QCCheckValue'),
            )
            for field, label in fields.items()
        ]).alias('_ldsc_qc'),
    )
    return (
        studies
        .join(annotations, on='studyId', how='left')
        .with_columns(
            pl
            .when(pl.col('_ldsc_qc').is_not_null())
            .then(pl.concat_list(pl.col('sumstatQCValues').fill_null([]), pl.col('_ldsc_qc')))
            .otherwise(pl.col('sumstatQCValues'))
            .alias('sumstatQCValues')
        )
        .drop('_ldsc_qc')
    )


def study(
    source: dict[str, str | list[str]],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Validate and annotate studies, separating accepted and excluded records.

    Args:
        source: Study input paths under `study`, plus `target`, `disease`, and
            `biosample` lookup paths and the `heritability` path.
        destination: Dataset paths under `output` and `excluded`.
        settings: QC reason names under `invalid_qc_reasons` and project IDs
            under `deprecated_project_ids`.
        config: Pipeline configuration.
    """
    sources = StudySource.model_validate(source)
    destinations = StudyDestination.model_validate(destination)
    options = StudySettings.model_validate(settings)
    target = scan_dataset(sources.target)
    biosample = scan_dataset(sources.biosample)
    disease = disease_lut(scan_dataset(sources.disease))
    heritability = scan_dataset(sources.heritability)
    result = (
        scan_datasets(sources.study, schema=study_schema)
        .pipe(deconvolute_studies)
        .pipe(validate_study_type)
        .pipe(validate_project_id, options.deprecated_project_ids)
        .pipe(validate_target, target)
        .pipe(validate_disease, disease)
        .pipe(validate_biosample, biosample)
        .pipe(validate_analysis_flags)
        .pipe(validate_ccs)
        .pipe(collect_heritability, heritability)
        .select(study_schema.names())
        .pipe(valid_rows, options.invalid_qc_reasons)
    )
    logger.info('Writing accepted studies to {}', destinations.output)
    write_dataset(result.valid, destinations.output, single_file=True)
    logger.info('Writing excluded studies to {}', destinations.excluded)
    write_dataset(result.invalid, destinations.excluded, single_file=True)
