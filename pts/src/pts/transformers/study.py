"""Study validation and annotation."""

from enum import StrEnum
from typing import Any

import polars as pl
from loguru import logger
from ot_datamodels.outputs.study import Study
from otter.config.model import Config
from pydantic import BaseModel, ConfigDict, Field, field_validator

from pts.schemas.dataset.conform import conform
from pts.schemas.study import STUDY_TYPES, study_schema, superseded_studies_schema
from pts.task_outputs import writes
from pts.transformers.evidence.utils.luts import build_disease_lut
from pts.transformers.utils.dataset import scan_dataset, sink_dataset
from pts.transformers.utils.quality_flags import flag_rows, resolve_flag_names, valid_rows


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
    SUPERSEDED_STUDY = 'Study superseded by another study'


class StudySource(BaseModel):
    """Input dataset paths for study validation."""

    model_config = ConfigDict(extra='forbid')

    study: list[str]
    target: str
    disease: str
    biosample: str
    heritability: str
    superseded_studies: str | None = None


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
        return resolve_flag_names(QualityFlag, value)


def sort_study_collections(studies: pl.LazyFrame) -> pl.LazyFrame:
    """Canonicalize list order without changing nulls, values, or multiplicities."""
    return studies.with_columns(
        pl.col(name).list.sort() for name, dtype in studies.collect_schema().items() if isinstance(dtype, pl.List)
    )


def record_key(columns: list[str]) -> pl.Expr:
    """The canonical content of a study record: `columns` in name order, encoded as JSON.

    Ranking records by it makes a choice between them independent of file and row order.
    """
    return pl.struct(sorted(columns)).struct.json_encode()


def validate_duplicated_studies(studies: pl.LazyFrame) -> pl.LazyFrame:
    """Keep one record of each study ID that several records hold, and flag the others.

    The kept record is the first by `record_key`, after list columns are put in canonical order.
    """
    logger.info('Validating study ID uniqueness')
    studies = studies.pipe(sort_study_collections)
    later = record_key(studies.collect_schema().names()).rank('ordinal').over('studyId') > 1
    return flag_rows(studies, later, QualityFlag.DUPLICATED_STUDY)


def validate_study_type(studies: pl.LazyFrame) -> pl.LazyFrame:
    """Flag study types that are not one of `STUDY_TYPES`, the gentropy study types."""
    logger.info('Validating study types')
    supported = pl.col('studyType').is_in(STUDY_TYPES)
    return flag_rows(studies, ~supported.fill_null(False), QualityFlag.UNKNOWN_STUDY_TYPE)


def validate_project_id(studies: pl.LazyFrame, deprecated_project_ids: list[str]) -> pl.LazyFrame:
    """Flag studies from deprecated projects."""
    logger.info('Validating deprecated project IDs')
    if not deprecated_project_ids:
        return studies
    return flag_rows(studies, pl.col('projectId').is_in(deprecated_project_ids), QualityFlag.DEPRECATED_PROJECT)


def validate_target(studies: pl.LazyFrame, target: pl.LazyFrame) -> pl.LazyFrame:
    """Flag non-GWAS studies whose gene identifier is absent from the target index."""
    logger.info('Validating study targets')
    lookup = target.select(pl.col('id').alias('geneId'), pl.lit(True).alias('_target_found'))
    return flag_rows(
        studies.join(lookup, on='geneId', how='left'),
        (pl.col('studyType') != 'gwas') & pl.col('_target_found').is_null(),
        QualityFlag.UNRESOLVED_TARGET,
    ).drop('_target_found')


def validate_disease(studies: pl.LazyFrame, disease_map: pl.LazyFrame) -> pl.LazyFrame:
    """Resolve foreground and background diseases and flag unresolved GWAS studies."""
    logger.info('Normalizing and validating study diseases')
    # Some sources (gentropy v4 study index) already carry these columns; replace them, or the
    # joins below would add `*_right` columns and leave the source values in place.
    studies = studies.drop('diseaseIds', 'backgroundDiseaseIds', strict=False)
    result = studies
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
            .agg(pl.col('diseaseId').drop_nulls().unique().sort().alias(destination))
        )
        result = result.join(mapped, on='studyId', how='left').with_columns(pl.col(destination).fill_null([]))
    return flag_rows(
        result,
        (pl.col('studyType') == 'gwas') & (pl.col('diseaseIds').list.len() == 0),
        QualityFlag.UNRESOLVED_DISEASE,
    )


def validate_biosample(studies: pl.LazyFrame, biosample: pl.LazyFrame) -> pl.LazyFrame:
    """Resolve source biosample IDs and flag unmatched non-GWAS studies."""
    logger.info('Validating study biosamples')
    # Replace a source `biosampleId` (gentropy v4 study index) with the resolved one.
    studies = studies.drop('biosampleId', strict=False)
    lookup = biosample.select(
        pl.col('biosampleId').alias('biosampleFromSourceId'),
        'biosampleId',
        pl.lit(True).alias('_biosample_found'),
    )
    return flag_rows(
        studies.join(lookup, on='biosampleFromSourceId', how='left'),
        (pl.col('studyType') != 'gwas') & pl.col('_biosample_found').is_null(),
        QualityFlag.UNKNOWN_BIOSAMPLE,
    ).drop('_biosample_found')


def validate_analysis_flags(studies: pl.LazyFrame) -> pl.LazyFrame:
    """Flag studies curated as case-case designs; a missing list of analysis flags becomes empty."""
    logger.info('Validating study analysis flags')
    return flag_rows(
        studies.with_columns(pl.col('analysisFlags').fill_null([])),
        pl.col('analysisFlags').list.contains('Case-case study'),
        QualityFlag.CASE_CASE_STUDY_DESIGN,
    )


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
        studies = flag_rows(studies, condition, flag)
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
    # One estimate per study: several successful LDSC runs for a study would otherwise
    # duplicate the study in the join (Gentropy joins them all). Keep one, deterministically.
    successful = (
        heritability
        .filter(pl.col('runStatus') == 'success')
        .sort(['studyId', *fields], nulls_last=True)
        .unique('studyId', keep='first', maintain_order=True)
    )
    annotations = successful.select(
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


def read_superseded_studies(path: str) -> dict[str, str]:
    """Read the curation file of superseded studies into a map from each study to the study superseding it.

    Raises:
        ValueError: when a row lacks either study, or a study is listed more than once.
    """
    rows = scan_dataset(path, format='tsv', schema=superseded_studies_schema, has_header=True).collect()
    if rows.select(pl.any_horizontal(pl.all().is_null() | (pl.all().str.strip_chars() == '')).any()).item():
        msg = f'superseded studies in {path} must name both studies on every row'
        raise ValueError(msg)
    if repeated := rows.filter(pl.col('studyId').is_duplicated())['studyId'].unique().sort().to_list():
        msg = f'superseded studies in {path} list these studies more than once: {repeated}'
        raise ValueError(msg)
    return dict(rows.iter_rows())


def validate_superseded_studies(
    studies: pl.DataFrame,
    superseded_studies: dict[str, str],
    invalid_qc_reasons: list[QualityFlag],
) -> pl.DataFrame:
    """Flag each superseded study whose replacement is present and valid.

    A superseded study absent from `studies` is skipped. One whose replacement is absent or carries
    an invalidating flag is kept, and a warning names it.
    """
    logger.info('Validating superseded studies')
    invalidating = {reason.value for reason in invalid_qc_reasons} - {QualityFlag.SUPERSEDED_STUDY.value}
    flags = dict(zip(studies['studyId'], studies['qualityControls'].fill_null([]).to_list(), strict=True))
    superseded, kept = [], []
    for study_id, replacement in superseded_studies.items():
        if study_id not in flags:
            continue
        if replacement not in flags:
            kept.append(f'{study_id} -> {replacement} (absent)')
        elif failing := sorted(QualityFlag(flag).name for flag in invalidating.intersection(flags[replacement])):
            kept.append(f'{study_id} -> {replacement} ({", ".join(failing)})')
        else:
            superseded.append(study_id)
    if kept:
        logger.warning('Keeping {} studies without a valid replacement: {}', len(kept), '; '.join(kept))
    return flag_rows(studies, pl.col('studyId').is_in(superseded), QualityFlag.SUPERSEDED_STUDY)


@writes(output=Study)
def study(
    source: dict[str, str | list[str]],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Validate and annotate studies, separating accepted and excluded records.

    Args:
        source: Study input paths under `study`, plus `target`, `disease`, and
            `biosample` lookup paths, the `heritability` path and, optionally, the
            curation file of `superseded_studies`.
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
    disease = build_disease_lut(sources.disease).lazy().rename({'diseaseFromSourceMappedId': 'efo'})
    heritability = scan_dataset(sources.heritability)
    superseded = read_superseded_studies(sources.superseded_studies) if sources.superseded_studies else {}
    validated = (
        pl
        .concat([scan_dataset(path) for path in sources.study], how='diagonal_relaxed')
        .pipe(validate_duplicated_studies)
        .pipe(validate_study_type)
        .pipe(validate_project_id, options.deprecated_project_ids)
        .pipe(validate_target, target)
        .pipe(validate_disease, disease)
        .pipe(validate_biosample, biosample)
        .pipe(validate_analysis_flags)
        .pipe(validate_ccs)
        .pipe(collect_heritability, heritability)
        .pipe(sort_study_collections)
        .select(study_schema.names())
        # Sources carry different subsets of columns; check the result against the schema,
        # which also restores its struct field order.
        .match_to_schema(study_schema)
        .collect()
        .pipe(validate_superseded_studies, superseded, options.invalid_qc_reasons)
    )
    # Validate once, then write both outputs from the same frame.
    result = valid_rows(validated.lazy(), options.invalid_qc_reasons)
    logger.info('Writing accepted studies to {} and excluded studies to {}', destinations.output, destinations.excluded)
    pl.collect_all([
        sink_dataset(conform(result.valid, Study), destinations.output),
        sink_dataset(result.invalid, destinations.excluded),
    ])
