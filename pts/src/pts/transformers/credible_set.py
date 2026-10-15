"""Credible set validation and annotation."""

from enum import StrEnum
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from pydantic import BaseModel, ConfigDict, Field, field_validator

from pts.schemas.credible_set import credible_set_schema
from pts.transformers.utils.dataset import scan_dataset, scan_datasets, write_dataset
from pts.transformers.utils.quality_flags import valid_rows


class QualityFlag(StrEnum):
    """Quality control flags recorded on credible sets."""

    SUBSIGNIFICANT_FLAG = 'Subsignificant p-value'
    NO_GENOMIC_LOCATION_FLAG = 'Incomplete genomic mapping'
    COMPOSITE_FLAG = 'Composite association'
    INCONSISTENCY_FLAG = 'Variant inconsistency'
    NON_MAPPED_VARIANT_FLAG = 'No mapping in GnomAd'
    PALINDROMIC_ALLELE_FLAG = 'Palindrome alleles - cannot harmonize'
    AMBIGUOUS_STUDY = 'Association with ambiguous study'
    UNRESOLVED_LD = 'Variant not found in LD reference'
    LD_CLUMPED = 'Explained by a more significant variant in high LD'
    WINDOW_CLUMPED = 'Explained by a more significant variant in the same window'
    NO_POPULATION = 'Study does not have population annotation to resolve LD'
    FLAGGED_STUDY = 'Study has quality control flag(s)'
    MISSING_STUDY = 'Study not found in the study index'
    DUPLICATED_STUDYLOCUS_ID = 'Non-unique study locus identifier'
    INVALID_VARIANT_IDENTIFIER = 'Some variant identifiers of this locus were not found in variant index'
    IN_MHC = 'MHC region'
    REDUNDANT_PICS_TOP_HIT = 'PICS results from summary statistics available for this same study'
    TOP_HIT = 'Study locus from curated top hit'
    EXPLAINED_BY_SUSIE = 'Study locus in region explained by a SuSiE credible set'
    OUT_OF_SAMPLE_LD = 'Study locus finemapped without in-sample LD reference'
    ABNORMAL_PIPS = 'Study locus with a sum of PIPs that not in the expected range [0.95,1]'
    INVALID_CHROMOSOME = 'Chromosome not in 1:22, X, Y, XY or MT'
    TOP_HIT_AND_SUMMARY_STATS = 'Curated top hit is flagged because summary statistics are available for study'
    REPLICATED = 'Lead variant is replicated in an independent study'


class CredibleInterval(StrEnum):
    """Credible interval membership fields."""

    IS95 = 'is95CredibleSet'
    IS99 = 'is99CredibleSet'


class FinemappingMethod(StrEnum):
    """Fine-mapping method identifiers."""

    PICS = 'PICS'
    SUSIE = 'SuSie'
    SUSIE_INF = 'SuSiE-inf'


class CredibleSetConfidenceClasses(StrEnum):
    """Confidence labels for credible sets."""

    FINEMAPPED_IN_SAMPLE_LD = 'SuSiE fine-mapped credible set with in-sample LD'
    FINEMAPPED_OUT_OF_SAMPLE_LD = 'SuSiE fine-mapped credible set with out-of-sample LD'
    PICSED_SUMMARY_STATS = 'PICS fine-mapped credible set extracted from summary statistics'
    PICSED_TOP_HIT = 'PICS fine-mapped credible set based on reported top hit'
    UNKNOWN = 'Unknown confidence'


class CredibleSetSource(BaseModel):
    """Input dataset paths for credible set validation."""

    model_config = ConfigDict(extra='forbid')

    credible_set: list[str]
    study: str
    target: str


class CredibleSetDestination(BaseModel):
    """Output dataset paths for accepted and excluded credible sets."""

    model_config = ConfigDict(extra='forbid')

    output: str
    excluded: str


class CredibleSetSettings(BaseModel):
    """Settings for credible set validation."""

    model_config = ConfigDict(extra='forbid')

    trans_qtl_threshold: int = Field(ge=0)
    """Genomic distance above which a QTL is considered trans."""
    invalid_qc_reasons: list[QualityFlag] = Field(default_factory=list)
    """QC reason names that cause a credible set to be excluded."""

    @field_validator('invalid_qc_reasons', mode='before')
    @classmethod
    def resolve_qc_names(cls, value: Any) -> Any:
        """Resolve configured enum names before validating the QC flags."""
        if isinstance(value, list):
            return [QualityFlag.__members__.get(item, item) if isinstance(item, str) else item for item in value]
        return value


def _contains_flag(column: str, flag: str) -> pl.Expr:
    """Preserve nullable array membership when QC lists contain null elements."""
    values = pl.col(column)
    contains = values.list.contains(flag)
    return (
        pl
        .when(contains)
        .then(True)
        .when(values.list.eval(pl.element().is_null()).list.any())
        .then(None)
        .otherwise(contains)
    )


def _flag(frame: pl.LazyFrame, condition: pl.Expr, flag: QualityFlag) -> pl.LazyFrame:
    """Append a QC flag and normalize null QC lists."""
    qc = pl.col('qualityControls').fill_null([])
    return frame.with_columns(
        pl
        .when(condition)
        .then(pl.concat_list(qc, pl.lit([flag.value])).list.unique().list.sort())
        .otherwise(qc)
        .alias('qualityControls')
    )


def qc_mhc_region(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag lead variants within the inclusive MHC region."""
    logger.info('Flagging credible sets in the MHC region')
    return _flag(
        frame, (pl.col('chromosome') == '6') & pl.col('position').is_between(25_726_063, 33_400_556), QualityFlag.IN_MHC
    )


def validate_chromosome_label(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag unsupported chromosome labels, leaving null labels unflagged."""
    logger.info('Validating credible set chromosome labels')
    chromosomes = [str(value) for value in range(1, 23)] + ['X', 'Y', 'XY', 'MT']
    return _flag(frame, ~pl.col('chromosome').is_in(chromosomes), QualityFlag.INVALID_CHROMOSOME)


def validate_study(frame: pl.LazyFrame, study: pl.LazyFrame) -> pl.LazyFrame:
    """Flag missing or flagged studies and top hits with available summary statistics."""
    logger.info('Validating credible set studies')
    qc = (
        pl.col('qualityControls')
        if 'qualityControls' in study.collect_schema()
        else pl.lit(None, dtype=pl.List(pl.String))
    )
    lookup = study.select('studyId', qc.alias('_study_qc'), pl.lit(True).alias('_study_found'))
    result = frame.join(lookup, on='studyId', how='left')
    result = _flag(result, pl.col('_study_qc').list.len() > 0, QualityFlag.FLAGGED_STUDY)
    result = _flag(
        result,
        _contains_flag('qualityControls', QualityFlag.TOP_HIT.value)
        & ~_contains_flag('_study_qc', 'Harmonized summary statistics are not available or empty'),
        QualityFlag.TOP_HIT_AND_SUMMARY_STATS,
    )
    return _flag(result, pl.col('_study_found').is_null(), QualityFlag.MISSING_STUDY).drop('_study_qc', '_study_found')


def annotate_study_type(frame: pl.LazyFrame, study: pl.LazyFrame) -> pl.LazyFrame:
    """Replace input study types with reference study types."""
    logger.info('Annotating credible set study types')
    return frame.drop('studyType').join(study.select('studyId', 'studyType'), on='studyId', how='left')


def qc_redundant_top_hits_from_pics(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag top hits in studies having PICS results derived from summary statistics."""
    logger.info('Flagging redundant PICS top hits')
    lookup = (
        frame
        .filter(pl.col('finemappingMethod') == FinemappingMethod.PICS.value)
        .group_by('studyId')
        .agg((~_contains_flag('qualityControls', QualityFlag.TOP_HIT.value)).any().alias('_pics_summary'))
    )
    return _flag(
        frame.join(lookup, on='studyId', how='left'),
        _contains_flag('qualityControls', QualityFlag.TOP_HIT.value) & pl.col('_pics_summary'),
        QualityFlag.REDUNDANT_PICS_TOP_HIT,
    ).drop('_pics_summary')


def qc_explained_by_susie(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag non-SuSiE sets with any tag inside a same-study, same-chromosome SuSiE region."""
    logger.info('Flagging credible sets explained by SuSiE')
    methods = [FinemappingMethod.SUSIE.value, FinemappingMethod.SUSIE_INF.value]
    regions = (
        frame
        .filter(pl.col('finemappingMethod').is_in(methods))
        .select('studyId', 'chromosome', 'locusStart', 'locusEnd')
        .unique()
    )
    tags = (
        frame
        .filter(~pl.col('finemappingMethod').is_in(methods))
        .select('studyLocusId', 'studyId', 'chromosome', 'locus')
        .explode('locus')
        .with_columns(
            pl
            .col('locus')
            .struct.field('variantId')
            .str.split('_')
            .list.get(1, null_on_oob=True)
            .cast(pl.Int64, strict=False)
            .alias('_tag_position')
        )
        .filter(pl.col('locus').is_not_null())
    )
    overlaps = (
        tags
        .join(regions, on=['studyId', 'chromosome'], how='inner')
        .filter(pl.col('_tag_position').is_between(pl.col('locusStart'), pl.col('locusEnd')))
        .select('studyLocusId')
        .unique()
        .with_columns(pl.lit(True).alias('_in_susie'))
    )
    return _flag(
        frame.join(overlaps, on='studyLocusId', how='left'),
        pl.col('_in_susie') & ~pl.col('finemappingMethod').is_in(methods),
        QualityFlag.EXPLAINED_BY_SUSIE,
    ).drop('_in_susie')


def filter_credible_set(frame: pl.LazyFrame, credible_interval: CredibleInterval) -> pl.LazyFrame:
    """Sort tags by PIP and retain the interval, including its threshold-crossing tag."""
    logger.info('Filtering credible sets to {}', credible_interval.name)
    sorted_locus = pl.col('locus').list.eval(
        pl.element().sort_by(pl.element().struct.field('posteriorProbability'), descending=True, nulls_last=True)
    )
    result = frame.with_columns(
        pl.when(pl.col('locus').list.len() > 0).then(sorted_locus).otherwise(None).alias('locus')
    )
    pip = pl.element().struct.field('posteriorProbability')
    preceding = pip.cum_sum().shift(1, fill_value=0.0)
    no_null_before = pip.is_null().cast(pl.Int32).cum_sum().shift(1, fill_value=0) == 0
    annotated = (
        pl
        .when(pl.element().is_not_null())
        .then(
            pl.element().struct.with_fields(
                ((preceding < 0.95) & no_null_before).fill_null(False).alias(CredibleInterval.IS95.value),
                ((preceding < 0.99) & no_null_before).fill_null(False).alias(CredibleInterval.IS99.value),
            )
        )
        .otherwise(None)
    )
    return result.with_columns(
        pl
        .col('locus')
        .list.eval(annotated)
        .list.eval(pl.element().filter(pl.element().struct.field(credible_interval.value)))
    )


def qc_abnormal_pips(
    frame: pl.LazyFrame, sum_pips_lower_threshold: float = 0.99, sum_pips_upper_threshold: float = 1.0001
) -> pl.LazyFrame:
    """Flag PIP totals outside the bounds; a null tag PIP makes the total unknown."""
    logger.info('Validating credible set PIP totals')
    pips = pl.col('locus').list.eval(pl.element().struct.field('posteriorProbability'))
    total = pl.when(pips.list.eval(pl.element().is_null()).list.any()).then(None).otherwise(pips.list.sum())
    return _flag(
        frame, (total < sum_pips_lower_threshold) | (total > sum_pips_upper_threshold), QualityFlag.ABNORMAL_PIPS
    )


def assign_confidence(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Assign confidence from the fine-mapping method and provenance flags."""
    logger.info('Assigning credible set confidence')
    susie = pl.col('finemappingMethod').is_in([FinemappingMethod.SUSIE.value, FinemappingMethod.SUSIE_INF.value])
    pics = pl.col('finemappingMethod') == FinemappingMethod.PICS.value
    out_of_sample = _contains_flag('qualityControls', QualityFlag.OUT_OF_SAMPLE_LD.value)
    top_hit = _contains_flag('qualityControls', QualityFlag.TOP_HIT.value)
    return frame.with_columns(
        pl
        .when(susie & ~out_of_sample)
        .then(pl.lit(CredibleSetConfidenceClasses.FINEMAPPED_IN_SAMPLE_LD.value))
        .when(susie & out_of_sample)
        .then(pl.lit(CredibleSetConfidenceClasses.FINEMAPPED_OUT_OF_SAMPLE_LD.value))
        .when(pics & ~top_hit)
        .then(pl.lit(CredibleSetConfidenceClasses.PICSED_SUMMARY_STATS.value))
        .when(pics & top_hit)
        .then(pl.lit(CredibleSetConfidenceClasses.PICSED_TOP_HIT.value))
        .otherwise(pl.lit(CredibleSetConfidenceClasses.UNKNOWN.value))
        .alias('confidence')
    )


def flag_trans_qtls(
    frame: pl.LazyFrame, study: pl.LazyFrame, target: pl.LazyFrame, trans_threshold: int = 5_000_000
) -> pl.LazyFrame:
    """Annotate a QTL as trans only when every tag is trans; GWAS sets remain null."""
    logger.info('Annotating trans QTLs with distance threshold {}', trans_threshold)
    if 'geneId' not in study.collect_schema():
        return frame
    studies = study.filter(pl.col('studyType') != 'gwas').select('studyId', 'geneId')
    targets = target.select(
        pl.col('id').alias('geneId'),
        'tss',
        pl.col('genomicLocation').struct.field('chromosome').alias('_gene_chromosome'),
    )
    variant = pl.col('locus').struct.field('variantId').str.split('_')
    tags = frame.select('studyLocusId', 'studyId', 'locus').explode('locus', empty_as_null=False, keep_nulls=False)
    tags = (
        tags
        .with_columns(
            variant.list.get(0, null_on_oob=True).alias('_tag_chromosome'),
            variant.list.get(1, null_on_oob=True).cast(pl.Int64, strict=False).alias('_tag_position'),
        )
        .join(studies, on='studyId', how='inner')
        .join(targets, on='geneId', how='left')
    )
    trans = (
        (pl.col('_tag_chromosome') != pl.col('_gene_chromosome'))
        | ((pl.col('tss') - pl.col('_tag_position')).abs() > trans_threshold)
    ).fill_null(False)
    annotations = tags.group_by('studyLocusId').agg(trans.all().alias('isTransQtl'))
    return frame.drop('isTransQtl').join(annotations, on='studyLocusId', how='left')


def validate_unique_study_locus_id(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag duplicate IDs after other exclusions, keeping the smallest credible set.

    Equal-sized sets use a stable content hash so separate output executions agree.
    """
    logger.info('Validating credible set identifier uniqueness')
    ranked = (
        frame
        .with_row_index('_input_order')
        .with_columns(
            pl.col('locus').list.len().fill_null(-1).alias('_locus_size'),
            pl.struct(frame.collect_schema().names()).hash(seed=0).alias('_tie_breaker'),
        )
        .sort(['_locus_size', '_tie_breaker', '_input_order'])
        .with_columns(pl.int_range(pl.len()).over('studyLocusId').alias('_duplicate_rank'))
    )
    return (
        _flag(ranked, pl.col('_duplicate_rank') > 0, QualityFlag.DUPLICATED_STUDYLOCUS_ID)
        .sort('_input_order')
        .drop('_input_order', '_locus_size', '_tie_breaker', '_duplicate_rank')
    )


def qc_replication(frame: pl.LazyFrame, study: pl.LazyFrame) -> pl.LazyFrame:
    """Flag surviving loci replicated on the full disease set or the measured gene."""
    logger.info('Assessing replication among surviving credible sets')
    loci = frame.select('studyLocusId', 'studyId', 'variantId')
    gwas_studies = study.filter((pl.col('studyType') == 'gwas') & (pl.col('diseaseIds').list.len() > 0)).select(
        'studyId',
        pl.col('diseaseIds').list.unique().list.sort().alias('_disease_set'),
        'cohorts',
        'pubmedId',
        'ldPopulationStructure',
    )
    gwas = loci.join(gwas_studies, on='studyId', how='inner')
    independent = gwas.select('variantId', '_disease_set', 'cohorts', 'pubmedId', 'ldPopulationStructure').unique()
    replicated_gwas = (
        independent
        .group_by('variantId', '_disease_set')
        .len()
        .filter(pl.col('len') >= 2)
        .select('variantId', '_disease_set')
    )
    gwas_ids = gwas.join(replicated_gwas, on=['variantId', '_disease_set'], how='inner').select('studyLocusId')
    qtl_studies = study.filter((pl.col('studyType') != 'gwas') & pl.col('geneId').is_not_null()).select(
        'studyId', 'geneId'
    )
    qtl = loci.join(qtl_studies, on='studyId', how='inner')
    replicated_qtl = qtl.group_by('variantId', 'geneId').len().filter(pl.col('len') >= 2).select('variantId', 'geneId')
    qtl_ids = qtl.join(replicated_qtl, on=['variantId', 'geneId'], how='inner').select('studyLocusId')
    replicated = pl.concat([gwas_ids, qtl_ids]).unique().with_columns(pl.lit(True).alias('_replicated'))
    return _flag(
        frame.join(replicated, on='studyLocusId', how='left'),
        pl.col('_replicated').is_not_null(),
        QualityFlag.REPLICATED,
    ).drop('_replicated')


def credible_set(
    source: dict[str, str | list[str]],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Validate and annotate credible sets, separating accepted and excluded records.

    Args:
        source: Credible set input paths under `credible_set`, plus `study` and
            `target` lookup paths.
        destination: Dataset paths under `output` and `excluded`.
        settings: Genomic distance under `trans_qtl_threshold` and QC reason
            names under `invalid_qc_reasons`.
        config: Pipeline configuration.
    """
    sources = CredibleSetSource.model_validate(source)
    destinations = CredibleSetDestination.model_validate(destination)
    options = CredibleSetSettings.model_validate(settings)
    study = scan_dataset(sources.study)
    target = scan_dataset(sources.target)
    result = (
        scan_datasets(sources.credible_set, schema=credible_set_schema)
        .pipe(qc_mhc_region)
        .pipe(validate_chromosome_label)
        .pipe(validate_study, study)
        .pipe(annotate_study_type, study)
        .pipe(qc_redundant_top_hits_from_pics)
        .pipe(qc_explained_by_susie)
        .pipe(filter_credible_set, CredibleInterval.IS95)
        .pipe(qc_abnormal_pips, sum_pips_lower_threshold=0.95, sum_pips_upper_threshold=1.0001)
        .pipe(assign_confidence)
        .pipe(flag_trans_qtls, study, target, options.trans_qtl_threshold)
        .select(credible_set_schema.names())
        .pipe(valid_rows, options.invalid_qc_reasons)
    )
    unique = result.valid.pipe(validate_unique_study_locus_id).pipe(valid_rows, options.invalid_qc_reasons)
    accepted = (
        unique.valid
        .pipe(qc_replication, study)
        .select(credible_set_schema.names())
        .sort('chromosome', 'position', nulls_last=False)
    )
    excluded = pl.concat([result.invalid, unique.invalid.select(credible_set_schema.names())], how='vertical')
    logger.info('Writing accepted credible sets to {}', destinations.output)
    write_dataset(accepted, destinations.output)
    logger.info('Writing excluded credible sets to {}', destinations.excluded)
    write_dataset(excluded, destinations.excluded)
