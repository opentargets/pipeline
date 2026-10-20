"""Enhancer-to-gene annotation and validation."""

from enum import StrEnum
from typing import Any, Self

import polars as pl
import polars_hash as plh
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pts.schemas.enhancer_to_gene import enhancer_to_gene_schema
from pts.transformers.utils.dataset import scan_dataset, sink_dataset
from pts.transformers.utils.quality_flags import flag_rows, resolve_flag_names, valid_rows
from pts.transformers.utils.target import tss_lut

PMID = '38014075'
DATASOURCE_ID = 'E2G'
DNASE_SCORE_NAME = 'DNase'
HIC_SCORE_NAME = 'HiC_contacts'
#: Columns hashed, in this order, into `intervalId`.
IDENTIFIER_COLUMNS = ('chromosome', 'start', 'end', 'geneId', 'studyId', 'intervalType')


class QualityFlag(StrEnum):
    """Quality control flags recorded on enhancer-to-gene evidence."""

    UNRESOLVED_TARGET = 'Target/gene identifier could not match to reference'
    UNKNOWN_BIOSAMPLE = 'Biosample identifier was not found in the reference'
    SCORE_OUTSIDE_BOUNDS = 'Score was above or below specified thresholds'
    UNKNOWN_INTERVAL_TYPE = 'Interval type is not supported'
    AMBIGUOUS_SCORE = 'Interval has a duplicate with different score'
    INVALID_CHROMOSOME = 'Interval chromosome was not found in contig index'
    INVALID_RANGE = 'Interval range exceeded chromosome bounds'
    AMBIGUOUS_INTERVAL_TYPE = 'Multiple interval types for the same (region, geneId) pair'
    TRANS_CHROMOSOMAL_EFFECT = 'Interval chromosome does not match associated gene chromosome'


class EnhancerToGeneSource(BaseModel):
    """Input paths for enhancer-to-gene annotation."""

    model_config = ConfigDict(extra='forbid')

    enhancer_to_gene: str
    biosample_lut: str
    target: str
    biosample: str
    chromosome: str


class EnhancerToGeneDestination(BaseModel):
    """Output paths for accepted and excluded enhancer-to-gene records."""

    model_config = ConfigDict(extra='forbid')

    output: str
    excluded: str


class EnhancerToGeneSettings(BaseModel):
    """Settings for enhancer-to-gene validation."""

    model_config = ConfigDict(extra='forbid')

    min_valid_score: float = Field(default=0.6, allow_inf_nan=False)
    max_valid_score: float = Field(default=1.0, allow_inf_nan=False)
    invalid_qc_reasons: list[QualityFlag] = Field(default_factory=list)

    @field_validator('invalid_qc_reasons', mode='before')
    @classmethod
    def resolve_qc_names(cls, value: Any) -> Any:
        """Resolve configured enum names before validating the QC flags."""
        return resolve_flag_names(QualityFlag, value)

    @model_validator(mode='after')
    def validate_score_bounds(self) -> Self:
        """Require an ordered, finite score range."""
        if not self.min_valid_score <= self.max_valid_score:
            raise ValueError('min_valid_score must be less than or equal to max_valid_score')
        return self


def read_enhancer_to_gene(path: str) -> pl.LazyFrame:
    """Scan headered TSV evidence, unifying source column names across files."""
    logger.info('Reading enhancer-to-gene evidence from {}', path)
    parent, _, name = path.rpartition('/')
    if any(char in name for char in '*?['):
        paths = sorted(StorageHandle(parent or '.').glob(name))
        if not paths:
            raise ValueError(f'{path!r} matched no enhancer-to-gene files')
    else:
        paths = [path]
    frames = []
    for file in paths:
        raw = scan_dataset(file, format='tsv', infer_schema=False, has_header=True, include_file_paths='filePath')
        chromosome = '#chr' if '#chr' in raw.collect_schema() else 'chr'
        frames.append(
            raw.select(
                pl.col(chromosome).alias('chromosome'),
                'start',
                'end',
                pl.col('TargetGeneEnsemblID').alias('geneId'),
                pl.col('CellType').alias('biosampleName'),
                pl.col('Score').alias('score'),
                pl.col('class').alias('intervalType'),
                pl.col('normalizedDNase_prom.Feature').alias('dnase'),
                pl.col('3DContact.Feature').alias('hicContacts'),
                'filePath',
            )
        )
    return pl.concat(frames, how='vertical')


def parse_enhancer_to_gene(
    raw: pl.LazyFrame,
    biosample_mapping: pl.LazyFrame,
    target: pl.LazyFrame,
) -> pl.LazyFrame:
    """Parse unified evidence and annotate biosamples, TSS distances and interval IDs."""
    logger.info('Parsing enhancer-to-gene evidence')
    chromosome = pl.col('chromosome').str.strip_prefix('chr').replace({'M': 'MT', '23': 'X', '24': 'Y'})
    base = (
        raw
        .with_columns(
            chromosome.alias('chromosome'),
            pl.col('start').cast(pl.Int64, strict=False),
            pl.col('end').cast(pl.Int64, strict=False),
            pl.col('score').cast(pl.Float64, strict=False),
            pl.col('intervalType').str.strip_chars().str.to_lowercase(),
            pl.col('filePath').str.extract(r'([^/]+)\.bed\.gz$', 1).fill_null('').alias('studyId'),
        )
        .join(tss_lut(target), on='geneId', how='left')
        .join(
            biosample_mapping.select('biosampleName', 'biosampleId'),
            on='biosampleName',
            how='left',
        )
    )
    distance = (
        pl
        .when((pl.col('intervalType') == 'promoter') | pl.col('tss').is_between(pl.col('start'), pl.col('end')))
        .then(0)
        .otherwise(pl.min_horizontal((pl.col('tss') - pl.col('start')).abs(), (pl.col('tss') - pl.col('end')).abs()))
        .cast(pl.Int32, strict=False)
    )
    # Gentropy hashed the columns concatenated without a separator, so different intervals
    # could collide (chromosome '1' + start '12' == chromosome '11' + start '2'). The
    # separator intentionally changes every intervalId relative to Gentropy.
    identifier = plh.concat_str(
        [pl.col(column).cast(pl.String).fill_null('None') for column in IDENTIFIER_COLUMNS],
        separator='|',
    ).nchash.md5()
    return base.select(
        'chromosome',
        'start',
        'end',
        'geneId',
        'score',
        distance.alias('distanceToTss'),
        pl.concat_list(
            pl.struct(
                pl.lit(DNASE_SCORE_NAME).alias('name'),
                pl.col('dnase').cast(pl.Float32, strict=False).alias('value'),
            ),
            pl.struct(
                pl.lit(HIC_SCORE_NAME).alias('name'),
                pl.col('hicContacts').cast(pl.Float32, strict=False).alias('value'),
            ),
        ).alias('resourceScore'),
        pl.lit(DATASOURCE_ID).alias('datasourceId'),
        'intervalType',
        pl.lit(PMID).alias('pmid'),
        'biosampleName',
        pl.lit(None, dtype=pl.String).alias('biosampleFromSourceId'),
        'biosampleId',
        'studyId',
        identifier.alias('intervalId'),
        pl.lit([], dtype=pl.List(pl.String)).alias('qualityControls'),
    ).match_to_schema(enhancer_to_gene_schema)


def validate_interval_range(frame: pl.LazyFrame, contig: pl.LazyFrame) -> pl.LazyFrame:
    """Flag noncanonical chromosomes and bounds outside their canonical contig."""
    logger.info('Validating enhancer-to-gene chromosome bounds')
    chromosomes = [str(value) for value in range(1, 23)] + ['X', 'Y', 'MT']
    lookup = contig.filter(pl.col('id').is_in(chromosomes)).select(
        pl.col('id').alias('chromosome'),
        pl.col('start').alias('_contig_start'),
        pl.col('end').alias('_contig_end'),
    )
    result = flag_rows(
        frame.join(lookup, on='chromosome', how='left'),
        pl.col('_contig_start').is_null() | pl.col('_contig_end').is_null(),
        QualityFlag.INVALID_CHROMOSOME,
    )
    return flag_rows(
        result,
        (pl.col('start') < pl.col('_contig_start')) | (pl.col('end') > pl.col('_contig_end')),
        QualityFlag.INVALID_RANGE,
    ).drop('_contig_start', '_contig_end')


def validate_target(frame: pl.LazyFrame, target: pl.LazyFrame) -> pl.LazyFrame:
    """Flag unknown genes and interval chromosomes differing from their genomic location."""
    logger.info('Validating enhancer-to-gene targets')
    lookup = target.select(
        pl.col('id').alias('geneId'),
        pl.col('genomicLocation').struct.field('chromosome').alias('_gene_chromosome'),
        pl.lit(True).alias('_target_found'),
    )
    result = flag_rows(
        frame.join(lookup, on='geneId', how='left'), pl.col('_target_found').is_null(), QualityFlag.UNRESOLVED_TARGET
    )
    return flag_rows(
        result, pl.col('chromosome') != pl.col('_gene_chromosome'), QualityFlag.TRANS_CHROMOSOMAL_EFFECT
    ).drop('_target_found', '_gene_chromosome')


def validate_biosample(frame: pl.LazyFrame, biosample: pl.LazyFrame) -> pl.LazyFrame:
    """Flag biosample IDs absent from the reference index."""
    logger.info('Validating enhancer-to-gene biosamples')
    lookup = biosample.select('biosampleId', pl.lit(True).alias('_biosample_found'))
    return flag_rows(
        frame.join(lookup, on='biosampleId', how='left'),
        pl.col('_biosample_found').is_null(),
        QualityFlag.UNKNOWN_BIOSAMPLE,
    ).drop('_biosample_found')


def validate_interval_type(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag unsupported types and multiple non-null types for the same region and gene."""
    logger.info('Validating enhancer-to-gene interval types')
    types = ['promoter', 'enhancer', 'intragenic', 'intergenic', 'genic']
    result = flag_rows(frame, ~pl.col('intervalType').is_in(types), QualityFlag.UNKNOWN_INTERVAL_TYPE)
    ambiguous = pl.col('intervalType').drop_nulls().n_unique().over(['chromosome', 'start', 'end', 'geneId']) > 1
    return flag_rows(result, ambiguous, QualityFlag.AMBIGUOUS_INTERVAL_TYPE)


def validate_score(frame: pl.LazyFrame, min_score: float, max_score: float) -> pl.LazyFrame:
    """Flag null scores and scores outside the inclusive bounds."""
    logger.info('Validating enhancer-to-gene scores in [{}, {}]', min_score, max_score)
    return flag_rows(
        frame,
        ~pl.col('score').is_between(min_score, max_score) | pl.col('score').is_null(),
        QualityFlag.SCORE_OUTSIDE_BOUNDS,
    )


def validate_id_has_unique_score(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Flag interval identities with more than one distinct non-null score."""
    logger.info('Validating enhancer-to-gene score uniqueness')
    columns = ['chromosome', 'start', 'end', 'biosampleId', 'geneId', 'studyId', 'intervalType']
    return flag_rows(frame, pl.col('score').drop_nulls().n_unique().over(columns) > 1, QualityFlag.AMBIGUOUS_SCORE)


def enhancer_to_gene(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Parse enhancer-to-gene evidence and separate accepted and excluded records.

    Args:
        source: Evidence glob under `enhancer_to_gene`, plus `biosample_lut`,
            `target`, `biosample`, and `chromosome` lookup paths.
        destination: Dataset paths under `output` and `excluded`.
        settings: Score bounds and QC reason names under `invalid_qc_reasons`.
        config: Pipeline configuration.
    """
    sources = EnhancerToGeneSource.model_validate(source)
    destinations = EnhancerToGeneDestination.model_validate(destination)
    options = EnhancerToGeneSettings.model_validate(settings)
    target = scan_dataset(sources.target)
    biosample = scan_dataset(sources.biosample)
    contig = scan_dataset(sources.chromosome)
    biosample_mapping = scan_dataset(sources.biosample_lut, format='csv', has_header=True)
    validated = (
        read_enhancer_to_gene(sources.enhancer_to_gene)
        .pipe(parse_enhancer_to_gene, biosample_mapping, target)
        .pipe(validate_interval_range, contig)
        .pipe(validate_target, target)
        .pipe(validate_biosample, biosample)
        .pipe(validate_interval_type)
        .pipe(validate_score, options.min_valid_score, options.max_valid_score)
        .pipe(validate_id_has_unique_score)
        .select(enhancer_to_gene_schema.names())
        .collect()
    )
    # Validate once (every rE2G file is read once), then write both outputs from the frame.
    result = valid_rows(validated.lazy(), options.invalid_qc_reasons)
    logger.info(
        'Writing accepted enhancer-to-gene records to {} and excluded records to {}',
        destinations.output,
        destinations.excluded,
    )
    pl.collect_all([
        sink_dataset(result.valid.sort('chromosome', 'start', nulls_last=False), destinations.output),
        sink_dataset(result.invalid.sort('chromosome', 'start', nulls_last=False), destinations.excluded),
    ])
