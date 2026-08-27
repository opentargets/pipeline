"""PTS task for cell-type-specific LDSC regression of one GWAS."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pyspark.sql.functions as f
from gentropy.method.ldsc import infer_ld_ancestry, run_ldsc_cts_from_arrays
from gentropy.method.ldsc.cell_type_annotation import CONTROL_ANNOTATION
from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.common.session import Session
from pts.pyspark.ldsc_cts_utils import (
    finite_or_none,
    normalise_chromosome,
    quoted_col,
    read_table,
    success_exists,
)

_KEY_AND_META = {
    'variantId',
    'chromosome',
    'position',
    'ref',
    'alt',
    'referenceAllele',
    'alternateAllele',
    'SNP',
    'CM',
    'BP',
    'BP_hg38',
    'CHR',
    'MAF',
}
_AVAILABLE_REFERENCE_POPULATIONS = {'nfe', 'eas'}


def ldsc_cts_regression(
    source: dict[str, str],
    destination: str | dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Run CTS regressions for one GWAS and one specificity dataset."""
    effective_properties = {
        'spark.driver.memory': '30g',
        'spark.executor.memory': '30g',
        'spark.driver.memoryOverhead': '3g',
        'spark.executor.memoryOverhead': '3g',
        'spark.sql.shuffle.partitions': '1024',
        'spark.default.parallelism': '16',
        'spark.dynamicAllocation.enabled': 'false',
        'spark.sql.adaptive.enabled': 'true',
        'spark.sql.adaptive.coalescePartitions.enabled': 'false',
        **properties,
    }
    spark_uri = str(settings.get('spark_uri', 'local[4]'))
    session = Session(
        app_name='ldsc_cts_regression',
        spark_uri=spark_uri,
        properties=effective_properties,
    )
    spark = session.spark
    output = destination['results'] if isinstance(destination, dict) else destination
    if success_exists(spark, output):
        logger.info('Skipping completed LDSC-CTS regression: {}', output)
        return

    specificity_id = str(settings['specificity_id'])
    analysis_id = str(settings.get('analysis_id', specificity_id))
    sumstats = _read_sumstats(spark, source['summary_statistics'])
    study_id = _extract_single_study_id(sumstats)
    expected_study_id = settings.get('study_id')
    if expected_study_id is not None and study_id != expected_study_id:
        raise ValueError(
            f'Summary-statistics studyId {study_id} does not match configured '
            f'study_id {expected_study_id}'
        )

    study_index = _read_study_index(spark, source['study_index'])
    study_matches = study_index.filter(f.col('studyId') == study_id).limit(2).collect()
    if not study_matches:
        _write_results(
            spark,
            output,
            study_id,
            analysis_id,
            specificity_id,
            None,
            ['study_not_found_in_index'],
            'skipped',
            ['Study not found in Open Targets study index'],
            [],
        )
        return
    if len(study_matches) > 1:
        raise ValueError(f'Study index contains multiple records for studyId {study_id}')
    study_row = study_matches[0]

    flags = _normalise_flags(study_row['analysisFlags'])
    skip_reasons: list[str] = []
    if not set(flags).issubset({'exwas', 'wgsgwas', 'metabolite'}):
        skip_reasons.append('Invalid study design')

    n_samples = _effective_sample_size(study_row)
    min_samples = int(settings.get('min_samples', 10_000))
    if n_samples is None or n_samples < min_samples:
        skip_reasons.append('Sample size missing' if n_samples is None else 'Sample size too small')

    ancestry: str | None = None
    try:
        ancestry = infer_ld_ancestry(study_row['ldPopulationStructure'])
    except (TypeError, ValueError) as exc:
        skip_reasons.append(str(exc))
        flags = [*flags, 'ancestry_inference_failed']
    else:
        flags = [*flags, f'ld_ancestry_inferred:{ancestry}']

    if skip_reasons:
        _write_results(
            spark,
            output,
            study_id,
            analysis_id,
            specificity_id,
            ancestry,
            flags,
            'skipped',
            skip_reasons,
            [],
        )
        return

    assert ancestry is not None
    if ancestry not in _AVAILABLE_REFERENCE_POPULATIONS:
        _write_results(
            spark,
            output,
            study_id,
            analysis_id,
            specificity_id,
            ancestry,
            [*flags, 'unsupported_ld_ancestry'],
            'skipped',
            [f'No LDSC-CTS reference artifacts are available for ancestry {ancestry}'],
            [],
        )
        return

    chromosomes = list(
        dict.fromkeys(
            normalise_chromosome(c)
            for c in settings.get('chromosomes', range(1, 23))
        )
    )
    annotation_root = source['annotation_root'].rstrip('/')
    annotation_paths = [
        f'{annotation_root}/{ancestry}/chr{chromosome}/ld_scores' for chromosome in chromosomes
    ]
    m_paths = [
        f'{annotation_root}/{ancestry}/chr{chromosome}/m_annot' for chromosome in chromosomes
    ]
    baseline_root = source['baseline_root'].rstrip('/')
    baseline_path = f'{baseline_root}/{ancestry}/baseline_ld_scores.tsv.gz'
    baseline_m_path = f'{baseline_root}/{ancestry}/baseline_m'

    try:
        prepared = _prepare_sumstats(sumstats, study_row)
        baseline, baseline_key, baseline_columns, baseline_m = _read_baseline(
            spark, baseline_path, baseline_m_path, settings
        )
        annotation_wide, cell_types = _read_annotations(
            spark, annotation_paths, specificity_id, settings
        )
        annotation_m = _read_annotation_m(spark, m_paths)
        joined = prepared.join(baseline, on=baseline_key, how='inner')
        joined = joined.join(annotation_wide, on='variantId', how='left')
        for column in [CONTROL_ANNOTATION, *cell_types]:
            if column not in joined.columns:
                joined = joined.withColumn(column, f.lit(0.0))
            else:
                joined = joined.withColumn(column, f.coalesce(quoted_col(column), f.lit(0.0)))

        joined = joined.dropDuplicates(['variantId'])
        joined = joined.filter(
            f.col('beta').isNotNull()
            & f.col('standardError').isNotNull()
            & (f.col('standardError') > 0)
            & f.col('sampleSize').isNotNull()
            & (f.col('sampleSize') > 0)
        )
        n_rows = joined.count()
        if n_rows == 0:
            raise _SkipRegressionError('No overlapping SNPs between summary statistics and LD scores')
        max_rows = int(settings.get('max_rows_for_collection', 20_000_000))
        if n_rows > max_rows:
            raise _SkipRegressionError('Too many joined SNPs for collection')

        select_columns = [
            'beta',
            'standardError',
            'sampleSize',
            *baseline_columns,
            CONTROL_ANNOTATION,
            *cell_types,
        ]
        pdf = joined.select(*[quoted_col(column).alias(column) for column in select_columns]).toPandas()
        rows = _regress_cell_types(
            pdf=pdf,
            baseline_columns=baseline_columns,
            baseline_m=baseline_m,
            cell_types=cell_types,
            annotation_m=annotation_m,
            n_blocks=int(settings.get('n_blocks', 200)),
            intercept=settings.get('intercept'),
        )
    except _SkipRegressionError as exc:
        _write_results(
            spark,
            output,
            study_id,
            analysis_id,
            specificity_id,
            ancestry,
            flags,
            'skipped',
            [str(exc)],
            [],
        )
        return

    _write_results(
        spark,
        output,
        study_id,
        analysis_id,
        specificity_id,
        ancestry,
        flags,
        'success',
        [],
        rows,
    )


class _SkipRegressionError(Exception):
    """Expected data-availability condition represented in the output."""


def _read_sumstats(spark: SparkSession, path: str) -> DataFrame:
    raw = spark.read.parquet(path)
    required = {'studyId', 'variantId', 'chromosome', 'position', 'beta', 'standardError', 'sampleSize'}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f'Summary statistics are missing required columns: {sorted(missing)}')
    return raw.select(
        'studyId',
        f.col('variantId').cast('string').alias('variantId'),
        f.upper(
            f.regexp_replace(f.col('chromosome').cast('string'), r'(?i)^chr', '')
        ).alias(
            'chromosome'
        ),
        f.col('position').cast('long').alias('position'),
        f.col('beta').cast('double').alias('beta'),
        f.col('standardError').cast('double').alias('standardError'),
        f.col('sampleSize').cast('double').alias('sampleSize'),
    )


def _read_study_index(spark: SparkSession, path: str) -> DataFrame:
    raw = spark.read.parquet(path)
    required = {'studyId', 'nCases', 'nControls', 'nSamples', 'ldPopulationStructure', 'analysisFlags'}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f'Study index is missing required columns: {sorted(missing)}')
    return raw.select(
        'studyId',
        'nCases',
        'nControls',
        'nSamples',
        'ldPopulationStructure',
        'analysisFlags',
    )


def _extract_single_study_id(sumstats: DataFrame) -> str:
    ids = [row['studyId'] for row in sumstats.select('studyId').distinct().collect() if row['studyId']]
    if len(ids) != 1:
        raise ValueError(f'Expected exactly one studyId, found {ids}')
    return str(ids[0])


def _effective_sample_size(row: Any) -> float | None:
    n_samples = finite_or_none(row['nSamples'])
    if n_samples is not None and n_samples > 0:
        return n_samples
    if row['nCases'] is not None and row['nControls'] is not None:
        cases = finite_or_none(row['nCases'])
        controls = finite_or_none(row['nControls'])
        if cases is None or controls is None:
            return None
        if cases > 0 and controls > 0:
            return 4.0 * cases * controls / (cases + controls)
    return None


def _normalise_flags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip().lower()] if value.strip() else []
    return [str(flag).strip().lower() for flag in value if flag is not None]


def _prepare_sumstats(sumstats: DataFrame, study_row: Any) -> DataFrame:
    fallback = _effective_sample_size(study_row)
    parts = f.split(f.col('variantId'), '_')
    variant_chromosome = f.upper(
        f.regexp_replace(parts.getItem(0), r'(?i)^chr', '')
    )
    sample_size = f.coalesce(f.col('sampleSize'), f.lit(fallback)) if fallback is not None else f.col('sampleSize')
    prepared = sumstats.withColumn('sampleSize', sample_size)
    prepared = prepared.withColumn('ref', parts.getItem(2)).withColumn('alt', parts.getItem(3))
    invalid_coordinates = prepared.filter(
        f.col('chromosome').isNull()
        | f.col('position').isNull()
        | (f.col('position') <= 0)
        | (f.col('chromosome') != variant_chromosome)
        | f.col('ref').isNull()
        | f.col('alt').isNull()
    ).limit(1).count()
    if invalid_coordinates:
        raise ValueError('Summary statistics contain genome-build-incompatible variant coordinates')
    return (
        prepared.filter(f.col('beta').isNotNull() & f.col('standardError').isNotNull())
        .filter(~f.isnan('beta') & ~f.isnan('standardError'))
        .dropDuplicates(['variantId'])
    )


def _read_baseline(
    spark: SparkSession,
    path: str,
    m_path: str,
    settings: dict[str, Any],
) -> tuple[DataFrame, list[str], list[str], dict[str, float]]:
    raw = read_table(
        spark,
        path,
        fmt=str(settings.get('baseline_format', 'csv')),
        sep=str(settings.get('baseline_sep', '\t')),
    )
    if 'variantId' not in raw.columns:
        aliases = {
            'CHR': 'chromosome',
            'BP_hg38': 'position',
            'BP': 'position',
        }
        for source_column, target_column in aliases.items():
            if source_column in raw.columns and target_column not in raw.columns:
                raw = raw.withColumnRenamed(source_column, target_column)
    key = ['variantId'] if 'variantId' in raw.columns else ['chromosome', 'position', 'ref', 'alt']
    if key != ['variantId'] and not set(key).issubset(raw.columns):
        raise ValueError('Baseline LD scores need variantId or chromosome/position/ref/alt')
    if key == ['variantId']:
        raw = raw.withColumn('variantId', f.col('variantId').cast('string'))
    else:
        raw = raw.withColumn(
            'chromosome',
            f.upper(f.regexp_replace(f.col('chromosome').cast('string'), r'(?i)^chr', '')),
        ).withColumn('position', f.col('position').cast('long'))
    configured = settings.get('baseline_annotation_columns')
    columns = list(configured) if configured else [column for column in raw.columns if column not in _KEY_AND_META]
    if not columns:
        raise ValueError('Baseline LD scores contain no annotation columns')
    missing = set(columns) - set(raw.columns)
    if missing:
        raise ValueError(f'Baseline LD scores are missing columns: {sorted(missing)}')
    selected = raw.select(
        *key,
        *[quoted_col(column).cast('double').alias(column) for column in columns],
    ).dropDuplicates(key)
    m_values = {}
    if m_path:
        m_values = {
            row['annotation']: float(row['M'])
            for row in spark.read.parquet(m_path)
            .groupBy('annotation')
            .agg(f.sum(f.col('M').cast('double')).alias('M'))
            .collect()
        }
    default_m = float(selected.count())
    return selected, key, columns, {column: m_values.get(column, default_m) for column in columns}


def _read_annotations(
    spark: SparkSession,
    paths: list[str],
    specificity_id: str,
    settings: dict[str, Any],
) -> tuple[DataFrame, list[str]]:
    raw = spark.read.parquet(*paths)
    required = {'variantId', 'annotation', 'ldScore'}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f'Annotation LD scores are missing columns: {sorted(missing)}')
    if 'specificityId' in raw.columns:
        raw = raw.filter(f.col('specificityId') == specificity_id)
    annotations = sorted(
        row['annotation'] for row in raw.select('annotation').distinct().collect()
    )
    if CONTROL_ANNOTATION not in annotations:
        raise ValueError(f'Annotation LD scores do not contain {CONTROL_ANNOTATION}')
    cell_types = [annotation for annotation in annotations if annotation != CONTROL_ANNOTATION]
    return (
        raw.select('variantId', 'annotation', f.col('ldScore').cast('double').alias('ldScore'))
        .groupBy('variantId')
        .pivot('annotation', annotations)
        .agg(f.first('ldScore')),
        cell_types,
    )


def _read_annotation_m(spark: SparkSession, paths: list[str]) -> dict[str, float]:
    raw = spark.read.parquet(*paths)
    required = {'annotation', 'M'}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f'Annotation M values are missing columns: {sorted(missing)}')
    return {
        row['annotation']: float(row['M'])
        for row in raw.groupBy('annotation')
        .agg(f.sum(f.col('M').cast('double')).alias('M'))
        .collect()
    }


def _regress_cell_types(
    pdf: pd.DataFrame,
    baseline_columns: list[str],
    baseline_m: dict[str, float],
    cell_types: list[str],
    annotation_m: dict[str, float],
    n_blocks: int,
    intercept: float | None,
) -> list[dict[str, Any]]:
    beta = pdf['beta'].to_numpy(dtype=float)
    se = pdf['standardError'].to_numpy(dtype=float)
    sample_size = pdf['sampleSize'].to_numpy(dtype=float)
    baseline = pdf[baseline_columns].to_numpy(dtype=float)
    control = pdf[CONTROL_ANNOTATION].to_numpy(dtype=float)
    base_column = next((column for column in baseline_columns if column.lower() in {'base', 'basel2'}), None)
    weights = pdf[base_column].to_numpy(dtype=float) if base_column else baseline.sum(axis=1)
    rows: list[dict[str, Any]] = []
    for cell_type in cell_types:
        ref_ld = np.column_stack([baseline, control, pdf[cell_type].to_numpy(dtype=float)])
        m_annot = np.asarray(
            [
                *(baseline_m[column] for column in baseline_columns),
                annotation_m.get(CONTROL_ANNOTATION, float(len(control))),
                annotation_m.get(cell_type, float('nan')),
            ],
            dtype=float,
        )
        try:
            result = run_ldsc_cts_from_arrays(
                beta=beta,
                se=se,
                N=sample_size,
                ref_ld=ref_ld,
                w_ld=weights,
                M_annot=m_annot,
                focal_index=-1,
                intercept=intercept,
                n_blocks=n_blocks,
            )
            rows.append({
                'cellType': cell_type,
                'coefficient': result['coefficient'],
                'coefficient_se': result['coefficient_se'],
                'coefficient_z': result['coefficient_z'],
                'pvalue': result['coefficient_p_value'],
                'h2': result['h2'],
                'intercept': result['intercept'],
                'mean_chisq': result['mean_chisq'],
                'lambda_gc': result['lambda_gc'],
                'n_snps_used': int(result['n_snps']),
                'cellTypeStatus': 'success',
            })
        except Exception as exc:
            rows.append({
                'cellType': cell_type,
                'coefficient': None,
                'coefficient_se': None,
                'coefficient_z': None,
                'pvalue': None,
                'h2': None,
                'intercept': None,
                'mean_chisq': None,
                'lambda_gc': None,
                'n_snps_used': len(beta),
                'cellTypeStatus': f'failed: {exc}',
            })
    return rows


def _write_results(
    spark: SparkSession,
    path: str,
    study_id: str,
    analysis_id: str,
    specificity_id: str,
    ancestry: str | None,
    flags: list[str],
    status: str,
    reasons: list[str],
    rows: list[dict[str, Any]],
) -> None:
    schema = StructType([
        StructField('studyId', StringType(), True),
        StructField('analysisId', StringType(), True),
        StructField('specificityId', StringType(), True),
        StructField('cellType', StringType(), True),
        StructField('runStatus', StringType(), True),
        StructField('cellTypeStatus', StringType(), True),
        StructField('skipReasons', ArrayType(StringType()), True),
        StructField('analysisFlags', ArrayType(StringType()), True),
        StructField('ld_ancestry', StringType(), True),
        StructField('coefficient', DoubleType(), True),
        StructField('coefficient_se', DoubleType(), True),
        StructField('coefficient_z', DoubleType(), True),
        StructField('pvalue', DoubleType(), True),
        StructField('h2', DoubleType(), True),
        StructField('intercept', DoubleType(), True),
        StructField('mean_chisq', DoubleType(), True),
        StructField('lambda_gc', DoubleType(), True),
        StructField('n_snps_used', LongType(), True),
    ])
    records = rows or [
        {
            'cellType': None,
            'coefficient': None,
            'coefficient_se': None,
            'coefficient_z': None,
            'pvalue': None,
            'h2': None,
            'intercept': None,
            'mean_chisq': None,
            'lambda_gc': None,
            'n_snps_used': None,
        }
    ]
    output = [
        {
            'studyId': study_id,
            'analysisId': analysis_id,
            'specificityId': specificity_id,
            'cellType': row.get('cellType'),
            'runStatus': status,
            'cellTypeStatus': row.get('cellTypeStatus'),
            'skipReasons': reasons,
            'analysisFlags': flags,
            'ld_ancestry': ancestry,
            'coefficient': finite_or_none(row.get('coefficient')),
            'coefficient_se': finite_or_none(row.get('coefficient_se')),
            'coefficient_z': finite_or_none(row.get('coefficient_z')),
            'pvalue': finite_or_none(row.get('pvalue')),
            'h2': finite_or_none(row.get('h2')),
            'intercept': finite_or_none(row.get('intercept')),
            'mean_chisq': finite_or_none(row.get('mean_chisq')),
            'lambda_gc': finite_or_none(row.get('lambda_gc')),
            'n_snps_used': row.get('n_snps_used'),
        }
        for row in records
    ]
    spark.createDataFrame(output, schema=schema).write.mode('overwrite').parquet(path)
