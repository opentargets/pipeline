"""PTS task for data-driven, batched LDSC-CTS regression."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
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
    discover_completed_study_paths,
    finite_or_none,
    normalise_chromosome,
    normalise_chromosomes,
    quoted_col,
    read_table,
    reference_path,
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
_ALLOWED_ANALYSIS_FLAGS = {'exwas', 'wgsgwas', 'metabolite'}
_REQUIRED_SUMSTATS_COLUMNS = {
    'studyId',
    'variantId',
    'chromosome',
    'position',
    'beta',
    'standardError',
    'sampleSize',
}


@dataclass(frozen=True)
class _StudyMetadata:
    """Study metadata and preflight state for one summary-statistics dataset."""

    study_id: str
    path: str
    ancestry: str | None
    flags: list[str]
    skip_reasons: list[str]
    n_cases: float | None = None
    n_controls: float | None = None
    n_samples: float | None = None


@dataclass(frozen=True)
class _AnnotationReference:
    """A complete specificity/population annotation reference."""

    specificity_id: str
    population: str
    ld_score_paths: tuple[str, ...]
    m_paths: tuple[str, ...]


def ldsc_cts_regression(
    source: dict[str, str],
    destination: str | dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Run all configured LDSC-CTS regressions in resumable study batches.

    Summary-statistics children are discovered below one root, while completed
    specificity/population references are discovered from the annotation
    catalog. A single Spark application processes each reference in batches and
    runs one grouped Pandas regression per study on Spark workers.
    """
    spark_uri = str(settings.get('spark_uri', 'local[*]'))
    session = Session(
        app_name='ldsc_cts_regression',
        spark_uri=spark_uri,
        properties=properties,
    )
    spark = session.spark

    summary_root = source.get('summary_statistics_root')
    annotation_root = source.get('annotations')
    study_index_path = source.get('study_index')
    reference_root = source.get('reference_root')
    if not summary_root:
        raise ValueError("Regression source is missing 'summary_statistics_root'")
    if not annotation_root:
        raise ValueError("Regression source is missing 'annotations'")
    if not study_index_path:
        raise ValueError("Regression source is missing 'study_index'")
    if not reference_root:
        raise ValueError("Regression source is missing 'reference_root'")

    output = destination['results'] if isinstance(destination, dict) else destination
    if not output:
        raise ValueError('Regression destination is missing a results path')
    output = output.rstrip('/')
    chromosomes = normalise_chromosomes(settings.get('chromosomes'))
    batch_size = int(settings.get('study_batch_size', 8))
    if batch_size <= 0:
        raise ValueError('study_batch_size must be positive')

    entries = discover_completed_study_paths(
        spark=spark,
        root=summary_root,
        study_ids=settings.get('study_ids'),
    )
    discovery_issues: dict[str, str] = {}
    entries = _resolve_root_dataset_ids(spark, entries, discovery_issues)
    study_index = _read_study_index(spark, study_index_path)
    metadata = _build_study_metadata(
        entries=entries,
        study_index=study_index,
        min_samples=int(settings.get('min_samples', 10_000)),
        discovery_issues=discovery_issues,
    )
    references = _read_annotation_catalog(
        spark=spark,
        annotation_root=annotation_root,
        chromosomes=chromosomes,
    )
    completed = _read_completed_keys(spark, output)

    for specificity_id in sorted({reference.specificity_id for reference in references}):
        refs_for_specificity = {
            reference.population: reference
            for reference in references
            if reference.specificity_id == specificity_id
        }
        pending = [
            study
            for study in metadata
            if (study.study_id, specificity_id) not in completed
        ]
        if not pending:
            continue

        preflight_skips: list[dict[str, Any]] = []
        grouped: dict[str, list[_StudyMetadata]] = defaultdict(list)
        for study in pending:
            if study.skip_reasons:
                preflight_skips.append(_skip_record(study, specificity_id, study.skip_reasons))
            elif study.ancestry not in refs_for_specificity:
                preflight_skips.append(
                    _skip_record(
                        study,
                        specificity_id,
                        [
                            f'No complete annotation reference for inferred ancestry '
                            f'{study.ancestry}'
                        ],
                    )
                )
            else:
                grouped[study.ancestry].append(study)

        if preflight_skips:
            _append_results(spark, output, preflight_skips)
            completed.update(
                (record['studyId'], record['specificityId']) for record in preflight_skips
            )

        for population, studies in sorted(grouped.items()):
            reference = refs_for_specificity[population]
            baseline_path = reference_path(
                reference_root,
                population,
                settings,
                'baseline_ld_scores',
                'baseline_ld_scores.tsv.gz',
            )
            baseline_m_path = reference_path(
                reference_root,
                population,
                settings,
                'baseline_m',
                'baseline_m',
            )
            baseline, baseline_key, baseline_columns, baseline_m = _read_baseline(
                spark,
                baseline_path,
                baseline_m_path,
                settings,
            )
            annotation_wide, cell_types = _read_annotations(
                spark,
                list(reference.ld_score_paths),
                specificity_id,
            )
            annotation_m = _read_annotation_m(spark, list(reference.m_paths))
            baseline = baseline.persist()
            annotation_wide = annotation_wide.persist()
            bad_inputs: dict[str, str] = {}
            try:
                for batch in _chunks(studies, batch_size):
                    batch_ids = {study.study_id for study in batch}
                    raw, input_issues = _read_sumstats_batch_safe(
                        spark=spark,
                        entries=[(study.study_id, study.path) for study in batch],
                        root=summary_root,
                    )
                    bad_inputs.update(dict(input_issues))
                    if raw is None:
                        skipped = [
                            _skip_record(
                                study,
                                specificity_id,
                                [bad_inputs.get(study.study_id, 'Malformed summary statistics')],
                            )
                            for study in batch
                        ]
                        _append_results(spark, output, skipped)
                        completed.update(
                            (record['studyId'], record['specificityId']) for record in skipped
                        )
                        continue

                    batch_metadata = _metadata_frame(spark, batch)
                    prepared, invalid_ids = _prepare_sumstats_batch(raw, batch_metadata)
                    for study_id in invalid_ids:
                        bad_inputs[study_id] = 'Summary statistics contain invalid variant coordinates'
                    valid_ids = batch_ids - set(bad_inputs)
                    if not valid_ids:
                        continue
                    prepared = prepared.filter(f.col('studyId').isin(sorted(valid_ids)))
                    result, overlap_ids = _join_and_regress_batch(
                        prepared=prepared,
                        baseline=baseline,
                        baseline_key=baseline_key,
                        baseline_columns=baseline_columns,
                        baseline_m=baseline_m,
                        annotation_wide=annotation_wide,
                        cell_types=cell_types,
                        annotation_m=annotation_m,
                        n_blocks=int(settings.get('n_blocks', 200)),
                        intercept=settings.get('intercept'),
                        max_rows=int(settings.get('max_rows_for_collection', 20_000_000)),
                    )
                    if result is not None:
                        result = _add_result_metadata(
                            result,
                            specificity_id=specificity_id,
                            population=population,
                            metadata=batch_metadata,
                        )
                        _append_results(spark, output, result)
                        completed.update(_collect_result_keys(result))

                    processed_ids = set() if result is None else {
                        row['studyId']
                        for row in result.select('studyId').distinct().collect()
                    }
                    skipped = []
                    for study in batch:
                        if study.study_id not in valid_ids:
                            reason = bad_inputs.get(study.study_id, 'Malformed summary statistics')
                            skipped.append(_skip_record(study, specificity_id, [reason]))
                        elif study.study_id not in overlap_ids:
                            skipped.append(
                                _skip_record(
                                    study,
                                    specificity_id,
                                    ['No overlapping SNPs between summary statistics and LD scores'],
                                )
                            )
                        elif study.study_id not in processed_ids:
                            skipped.append(
                                _skip_record(study, specificity_id, ['No regression rows produced'])
                            )
                    if skipped:
                        _append_results(spark, output, skipped)
                        completed.update(
                            (record['studyId'], record['specificityId']) for record in skipped
                        )
            finally:
                baseline.unpersist()
                annotation_wide.unpersist()


def _chunks(values: list[_StudyMetadata], size: int) -> list[list[_StudyMetadata]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _resolve_root_dataset_ids(
    spark: SparkSession,
    entries: list[tuple[str | None, str]],
    discovery_issues: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    resolved: list[tuple[str, str]] = []
    for candidate, path in entries:
        try:
            raw_ids = [
                row['studyId']
                for row in spark.read.parquet(path).select('studyId').distinct().collect()
            ]
        except Exception as exc:
            if candidate is not None and discovery_issues is not None:
                study_id = str(candidate)
                resolved.append((study_id, path))
                discovery_issues[study_id] = f'Could not validate summary-statistics studyId: {exc}'
                continue
            raise ValueError(
                f'Could not read studyId from completed summary-statistics dataset {path}: {exc}'
            ) from exc
        ids = [str(value) for value in raw_ids if value is not None]
        if len(ids) != 1 or len(raw_ids) != 1:
            if candidate is not None and discovery_issues is not None:
                study_id = str(candidate)
                resolved.append((study_id, path))
                discovery_issues[study_id] = (
                    'Summary-statistics dataset must contain exactly one non-null studyId'
                )
                continue
            raise ValueError(
                f'Completed summary-statistics dataset {path} must contain exactly one studyId'
            )
        study_id = ids[0]
        if candidate is not None and str(candidate) != study_id:
            if discovery_issues is not None:
                candidate_id = str(candidate)
                resolved.append((candidate_id, path))
                discovery_issues[candidate_id] = (
                    f'Summary-statistics directory ID {candidate_id} does not match '
                    f'studyId {study_id}'
                )
                continue
            raise ValueError(
                f'Summary-statistics directory ID {candidate} does not match studyId {study_id} '
                f'in {path}'
            )
        resolved.append((study_id, path))
    duplicate_ids = {
        study_id
        for study_id in {entry[0] for entry in resolved}
        if sum(candidate == study_id for candidate, _ in resolved) > 1
    }
    if duplicate_ids:
        raise ValueError(f'Duplicate summary-statistics datasets for study IDs: {sorted(duplicate_ids)}')
    return resolved


def _read_study_index(spark: SparkSession, path: str) -> DataFrame:
    raw = spark.read.parquet(path)
    required = {'studyId', 'nCases', 'nControls', 'nSamples', 'ldPopulationStructure', 'analysisFlags'}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f'Study index is missing required columns: {sorted(missing)}')
    return raw.select(
        f.col('studyId').cast('string').alias('studyId'),
        'nCases',
        'nControls',
        'nSamples',
        'ldPopulationStructure',
        'analysisFlags',
    )


def _build_study_metadata(
    entries: list[tuple[str, str]],
    study_index: DataFrame,
    min_samples: int,
    discovery_issues: dict[str, str] | None = None,
) -> list[_StudyMetadata]:
    ids = study_index.sparkSession.createDataFrame(
        [(study_id,) for study_id, _ in entries],
        schema=StructType([StructField('studyId', StringType(), False)]),
    )
    rows = study_index.join(ids, on='studyId', how='inner').collect()
    by_id: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        if row['studyId'] is not None:
            by_id[str(row['studyId'])].append(row)

    metadata: list[_StudyMetadata] = []
    for study_id, path in entries:
        rows_for_id = by_id.get(study_id, [])
        discovery_issue = (discovery_issues or {}).get(study_id)
        if not rows_for_id:
            metadata.append(
                _StudyMetadata(
                    study_id=study_id,
                    path=path,
                    ancestry=None,
                    flags=[],
                    skip_reasons=[
                        reason
                        for reason in (
                            discovery_issue,
                            'Study not found in Open Targets study index',
                        )
                        if reason is not None
                    ],
                )
            )
            continue
        if len(rows_for_id) > 1:
            metadata.append(
                _StudyMetadata(
                    study_id=study_id,
                    path=path,
                    ancestry=None,
                    flags=[],
                    skip_reasons=[
                        reason
                        for reason in (
                            discovery_issue,
                            'Study index contains multiple records for studyId',
                        )
                        if reason is not None
                    ],
                )
            )
            continue

        row = rows_for_id[0]
        flags = _normalise_flags(row['analysisFlags'])
        reasons: list[str] = [discovery_issue] if discovery_issue is not None else []
        if not set(flags).issubset(_ALLOWED_ANALYSIS_FLAGS):
            reasons.append('Invalid study design')
        sample_size = _effective_sample_size(row)
        if sample_size is None:
            reasons.append('Sample size missing')
        elif sample_size < min_samples:
            reasons.append('Sample size too small')

        ancestry: str | None = None
        try:
            ancestry = infer_ld_ancestry(row['ldPopulationStructure'])
        except (TypeError, ValueError) as exc:
            reasons.append(str(exc))
            flags.append('ancestry_inference_failed')
        else:
            flags.append(f'ld_ancestry_inferred:{ancestry}')
        metadata.append(
            _StudyMetadata(
                study_id=study_id,
                path=path,
                ancestry=ancestry,
                flags=flags,
                skip_reasons=reasons,
                n_cases=finite_or_none(row['nCases']),
                n_controls=finite_or_none(row['nControls']),
                n_samples=finite_or_none(row['nSamples']),
            )
        )
    return metadata


def _effective_sample_size(row: Any) -> float | None:
    n_samples = finite_or_none(row['nSamples'])
    if n_samples is not None and n_samples > 0:
        return n_samples
    cases = finite_or_none(row['nCases'])
    controls = finite_or_none(row['nControls'])
    if cases is not None and controls is not None and cases > 0 and controls > 0:
        return 4.0 * cases * controls / (cases + controls)
    return None


def _normalise_flags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip().lower()] if value.strip() else []
    return [str(flag).strip().lower() for flag in value if flag is not None]


def _metadata_frame(spark: SparkSession, studies: list[_StudyMetadata]) -> DataFrame:
    schema = StructType([
        StructField('studyId', StringType(), False),
        StructField('analysisFlags', ArrayType(StringType()), False),
        StructField('nCases', DoubleType(), True),
        StructField('nControls', DoubleType(), True),
        StructField('nSamples', DoubleType(), True),
    ])
    return spark.createDataFrame(
        [
            (
                study.study_id,
                study.flags,
                study.n_cases,
                study.n_controls,
                study.n_samples,
            )
            for study in studies
        ],
        schema=schema,
    )


def _read_sumstats_batch(
    spark: SparkSession,
    entries: list[tuple[str, str]],
    root: str,
) -> tuple[DataFrame | None, list[tuple[str, str]]]:
    paths = [path for _, path in entries]
    raw = spark.read.parquet(*paths)
    missing = _REQUIRED_SUMSTATS_COLUMNS - set(raw.columns)
    if missing:
        reason = f'Summary statistics are missing required columns: {sorted(missing)}'
        return None, [(study_id, reason) for study_id, _ in entries]

    pattern = re.escape(root.rstrip('/')) + r'/([^/]+)/'
    source_id = f.regexp_extract(f.input_file_name(), pattern, 1)
    if len(entries) == 1:
        source_id = f.when(source_id == '', f.lit(entries[0][0])).otherwise(source_id)
    raw = raw.withColumn('_sourceStudyId', source_id)
    pairs = raw.select('_sourceStudyId', 'studyId').distinct().collect()
    by_source: dict[str, set[str]] = defaultdict(set)
    for row in pairs:
        source_value = str(row['_sourceStudyId'] or '')
        actual_value = row['studyId']
        by_source[source_value].add(str(actual_value) if actual_value is not None else '')

    issues: list[tuple[str, str]] = []
    invalid_sources: set[str] = set()
    expected = {study_id for study_id, _ in entries}
    for source_value in expected:
        actual = by_source.get(source_value, set())
        if actual != {source_value}:
            invalid_sources.add(source_value)
            issues.append(
                (
                    source_value,
                    f'Summary-statistics dataset contains study IDs {sorted(actual)}; '
                    f'expected exactly {source_value}',
                )
            )
    valid = raw.filter(~f.col('_sourceStudyId').isin(sorted(invalid_sources)))
    return _normalise_sumstats_frame(valid), issues


def _read_sumstats(spark: SparkSession, path: str) -> DataFrame:
    """Read one completed summary-statistics dataset (compatibility helper)."""
    return _normalise_sumstats_frame(spark.read.parquet(path))


def _normalise_sumstats_frame(raw: DataFrame) -> DataFrame:
    missing = _REQUIRED_SUMSTATS_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f'Summary statistics are missing required columns: {sorted(missing)}')
    return raw.select(
        f.col('studyId').cast('string').alias('studyId'),
        f.col('variantId').cast('string').alias('variantId'),
        f.upper(
            f.regexp_replace(f.col('chromosome').cast('string'), r'(?i)^chr', '')
        ).alias('chromosome'),
        f.col('position').cast('long').alias('position'),
        f.col('beta').cast('double').alias('beta'),
        f.col('standardError').cast('double').alias('standardError'),
        f.col('sampleSize').cast('double').alias('sampleSize'),
    )


def _read_sumstats_batch_safe(
    spark: SparkSession,
    entries: list[tuple[str, str]],
    root: str,
) -> tuple[DataFrame | None, list[tuple[str, str]]]:
    """Read a batch, isolating a corrupt dataset when a shared read fails."""
    try:
        frame, batch_issues = _read_sumstats_batch(spark, entries, root)
        if frame is not None or len(entries) == 1:
            return frame, batch_issues
    except Exception as exc:
        if len(entries) == 1:
            return None, [(entries[0][0], f'Could not read summary statistics: {exc}')]
    frames: list[DataFrame] = []
    isolated_issues: list[tuple[str, str]] = []
    for entry in entries:
        try:
            frame, entry_issues = _read_sumstats_batch(spark, [entry], root)
        except Exception as entry_exc:
            isolated_issues.append((entry[0], f'Could not read summary statistics: {entry_exc}'))
            continue
        isolated_issues.extend(entry_issues)
        if frame is not None:
            frames.append(frame)
    if not frames:
        return None, isolated_issues
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.unionByName(frame, allowMissingColumns=True)
    return combined, isolated_issues


def _prepare_sumstats_batch(
    sumstats: DataFrame,
    study_index: DataFrame,
) -> tuple[DataFrame, set[str]]:
    fallback = (
        f.when(
            f.col('nCases').isNotNull()
            & f.col('nControls').isNotNull()
            & (f.col('nCases') > 0)
            & (f.col('nControls') > 0),
            4.0 * f.col('nCases') * f.col('nControls') / (f.col('nCases') + f.col('nControls')),
        )
        .otherwise(f.col('nSamples').cast('double'))
    )
    prepared = (
        sumstats.join(
            study_index.select('studyId', 'nCases', 'nControls', 'nSamples'),
            on='studyId',
            how='left',
        )
        .withColumn('sampleSize', f.coalesce(f.col('sampleSize'), fallback))
    )
    parts = f.split(f.col('variantId'), '_')
    prepared = (
        prepared.withColumn('variantChromosome', f.upper(f.regexp_replace(parts.getItem(0), r'(?i)^chr', '')))
        .withColumn('ref', parts.getItem(2))
        .withColumn('alt', parts.getItem(3))
    )
    invalid_condition = (
        f.col('studyId').isNull()
        | f.col('variantId').isNull()
        | f.col('variantChromosome').isNull()
        | ~f.col('variantChromosome').rlike(r'^(?:[1-9]|1[0-9]|2[0-2])$')
        | f.col('chromosome').isNull()
        | (f.col('chromosome') != f.col('variantChromosome'))
        | f.col('position').isNull()
        | (f.col('position') <= 0)
        | f.col('ref').isNull()
        | f.col('alt').isNull()
    )
    invalid_ids = {
        str(row['studyId'])
        for row in prepared.filter(invalid_condition).select('studyId').distinct().collect()
        if row['studyId'] is not None
    }
    valid = (
        prepared.filter(~f.col('studyId').isin(sorted(invalid_ids)))
        .filter(f.col('beta').isNotNull() & f.col('standardError').isNotNull())
        .filter(~f.isnan('beta') & ~f.isnan('standardError'))
        .drop('variantChromosome', 'nCases', 'nControls', 'nSamples')
        .dropDuplicates(['studyId', 'variantId'])
    )
    return valid, invalid_ids


def _join_and_regress_batch(
    prepared: DataFrame,
    baseline: DataFrame,
    baseline_key: list[str],
    baseline_columns: list[str],
    baseline_m: dict[str, float],
    annotation_wide: DataFrame,
    cell_types: list[str],
    annotation_m: dict[str, float],
    n_blocks: int,
    intercept: float | None,
    max_rows: int,
) -> tuple[DataFrame | None, set[str]]:
    joined = prepared.join(baseline, on=baseline_key, how='inner')
    joined = joined.join(annotation_wide, on='variantId', how='left')
    for column in [CONTROL_ANNOTATION, *cell_types]:
        if column not in joined.columns:
            joined = joined.withColumn(column, f.lit(0.0))
        else:
            value = quoted_col(column)
            joined = joined.withColumn(
                column,
                f.when(f.isnan(value), f.lit(0.0)).otherwise(f.coalesce(value, f.lit(0.0))),
            )
    for column in baseline_columns:
        value = quoted_col(column)
        joined = joined.filter(value.isNotNull() & ~f.isnan(value))
    base_column = next(
        (column for column in baseline_columns if column.lower() in {'base', 'basel2'}),
        None,
    )
    if base_column is not None:
        joined = joined.filter(quoted_col(base_column) > 0)
    joined = joined.filter(
        f.col('beta').isNotNull()
        & f.col('standardError').isNotNull()
        & (f.col('standardError') > 0)
        & f.col('sampleSize').isNotNull()
        & (f.col('sampleSize') > 0)
    ).dropDuplicates(['studyId', 'variantId'])
    overlap_ids = {
        str(row['studyId'])
        for row in joined.select('studyId').distinct().collect()
        if row['studyId'] is not None
    }
    if not overlap_ids:
        return None, set()

    select_columns = [
        'studyId',
        'beta',
        'standardError',
        'sampleSize',
        *baseline_columns,
        CONTROL_ANNOTATION,
        *cell_types,
    ]
    selected = joined.select(*[quoted_col(column).alias(column) for column in select_columns])

    def regress_group(pdf: pd.DataFrame) -> pd.DataFrame:
        study_id = str(pdf['studyId'].iloc[0])
        if len(pdf) > max_rows:
            return pd.DataFrame([_group_skip_row(study_id, 'Too many joined SNPs for collection')])
        if not cell_types:
            return pd.DataFrame([_group_skip_row(study_id, 'No focal specificity annotations')])
        try:
            rows = _regress_cell_types(
                pdf=pdf,
                baseline_columns=baseline_columns,
                baseline_m=baseline_m,
                cell_types=cell_types,
                annotation_m=annotation_m,
                n_blocks=n_blocks,
                intercept=intercept,
            )
        except Exception as exc:  # pragma: no cover - defensive worker boundary
            return pd.DataFrame([_group_skip_row(study_id, f'Regression failed: {exc}')])
        status = 'success' if all(row['cellTypeStatus'] == 'success' for row in rows) else 'partial_failure'
        return pd.DataFrame(
            [
                {
                    'studyId': study_id,
                    'cellType': row['cellType'],
                    'runStatus': status,
                    'cellTypeStatus': row['cellTypeStatus'],
                    'skipReasons': [],
                    'coefficient': row['coefficient'],
                    'coefficient_se': row['coefficient_se'],
                    'coefficient_z': row['coefficient_z'],
                    'pvalue': row['pvalue'],
                    'h2': row['h2'],
                    'intercept': row['intercept'],
                    'mean_chisq': row['mean_chisq'],
                    'lambda_gc': row['lambda_gc'],
                    'n_snps_used': row['n_snps_used'],
                }
                for row in rows
            ]
        )

    metrics = selected.groupBy('studyId').applyInPandas(
        regress_group,
        schema=_GROUP_RESULT_SCHEMA,
    )
    return metrics, overlap_ids


def _group_skip_row(study_id: str, reason: str) -> dict[str, Any]:
    return {
        'studyId': study_id,
        'cellType': None,
        'runStatus': 'skipped',
        'cellTypeStatus': None,
        'skipReasons': [reason],
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


_GROUP_RESULT_SCHEMA = StructType([
    StructField('studyId', StringType(), False),
    StructField('cellType', StringType(), True),
    StructField('runStatus', StringType(), False),
    StructField('cellTypeStatus', StringType(), True),
    StructField('skipReasons', ArrayType(StringType()), False),
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


def _add_result_metadata(
    result: DataFrame,
    specificity_id: str,
    population: str,
    metadata: DataFrame,
) -> DataFrame:
    return (
        result.join(metadata, on='studyId', how='left')
        .withColumn('analysisId', f.col('studyId'))
        .withColumn('specificityId', f.lit(specificity_id))
        .withColumn('ld_ancestry', f.lit(population))
        .select(*_RESULT_SCHEMA.fieldNames())
    )


_RESULT_SCHEMA = StructType([
    StructField('studyId', StringType(), False),
    StructField('analysisId', StringType(), False),
    StructField('specificityId', StringType(), False),
    StructField('cellType', StringType(), True),
    StructField('runStatus', StringType(), False),
    StructField('cellTypeStatus', StringType(), True),
    StructField('skipReasons', ArrayType(StringType()), False),
    StructField('analysisFlags', ArrayType(StringType()), False),
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


def _skip_record(
    study: _StudyMetadata,
    specificity_id: str,
    reasons: list[str],
) -> dict[str, Any]:
    return {
        'studyId': study.study_id,
        'analysisId': study.study_id,
        'specificityId': specificity_id,
        'cellType': None,
        'runStatus': 'skipped',
        'cellTypeStatus': None,
        'skipReasons': reasons,
        'analysisFlags': study.flags,
        'ld_ancestry': study.ancestry,
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


def _append_results(spark: SparkSession, path: str, records: DataFrame | list[dict[str, Any]]) -> None:
    if isinstance(records, list):
        if not records:
            return
        frame = spark.createDataFrame(records, schema=_RESULT_SCHEMA)
    else:
        frame = records
    frame.select(*_RESULT_SCHEMA.fieldNames()).write.mode('append').partitionBy('specificityId').parquet(path)


def _collect_result_keys(result: DataFrame) -> set[tuple[str, str]]:
    return {
        (str(row['studyId']), str(row['specificityId']))
        for row in result.select('studyId', 'specificityId').distinct().collect()
    }


def _read_completed_keys(spark: SparkSession, path: str) -> set[tuple[str, str]]:
    if not success_exists(spark, path):
        return set()
    raw = spark.read.parquet(path)
    required = {'studyId', 'specificityId'}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f'Existing LDSC-CTS result dataset is missing columns: {sorted(missing)}')
    return {
        (str(row['studyId']), str(row['specificityId']))
        for row in raw.select('studyId', 'specificityId').distinct().collect()
    }


def _read_annotation_catalog(
    spark: SparkSession,
    annotation_root: str,
    chromosomes: list[str],
) -> list[_AnnotationReference]:
    path = f'{annotation_root.rstrip("/")}/_catalog'
    if not success_exists(spark, path):
        raise ValueError(f'Annotation catalog is not a completed dataset: {path}')
    raw = spark.read.parquet(path)
    required = {
        'specificityId',
        'ldPopulation',
        'chromosome',
        'ldScoresPath',
        'mAnnotPath',
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f'Annotation catalog is missing columns: {sorted(missing)}')
    grouped: dict[tuple[str, str], dict[str, tuple[str, str]]] = defaultdict(dict)
    for row in raw.collect():
        key = (str(row['specificityId']), str(row['ldPopulation']).lower())
        chromosome = normalise_chromosome(row['chromosome'])
        if chromosome in grouped[key]:
            raise ValueError(f'Duplicate annotation catalog entry for {key}/chr{chromosome}')
        ld_path = row['ldScoresPath']
        m_path = row['mAnnotPath']
        if not isinstance(ld_path, str) or not isinstance(m_path, str) or not ld_path or not m_path:
            raise ValueError(f'Annotation catalog has invalid paths for {key}/chr{chromosome}')
        grouped[key][chromosome] = (ld_path, m_path)

    references: list[_AnnotationReference] = []
    for (specificity_id, population), values in sorted(grouped.items()):
        if any(chromosome not in values for chromosome in chromosomes):
            logger.warning(
                'Ignoring incomplete annotation reference {}/{}; requested chromosomes are {}',
                specificity_id,
                population,
                chromosomes,
            )
            continue
        selected = [values[chromosome] for chromosome in chromosomes]
        if any(
            not success_exists(spark, path)
            for pair in selected
            for path in pair
        ):
            raise ValueError(f'Annotation catalog reference {specificity_id}/{population} has incomplete outputs')
        references.append(
            _AnnotationReference(
                specificity_id=specificity_id,
                population=population,
                ld_score_paths=tuple(pair[0] for pair in selected),
                m_paths=tuple(pair[1] for pair in selected),
            )
        )
    if not references:
        raise ValueError('Annotation catalog has no complete references for requested chromosomes')
    return references


def _read_baseline(
    spark: SparkSession,
    path: str,
    m_path: str,
    settings: dict[str, Any],
) -> tuple[DataFrame, list[str], list[str], dict[str, float]]:
    if not success_exists(spark, m_path):
        raise ValueError(f'Baseline M input is not a completed parquet dataset: {m_path}')
    raw = read_table(
        spark,
        path,
        fmt=str(settings.get('baseline_format', 'csv')),
        sep=str(settings.get('baseline_sep', '\t')),
    )
    if 'variantId' not in raw.columns:
        aliases = {'CHR': 'chromosome', 'BP_hg38': 'position', 'BP': 'position'}
        for source_column, target_column in aliases.items():
            if source_column in raw.columns and target_column not in raw.columns:
                raw = raw.withColumnRenamed(source_column, target_column)
    key = ['variantId'] if 'variantId' in raw.columns else ['chromosome', 'position', 'ref', 'alt']
    if key != ['variantId'] and not set(key).issubset(raw.columns):
        raise ValueError('Baseline LD scores need variantId or chromosome/position/ref/alt')
    if key == ['variantId']:
        raw = raw.withColumn('variantId', f.col('variantId').cast('string'))
    else:
        raw = (
            raw.withColumn('chromosome', f.upper(f.regexp_replace(f.col('chromosome').cast('string'), r'(?i)^chr', '')))
            .withColumn('position', f.col('position').cast('long'))
        )
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
    m_values = _collect_m_values(
        spark.read.parquet(m_path)
        .groupBy('annotation')
        .agg(f.sum(f.col('M').cast('double')).alias('M'))
    )
    default_m = float(selected.count())
    return selected, key, columns, {column: m_values.get(column, default_m) for column in columns}


def _read_annotations(
    spark: SparkSession,
    paths: list[str],
    specificity_id: str,
) -> tuple[DataFrame, list[str]]:
    raw = spark.read.parquet(*paths)
    required = {'variantId', 'annotation', 'ldScore'}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f'Annotation LD scores are missing columns: {sorted(missing)}')
    if 'specificityId' in raw.columns:
        raw = raw.filter(f.col('specificityId') == specificity_id)
    annotations = sorted(
        row['annotation']
        for row in raw.select('annotation').distinct().collect()
        if row['annotation'] is not None
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
    return _collect_m_values(
        raw.groupBy('annotation').agg(f.sum(f.col('M').cast('double')).alias('M'))
    )


def _collect_m_values(grouped: DataFrame) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in grouped.collect():
        value = finite_or_none(row['M'])
        if row['annotation'] is None or value is None or value < 0:
            raise ValueError('Annotation M values must be finite and non-negative')
        values[str(row['annotation'])] = value
    return values


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


def _prepare_sumstats(sumstats: DataFrame, study_row: Any) -> DataFrame:
    """Prepare a single-study dataframe (kept as a compatibility helper)."""
    fallback = _effective_sample_size(study_row)
    parts = f.split(f.col('variantId'), '_')
    variant_chromosome = f.upper(
        f.regexp_replace(parts.getItem(0), r'(?i)^chr', '')
    )
    prepared = (
        sumstats.withColumn(
            'sampleSize',
            f.coalesce(f.col('sampleSize'), f.lit(fallback))
            if fallback is not None
            else f.col('sampleSize'),
        )
        .withColumn('variantChromosome', variant_chromosome)
        .withColumn('ref', parts.getItem(2))
        .withColumn('alt', parts.getItem(3))
    )
    invalid = (
        f.col('chromosome').isNull()
        | f.col('position').isNull()
        | (f.col('position') <= 0)
        | (f.col('chromosome') != f.col('variantChromosome'))
        | f.col('ref').isNull()
        | f.col('alt').isNull()
    )
    if prepared.filter(invalid).limit(1).count():
        raise ValueError('Summary statistics contain genome-build-incompatible variant coordinates')
    return (
        prepared.filter(f.col('beta').isNotNull() & f.col('standardError').isNotNull())
        .filter(~f.isnan('beta') & ~f.isnan('standardError'))
        .drop('variantChromosome')
        .dropDuplicates(['variantId'])
    )
