"""Release-level metrics generation in Polars."""

from __future__ import annotations

import re
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import polars as pl
from huggingface_hub import HfApi
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.utils import load_spark_schema_as_polars

ASSOCIATION_MINIMAL_SCHEMA: dict[str, Any] = {
    'diseaseId': pl.String,
    'targetId': pl.String,
    'datasourceId': pl.String,
    'aggregationType': pl.String,
    'aggregationValue': pl.String,
}

EVIDENCE_OUTPUT_PATTERN = '/output/evidence_*'
EVIDENCE_EXCLUDED_PATTERN = '/excluded/evidence/*'


def _empty_metrics_frame() -> pl.DataFrame:
    """Metrics frame following the canonical output schema."""
    return pl.DataFrame(
        schema={
            'datasourceId': pl.String,
            'variable': pl.String,
            'field': pl.String,
            'value': pl.Int64,
            'runId': pl.String,
        }
    )


def _document_total_count(df: pl.DataFrame, variable: str) -> pl.DataFrame:
    """Build a single total-count metric row from a dataframe."""
    return pl.DataFrame({'datasourceId': ['all'], 'variable': [variable], 'field': [None], 'value': [df.height]})


def _document_total_value(value: int, variable: str) -> pl.DataFrame:
    """Build a single total-count metric row from a scalar value."""
    return pl.DataFrame({'datasourceId': ['all'], 'variable': [variable], 'field': [None], 'value': [value]})


def _document_count_by(df: pl.DataFrame, column: str, variable: str) -> pl.DataFrame:
    """Build count-by-column metrics with datasource-compatible output columns."""
    return (
        df
        .group_by(column)
        .len()
        .rename({column: 'datasourceId', 'len': 'value'})
        .with_columns(variable=pl.lit(variable), field=pl.lit(None, dtype=pl.String))
        .select('datasourceId', 'variable', 'field', 'value')
    )


def _not_null_fields_count_top_level(df: pl.DataFrame, variable: str, group_by_datasource: bool) -> pl.DataFrame:
    """Compute non-null counts using only top-level dataframe columns."""
    columns_to_count = list(df.columns)
    if group_by_datasource:
        columns_to_count = [column for column in columns_to_count if column != 'datasourceId']

    count_exprs = [pl.col(column).is_not_null().sum().alias(column) for column in columns_to_count]

    if group_by_datasource:
        aggregated = df.group_by('datasourceId').agg(count_exprs)
        id_vars = ['datasourceId']
    else:
        aggregated = df.select(count_exprs)
        id_vars = []

    value_columns = [column for column in aggregated.columns if column not in id_vars]
    melted = aggregated.unpivot(
        on=value_columns,
        index=id_vars,
        variable_name='field',
        value_name='value',
    ).with_columns(variable=pl.lit(variable))

    if not group_by_datasource:
        melted = melted.with_columns(datasourceId=pl.lit('all'))

    return melted.select('datasourceId', 'variable', 'field', 'value')


def _distinct_fields_count_top_level(df: pl.DataFrame, variable: str) -> pl.DataFrame:
    """Compute distinct counts using only top-level dataframe columns."""
    value_columns = [column for column in df.columns if column != 'datasourceId']
    unique_exprs = [pl.col(column).n_unique().alias(column) for column in value_columns]

    aggregated = df.group_by('datasourceId').agg(unique_exprs)
    melted = aggregated.unpivot(
        on=[column for column in aggregated.columns if column != 'datasourceId'],
        index=['datasourceId'],
        variable_name='field',
        value_name='value',
    )

    return melted.with_columns(variable=pl.lit(variable)).select('datasourceId', 'variable', 'field', 'value')


def _get_evidence_columns_to_report(dataset_columns: list[str]) -> list[str]:
    """Return the evidence fields used for backward-compatible distinct metrics."""
    return [
        'datasourceId',
        'targetFromSourceId',
        'diseaseFromSourceMappedId' if 'diseaseFromSourceMappedId' in dataset_columns else 'diseaseFromSourceId',
        'drugId',
        'literature',
    ]


def _metric_prefix(dataset_rel_path: str) -> str:
    """Map dataset path names to metric variable name prefixes."""
    dataset_name = Path(dataset_rel_path).name
    aliases = {
        'association_by_datasource_direct': 'associationsDirect',
        'association_by_datasource_indirect': 'associationsIndirect',
        'disease': 'diseases',
        'target': 'targets',
        'drug_molecule': 'drugs',
    }
    alias = aliases.get(dataset_name)
    if alias:
        return alias

    parts = dataset_name.split('_')
    return parts[0] + ''.join(part.capitalize() for part in parts[1:])


def _global_rich_metrics(df: pl.DataFrame, metric_prefix: str) -> list[pl.DataFrame]:
    """Build rich top-level metrics (total, not-null, distinct) for a dataset."""
    global_df = df.with_columns(datasourceId=pl.lit('all'))
    return [
        _document_total_count(df, f'{metric_prefix}TotalCount'),
        _not_null_fields_count_top_level(global_df, f'{metric_prefix}NotNullCount', group_by_datasource=False),
        _distinct_fields_count_top_level(global_df, f'{metric_prefix}DistinctFieldsCount'),
    ]


def _metric_label_token(label: str) -> str:
    """Normalize arbitrary QC labels into deterministic metric-name tokens."""
    parts = [part for part in re.split(r'[^A-Za-z0-9]+', label) if part]
    token = ''.join(part.lower().capitalize() for part in parts)
    if token and token[0].isdigit():
        return f'Flag{token}'
    return token or 'Unknown'


def _quality_control_flag_total_metrics(df: pl.DataFrame, metric_prefix: str) -> list[pl.DataFrame]:
    """Build per-flag total-count metrics when a qualityControls column is present."""
    if 'qualityControls' not in df.columns:
        return []

    qc_dtype = df.schema['qualityControls']
    if isinstance(qc_dtype, pl.List):
        qc_counts = (
            df
            .select(pl.col('qualityControls').explode().alias('qc_flag'))
            .filter(pl.col('qc_flag').is_not_null())
            .group_by('qc_flag')
            .len()
            .sort('len', descending=True)
        )
    else:
        qc_counts = (
            df
            .select(pl.col('qualityControls').alias('qc_flag'))
            .filter(pl.col('qc_flag').is_not_null())
            .group_by('qc_flag')
            .len()
            .sort('len', descending=True)
        )

    if qc_counts.height == 0:
        return []

    metrics: list[pl.DataFrame] = []
    for label, count in qc_counts.iter_rows():
        token = _metric_label_token(str(label))
        metrics.append(_document_total_value(int(count), f'{metric_prefix}{token}TotalCount'))
    return metrics


def _build_run_id(ot_release: str) -> str:
    """Build a normalised run identifier from release labels.

    Expected canonical format is ``YY.MM-(ppp|pub).N-YYYY-MM-DD``. Legacy values are normalised when possible.
    """
    release_timestamp = datetime.today().strftime('%Y-%m-%d')
    canonical_pattern = re.compile(r'^(?P<yy>\d{2})\.(?P<mm>\d{2})-(?P<channel>ppp|pub)\.(?P<run>\d+)$')
    legacy_test_pattern = re.compile(r'^(?P<yy>\d{2})\.(?P<mm>\d{2})-test(?P<run>\d+)$')

    run_release_normalised = ot_release

    canonical_match = canonical_pattern.match(ot_release)
    if canonical_match:
        run_release_normalised = ot_release
    else:
        legacy_test_match = legacy_test_pattern.match(ot_release)
        if legacy_test_match:
            run_release_normalised = (
                f'{legacy_test_match.group("yy")}.{legacy_test_match.group("mm")}-pub.{legacy_test_match.group("run")}'
            )
        elif ot_release.startswith('partners/'):
            partner_release = ot_release.split('/', maxsplit=1)[1]
            canonical_partner_match = canonical_pattern.match(partner_release)
            if canonical_partner_match:
                run_release_normalised = (
                    f'{canonical_partner_match.group("yy")}.{canonical_partner_match.group("mm")}'
                    f'-ppp.{canonical_partner_match.group("run")}'
                )
            else:
                legacy_partner_match = legacy_test_pattern.match(partner_release)
                if legacy_partner_match:
                    run_release_normalised = (
                        f'{legacy_partner_match.group("yy")}.{legacy_partner_match.group("mm")}'
                        f'-ppp.{legacy_partner_match.group("run")}'
                    )
                else:
                    year_month_match = re.search(r'(?P<yy>\d{2})\.(?P<mm>\d{2})', partner_release)
                    if year_month_match:
                        suffix_after_year_month = partner_release[year_month_match.end() :]
                        run_match = re.search(r'(?P<run>\d+)', suffix_after_year_month)
                        run_number = run_match.group('run') if run_match else '1'
                        if not run_match:
                            logger.warning(
                                f'Partner release `{ot_release}` missing run number; defaulting to `{run_number}`'
                            )
                        run_release_normalised = (
                            f'{year_month_match.group("yy")}.{year_month_match.group("mm")}-ppp.{run_number}'
                        )
                    else:
                        logger.warning(
                            f'Could not normalise partner release `{ot_release}` to `YY.MM-ppp.N`; using raw value'
                        )
        else:
            logger.warning(f'Could not normalise ot_release `{ot_release}` to `YY.MM-(ppp|pub).N`; using raw value')

    return f'{run_release_normalised}-{release_timestamp}'


def _to_parquet_glob(path: str | Path) -> str:
    """Normalize a dataset path to a parquet glob consumable by Polars."""
    path_str = str(path)
    if '.parquet' in path_str:
        return path_str
    return f'{path_str.rstrip("/")}/*.parquet'


def _to_release_relative_path(path: str, release_uri: str) -> str:
    """Convert an absolute dataset path into a release-relative path key."""
    release_root = release_uri.rstrip('/')
    if path.startswith(release_root):
        relative = path[len(release_root) :]
    else:
        relative = path

    relative = relative.rstrip('/')
    if not relative.startswith('/'):
        relative = f'/{relative}'
    return relative


def _build_absolute_scope_pattern(release_uri: str, scope: str) -> str:
    """Build an absolute storage glob pattern from release URI and scope."""
    scope_path = scope if scope.startswith('/') else f'/{scope}'
    return f'{release_uri.rstrip("/")}{scope_path}'


def _scope_to_parquet_file_glob(scope: str) -> str:
    """Convert a dataset scope into a parquet-file scope for robust discovery."""
    normalized = scope.rstrip('/')
    if normalized.endswith('.parquet'):
        return normalized
    return f'{normalized}/*.parquet'


def _dataset_path_from_parquet_file(path: str) -> str:
    """Return the dataset directory path for a parquet file URI/path."""
    return path.rsplit('/', maxsplit=1)[0]


def _has_glob_wildcards(path_pattern: str) -> bool:
    """Return whether a path pattern contains glob wildcards."""
    return any(char in path_pattern for char in '*?[')


def _expand_storage_glob(path_pattern: str, config: Config) -> list[str]:
    """Expand a storage glob pattern into concrete dataset paths.

    Example:
        path_pattern='gs://bucket/release/output/*/*.parquet'
        -> root='gs://bucket/release/output'
        -> glob pattern='*/*.parquet'
        -> returns e.g.
           [
               'gs://bucket/release/output/disease/part-00000.parquet',
               'gs://bucket/release/output/target/part-00000.parquet',
           ]
    If no wildcards are present, returns [path_pattern] unchanged.
    """
    wildcard_positions = [
        idx for idx in (path_pattern.find('*'), path_pattern.find('?'), path_pattern.find('[')) if idx != -1
    ]
    if not wildcard_positions:
        return [path_pattern]

    first_wildcard = min(wildcard_positions)
    slash_idx = path_pattern.rfind('/', 0, first_wildcard)
    if slash_idx == -1:
        msg = f'Invalid scope pattern: {path_pattern}'
        raise ValueError(msg)

    root = path_pattern[:slash_idx]
    pattern = path_pattern[slash_idx + 1 :]
    return sorted(StorageHandle(root, config=config).glob(pattern))


def _discover_dataset_paths(release_uri: str, scope_globs: list[str], config: Config) -> dict[str, str]:
    """Discover datasets from configured scopes and key them by release-relative path."""
    discovered: dict[str, str] = {}
    for scope in scope_globs:
        abs_pattern = _build_absolute_scope_pattern(release_uri, scope)
        if not _has_glob_wildcards(scope):
            for match in _expand_storage_glob(abs_pattern, config):
                dataset_path = _dataset_path_from_parquet_file(match) if '.parquet' in match else match.rstrip('/')
                relative = _to_release_relative_path(dataset_path, release_uri)
                discovered[relative] = dataset_path

        abs_parquet_pattern = _build_absolute_scope_pattern(release_uri, _scope_to_parquet_file_glob(scope))
        for parquet_file in _expand_storage_glob(abs_parquet_pattern, config):
            dataset_path = _dataset_path_from_parquet_file(parquet_file)
            relative = _to_release_relative_path(dataset_path, release_uri)
            discovered[relative] = dataset_path
    return discovered


def _resolve_metric_lists(
    discovered_dataset_paths: dict[str, str],
    rich_dataset_whitelist: list[str],
) -> tuple[set[str], set[str]]:
    """Split discovered datasets into rich and minimal sets from whitelist globs."""
    discovered = set(discovered_dataset_paths)
    rich = {
        dataset
        for dataset in discovered
        if any(
            fnmatch(dataset, pattern if pattern.startswith('/') else f'/{pattern}')
            for pattern in rich_dataset_whitelist
        )
    }
    minimal = discovered - rich
    return rich, minimal


def _read_evidence_with_canonical_schema(path: str, schema: dict[str, Any]) -> pl.DataFrame:
    """Read evidence parquet files using a canonical schema for cross-source unioning."""
    return pl.scan_parquet(
        path,
        glob=True,
        schema=schema,
        missing_columns='insert',
        extra_columns='ignore',
    ).collect()


def _read_evidence_with_canonical_schema_from_paths(paths: list[str], schema: dict[str, Any]) -> pl.DataFrame:
    """Read multiple evidence parquet path globs using the canonical evidence schema."""
    if not paths:
        return pl.DataFrame(schema=schema)

    return pl.scan_parquet(
        paths,
        glob=True,
        schema=schema,
        missing_columns='insert',
        extra_columns='ignore',
    ).collect()


def _read_associations_minimal(path: str) -> pl.DataFrame:
    """Read only lightweight association columns needed for metrics."""
    return pl.scan_parquet(
        path,
        glob=True,
        schema=ASSOCIATION_MINIMAL_SCHEMA,
        missing_columns='insert',
        extra_columns='ignore',
    ).collect()


def _load_parquet_dataset(path: str) -> pl.DataFrame:
    """Read a dataset parquet path (or directory) eagerly into a dataframe."""
    return pl.read_parquet(_to_parquet_glob(path))


def _count_parquet_rows(path: str) -> int:
    """Count rows from a parquet dataset lazily without loading full data."""
    return int(pl.scan_parquet(_to_parquet_glob(path), glob=True).select(pl.len()).collect().item())


def _single_discovered_path(discovered: dict[str, str], rel_path: str) -> str | None:
    """Return one discovered dataset path and warn if it is missing."""
    path = discovered.get(rel_path)
    if not path:
        logger.warning(f'Requested dataset `{rel_path}` not found in discovered scopes')
    return path


def _datasource_view_for_dataset(df: pl.DataFrame) -> pl.DataFrame | None:
    """Extract a datasource-only view when a dataset exposes datasource information."""
    columns = set(df.columns)
    if 'datasourceId' in columns:
        return df.select('datasourceId')

    if {'aggregationType', 'aggregationValue'}.issubset(columns):
        return (
            df
            .with_columns(
                datasourceId=(
                    pl
                    .when(pl.col('aggregationType') == 'datasourceId')
                    .then(pl.col('aggregationValue'))
                    .otherwise(None)
                )
            )
            .filter(pl.col('datasourceId').is_not_null())
            .select('datasourceId')
        )

    return None


def _association_datasource_view(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize association datasource representation across old and new schemas."""
    columns = set(df.columns)

    if {'datasourceId', 'aggregationType', 'aggregationValue'}.issubset(columns):
        datasource_expr = pl.coalesce([
            pl.col('datasourceId'),
            pl.when(pl.col('aggregationType') == 'datasourceId').then(pl.col('aggregationValue')).otherwise(None),
        ])
    elif 'datasourceId' in columns:
        datasource_expr = pl.col('datasourceId')
    elif {'aggregationType', 'aggregationValue'}.issubset(columns):
        datasource_expr = (
            pl.when(pl.col('aggregationType') == 'datasourceId').then(pl.col('aggregationValue')).otherwise(None)
        )
    else:
        msg = 'Association datasource columns not found. Expected datasourceId or aggregationType/aggregationValue.'
        raise ValueError(msg)

    return (
        df
        .with_columns(datasourceId=datasource_expr)
        .filter(pl.col('datasourceId').is_not_null())
        .select('datasourceId', 'diseaseId', 'targetId')
    )


def _emit_evidence_metrics(evidence: pl.DataFrame) -> list[pl.DataFrame]:
    """Emit backward-compatible metrics for merged evidence outputs."""
    columns_to_report = _get_evidence_columns_to_report(evidence.columns)
    return [
        _document_total_count(evidence, 'evidenceTotalCount'),
        _document_count_by(evidence, 'datasourceId', 'evidenceCountByDatasource'),
        _not_null_fields_count_top_level(
            evidence,
            'evidenceFieldNotNullCountByDatasource',
            group_by_datasource=True,
        ),
        _distinct_fields_count_top_level(
            evidence.select(columns_to_report),
            'evidenceDistinctFieldsCountByDatasource',
        ),
    ]


def _emit_evidence_failed_metrics(evidence_failed: pl.DataFrame) -> list[pl.DataFrame]:
    """Emit backward-compatible metrics for excluded/failed evidence outputs."""
    return [
        _document_total_count(evidence_failed, 'evidenceInvalidTotalCount'),
        _document_total_count(
            evidence_failed.filter(pl.col('qualityControls').list.contains('Duplicated')),
            'evidenceDuplicateTotalCount',
        ),
        _document_total_count(
            evidence_failed.filter(pl.col('qualityControls').list.contains('No valid score')),
            'evidenceNullifiedScoreTotalCount',
        ),
        _document_total_count(
            evidence_failed.filter(pl.col('qualityControls').list.contains('No valid target')),
            'evidenceUnresolvedTargetTotalCount',
        ),
        _document_total_count(
            evidence_failed.filter(pl.col('qualityControls').list.contains('No valid disease')),
            'evidenceUnresolvedDiseaseTotalCount',
        ),
        _document_count_by(evidence_failed, 'datasourceId', 'evidenceInvalidCountByDatasource'),
        _document_count_by(
            evidence_failed.filter(pl.col('qualityControls').list.contains('Duplicated')),
            'datasourceId',
            'evidenceDuplicateCountByDatasource',
        ),
        _document_count_by(
            evidence_failed.filter(pl.col('qualityControls').list.contains('No valid score')),
            'datasourceId',
            'evidenceNullifiedScoreCountByDatasource',
        ),
        _document_count_by(
            evidence_failed.filter(pl.col('qualityControls').list.contains('No valid target')),
            'datasourceId',
            'evidenceUnresolvedTargetCountByDatasource',
        ),
        _document_count_by(
            evidence_failed.filter(pl.col('qualityControls').list.contains('No valid disease')),
            'datasourceId',
            'evidenceUnresolvedDiseaseCountByDatasource',
        ),
    ]


def _emit_association_metrics(df: pl.DataFrame, kind: str) -> list[pl.DataFrame]:
    """Emit backward-compatible metrics for association by-datasource datasets."""
    datasource_view = _association_datasource_view(df)
    return [
        _document_total_count(df.select('diseaseId', 'targetId').unique(), f'associations{kind}TotalCount'),
        _document_count_by(datasource_view, 'datasourceId', f'associations{kind}ByDatasource'),
        _not_null_fields_count_top_level(
            datasource_view,
            f'associations{kind}NotNullCountByDatasource',
            group_by_datasource=True,
        ),
        _distinct_fields_count_top_level(datasource_view, f'associations{kind}DistinctFieldsCountByDatasource'),
    ]


def _compute_metrics(
    settings: dict[str, Any],
    config: Config,
    run_id: str,
) -> pl.DataFrame:
    """Discover datasets and compute rich/minimal metrics according to whitelist rules."""
    data_root_uri = config.release_uri or str(config.work_path)

    scope_globs = list(settings.get('metric_scopes', ['/output/*', '/excluded/evidence/*']))
    rich_patterns = list(settings.get('rich_dataset_whitelist', []))
    evidence_schema = load_spark_schema_as_polars('evidence.json')

    discovered = _discover_dataset_paths(data_root_uri, scope_globs, config)
    rich_dataset_list, minimal_dataset_list = _resolve_metric_lists(discovered, rich_patterns)

    metric_frames: list[pl.DataFrame] = []
    handled_paths: set[str] = set()

    if any(fnmatch(dataset, EVIDENCE_OUTPUT_PATTERN) for dataset in rich_dataset_list):
        evidence_paths = [
            _to_parquet_glob(path) for rel, path in discovered.items() if fnmatch(rel, EVIDENCE_OUTPUT_PATTERN)
        ]
        if evidence_paths:
            evidence = _read_evidence_with_canonical_schema_from_paths(evidence_paths, evidence_schema)
            metric_frames.extend(_emit_evidence_metrics(evidence))
            handled_paths.update(rel for rel in discovered if fnmatch(rel, EVIDENCE_OUTPUT_PATTERN))
        else:
            logger.warning(f'Evidence requested in whitelist but no {EVIDENCE_OUTPUT_PATTERN} datasets were discovered')

    if any(fnmatch(dataset, EVIDENCE_EXCLUDED_PATTERN) for dataset in rich_dataset_list):
        evidence_failed_paths = [
            _to_parquet_glob(path) for rel, path in discovered.items() if fnmatch(rel, EVIDENCE_EXCLUDED_PATTERN)
        ]
        if evidence_failed_paths:
            evidence_failed = _read_evidence_with_canonical_schema_from_paths(evidence_failed_paths, evidence_schema)
            metric_frames.extend(_emit_evidence_failed_metrics(evidence_failed))
            handled_paths.update(rel for rel in discovered if fnmatch(rel, EVIDENCE_EXCLUDED_PATTERN))
        else:
            logger.warning(
                f'Excluded evidence requested in whitelist but no {EVIDENCE_EXCLUDED_PATTERN} datasets were discovered'
            )

    generic_rich = sorted(dataset for dataset in rich_dataset_list if dataset not in handled_paths)
    generic_minimal = sorted(dataset for dataset in minimal_dataset_list if dataset not in handled_paths)

    for rel_path in generic_rich:
        dataset_path = _single_discovered_path(discovered, rel_path)
        if not dataset_path:
            continue

        if rel_path == '/output/association_by_datasource_direct':
            df = _read_associations_minimal(_to_parquet_glob(dataset_path))
            metric_frames.extend(_emit_association_metrics(df, 'Direct'))
            continue

        if rel_path == '/output/association_by_datasource_indirect':
            df = _read_associations_minimal(_to_parquet_glob(dataset_path))
            metric_frames.extend(_emit_association_metrics(df, 'Indirect'))
            continue

        df = _load_parquet_dataset(dataset_path)
        prefix = _metric_prefix(rel_path)
        metric_frames.extend(_global_rich_metrics(df, prefix))
        metric_frames.extend(_quality_control_flag_total_metrics(df, prefix))

        datasource_view = _datasource_view_for_dataset(df)
        if datasource_view is not None:
            metric_frames.append(_document_count_by(datasource_view, 'datasourceId', f'{prefix}ByDatasource'))

    for rel_path in generic_minimal:
        total = _count_parquet_rows(discovered[rel_path])
        metric_frames.append(_document_total_value(total, f'{_metric_prefix(rel_path)}TotalCount'))

    logger.info(
        'Metrics execution summary: '
        f'discovered={len(discovered)}, '
        f'rich={len(rich_dataset_list)}, '
        f'minimal={len(minimal_dataset_list)}, '
        f'handled_special={len(handled_paths)}, '
        f'generic_rich={len(generic_rich)}, '
        f'generic_minimal={len(generic_minimal)}'
    )

    if not metric_frames:
        return _empty_metrics_frame()

    return (
        pl
        .concat(metric_frames, how='vertical_relaxed')
        .with_columns(runId=pl.lit(run_id))
        .select('datasourceId', 'variable', 'field', 'value', 'runId')
    )


def _upload_metrics_to_hf_hub(
    csv_data: bytes,
    csv_filename: str,
    token_filename: Path,
    repo_id: str,
    data_dir: str,
) -> None:
    """Upload CSV metrics bytes to a Hugging Face dataset repository."""
    hf_token = token_filename.read_text().strip()
    if not hf_token:
        msg = f'HF token file is empty: {token_filename}'
        raise ValueError(msg)

    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=csv_data,
        path_in_repo=f'{data_dir}/{csv_filename}',
        repo_id=repo_id,
        repo_type='dataset',
    )


def release_metrics(
    source: dict[str, Path],
    destination: dict[str, Path],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate release metrics parquet and optionally upload CSV to Hugging Face.

    Args:
        source: Source paths (unused).
        destination: Destination paths (parquet). Relative paths resolve to ``release_uri`` when set.
        settings: Step settings. ``ot_release`` is used to build ``runId``.
        config: Application configuration (automatically injected). Dataset discovery reads from
            ``release_uri`` when set, otherwise from ``work_path``.
    """
    del source

    ot_release = settings['ot_release']
    run_id = _build_run_id(ot_release)

    logger.info(f'Loading and calculating metrics for release {ot_release}')
    metrics = _compute_metrics(settings, config, run_id)

    logger.info(f'Writing metrics parquet to {destination["parquet"]}')
    metrics.write_parquet(destination['parquet'], mkdir=True)

    if settings.get('upload_to_hf_hub'):
        hf_token_filename = settings.get('hf_token_filename')
        if not hf_token_filename:
            logger.warning('HF upload requested but settings["hf_token_filename"] is missing; skipping upload')
            return

        hf_repo_id = settings.get('hf_repo_id', 'opentargets/ot-release-metrics')
        hf_data_dir = settings.get('hf_data_dir', 'metrics')
        csv_filename = f'{run_id}.csv'
        csv_data = metrics.write_csv(include_header=True).encode('utf-8')

        try:
            _upload_metrics_to_hf_hub(
                csv_data=csv_data,
                csv_filename=csv_filename,
                token_filename=Path(hf_token_filename),
                repo_id=hf_repo_id,
                data_dir=hf_data_dir,
            )
            logger.info(f'Uploaded metrics CSV to hf://datasets/{hf_repo_id}/{hf_data_dir}/{csv_filename}')
        except Exception:
            logger.exception('HF upload failed; continuing without failing the step')
