from __future__ import annotations

from datetime import datetime
from typing import cast

import polars as pl
import pytest
from otter.config.model import Config

from pts.transformers.release_metrics import (
    _build_run_id,
    _discover_dataset_paths,
    _emit_association_metrics,
    _emit_evidence_failed_metrics,
    _emit_evidence_metrics,
    _global_rich_metrics,
    _quality_control_flag_total_metrics,
)


def _input_tables() -> dict[str, pl.DataFrame]:
    evidence = pl.DataFrame({
        'datasourceId': ['ds1', 'ds1', 'ds2'],
        'targetFromSourceId': ['T1', 'T2', None],
        'diseaseFromSourceMappedId': ['D1', None, 'D2'],
        'drugId': ['DR1', 'DR1', None],
        'literature': [['PMID:1'], [], None],
    })

    evidence_failed = pl.DataFrame({
        'id': ['e1', 'e2', 'e3'],
        'datasourceId': ['ds1', 'ds2', 'ds2'],
        'qualityControls': [
            ['Duplicated', 'No valid score'],
            ['No valid target'],
            ['No valid disease'],
        ],
    })

    associations_direct = pl.DataFrame({
        'datasourceId': ['ds1', 'ds1', 'ds2'],
        'diseaseId': ['D1', 'D1', 'D2'],
        'targetId': ['T1', 'T1', 'T2'],
    })

    associations_indirect = pl.DataFrame({
        'datasourceId': ['ds1', 'ds2'],
        'diseaseId': ['D3', 'D2'],
        'targetId': ['T3', 'T2'],
    })

    diseases = pl.DataFrame({'id': ['D1', 'D2'], 'name': ['n1', None]})
    targets = pl.DataFrame({'id': ['T1', 'T2', 'T3']})
    drugs = pl.DataFrame({'id': ['DR1', 'DR2'], 'description': ['d1', None]})

    return {
        'evidence': evidence,
        'evidence_failed': evidence_failed,
        'associations_direct': associations_direct,
        'associations_indirect': associations_indirect,
        'diseases': diseases,
        'targets': targets,
        'drugs': drugs,
    }


def _metric_value(metrics: pl.DataFrame, variable: str, datasource: str = 'all') -> int:
    return int(
        metrics.filter(pl.col('variable') == variable, pl.col('datasourceId') == datasource).select('value').item()
    )


def _core_metrics_frame(tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    metrics = [
        *_emit_evidence_metrics(tables['evidence']),
        *_emit_evidence_failed_metrics(tables['evidence_failed']),
        *_emit_association_metrics(tables['associations_direct'], 'Direct'),
        *_emit_association_metrics(tables['associations_indirect'], 'Indirect'),
        *_global_rich_metrics(tables['diseases'], 'diseases'),
        *_global_rich_metrics(tables['targets'], 'targets'),
        *_global_rich_metrics(tables['drugs'], 'drugs'),
    ]
    return pl.concat(metrics, how='vertical_relaxed').with_columns(runId=pl.lit('26.03_2026-03-13'))


def test_release_metrics_schema_and_variables() -> None:
    metrics = _core_metrics_frame(_input_tables())

    assert metrics.columns == ['datasourceId', 'variable', 'field', 'value', 'runId']
    assert set(metrics['variable'].unique()) == {
        'associationsDirectDistinctFieldsCountByDatasource',
        'associationsDirectByDatasource',
        'associationsDirectNotNullCountByDatasource',
        'associationsDirectTotalCount',
        'associationsIndirectDistinctFieldsCountByDatasource',
        'associationsIndirectByDatasource',
        'associationsIndirectNotNullCountByDatasource',
        'associationsIndirectTotalCount',
        'diseasesDistinctFieldsCount',
        'diseasesNotNullCount',
        'diseasesTotalCount',
        'drugsDistinctFieldsCount',
        'drugsNotNullCount',
        'drugsTotalCount',
        'evidenceCountByDatasource',
        'evidenceDistinctFieldsCountByDatasource',
        'evidenceDuplicateCountByDatasource',
        'evidenceDuplicateTotalCount',
        'evidenceFieldNotNullCountByDatasource',
        'evidenceInvalidCountByDatasource',
        'evidenceInvalidTotalCount',
        'evidenceNullifiedScoreCountByDatasource',
        'evidenceNullifiedScoreTotalCount',
        'evidenceTotalCount',
        'evidenceUnresolvedDiseaseCountByDatasource',
        'evidenceUnresolvedDiseaseTotalCount',
        'evidenceUnresolvedTargetCountByDatasource',
        'evidenceUnresolvedTargetTotalCount',
        'targetsDistinctFieldsCount',
        'targetsNotNullCount',
        'targetsTotalCount',
    }


def test_release_metrics_core_counts() -> None:
    metrics = _core_metrics_frame(_input_tables())

    assert _metric_value(metrics, 'evidenceTotalCount') == 3
    assert _metric_value(metrics, 'evidenceCountByDatasource', datasource='ds1') == 2
    assert _metric_value(metrics, 'evidenceInvalidTotalCount') == 3
    assert _metric_value(metrics, 'evidenceDuplicateTotalCount') == 1
    assert _metric_value(metrics, 'associationsDirectTotalCount') == 2
    assert _metric_value(metrics, 'associationsIndirectTotalCount') == 2
    assert _metric_value(metrics, 'diseasesTotalCount') == 2
    assert _metric_value(metrics, 'targetsTotalCount') == 3
    assert _metric_value(metrics, 'drugsTotalCount') == 2


def test_release_metrics_top_level_only_fields() -> None:
    metrics = _core_metrics_frame(_input_tables())

    evidence_fields = (
        metrics
        .filter(pl.col('variable') == 'evidenceFieldNotNullCountByDatasource')['field']
        .drop_nulls()
    )
    assert all('.' not in field for field in evidence_fields)


def test_release_metrics_association_new_schema_counts() -> None:
    associations_direct = pl.DataFrame({
        'diseaseId': ['D1', 'D1', 'D2'],
        'targetId': ['T1', 'T1', 'T2'],
        'aggregationType': ['datasourceId', 'datasourceId', 'datasourceId'],
        'aggregationValue': ['ds1', 'ds1', 'ds2'],
    })
    associations_indirect = pl.DataFrame({
        'diseaseId': ['D3', 'D2'],
        'targetId': ['T3', 'T2'],
        'aggregationType': ['datasourceId', 'datasourceId'],
        'aggregationValue': ['ds1', 'ds2'],
    })

    metrics = pl.concat(
        [
            *_emit_association_metrics(associations_direct, 'Direct'),
            *_emit_association_metrics(associations_indirect, 'Indirect'),
        ],
        how='vertical_relaxed',
    ).with_columns(runId=pl.lit('26.03_2026-03-13'))

    assert _metric_value(metrics, 'associationsDirectByDatasource', datasource='ds1') == 2
    assert _metric_value(metrics, 'associationsIndirectByDatasource', datasource='ds2') == 1


def test_generic_rich_metrics_profile() -> None:
    df = pl.DataFrame({'id': ['b1', 'b2'], 'name': ['x', None]})
    metrics = pl.concat(_global_rich_metrics(df, 'biosample'), how='vertical_relaxed').with_columns(
        runId=pl.lit('26.03_2026-03-13')
    )

    assert _metric_value(metrics, 'biosampleTotalCount') == 2
    assert metrics.filter(pl.col('variable') == 'biosampleDistinctFieldsCount').height > 0


def test_generic_rich_metrics_with_heterogeneous_list_struct() -> None:
    df = pl.DataFrame({
        'studyId': ['s1', 's2'],
        'ldPopulationStructure': [
            [{'ldPopulation': 'nfe'}],
            [{'relativeSampleSize': 0.5}],
        ],
    })

    metrics = pl.concat(_global_rich_metrics(df, 'study'), how='vertical_relaxed').with_columns(
        runId=pl.lit('26.03_2026-03-13')
    )

    assert _metric_value(metrics, 'studyTotalCount') == 2
    assert (
        metrics.filter(pl.col('variable') == 'studyNotNullCount', pl.col('field') == 'ldPopulationStructure').height
        == 1
    )


def test_quality_control_flag_total_metrics() -> None:
    df = pl.DataFrame({
        'id': ['r1', 'r2', 'r3'],
        'qualityControls': [
            ['PHASE_IV_NOT_APPROVED', 'UNVALIDATED_INDICATION'],
            ['UNVALIDATED_INDICATION'],
            ['INDIRECT_PRIMARY_PURPOSE'],
        ],
    })

    metrics = pl.concat(
        _quality_control_flag_total_metrics(df, 'clinicalReport'),
        how='vertical_relaxed',
    ).with_columns(runId=pl.lit('26.03_2026-03-13'))

    assert _metric_value(metrics, 'clinicalReportPhaseIvNotApprovedTotalCount') == 1
    assert _metric_value(metrics, 'clinicalReportUnvalidatedIndicationTotalCount') == 2
    assert _metric_value(metrics, 'clinicalReportIndirectPrimaryPurposeTotalCount') == 1


def test_discover_dataset_paths_recovers_missing_directory_markers(monkeypatch) -> None:
    release_uri = 'gs://bucket/release'
    config = cast(Config, object())

    def fake_expand_storage_glob(path_pattern: str, _config) -> list[str]:
        if path_pattern == f'{release_uri}/output/*/':
            return [f'{release_uri}/output/target/']
        if path_pattern == f'{release_uri}/output/*/*.parquet':
            return [
                f'{release_uri}/output/target/part-00000.parquet',
                f'{release_uri}/output/disease/part-00000.parquet',
                f'{release_uri}/output/go/part-00000.parquet',
            ]
        return []

    monkeypatch.setattr('pts.transformers.release_metrics._expand_storage_glob', fake_expand_storage_glob)

    discovered = _discover_dataset_paths(release_uri, ['/output/*/'], config=config)

    assert set(discovered) == {'/output/disease', '/output/go', '/output/target'}


def test_discover_dataset_paths_scope_trailing_slash_equivalent(monkeypatch) -> None:
    release_uri = 'gs://bucket/release'
    config = cast(Config, object())

    def fake_expand_storage_glob(path_pattern: str, _config) -> list[str]:
        if path_pattern in {f'{release_uri}/output/*', f'{release_uri}/output/*/'}:
            return []
        if path_pattern == f'{release_uri}/output/*/*.parquet':
            return [
                f'{release_uri}/output/target/part-00000.parquet',
                f'{release_uri}/output/disease/part-00000.parquet',
            ]
        return []

    monkeypatch.setattr('pts.transformers.release_metrics._expand_storage_glob', fake_expand_storage_glob)

    discovered_without_slash = _discover_dataset_paths(release_uri, ['/output/*'], config=config)
    discovered_with_slash = _discover_dataset_paths(release_uri, ['/output/*/'], config=config)

    assert discovered_without_slash == discovered_with_slash
    assert set(discovered_without_slash) == {'/output/disease', '/output/target'}


@pytest.mark.parametrize(
    ('ot_release', 'expected_run_id'),
    [
        ('26.03-pub.4', '26.03-pub.4-2026-04-02'),
        ('26.03-test4', '26.03-pub.4-2026-04-02'),
        ('partners/26.03-test4', '26.03-ppp.4-2026-04-02'),
        ('partners/26.03-pub.7', '26.03-ppp.7-2026-04-02'),
        ('partners/26.03', '26.03-ppp.1-2026-04-02'),
        ('custom-release', 'custom-release-2026-04-02'),
    ],
)
def test_build_run_id_normalisation(monkeypatch, ot_release: str, expected_run_id: str) -> None:
    class _FixedDateTime:
        @classmethod
        def today(cls) -> datetime:
            return datetime(2026, 4, 2)

    monkeypatch.setattr('pts.transformers.release_metrics.datetime', _FixedDateTime)

    assert _build_run_id(ot_release) == expected_run_id
