from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from pts.transformers.dataset_metrics import (
    _config_for_dataset,
    _dataset_file_stats,
    compute_breakdown,
    compute_filter_count,
    dataset_metrics,
    profile_dataset,
)


def _write_dataset(directory: Path, df: pl.DataFrame, n_files: int = 1) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    rows_per = max(1, df.height // n_files)
    for i in range(n_files):
        chunk = df.slice(i * rows_per, rows_per if i < n_files - 1 else df.height - i * rows_per)
        chunk.write_parquet(directory / f'part-{i}.parquet')
    return str(directory)


def _row(rows: list[dict], kind: str, metric: str, group_value: str | None = None) -> dict:
    matches = [r for r in rows if r['kind'] == kind and r['metric'] == metric and r['group_value'] == group_value]
    assert len(matches) == 1, f'expected one {kind}/{metric}/{group_value} row, got {matches}'
    return matches[0]


def test_compute_breakdown_plain_column_with_null() -> None:
    df = pl.DataFrame({'studyType': ['gwas', 'eqtl', 'gwas', None]})
    assert compute_breakdown(df, 'studyType') == {'gwas': 2, 'eqtl': 1, 'null': 1}


def test_compute_breakdown_derived_expression() -> None:
    df = pl.DataFrame({'rightStudyType': ['gwas', 'eqtl', 'eqtl']})
    assert compute_breakdown(df, "concat('gwas-', rightStudyType)") == {'gwas-eqtl': 2, 'gwas-gwas': 1}


def test_compute_breakdown_explodes_list_column() -> None:
    df = pl.DataFrame({'tas': [['a', 'b'], ['a']]})
    assert compute_breakdown(df, 'tas') == {'a': 2, 'b': 1}


def test_compute_filter_count_rows() -> None:
    df = pl.DataFrame({'score': [0.9, 0.2, 0.7], 'geneId': ['g1', 'g2', 'g1']})
    assert compute_filter_count(df, 'score > 0.5') == 2


def test_compute_filter_count_distinct() -> None:
    df = pl.DataFrame({'score': [0.9, 0.2, 0.7], 'geneId': ['g1', 'g2', 'g1']})
    assert compute_filter_count(df, 'score > 0.5', distinct='geneId') == 1


def test_compute_filter_count_distinct_excludes_null() -> None:
    df = pl.DataFrame({'score': [0.9, 0.7, 0.8], 'geneId': ['g1', None, 'g1']})
    # all three rows pass score > 0.5; geneId is {g1, null} -> a null is not a gene
    assert compute_filter_count(df, 'score > 0.5', distinct='geneId') == 1


def test_dataset_file_stats(tmp_path: Path) -> None:
    dataset_dir = tmp_path / 'study'
    dataset_dir.mkdir()
    df = pl.DataFrame({'studyType': ['gwas', 'eqtl']})
    df.write_parquet(dataset_dir / 'part-0.parquet')  # Spark-style name
    df.write_parquet(dataset_dir / 'data-abc.parquet')  # single-file Polars-style name
    (dataset_dir / '_SUCCESS').write_text('')  # marker must be ignored
    (dataset_dir / '.part-0.parquet.crc').write_text('')  # checksum must be ignored

    file_size, n_partitions = _dataset_file_stats(str(dataset_dir))

    assert n_partitions == 2  # both .parquet files counted, regardless of prefix
    assert file_size > 0


def test_profile_dataset_emits_long_rows(tmp_path: Path) -> None:
    df = pl.DataFrame({
        'studyType': ['gwas', 'eqtl', 'gwas'],
        'score': [0.9, 0.2, 0.7],
        'geneId': ['g1', 'g2', 'g1'],
    })
    path = _write_dataset(tmp_path / 'l2g_like', df)
    config = {
        'groupings': {'studyType': 'studyType'},
        'filter_counts': [{'name': 'high', 'filter': 'score > 0.5', 'distinct': 'geneId'}],
    }

    rows = profile_dataset(path, 'l2g_like', config, run='26.03-test5')

    assert rows is not None
    assert all(r['run'] == '26.03-test5' and r['dataset'] == 'l2g_like' for r in rows)

    count_row = _row(rows, 'scalar', 'count')
    assert count_row['value'] == 3
    assert count_row['expression'] is None
    assert _row(rows, 'scalar', 'number_of_partitions')['value'] == 1
    assert _row(rows, 'scalar', 'file_size')['value'] > 0

    gwas = _row(rows, 'grouping', 'studyType', 'gwas')
    assert gwas['value'] == 2
    assert gwas['expression'] == 'studyType'
    assert _row(rows, 'grouping', 'studyType', 'eqtl')['value'] == 1

    high = _row(rows, 'filter', 'high')
    assert high['value'] == 1
    assert high['expression'] == 'distinct geneId where score > 0.5'
    assert high['group_value'] is None


def test_profile_dataset_counts_every_part(tmp_path: Path) -> None:
    df = pl.DataFrame({'id': [f'i{n}' for n in range(9)]})
    path = _write_dataset(tmp_path / 'multipart', df, n_files=3)

    rows = profile_dataset(path, 'multipart', {}, run='r1')

    assert rows is not None
    assert _row(rows, 'scalar', 'count')['value'] == 9
    assert _row(rows, 'scalar', 'number_of_partitions')['value'] == 3


def test_profile_dataset_base_stats_only(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path / 'biosample', pl.DataFrame({'id': ['b1', 'b2']}))
    rows = profile_dataset(path, 'biosample', {}, run='r1')
    assert rows is not None
    assert sorted((r['kind'], r['metric']) for r in rows) == [
        ('scalar', 'count'),
        ('scalar', 'file_size'),
        ('scalar', 'number_of_partitions'),
    ]
    assert _row(rows, 'scalar', 'count')['value'] == 2


def test_profile_dataset_unreadable_returns_none(tmp_path: Path) -> None:
    empty_dir = tmp_path / 'not_parquet'
    empty_dir.mkdir()
    (empty_dir / 'readme.txt').write_text('not parquet')
    assert profile_dataset(str(empty_dir), 'not_parquet', {}, run='r1') is None


def test_profile_dataset_bad_expression_fails_loud(tmp_path: Path) -> None:
    path = _write_dataset(tmp_path / 'study', pl.DataFrame({'studyType': ['gwas']}))
    config = {'groupings': {'oops': 'nonexistent_column'}}
    with pytest.raises(ValueError, match='oops'):
        profile_dataset(path, 'study', config, run='r1')


def test_profile_dataset_listing_failure_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_dataset(tmp_path / 'study', pl.DataFrame({'studyType': ['gwas']}))

    def boom(_path: str) -> tuple[int, int]:
        raise OSError('listing failed')

    monkeypatch.setattr('pts.transformers.dataset_metrics._dataset_file_stats', boom)
    assert profile_dataset(path, 'study', {}, run='r1') is None


def test_dataset_metrics_writes_one_json_line_per_measurement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    study = _write_dataset(tmp_path / 'study', pl.DataFrame({'studyType': ['gwas', 'eqtl', 'gwas']}))
    biosample = _write_dataset(tmp_path / 'biosample', pl.DataFrame({'id': ['b1', 'b2']}))
    discovered = {'/output/study': study, '/output/biosample': biosample}
    monkeypatch.setattr(
        'pts.transformers.dataset_metrics.discover_dataset_paths',
        lambda root, scopes, config: discovered,
    )

    out_dir = tmp_path / 'export' / 'dataset_metrics'
    config = SimpleNamespace(release_uri='gs://bucket/26.03-test5', work_path=tmp_path)
    settings = {'datasets': {'study': {'groupings': {'studyType': 'studyType'}}}}

    dataset_metrics({}, {'metrics': str(out_dir)}, settings, config)

    assert sorted(p.name for p in out_dir.iterdir()) == ['part-00000.json']
    rows = [json.loads(line) for line in (out_dir / 'part-00000.json').read_text().splitlines()]
    assert {row['run'] for row in rows} == {'26.03-test5'}  # derived from the release_uri basename
    assert [row for row in rows if row['metric'] != 'file_size'] == [
        {'run': '26.03-test5', 'dataset': 'biosample', 'kind': 'scalar', 'metric': 'count', 'value': 2},
        {'run': '26.03-test5', 'dataset': 'biosample', 'kind': 'scalar', 'metric': 'number_of_partitions', 'value': 1},
        {'run': '26.03-test5', 'dataset': 'study', 'kind': 'scalar', 'metric': 'count', 'value': 3},
        {'run': '26.03-test5', 'dataset': 'study', 'kind': 'scalar', 'metric': 'number_of_partitions', 'value': 1},
        {
            'run': '26.03-test5',
            'dataset': 'study',
            'kind': 'grouping',
            'metric': 'studyType',
            'expression': 'studyType',
            'group_value': 'gwas',
            'value': 2,
        },
        {
            'run': '26.03-test5',
            'dataset': 'study',
            'kind': 'grouping',
            'metric': 'studyType',
            'expression': 'studyType',
            'group_value': 'eqtl',
            'value': 1,
        },
    ]


def test_dataset_metrics_writes_to_the_destination_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    study = _write_dataset(tmp_path / 'study', pl.DataFrame({'id': ['s1']}))
    monkeypatch.setattr(
        'pts.transformers.dataset_metrics.discover_dataset_paths',
        lambda root, scopes, config: {'/output/study': study},
    )
    monkeypatch.setattr('pts.transformers.dataset_metrics._dataset_file_stats', lambda _path: (1, 1))
    written: dict[str, pl.DataFrame] = {}
    monkeypatch.setattr(
        'pts.transformers.dataset_metrics.write_json_dataset',
        lambda frame, path: written.__setitem__(path, frame),
    )

    config = SimpleNamespace(release_uri='gs://bucket/26.03-test5', work_path=tmp_path)
    out_uri = 'gs://bucket/26.03-test5/export/dataset_metrics'
    dataset_metrics({}, {'metrics': out_uri}, {}, config)

    assert list(written) == [out_uri]
    assert written[out_uri].filter(pl.col('metric') == 'count')['value'].to_list() == [1]


def test_config_for_dataset_pattern_match() -> None:
    cfg = {
        'study': {'groupings': {'a': 'a'}},
        'evidence_*': {'groupings': {'datatype': 'datatypeId'}},
    }
    assert _config_for_dataset('study', cfg) == {'groupings': {'a': 'a'}}
    assert _config_for_dataset('evidence_eva', cfg) == {'groupings': {'datatype': 'datatypeId'}}
    assert _config_for_dataset('biosample', cfg) == {}


def test_config_for_dataset_specificity_precedence() -> None:
    cfg = {
        '*': {'groupings': {'all': 'all'}},
        'evidence_*': {'groupings': {'datatype': 'datatypeId'}},
    }
    # the more specific (longer) pattern wins over the catch-all
    assert _config_for_dataset('evidence_eva', cfg) == {'groupings': {'datatype': 'datatypeId'}}
    assert _config_for_dataset('study', cfg) == {'groupings': {'all': 'all'}}
