"""Focused tests for the data-driven PTS LDSC-CTS tasks."""

from __future__ import annotations

import json
from pathlib import Path

import pyspark.sql.functions as f
import pytest
from gentropy.method.ldsc import infer_ld_ancestry
from pyspark.sql import Row

from pts.pyspark.ldsc_cts_annotation import _read_edges
from pts.pyspark.ldsc_cts_regression import (
    _chunks,
    _prepare_sumstats,
    _read_annotation_catalog,
    _resolve_root_dataset_ids,
)
from pts.pyspark.ldsc_cts_utils import (
    discover_completed_study_paths,
    discover_edge_manifest,
    normalise_dataset_registry,
    normalise_populations,
    success_exists,
)


def _write_edge_task(root: Path, name: str, ancestry: str, chromosome: int) -> Path:
    task = root / name
    edge_output = task / 'edges.parquet'
    edge_output.mkdir(parents=True)
    (edge_output / '_SUCCESS').write_text('')
    (task / '.command.sh').write_text(
        'python export_edges_gnomad.py '
        f'--ld-bm-path gs://bucket/gnomad.genomes.r2.1.1.{ancestry}.common.adj.ld.bm '
        f'--chromosome chr{chromosome} --min-r2 0.0 --ld-window-cm 1.0 '
        '--output-path edges.parquet'
    )
    return edge_output


def test_edge_manifest_requires_complete_unique_exports(tmp_path: Path) -> None:
    """The task stops before Spark work for incomplete or duplicate manifests."""
    _write_edge_task(tmp_path, 'one', 'nfe', 1)
    with pytest.raises(ValueError, match='Missing successful'):
        discover_edge_manifest(str(tmp_path), 'nfe', [1, 2])

    _write_edge_task(tmp_path, 'two', 'nfe', 2)
    _write_edge_task(tmp_path, 'retry', 'nfe', 2)
    with pytest.raises(ValueError, match='Duplicate successful'):
        discover_edge_manifest(str(tmp_path), 'nfe', [1, 2])


def test_dataset_registry_and_population_validation() -> None:
    assert normalise_dataset_registry([{'id': 'one', 'path': 'one.csv'}]) == [
        {'id': 'one', 'path': 'one.csv'}
    ]
    with pytest.raises(ValueError, match='duplicate id'):
        normalise_dataset_registry([
            {'id': 'one', 'path': 'one.csv'},
            {'id': 'one', 'path': 'two.csv'},
        ])
    with pytest.raises(ValueError, match='non-empty path'):
        normalise_dataset_registry([{'id': 'one', 'path': ''}])
    assert normalise_populations('nfe,eas') == ['nfe', 'eas']
    with pytest.raises(ValueError, match='duplicate population'):
        normalise_populations(['nfe', 'NFE'])


def test_study_index_plurality_is_independent_of_analysis_id() -> None:
    """A descriptive ID containing EUR/EAS cannot select the LD ancestry."""
    structure = [
        {'ldPopulation': 'eas', 'relativeSampleSize': 0.6},
        {'ldPopulation': 'nfe', 'relativeSampleSize': 0.4},
    ]
    assert infer_ld_ancestry(structure) == 'eas'


def test_completed_study_discovery_filters_success_markers(tmp_path: Path, spark) -> None:
    complete = tmp_path / 'GCST1'
    complete.mkdir()
    (complete / '_SUCCESS').write_text('')
    incomplete = tmp_path / 'GCST2'
    incomplete.mkdir()
    assert discover_completed_study_paths(spark, str(tmp_path)) == [('GCST1', str(complete))]
    with pytest.raises(ValueError, match='Missing completed'):
        discover_completed_study_paths(spark, str(tmp_path), ['GCST2'])


def test_summary_directory_id_must_match_single_study_id(spark, tmp_path: Path) -> None:
    path = tmp_path / 'GCST1'
    spark.createDataFrame([Row(studyId='GCST2')]).write.mode('overwrite').parquet(str(path))
    with pytest.raises(ValueError, match='does not match'):
        _resolve_root_dataset_ids(spark, [('GCST1', str(path))])


def test_malformed_child_can_be_isolated_with_directory_id(spark, tmp_path: Path) -> None:
    path = tmp_path / 'GCST1'
    spark.createDataFrame([Row(value=1)]).write.mode('overwrite').parquet(str(path))
    issues: dict[str, str] = {}
    assert _resolve_root_dataset_ids(spark, [('GCST1', str(path))], issues) == [
        ('GCST1', str(path))
    ]
    assert 'GCST1' in issues


def test_study_batch_boundaries() -> None:
    studies = [type('Study', (), {'study_id': str(i)})() for i in range(5)]
    assert [len(batch) for batch in _chunks(studies, 2)] == [2, 2, 1]


def test_score_edge_schema_and_chromosome_are_validated(spark, tmp_path: Path) -> None:
    """Malformed flat edges are rejected before annotation scoring."""
    missing = spark.createDataFrame([Row(variantId='1_1_A_G')])
    missing_path = str(tmp_path / 'missing-edge')
    missing.write.mode('overwrite').parquet(missing_path)
    with pytest.raises(ValueError, match='missing required'):
        _read_edges(spark, missing_path, '1')


def test_summary_statistics_coordinates_are_validated(spark) -> None:
    """Summary-statistics coordinates must agree with variantId."""
    valid = spark.createDataFrame(
        [
            Row(
                studyId='GCST1',
                variantId='1_100_A_G',
                chromosome='1',
                position=100,
                beta=0.1,
                standardError=0.2,
                sampleSize=100_000.0,
            )
        ]
    )
    study = Row(nSamples=100_000, nCases=None, nControls=None)
    assert _prepare_sumstats(valid, study).count() == 1
    invalid = valid.withColumn('chromosome', f.concat(f.col('chromosome'), f.lit('0')))
    with pytest.raises(ValueError, match='genome-build-incompatible'):
        _prepare_sumstats(invalid, study)


def test_success_marker_is_checked_for_local_outputs(spark, tmp_path: Path) -> None:
    """Resumability requires the explicit Spark success marker."""
    output = tmp_path / 'out'
    output.mkdir()
    assert not success_exists(spark, str(output))
    (output / '_SUCCESS').write_text('')
    assert success_exists(spark, str(output))


def test_two_chromosome_two_dataset_annotation_and_batched_regression(
    spark, tmp_path: Path, monkeypatch
) -> None:
    """Run both generic tasks on a fixture with two datasets and two studies."""
    import importlib

    annotation_module = importlib.import_module('pts.pyspark.ldsc_cts_annotation')
    regression_module = importlib.import_module('pts.pyspark.ldsc_cts_regression')

    class FakeSession:
        def __init__(self, **_kwargs):
            self.spark = spark

    monkeypatch.setattr(annotation_module, 'Session', FakeSession)
    monkeypatch.setattr(regression_module, 'Session', FakeSession)

    target_path = tmp_path / 'target'
    spark.createDataFrame([
        Row(id='GENE1', genomicLocation=Row(chromosome='1', start=95, end=105)),
        Row(id='GENE2', genomicLocation=Row(chromosome='1', start=115, end=125)),
        Row(id='GENE3', genomicLocation=Row(chromosome='2', start=195, end=205)),
        Row(id='GENE4', genomicLocation=Row(chromosome='2', start=215, end=225)),
    ]).write.mode('overwrite').parquet(str(target_path))

    specificity_root = tmp_path / 'specificity'
    for dataset_id, annotation in [('one', 'cell'), ('two', 'cell_two')]:
        path = specificity_root / f'{dataset_id}.csv.gz'
        spark.createDataFrame([
            Row(gene='GENE1', **{annotation: 1.0}),
            Row(gene='GENE2', **{annotation: 2.0}),
            Row(gene='GENE3', **{annotation: 1.0}),
            Row(gene='GENE4', **{annotation: 2.0}),
        ]).write.mode('overwrite').option('header', 'true').csv(str(path))

    variants = [
        ('1_100_A_G', '1', 100, 'A', 'G'),
        ('1_120_C_T', '1', 120, 'C', 'T'),
        ('2_200_G_A', '2', 200, 'G', 'A'),
        ('2_220_T_C', '2', 220, 'T', 'C'),
    ]
    score_path = tmp_path / 'baseline' / 'nfe' / 'baseline_ld_scores.tsv.gz'
    spark.createDataFrame([Row(variantId=value[0]) for value in variants]).write.mode(
        'overwrite'
    ).option('header', 'true').csv(str(score_path))

    edge_manifest = {'nfe': {}}
    edge_rows = {
        1: [
            Row(variantId='1_100_A_G', tagVariantId='1_100_A_G', r=1.0),
            Row(variantId='1_100_A_G', tagVariantId='1_120_C_T', r=0.5),
        ],
        2: [
            Row(variantId='2_200_G_A', tagVariantId='2_200_G_A', r=1.0),
            Row(variantId='2_200_G_A', tagVariantId='2_220_T_C', r=0.25),
        ],
    }
    for chromosome, rows in edge_rows.items():
        edge_path = tmp_path / f'edges-{chromosome}'
        spark.createDataFrame(rows).write.mode('overwrite').parquet(str(edge_path))
        edge_manifest['nfe'][f'chr{chromosome}'] = str(edge_path)
    manifest_path = tmp_path / 'edges.json'
    manifest_path.write_text(json.dumps(edge_manifest))

    from pts.pyspark.ldsc_cts_annotation import ldsc_cts_annotation

    annotation_root = tmp_path / 'annotations'
    annotation_source = {
        'specificity_root': str(specificity_root),
        'target_index': str(target_path),
        'reference_root': str(tmp_path / 'baseline'),
        'edge_manifest': str(manifest_path),
    }
    annotation_settings = {
        'datasets': [
            {'id': 'one', 'path': 'one.csv.gz'},
            {'id': 'two', 'path': 'two.csv.gz'},
        ],
        'populations': ['nfe'],
        'chromosomes': [1, 2],
        'format': 'csv',
        'separator': ',',
        'score_variant_format': 'csv',
        'score_variant_sep': ',',
        'window_kb': 0,
    }
    ldsc_cts_annotation(annotation_source, {'annotations': str(annotation_root)}, annotation_settings, {})

    catalog = spark.read.parquet(str(annotation_root / '_catalog'))
    assert catalog.count() == 4
    assert set(catalog.select('specificityId').distinct().toPandas()['specificityId']) == {'one', 'two'}
    # The second run sees both output markers and must not recompute units.
    ldsc_cts_annotation(annotation_source, {'annotations': str(annotation_root)}, annotation_settings, {})
    assert spark.read.parquet(str(annotation_root / '_catalog')).count() == 4

    baseline_root = tmp_path / 'baseline' / 'nfe'
    spark.createDataFrame([
        Row(
            variantId=variant_id,
            CHR=chromosome,
            BP_hg38=position,
            ref=ref,
            alt=alt,
            base=1.0 + index * 0.1,
        )
        for index, (variant_id, chromosome, position, ref, alt) in enumerate(variants)
    ]).write.mode('overwrite').option('header', 'true').option('sep', '\t').csv(
        str(baseline_root / 'baseline_ld_scores.tsv.gz')
    )
    spark.createDataFrame([Row(annotation='base', M=4.0)]).write.mode('overwrite').parquet(
        str(baseline_root / 'baseline_m')
    )
    summaries_root = tmp_path / 'summaries'
    for study_id, offset in [('GCSTFIX1', 0.0), ('GCSTFIX2', 0.01)]:
        spark.createDataFrame([
            Row(
                studyId=study_id,
                variantId=variant_id,
                chromosome=chromosome,
                position=position,
                beta=0.1 + index * 0.03 + offset,
                standardError=0.2,
                sampleSize=100_000.0,
            )
            for index, (variant_id, chromosome, position, _ref, _alt) in enumerate(variants)
        ]).write.mode('overwrite').parquet(str(summaries_root / study_id))
    spark.createDataFrame([
        Row(
            studyId=study_id,
            nCases=50_000,
            nControls=50_000,
            nSamples=100_000,
            ldPopulationStructure=[Row(ldPopulation='nfe', relativeSampleSize=1.0)],
            analysisFlags=['exwas'],
        )
        for study_id in ('GCSTFIX1', 'GCSTFIX2')
    ]).write.mode('overwrite').parquet(str(tmp_path / 'study'))

    from pts.pyspark.ldsc_cts_regression import ldsc_cts_regression

    result_path = tmp_path / 'results'
    regression_source = {
        'summary_statistics_root': str(summaries_root),
        'study_index': str(tmp_path / 'study'),
        'annotations': str(annotation_root),
        'reference_root': str(tmp_path / 'baseline'),
    }
    regression_settings = {
        'study_batch_size': 1,
        'study_ids': ['GCSTFIX1', 'GCSTFIX2'],
        'chromosomes': [1, 2],
        'min_samples': 1,
        'n_blocks': 2,
        'intercept': 1.0,
        'baseline_format': 'csv',
        'baseline_sep': '\t',
    }
    ldsc_cts_regression(regression_source, {'results': str(result_path)}, regression_settings, {})
    result = spark.read.parquet(str(result_path)).collect()
    assert {row['studyId'] for row in result} == {'GCSTFIX1', 'GCSTFIX2'}
    assert {row['specificityId'] for row in result} == {'one', 'two'}
    assert {row['analysisId'] for row in result} == {'GCSTFIX1', 'GCSTFIX2'}
    assert {row['ld_ancestry'] for row in result} == {'nfe'}
    assert {row['runStatus'] for row in result} == {'success'}
    assert {row['n_snps_used'] for row in result} == {4}
    ldsc_cts_regression(regression_source, {'results': str(result_path)}, regression_settings, {})
    assert spark.read.parquet(str(result_path)).count() == len(result)


def test_annotation_catalog_requires_complete_chromosomes(spark, tmp_path: Path) -> None:
    root = tmp_path / 'annotations'
    catalog = root / '_catalog'
    catalog.parent.mkdir(parents=True)
    spark.createDataFrame([
        Row(
            specificityId='one',
            ldPopulation='nfe',
            chromosome=1,
            ldScoresPath=str(tmp_path / 'chr1' / 'ld_scores'),
            mAnnotPath=str(tmp_path / 'chr1' / 'm_annot'),
        )
    ]).write.mode('overwrite').parquet(str(catalog))
    with pytest.raises(ValueError, match='no complete references'):
        _read_annotation_catalog(spark, str(root), ['1', '2'])
