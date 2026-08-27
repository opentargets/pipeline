"""Focused tests for the PTS LDSC-CTS orchestration helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pyspark.sql.functions as f
import pytest
from gentropy.method.ldsc import infer_ld_ancestry
from pyspark.sql import Row

from pts.pyspark.ldsc_cts_annotation import _read_edges
from pts.pyspark.ldsc_cts_regression import _prepare_sumstats
from pts.pyspark.ldsc_cts_utils import discover_edge_manifest, success_exists


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


def test_study_index_plurality_is_independent_of_analysis_id() -> None:
    """A descriptive ID containing EUR/EAS cannot select the LD ancestry."""
    structure = [
        {'ldPopulation': 'eas', 'relativeSampleSize': 0.6},
        {'ldPopulation': 'nfe', 'relativeSampleSize': 0.4},
    ]
    assert infer_ld_ancestry(structure) == 'eas'


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


def test_two_chromosome_annotation_and_regression_fixture(spark, tmp_path: Path, monkeypatch) -> None:
    """Run both PTS tasks on a tiny fixture with manually checkable dimensions."""
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
        Row(id='GENE2', genomicLocation=Row(chromosome='2', start=195, end=205)),
    ]).write.mode('overwrite').parquet(str(target_path))

    specificity_path = tmp_path / 'specificity'
    spark.createDataFrame([
        Row(gene='GENE1', cell=1.0),
        Row(gene='GENE2', cell=1.0),
    ]).write.mode('overwrite').option('header', 'true').csv(str(specificity_path))

    score_path = tmp_path / 'score_variants'
    spark.createDataFrame([
        Row(variantId='1_100_A_G'),
        Row(variantId='1_120_C_T'),
        Row(variantId='2_200_G_A'),
        Row(variantId='2_220_T_C'),
    ]).write.mode('overwrite').option('header', 'true').csv(str(score_path))

    edge_manifest = {'nfe': {}}
    for chromosome, rows in {
        '1': [
            Row(variantId='1_100_A_G', tagVariantId='1_100_A_G', r=1.0),
            Row(variantId='1_100_A_G', tagVariantId='1_120_C_T', r=0.5),
        ],
        '2': [
            Row(variantId='2_200_G_A', tagVariantId='2_200_G_A', r=1.0),
            Row(variantId='2_200_G_A', tagVariantId='2_220_T_C', r=0.25),
        ],
    }.items():
        edge_path = tmp_path / f'edges-{chromosome}'
        spark.createDataFrame(rows).write.mode('overwrite').parquet(str(edge_path))
        edge_manifest['nfe'][f'chr{chromosome}'] = str(edge_path)
    manifest_path = tmp_path / 'edges.json'
    manifest_path.write_text(json.dumps(edge_manifest))

    from pts.pyspark.ldsc_cts_annotation import ldsc_cts_annotation

    annotation_root = tmp_path / 'annotations'
    ldsc_cts_annotation(
        {
            'specificity': str(specificity_path),
            'target_index': str(target_path),
            'score_variants': str(score_path),
            'edge_manifest': str(manifest_path),
        },
        {'annotations': str(annotation_root)},
        {
            'specificity_id': 'fixture',
            'ancestry': 'nfe',
            'chromosomes': [1, 2],
            'specificity_format': 'csv',
            'specificity_sep': ',',
            'score_variant_format': 'csv',
            'score_variant_sep': ',',
            'window_kb': 1,
        },
        {},
    )

    for chromosome, expected_m, expected_score in [('1', 2.0, 1.25), ('2', 2.0, 1.0625)]:
        output = annotation_root / 'nfe' / f'chr{chromosome}'
        scores = spark.read.parquet(str(output / 'ld_scores'))
        m_values = {
            row['annotation']: row['M']
            for row in spark.read.parquet(str(output / 'm_annot')).collect()
        }
        lead_variant = '1_100_A_G' if chromosome == '1' else '2_200_G_A'
        score_row = scores.filter(f"variantId = '{lead_variant}'").filter(
            "annotation = 'cell'"
        ).first()
        assert score_row is not None
        assert score_row['ldScore'] == pytest.approx(expected_score)
        assert m_values['cell'] == pytest.approx(expected_m)

    from pts.pyspark.ldsc_cts_regression import ldsc_cts_regression

    variant_rows = [
        ('1_100_A_G', '1', 100, 'A', 'G'),
        ('1_120_C_T', '1', 120, 'C', 'T'),
        ('2_200_G_A', '2', 200, 'G', 'A'),
        ('2_220_T_C', '2', 220, 'T', 'C'),
    ]
    spark.createDataFrame([
        Row(
            studyId='GCSTFIX',
            variantId=variant_id,
            chromosome=chromosome,
            position=position,
            beta=0.1 + index * 0.03,
            standardError=0.2,
            sampleSize=100_000.0,
        )
        for index, (variant_id, chromosome, position, _ref, _alt) in enumerate(variant_rows)
    ]).write.mode('overwrite').parquet(str(tmp_path / 'sumstats'))

    baseline_root = tmp_path / 'baseline' / 'nfe'
    baseline_root.mkdir(parents=True)
    spark.createDataFrame([
        Row(
            variantId=variant_id,
            CHR=chromosome,
            BP_hg38=position,
            ref=ref,
            alt=alt,
            base=1.0 + index * 0.1,
        )
        for index, (variant_id, chromosome, position, ref, alt) in enumerate(variant_rows)
    ]).write.mode('overwrite').option('header', 'true').option('sep', '\t').csv(
        str(baseline_root / 'baseline_ld_scores.tsv.gz')
    )
    spark.createDataFrame([Row(annotation='base', M=4.0)]).write.mode('overwrite').parquet(
        str(baseline_root / 'baseline_m')
    )
    spark.createDataFrame([
        Row(
            studyId='GCSTFIX',
            nCases=50_000,
            nControls=50_000,
            nSamples=100_000,
            ldPopulationStructure=[Row(ldPopulation='nfe', relativeSampleSize=1.0)],
            analysisFlags=['exwas'],
        )
    ]).write.mode('overwrite').parquet(str(tmp_path / 'study'))

    result_path = tmp_path / 'results'
    ldsc_cts_regression(
        {
            'summary_statistics': str(tmp_path / 'sumstats'),
            'study_index': str(tmp_path / 'study'),
            'annotation_root': str(annotation_root),
            'baseline_root': str(tmp_path / 'baseline'),
        },
        {'results': str(result_path)},
        {
            'study_id': 'GCSTFIX',
            'analysis_id': 'contains-EUR-but-not-used',
            'specificity_id': 'fixture',
            'chromosomes': [1, 2],
            'min_samples': 1,
            'n_blocks': 2,
            'baseline_format': 'csv',
            'baseline_sep': '\t',
        },
        {},
    )
    result = spark.read.parquet(str(result_path)).collect()
    assert len(result) == 1
    assert {row['ld_ancestry'] for row in result} == {'nfe'}
    assert {row['analysisId'] for row in result} == {'contains-EUR-but-not-used'}
    assert {row['runStatus'] for row in result} == {'success'}
    assert {row['n_snps_used'] for row in result} == {4}
