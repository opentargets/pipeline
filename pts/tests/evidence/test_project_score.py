"""Tests for Project Score (crispr) evidence."""

from __future__ import annotations

import gzip
import random
from math import isnan
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as f

from pts.transformers.evidence.project_score import (
    CRISPR,
    PMID,
    SCORE,
    disease_cell_lines,
    generate,
    parse_msi_status,
    passport_cell_lines,
    project_score,
)
from tests.evidence.fixtures import write_disease, write_literature, write_target

GENE_SCORES_HEADER = ['cancerType', 'targetSymbol', 'diseaseName', 'diseaseId', 'targetId', 'PRIORITY']
CELL_TYPES_HEADER = ['SAMPLE_ID', 'CellLineName', 'TISSUE', 'CANCER_TYPE', 'CMP_ID']
PASSPORT_HEADER = ['model_id', 'model_name', 'model_treatment', 'msi_status', 'tissue']
MAPPING_HEADER = ['tissueFromSource', 'tissueId', 'tissueName']

CELL_LINE = pl.Struct({'tissue': pl.String, 'name': pl.String, 'id': pl.String, 'tissueId': pl.String})
RAW_SCHEMA = {
    'targetFromSource': pl.String,
    'diseaseFromSource': pl.String,
    'diseaseFromSourceMappedId': pl.String,
    'resourceScore': pl.Float32,
    'targetFromSourceId': pl.String,
    'literature': pl.List(pl.String),
    'datasourceId': pl.String,
    'datatypeId': pl.String,
    'diseaseCellLines': pl.List(CELL_LINE),
}

SKIN = {'tissue': 'skin of body', 'tissueId': 'UBERON_0002097'}
A375 = {**SKIN, 'name': 'A-375', 'id': 'SIDM1'}
SKMEL = {**SKIN, 'name': 'SK-MEL-2', 'id': 'SIDM2'}
HT29 = {'tissue': 'large intestine', 'name': 'HT-29', 'id': 'SIDM3', 'tissueId': 'UBERON_0000059'}


def _write_delimited(path: Path, header: list[str], rows: list[list[str]], separator: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = '\n'.join(separator.join(row) for row in [header, *rows]) + '\n'
    if path.suffix == '.gz':
        path.write_bytes(gzip.compress(text.encode()))
    else:
        path.write_text(text)
    return str(path)


def _write_inputs(
    src: Path,
    gene_scores: list[list[str]],
    cell_types: list[list[str]],
    passport: list[list[str]],
    mapping: list[list[str]],
) -> dict:
    return {
        'gene_scores': _write_delimited(src / 'mapped_diseases.tsv', GENE_SCORES_HEADER, gene_scores, '\t'),
        'cell_types': _write_delimited(src / 'cell_types.tsv', CELL_TYPES_HEADER, cell_types, '\t'),
        'cell_passport': _write_delimited(src / 'cell_passport.csv.gz', PASSPORT_HEADER, passport, ','),
        'cell_line_mapping': _write_delimited(src / 'mapping.csv', MAPPING_HEADER, mapping, ','),
    }


def _sources(src: Path) -> dict:
    return _write_inputs(
        src,
        gene_scores=[
            ['Melanoma', 'BRAF', 'melanoma', 'EFO_1', 'ENSG1', '83.8'],
            ['Colorectal.Carcinoma', 'WRN', 'colorectal carcinoma', 'EFO_2', 'ENSG2', '52.8'],
            ['panCancer', 'BRAF', 'cancer', 'EFO_3', 'ENSG1', '90'],
            ['', 'BRAF', 'cancer', 'EFO_4', 'ENSG1', '90'],
            ['Glioma', 'EGFR', 'glioma', 'EFO_5', 'ENSG3', '41.5'],
        ],
        cell_types=[
            ['A375_R1', 'A375', 'Skin', 'Melanoma', 'SIDM1'],
            ['A375_R2', 'A375', 'Skin', 'Melanoma', 'SIDM1'],
            ['SKMEL2_R1', 'SKMEL2', 'Skin', 'Melanoma', 'SIDM2'],
            ['HT29_R1', 'HT29', 'Large Intestine', 'Colorectal.Carcinoma', 'SIDM3'],
            ['GONE_R1', 'GONE', 'Skin', 'Melanoma', 'SIDM9'],
        ],
        passport=[
            ['SIDM2', 'SK-MEL-2', '"Vemurafenib\nDabrafenib"', 'MSS', 'Skin'],
            ['SIDM1', 'A-375', '', 'MSS', 'Skin'],
            ['SIDM3', 'HT-29', '', 'MSI', 'Large Intestine'],
            ['SIDM4', 'UNUSED', '', '', 'Skin'],
        ],
        mapping=[
            ['skin', 'UBERON_0002097', 'skin of body'],
            ['large intestine', 'UBERON_0000059', 'large intestine'],
        ],
    )


def test_generate_gives_one_row_per_hit_with_its_cancer_type_cell_lines(tmp_path: Path) -> None:
    raw = generate(**_sources(tmp_path)).collect()

    assert raw.schema == pl.Schema(RAW_SCHEMA)
    assert raw.to_dicts() == [
        {
            'targetFromSource': 'BRAF',
            'diseaseFromSource': 'melanoma',
            'diseaseFromSourceMappedId': 'EFO_1',
            'resourceScore': pytest.approx(83.8, rel=1e-7),
            'targetFromSourceId': 'ENSG1',
            'literature': [PMID],
            'datasourceId': 'crispr',
            'datatypeId': 'affected_pathway',
            'diseaseCellLines': [A375, SKMEL],
        },
        {
            'targetFromSource': 'WRN',
            'diseaseFromSource': 'colorectal carcinoma',
            'diseaseFromSourceMappedId': 'EFO_2',
            'resourceScore': pytest.approx(52.8, rel=1e-7),
            'targetFromSourceId': 'ENSG2',
            'literature': [PMID],
            'datasourceId': 'crispr',
            'datatypeId': 'affected_pathway',
            'diseaseCellLines': [HT29],
        },
        {
            'targetFromSource': 'EGFR',
            'diseaseFromSource': 'glioma',
            'diseaseFromSourceMappedId': 'EFO_5',
            'resourceScore': 41.5,
            'targetFromSourceId': 'ENSG3',
            'literature': [PMID],
            'datasourceId': 'crispr',
            'datatypeId': 'affected_pathway',
            'diseaseCellLines': None,
        },
    ]


def test_generate_drops_pancancer_and_null_cancer_type_hits(tmp_path: Path) -> None:
    raw = generate(**_sources(tmp_path)).collect()

    assert 'EFO_3' not in raw['diseaseFromSourceMappedId'].to_list()
    assert 'EFO_4' not in raw['diseaseFromSourceMappedId'].to_list()


def test_generate_casts_priority_to_float32(tmp_path: Path) -> None:
    raw = generate(**_sources(tmp_path)).collect()

    assert raw['resourceScore'].to_list()[1] == float(np.float32(52.8))
    assert raw['resourceScore'].to_list()[1] != 52.8


def test_disease_cell_lines_are_sorted_and_unique_per_lower_cased_cancer_type() -> None:
    passport = pl.DataFrame({
        'model_id': ['SIDM2', 'SIDM1', 'SIDM3'],
        'model_name': ['SK-MEL-2', 'A-375', 'HT-29'],
        'msi_status': ['MSS', 'MSS', 'MSI'],
        'tissue': ['Skin', 'Skin', 'Large Intestine'],
    })
    mapping = pl.DataFrame({
        'tissueFromSource': ['skin', 'large intestine'],
        'tissueId': ['UBERON_0002097', 'UBERON_0000059'],
        'tissueName': ['skin of body', 'large intestine'],
    })
    cell_types = pl.DataFrame({
        'CANCER_TYPE': ['Melanoma', 'MELANOMA', 'Melanoma', 'Colorectal.Carcinoma', 'Melanoma'],
        'CMP_ID': ['SIDM2', 'SIDM1', 'SIDM1', 'SIDM3', 'SIDM9'],
    })

    got = disease_cell_lines(passport, mapping, cell_types).sort('cancerType', nulls_last=True)

    assert got.to_dicts() == [
        {'cancerType': 'colorectal.carcinoma', 'diseaseCellLines': [HT29]},
        {'cancerType': 'melanoma', 'diseaseCellLines': [A375, SKMEL]},
    ]


def test_passport_cell_lines_strip_dashes_from_the_name_only() -> None:
    passport = pl.DataFrame({
        'model_id': ['SIDM1', 'SIDM3', 'SIDM4', 'SIDM5'],
        'model_name': ['A-375', 'HT-29', 'X-1', 'Y'],
        'msi_status': ['MSS', 'MSI', 'MSI-H', None],
        'tissue': ['Skin', 'Large Intestine', 'Unmapped', None],
    })
    mapping = pl.DataFrame({
        'tissueFromSource': ['skin', 'large intestine'],
        'tissueId': ['UBERON_0002097', 'UBERON_0000059'],
        'tissueName': ['skin of body', 'large intestine'],
    })

    got = passport_cell_lines(passport, mapping)

    assert got.columns == ['name', 'id', 'biomarkerList', 'diseaseCellLine']
    assert got.to_dicts() == [
        {
            'name': 'A375',
            'id': 'SIDM1',
            'biomarkerList': [{'name': 'MSS', 'description': 'Microsatellite stable'}],
            'diseaseCellLine': A375,
        },
        {
            'name': 'HT29',
            'id': 'SIDM3',
            'biomarkerList': [{'name': 'MSI', 'description': 'Microsatellite instable'}],
            'diseaseCellLine': HT29,
        },
        {
            'name': 'X1',
            'id': 'SIDM4',
            'biomarkerList': [None],
            'diseaseCellLine': {'tissue': None, 'name': 'X-1', 'id': 'SIDM4', 'tissueId': None},
        },
        {
            'name': 'Y',
            'id': 'SIDM5',
            'biomarkerList': [None],
            'diseaseCellLine': {'tissue': None, 'name': 'Y', 'id': 'SIDM5', 'tissueId': None},
        },
    ]


def test_parse_msi_status_is_null_for_anything_but_msi_or_mss() -> None:
    df = pl.DataFrame({'s': ['MSI', 'MSS', 'msi', '', None]})
    got = df.select(parse_msi_status(pl.col('s')).alias('b'))['b'].to_list()
    assert got == [
        {'name': 'MSI', 'description': 'Microsatellite instable'},
        {'name': 'MSS', 'description': 'Microsatellite stable'},
        None,
        None,
        None,
    ]


def test_score_clamps_to_the_output_range() -> None:
    # measured against spark
    df = pl.DataFrame({'resourceScore': [0.0, 41.5, 100.0, 150.0, None, float('nan')]})
    got = df.select(SCORE.alias('s'))['s'].to_list()
    assert got[:4] == pytest.approx([0.415, 0.415, 1.0, 1.0])
    assert got[4:] == [None, None]


def test_score_of_a_float32_priority_differs_from_its_float64_value() -> None:
    as_float32 = pl.DataFrame({'resourceScore': [52.8]}, schema={'resourceScore': pl.Float32})
    as_float64 = pl.DataFrame({'resourceScore': [52.8]}, schema={'resourceScore': pl.Float64})
    widened = pl.DataFrame({'resourceScore': [float(np.float32(52.8))]})
    got = as_float32.select(SCORE.alias('s'))['s'][0]
    assert got != as_float64.select(SCORE.alias('s'))['s'][0]
    assert got == widened.select(SCORE.alias('s'))['s'][0]


def test_crispr_unique_fields_are_the_config_ones() -> None:
    assert CRISPR.datasource_id == 'crispr'
    assert CRISPR.unique_fields == ['targetId', 'targetFromSourceId', 'diseaseId', 'datasourceId']


def test_project_score_step_scores_and_splits(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    sources = _sources(src)
    dst = tmp_path / 'out'

    project_score(
        {
            **sources,
            'disease': write_disease(src / 'disease', ['EFO_1', 'EFO_2']),
            'target': write_target(src / 'target', ['ENSG1', 'ENSG2']),
            'literature': write_literature(src / 'literature', {PMID: '2024-01-16'}),
        },
        {'evidence': str(dst / 'valid'), 'failed_evidence': str(dst / 'failed')},
        {},
        None,
    )

    valid = pl.read_parquet(dst / 'valid').sort('targetId')
    failed = pl.read_parquet(dst / 'failed')
    assert valid.select('targetId', 'diseaseId', 'datasourceId').rows() == [
        ('ENSG1', 'EFO_1', 'crispr'),
        ('ENSG2', 'EFO_2', 'crispr'),
    ]
    widened = float(np.float32(52.8))
    assert valid['score'].to_list() == [
        pytest.approx(0.585 * (float(np.float32(83.8)) - 41.5) / 58.5 + 0.415),
        pl.DataFrame({'resourceScore': [widened]}).select(SCORE.alias('s'))['s'][0],
    ]
    assert valid['publicationDate'].to_list() == ['2024-01-16', '2024-01-16']
    assert valid['diseaseCellLines'].to_list() == [[A375, SKMEL], [HT29]]
    assert failed['diseaseFromSourceMappedId'].to_list() == ['EFO_5']


# ------------------------------------------------------------------------------------------------
# Exact parity with the score SQL the generic step ran for crispr.
# ------------------------------------------------------------------------------------------------

CRISPR_SQL = """
CASE
  WHEN resourceScore IS NULL OR isnan(resourceScore) THEN NULL
  ELSE LEAST(1.0, GREATEST(0.415,
    (1.0 - 0.415) * (CAST(resourceScore AS DOUBLE) - 41.5) / (100 - 41.5) + 0.415
  ))
END
"""


def _spark_scores(spark: SparkSession, values: list, ddl_type: str) -> list:
    df = spark.createDataFrame([(v,) for v in values], f'resourceScore {ddl_type}')
    return [row.s for row in df.select(f.expr(CRISPR_SQL).cast('double').alias('s')).collect()]


def _polars_scores(values: list, dtype: pl.DataType) -> list:
    df = pl.DataFrame({'resourceScore': values}, schema={'resourceScore': dtype})
    return df.select(SCORE.alias('s'))['s'].to_list()


def _same(a: list, b: list) -> bool:
    return len(a) == len(b) and all(
        (x is None and y is None) or (x is not None and y is not None and (x == y or (isnan(x) and isnan(y))))
        for x, y in zip(a, b, strict=True)
    )


def test_crispr_score_matches_spark_bit_for_bit(spark: SparkSession) -> None:
    rng = random.Random(0)
    values: list[float | None] = [0.0, -5.0, 36.0, 41.4, 41.5, 41.6, 44.8, 52.8, 83.8, 99.99, 100.0, 100.1, 1e30]
    values += [-1e30, float('inf'), float('-inf'), float('nan'), None]
    values += [rng.uniform(30.0, 110.0) for _ in range(5000)]
    expected = _spark_scores(spark, values, 'DOUBLE')
    assert _same(_polars_scores(values, pl.Float64), expected)


def test_crispr_float_score_matches_spark_bit_for_bit(spark: SparkSession) -> None:
    rng = random.Random(1)
    values: list[float | None] = [41.5, 52.8, 78.98181818, 83.8, float('nan'), None]
    values += [rng.uniform(30.0, 110.0) for _ in range(5000)]
    expected = _spark_scores(spark, values, 'FLOAT')
    assert _same(_polars_scores(values, pl.Float32), expected)


def test_crispr_integer_score_matches_spark(spark: SparkSession) -> None:
    values = [*range(-10, 151), None]
    expected = _spark_scores(spark, values, 'BIGINT')
    assert _same(_polars_scores(values, pl.Int64), expected)
