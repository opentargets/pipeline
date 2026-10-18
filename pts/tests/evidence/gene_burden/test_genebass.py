"""Tests for the Genebass gene burden parser."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from pyspark.sql import SparkSession

from pts.pyspark.gene_burden import process_genebass_gene_burden
from pts.schemas.gene_burden import GENE_BURDEN_PARSED
from pts.transformers.evidence.gene_burden.genebass import parse_genebass
from tests.evidence.gene_burden.fixtures import genebass_row, write_genebass
from tests.evidence.gene_burden.logs import logged
from tests.evidence.gene_burden.spark_rows import multiset


def _parse(tmp: Path, *files: list[dict[str, Any]]) -> pl.DataFrame:
    return parse_genebass(write_genebass(tmp, *files)).collect()


def test_output_has_the_parsed_schema(tmp_path: Path) -> None:
    assert _parse(tmp_path, [genebass_row()]).schema == GENE_BURDEN_PARSED


def test_columns(tmp_path: Path) -> None:
    row = _parse(tmp_path, [genebass_row()]).row(0, named=True)
    assert row == {
        'datasourceId': 'gene_burden',
        'datatypeId': 'genetic_association',
        'allelicRequirements': None,
        'targetFromSourceId': 'ENSG1',
        'diseaseFromSource': 'White blood cell count',
        'pValueMantissa': 2.0,
        'pValueExponent': -9,
        'beta': 0.25,
        'betaConfidenceIntervalLower': 0.125,
        'betaConfidenceIntervalUpper': 0.375,
        'oddsRatio': None,
        'oddsRatioConfidenceIntervalLower': None,
        'oddsRatioConfidenceIntervalUpper': None,
        'resourceScore': 2e-9,
        'ancestry': 'EUR',
        'ancestryId': 'HANCESTRO_0005',
        'literature': ['36778668'],
        'projectId': 'Genebass',
        'cohortId': 'UK Biobank 450k',
        'releaseVersion': None,
        'studySampleSize': 10000,
        'studyCases': 1000,
        'studyCasesWithQualifyingVariants': None,
        'statisticalMethod': 'pLoF',
        'statisticalMethodOverview': 'Burden test carried out with rare pLOF variants.',
        'urls': None,
        'sex': None,
        'diseaseFromSourceId': '30000',
        'curatedDiseaseFromSourceMappedId': None,
    }


def test_binary_traits_report_beta(tmp_path: Path) -> None:
    row = _parse(tmp_path, [genebass_row(trait_type='icd_first_occurrence', BETA_Burden=-0.5, SE_Burden=0.25)]).row(
        0, named=True
    )
    assert (row['beta'], row['betaConfidenceIntervalLower'], row['betaConfidenceIntervalUpper']) == (-0.5, -0.75, -0.25)
    assert row['oddsRatio'] is None


@pytest.mark.parametrize(
    ('n_cases', 'n_controls', 'sample_size'),
    [(1000, 9000, 10000), (1000, None, 1000), (None, 9000, None)],
)
def test_sample_size_counts_missing_controls_as_zero(
    tmp_path: Path, n_cases: int | None, n_controls: int | None, sample_size: int | None
) -> None:
    row = _parse(tmp_path, [genebass_row(n_cases=n_cases, n_controls=n_controls)]).row(0, named=True)
    assert (row['studySampleSize'], row['studyCases']) == (sample_size, n_cases)


@pytest.mark.parametrize(
    ('pvalue', 'kept'),
    [(6.7e-7, True), (math.nextafter(6.7e-7, 1.0), False), (0.5, False), (math.nan, False), (None, False)],
)
def test_pvalue_filter(tmp_path: Path, pvalue: float | None, kept: bool) -> None:
    assert _parse(tmp_path, [genebass_row(), genebass_row(gene_id='ENSG2', Pvalue_Burden=pvalue)]).height == 1 + int(
        kept
    )


@pytest.mark.parametrize(('trait_type', 'kept'), [('categorical', False), (None, False), ('continuous', True)])
def test_trait_type_filter(tmp_path: Path, trait_type: str | None, kept: bool) -> None:
    assert _parse(tmp_path, [genebass_row(trait_type=trait_type)]).height == int(kept)


def test_zero_pvalue_minimum_is_over_the_whole_input(tmp_path: Path) -> None:
    # the minimum comes from a categorical row in another file, dropped afterwards
    rows, log = logged(
        lambda: _parse(
            tmp_path,
            [genebass_row(Pvalue_Burden=0.0), genebass_row(gene_id='ENSG2', Pvalue_Burden=1e-10)],
            [
                genebass_row(gene_id='ENSG3', trait_type='categorical', Pvalue_Burden=1e-30),
                genebass_row(Pvalue_Burden=0.5),
            ],
        )
    )
    assert dict(zip(rows['targetFromSourceId'], rows['resourceScore'], strict=True)) == {
        'ENSG1': 1e-30,
        'ENSG2': 1e-10,
    }
    assert 'There are 1 Genebass evidence with a p-value of 0.0; they take the minimum, 1e-30.' in log


def test_zero_pvalue_without_a_significant_minimum_is_dropped(tmp_path: Path) -> None:
    assert _parse(tmp_path, [genebass_row(Pvalue_Burden=0.0), genebass_row(Pvalue_Burden=0.01)]).height == 0


def test_raw_mantissa_at_a_power_of_ten(tmp_path: Path) -> None:
    row = _parse(tmp_path, [genebass_row(Pvalue_Burden=1e-8)]).row(0, named=True)
    assert (row['pValueMantissa'], row['pValueExponent']) == (10.0, -9)


def test_method_overview(tmp_path: Path) -> None:
    annotations = ['pLoF', 'missense|LC', 'synonymous', 'pLoF|missense|LC', 'other', None]
    rows = _parse(tmp_path, [genebass_row(gene_id=str(i), annotation=a) for i, a in enumerate(annotations)])
    assert dict(zip(rows['statisticalMethod'], rows['statisticalMethodOverview'], strict=True)) == {
        'pLoF': 'Burden test carried out with rare pLOF variants.',
        'missense|LC': (
            'Burden test carried out with rare missense variants including low-confidence pLOF and in-frame indels.'
        ),
        'synonymous': 'Burden test carried out with rare synonymous variants.',
        'pLoF|missense|LC': 'Burden test carried out with pLOF or missense variants.',
        'other': 'other',
        None: None,
    }


def test_rows_are_distinct(tmp_path: Path) -> None:
    # differing only in columns the parser drops, across files
    rows = _parse(
        tmp_path, [genebass_row(), genebass_row(category='other')], [genebass_row(Pvalue=0.1, gene_symbol='X')]
    )
    assert rows.height == 1


def test_negative_zero_beta_is_a_duplicate_of_zero(tmp_path: Path) -> None:
    rows = _parse(
        tmp_path, [genebass_row(BETA_Burden=-0.0, SE_Burden=0.5), genebass_row(BETA_Burden=0.0, SE_Burden=0.5)]
    )
    assert rows.height == 1
    assert math.copysign(1.0, rows['beta'].item()) == 1.0


def test_no_significant_rows_give_an_empty_frame(tmp_path: Path) -> None:
    rows = _parse(tmp_path, [genebass_row(Pvalue_Burden=0.5)])
    assert rows.height == 0
    assert rows.schema == GENE_BURDEN_PARSED


def test_matches_spark(tmp_path: Path, spark: SparkSession) -> None:
    files = [
        [
            genebass_row(),
            genebass_row(),
            genebass_row(gene_id='ENSG2', Pvalue_Burden=0.0, trait_type='icd_first_occurrence', n_controls=None),
            genebass_row(gene_id='ENSG3', Pvalue_Burden=1e-8, annotation='missense|LC', BETA_Burden=-0.0),
            genebass_row(gene_id='ENSG4', Pvalue_Burden=6.7e-7, annotation='synonymous', BETA_Burden=math.nan),
            genebass_row(gene_id='ENSG5', Pvalue_Burden=5.498e-320, annotation='pLoF|missense|LC', SE_Burden=None),
        ],
        [
            genebass_row(gene_id='ENSG6', trait_type='categorical', Pvalue_Burden=1e-12),
            genebass_row(gene_id='ENSG7', trait_type=None),
            genebass_row(gene_id='ENSG8', Pvalue_Burden=math.nan),
            genebass_row(gene_id='ENSG9', Pvalue_Burden=9.9995e-9, annotation='new', n_cases=None, description=None),
            genebass_row(gene_id='ENSG10', Pvalue_Burden=3.0000005e-9, phenocode=None, BETA_Burden=math.inf),
            genebass_row(gene_id='ENSG11', Pvalue_Burden=1e-3),
        ],
    ]
    path = write_genebass(tmp_path, *files)

    old = process_genebass_gene_burden(spark.read.option('mergeSchema', True).parquet(path))
    columns = old.columns
    old_rows = [row.asDict(recursive=True) for row in old.collect()]
    new = parse_genebass(path).collect()

    absent = [
        'allelicRequirements',
        'oddsRatio',
        'oddsRatioConfidenceIntervalLower',
        'oddsRatioConfidenceIntervalUpper',
        'releaseVersion',
        'studyCasesWithQualifyingVariants',
        'urls',
        'sex',
        'curatedDiseaseFromSourceMappedId',
    ]
    assert set(new.columns) - set(columns) == set(absent)
    assert new.select(absent).null_count().row(0) == (new.height,) * len(absent)
    assert multiset(new.rows(named=True), columns) == multiset(old_rows, columns)
