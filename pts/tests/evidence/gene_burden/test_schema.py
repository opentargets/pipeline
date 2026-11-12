"""Tests for the SCHEMA2 gene burden parser."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from pts.schemas.gene_burden import GENE_BURDEN_PARSED
from pts.transformers.evidence.gene_burden.schema import parse_schema
from tests.evidence.gene_burden.fixtures import schema_row, write_schema


def _parse(tmp: Path, *rows: dict[str, str], release: str = '2026-07-15') -> pl.DataFrame:
    return parse_schema(write_schema(tmp, list(rows)), release).collect()


def test_output_has_the_parsed_schema(tmp_path: Path) -> None:
    assert _parse(tmp_path, schema_row()).schema == GENE_BURDEN_PARSED


def test_columns(tmp_path: Path) -> None:
    row = _parse(tmp_path, schema_row()).row(0, named=True)
    assert row == {
        'datasourceId': 'gene_burden',
        'datatypeId': 'genetic_association',
        'allelicRequirements': None,
        'targetFromSourceId': 'ENSG1',
        'diseaseFromSource': 'schizophrenia',
        'pValueMantissa': 2.5,
        'pValueExponent': -9,
        'beta': None,
        'betaConfidenceIntervalLower': None,
        'betaConfidenceIntervalUpper': None,
        'oddsRatio': 4.25,
        'oddsRatioConfidenceIntervalLower': 2.5,
        'oddsRatioConfidenceIntervalUpper': 7.75,
        'resourceScore': 2.5e-9,
        'ancestry': None,
        'ancestryId': None,
        'literature': None,
        'projectId': 'SCHEMA',
        'cohortId': 'SCHEMA2',
        'releaseVersion': '2026-07-15',
        'studySampleSize': 87959 + 150587,
        'studyCases': 87959,
        'studyCasesWithQualifyingVariants': 90,
        'statisticalMethod': 'ptv_mis',
        'statisticalMethodOverview': (
            'Case-control burden test carried out with rare protein-truncating and damaging missense variants.'
        ),
        'urls': [{'url': 'https://schema.broadinstitute.org/gene/ENSG1', 'niceName': 'SCHEMA browser'}],
        'sex': None,
        'diseaseFromSourceId': None,
        'curatedDiseaseFromSourceMappedId': 'MONDO_0005090',
    }


def test_reports_the_more_significant_test(tmp_path: Path) -> None:
    row = _parse(
        tmp_path, schema_row(ptv_p_value='1.0000e-10', ptv_mis_p_value='3.0000e-08', schema_case_control_p_value='NA')
    ).row(0, named=True)
    assert (
        row['statisticalMethod'],
        row['resourceScore'],
        row['oddsRatio'],
        row['oddsRatioConfidenceIntervalLower'],
        row['oddsRatioConfidenceIntervalUpper'],
        row['studyCasesWithQualifyingVariants'],
        row['statisticalMethodOverview'],
    ) == (
        'ptv',
        1e-10,
        5.5,
        1.5,
        20.25,
        40,
        'Case-control burden test carried out with rare protein-truncating variants.',
    )


def test_a_tie_reports_the_ptv_test(tmp_path: Path) -> None:
    row = _parse(tmp_path, schema_row(ptv_p_value='2.5000e-09')).row(0, named=True)
    assert row['statisticalMethod'] == 'ptv'


def test_a_missing_test_is_skipped(tmp_path: Path) -> None:
    row = _parse(tmp_path, schema_row(ptv_mis_p_value='NA', ptv_p_value='5.0000e-07')).row(0, named=True)
    assert (row['statisticalMethod'], row['resourceScore']) == ('ptv', 5e-7)


@pytest.mark.parametrize(
    ('ptv', 'ptv_mis', 'kept'),
    [
        ('6.8399e-05', 'NA', True),
        ('6.8400e-05', 'NA', False),
        ('NA', '1.0000e-05', True),
        ('1.0000e-01', '7.0000e-05', False),
        ('NA', 'NA', False),
    ],
)
def test_keeps_the_five_percent_fdr_genes(tmp_path: Path, ptv: str, ptv_mis: str, kept: bool) -> None:
    rows = _parse(tmp_path, schema_row(gene_id='ENSG2', ptv_p_value=ptv, ptv_mis_p_value=ptv_mis))
    assert rows.height == int(kept)


def test_only_the_meta_analysis_is_read(tmp_path: Path) -> None:
    rows = _parse(tmp_path, schema_row(), schema_row(gene_id='ENSG2', group='EUR'))
    assert rows['targetFromSourceId'].to_list() == ['ENSG1']


def test_infinite_odds_ratio_with_no_interval(tmp_path: Path) -> None:
    row = _parse(tmp_path, schema_row(ptv_mis_odds_ratio='Inf', ptv_mis_odds_ratio_95_ci='NaN - NaN')).row(
        0, named=True
    )
    assert row['oddsRatio'] == math.inf
    assert math.isnan(row['oddsRatioConfidenceIntervalLower'])
    assert math.isnan(row['oddsRatioConfidenceIntervalUpper'])


def test_missing_interval_is_null(tmp_path: Path) -> None:
    row = _parse(tmp_path, schema_row(ptv_mis_odds_ratio_95_ci='NA')).row(0, named=True)
    assert (row['oddsRatioConfidenceIntervalLower'], row['oddsRatioConfidenceIntervalUpper']) == (None, None)


def test_one_row_per_gene(tmp_path: Path) -> None:
    rows = _parse(tmp_path, schema_row(), schema_row(gene_id='ENSG2'), schema_row(gene_id='ENSG3', ptv_p_value='1e-7'))
    assert rows.sort('targetFromSourceId').select('targetFromSourceId', 'statisticalMethod').rows() == [
        ('ENSG1', 'ptv_mis'),
        ('ENSG2', 'ptv_mis'),
        ('ENSG3', 'ptv_mis'),
    ]


def test_no_significant_rows_give_an_empty_frame(tmp_path: Path) -> None:
    rows = _parse(tmp_path, schema_row(ptv_p_value='0.5', ptv_mis_p_value='0.5'))
    assert rows.height == 0
    assert rows.schema == GENE_BURDEN_PARSED
