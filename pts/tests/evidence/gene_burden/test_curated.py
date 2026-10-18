"""Tests for the curated gene burden evidence parser."""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from pts.schemas.gene_burden import CURATED_INPUT_COLUMNS, GENE_BURDEN_PARSED
from pts.transformers.evidence.gene_burden.curated import parse_curated
from tests.evidence.gene_burden.fixtures import HEADER, curated_line, write_curated
from tests.evidence.gene_burden.logs import logged
from tests.evidence.gene_burden.spark_rows import expand, multiset


def _parse(tmp: Path, *lines: str, header: str = HEADER) -> pl.DataFrame:
    return parse_curated(write_curated(tmp / 'curated_evidence.tsv', *lines, header=header)).collect()


def test_output_has_the_parsed_schema(tmp_path: Path) -> None:
    assert _parse(tmp_path, curated_line()).schema == GENE_BURDEN_PARSED


def test_columns_are_read_by_position(tmp_path: Path) -> None:
    header = '\t'.join(f'renamed{i}' for i in range(len(CURATED_INPUT_COLUMNS))) + '\n'
    row = _parse(tmp_path, curated_line(), header=header).row(0, named=True)
    assert row['projectId'] == 'REGENERON'
    assert row['targetFromSourceId'] == 'ENSG1'
    assert row['diseaseFromSource'] == 'Disease A'
    assert row['datasourceId'] == 'gene_burden'
    assert row['datatypeId'] == 'genetic_association'
    assert row['studySampleSize'] == 1000
    assert row['statisticalMethod'] == 'M1'
    assert row['literature'] == ['12345']
    assert (row['releaseVersion'], row['urls'], row['diseaseFromSourceId']) == (None, None, None)


def test_curated_disease_id_is_kept_apart_from_the_mapped_id(tmp_path: Path) -> None:
    parsed = _parse(tmp_path, curated_line())
    assert parsed['curatedDiseaseFromSourceMappedId'].to_list() == ['EFO_0000001']
    assert 'diseaseFromSourceMappedId' not in parsed.columns


def test_float_case_counts_parse(tmp_path: Path) -> None:
    parsed, log = logged(
        lambda: _parse(
            tmp_path,
            curated_line(targetFromSourceId='A', studyCases='13885.0', studyCasesWithQualifyingVariants='12.0'),
            curated_line(targetFromSourceId='B', studyCases='1.5', studyCasesWithQualifyingVariants='x'),
            curated_line(targetFromSourceId='C', studyCases='42', studyCasesWithQualifyingVariants='7.00'),
            curated_line(targetFromSourceId='D'),
        )
    )
    rows = parsed.sort('targetFromSourceId')
    assert rows.select('studyCases', 'studyCasesWithQualifyingVariants').rows() == [
        (13885, 12),
        (None, None),
        (42, 7),
        (None, None),
    ]
    assert '1 curated evidence have studyCases written as N.0, read as N' in log
    assert '2 curated evidence have studyCasesWithQualifyingVariants written as N.0, read as N' in log


def test_mantissa_and_exponent_are_taken_verbatim(tmp_path: Path) -> None:
    row = _parse(tmp_path, curated_line(resourceScore='1e-12', pValueMantissa='10.0', pValueExponent='-13')).row(
        0, named=True
    )
    assert (row['resourceScore'], row['pValueMantissa'], row['pValueExponent']) == (1e-12, 10.0, -13)


@pytest.mark.parametrize(
    ('effects', 'odds_ratio_ci', 'beta_ci'),
    [
        ({'oddsRatio': '2.0'}, (1.0, 3.0), (None, None)),
        ({'beta': '0.5'}, (None, None), (1.0, 3.0)),
        ({'oddsRatio': '2.0', 'beta': '0.5'}, (1.0, 3.0), (1.0, 3.0)),
        ({}, (None, None), (None, None)),
    ],
)
def test_confidence_intervals_follow_the_effect(
    tmp_path: Path,
    effects: dict[str, str],
    odds_ratio_ci: tuple[float | None, float | None],
    beta_ci: tuple[float | None, float | None],
) -> None:
    row = _parse(tmp_path, curated_line(ConfidenceIntervalLower='1.0', ConfidenceIntervalUpper='3.0', **effects)).row(
        0, named=True
    )
    assert (row['oddsRatioConfidenceIntervalLower'], row['oddsRatioConfidenceIntervalUpper']) == odds_ratio_ci
    assert (row['betaConfidenceIntervalLower'], row['betaConfidenceIntervalUpper']) == beta_ci


def test_lists(tmp_path: Path) -> None:
    rows = _parse(
        tmp_path,
        curated_line(targetFromSourceId='A', sex='male, female', allelicRequirements='dominant'),
        curated_line(targetFromSourceId='B', sex='both,male', literature=''),
    ).sort('targetFromSourceId')
    assert rows['sex'].to_list() == [['male', 'female'], ['both,male']]
    assert rows['allelicRequirements'].to_list() == [['dominant'], None]
    # spark's array of a null literature is a list holding null
    assert rows['literature'].to_list() == [['12345'], [None]]


def test_rows_are_distinct(tmp_path: Path) -> None:
    # differing only in columns the parser drops
    rows = _parse(
        tmp_path, curated_line(), curated_line(), curated_line(targetFromSource='OTHER', studyId='S2', url='https://x')
    )
    assert rows.height == 1


def test_matches_spark_baseline(tmp_path: Path) -> None:
    path = write_curated(
        tmp_path / 'curated_evidence.tsv',
        curated_line(),
        curated_line(),
        curated_line(projectId='', targetFromSourceId='ENSG2', sex='male, female', allelicRequirements='recessive'),
        curated_line(
            targetFromSourceId='ENSG3', oddsRatio=' 2.5', ConfidenceIntervalLower='1e-3', ConfidenceIntervalUpper='5.'
        ),
        curated_line(
            targetFromSourceId='ENSG4', beta='-0.0', ConfidenceIntervalLower='.5', ConfidenceIntervalUpper='2.5d'
        ),
        curated_line(
            targetFromSourceId='ENSG5', oddsRatio='NaN', ConfidenceIntervalLower='Inf', ConfidenceIntervalUpper='-Inf'
        ),
        curated_line(targetFromSourceId='ENSG6', oddsRatio='Infinity', ConfidenceIntervalLower='-NaN', beta='nan'),
        curated_line(targetFromSourceId='ENSG7', oddsRatio='inf', ConfidenceIntervalLower='1,5', beta='1.0E+2F'),
        curated_line(targetFromSourceId='ENSG8', pValueExponent='+5', studySampleSize=' 5', resourceScore='abc'),
        curated_line(
            targetFromSourceId='ENSG9', pValueExponent='5.0', studySampleSize='2147483648', pValueMantissa='0x1p3'
        ),
        curated_line(targetFromSourceId='ENSG10', pValueExponent='-2147483648', studySampleSize='1e3', literature=''),
        curated_line(
            targetFromSourceId='ENSG11', studyCases='1.5', studyCasesWithQualifyingVariants='x', sex='male,female'
        ),
        curated_line(targetFromSourceId='ENSG12', studyCases='12', studyCasesWithQualifyingVariants='-3', beta='1.0f '),
        'REGENERON\tGENE\tENSG13\tShort row\n',
        curated_line(targetFromSourceId='ENSG14').rstrip('\n') + '\textra\tcolumns\n',
        '\n',
    )
    new = parse_curated(path).collect()

    # expected rows captured from the removed pyspark implementation
    constant = {
        'datasourceId': 'gene_burden',
        'datatypeId': 'genetic_association',
        'ancestry': None,
        'ancestryId': None,
        'statisticalMethodOverview': None,
    }
    varying = [
        'projectId',
        'targetFromSourceId',
        'diseaseFromSource',
        'curatedDiseaseFromSourceMappedId',
        'resourceScore',
        'pValueMantissa',
        'pValueExponent',
        'oddsRatio',
        'oddsRatioConfidenceIntervalLower',
        'oddsRatioConfidenceIntervalUpper',
        'beta',
        'betaConfidenceIntervalLower',
        'betaConfidenceIntervalUpper',
        'sex',
        'cohortId',
        'studySampleSize',
        'studyCases',
        'studyCasesWithQualifyingVariants',
        'allelicRequirements',
        'statisticalMethod',
        'literature',
    ]
    rows = [
        (
            'REGENERON',
            'ENSG1',
            'Disease A',
            'EFO_0000001',
            1.5e-09,
            1.5,
            -9,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            'UK Biobank',
            1000,
            None,
            None,
            None,
            'M1',
            ['12345'],
        ),
        (
            None,
            'ENSG2',
            'Disease A',
            'EFO_0000001',
            1.5e-09,
            1.5,
            -9,
            None,
            None,
            None,
            None,
            None,
            None,
            ['male', 'female'],
            'UK Biobank',
            1000,
            None,
            None,
            ['recessive'],
            'M1',
            ['12345'],
        ),
        (
            'REGENERON',
            'ENSG3',
            'Disease A',
            'EFO_0000001',
            1.5e-09,
            1.5,
            -9,
            2.5,
            0.001,
            5.0,
            None,
            None,
            None,
            None,
            'UK Biobank',
            1000,
            None,
            None,
            None,
            'M1',
            ['12345'],
        ),
        (
            'REGENERON',
            'ENSG4',
            'Disease A',
            'EFO_0000001',
            1.5e-09,
            1.5,
            -9,
            None,
            None,
            None,
            0.0,
            0.5,
            2.5,
            None,
            'UK Biobank',
            1000,
            None,
            None,
            None,
            'M1',
            ['12345'],
        ),
        (
            'REGENERON',
            'ENSG5',
            'Disease A',
            'EFO_0000001',
            1.5e-09,
            1.5,
            -9,
            math.nan,
            math.inf,
            -math.inf,
            None,
            None,
            None,
            None,
            'UK Biobank',
            1000,
            None,
            None,
            None,
            'M1',
            ['12345'],
        ),
        (
            'REGENERON',
            'ENSG6',
            'Disease A',
            'EFO_0000001',
            1.5e-09,
            1.5,
            -9,
            math.inf,
            math.nan,
            None,
            None,
            None,
            None,
            None,
            'UK Biobank',
            1000,
            None,
            None,
            None,
            'M1',
            ['12345'],
        ),
        (
            'REGENERON',
            'ENSG7',
            'Disease A',
            'EFO_0000001',
            1.5e-09,
            1.5,
            -9,
            None,
            None,
            None,
            100.0,
            None,
            None,
            None,
            'UK Biobank',
            1000,
            None,
            None,
            None,
            'M1',
            ['12345'],
        ),
        (
            'REGENERON',
            'ENSG8',
            'Disease A',
            'EFO_0000001',
            None,
            1.5,
            5,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            'UK Biobank',
            None,
            None,
            None,
            None,
            'M1',
            ['12345'],
        ),
        (
            'REGENERON',
            'ENSG9',
            'Disease A',
            'EFO_0000001',
            1.5e-09,
            8.0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            'UK Biobank',
            None,
            None,
            None,
            None,
            'M1',
            ['12345'],
        ),
        (
            'REGENERON',
            'ENSG10',
            'Disease A',
            'EFO_0000001',
            1.5e-09,
            1.5,
            -2147483648,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            'UK Biobank',
            None,
            None,
            None,
            None,
            'M1',
            [None],
        ),
        (
            'REGENERON',
            'ENSG11',
            'Disease A',
            'EFO_0000001',
            1.5e-09,
            1.5,
            -9,
            None,
            None,
            None,
            None,
            None,
            None,
            ['male,female'],
            'UK Biobank',
            1000,
            None,
            None,
            None,
            'M1',
            ['12345'],
        ),
        (
            'REGENERON',
            'ENSG12',
            'Disease A',
            'EFO_0000001',
            1.5e-09,
            1.5,
            -9,
            None,
            None,
            None,
            1.0,
            None,
            None,
            None,
            'UK Biobank',
            1000,
            12,
            -3,
            None,
            'M1',
            ['12345'],
        ),
        (
            'REGENERON',
            'ENSG13',
            'Short row',
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            [None],
        ),
        (
            'REGENERON',
            'ENSG14',
            'Disease A',
            'EFO_0000001',
            1.5e-09,
            1.5,
            -9,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            'UK Biobank',
            1000,
            None,
            None,
            None,
            'M1',
            ['12345'],
        ),
    ]
    old_rows = expand(constant, varying, rows)
    columns = list(old_rows[0])

    absent = ['releaseVersion', 'urls', 'diseaseFromSourceId']
    assert set(new.columns) - set(columns) == set(absent)
    assert new.select(absent).null_count().row(0) == (new.height,) * len(absent)
    assert multiset(new.rows(named=True), columns) == multiset(old_rows, columns)
