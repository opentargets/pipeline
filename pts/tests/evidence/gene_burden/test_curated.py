"""Tests for the curated gene burden evidence parser."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from loguru import logger
from pyspark.sql import SparkSession

from pts.pyspark.gene_burden import CURATION_SCHEMA, process_gene_burden_curation
from pts.schemas.gene_burden import CURATED_INPUT_COLUMNS, GENE_BURDEN_PARSED
from pts.transformers.evidence.gene_burden.curated import parse_curated
from tests.evidence.gene_burden.spark_rows import multiset

HEADER = '\t'.join(CURATED_INPUT_COLUMNS) + '\n'


def _line(**fields: str) -> str:
    row = {
        'projectId': 'REGENERON',
        'targetFromSource': 'GENE1',
        'targetFromSourceId': 'ENSG1',
        'diseaseFromSource': 'Disease A',
        'diseaseFromSourceMappedId': 'EFO_0000001',
        'resourceScore': '1.5e-9',
        'pValueMantissa': '1.5',
        'pValueExponent': '-9',
        'cohortId': 'UK Biobank',
        'studySampleSize': '1000',
        'statisticalMethod': 'M1',
        'literature': '12345',
    }
    row.update(fields)
    unknown = set(row) - set(CURATED_INPUT_COLUMNS)
    assert not unknown, unknown
    return '\t'.join(row.get(c) or '' for c in CURATED_INPUT_COLUMNS) + '\n'


def _write(path: Path, *lines: str, header: str = HEADER) -> str:
    path.write_text(header + ''.join(lines))
    return str(path)


def _parse(tmp: Path, *lines: str, header: str = HEADER) -> pl.DataFrame:
    return parse_curated(_write(tmp / 'curated_evidence.tsv', *lines, header=header)).collect()


def test_output_has_the_parsed_schema(tmp_path: Path) -> None:
    assert _parse(tmp_path, _line()).schema == GENE_BURDEN_PARSED


def test_columns_are_read_by_position(tmp_path: Path) -> None:
    header = '\t'.join(f'renamed{i}' for i in range(len(CURATED_INPUT_COLUMNS))) + '\n'
    row = _parse(tmp_path, _line(), header=header).row(0, named=True)
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
    parsed = _parse(tmp_path, _line())
    assert parsed['curatedDiseaseFromSourceMappedId'].to_list() == ['EFO_0000001']
    assert 'diseaseFromSourceMappedId' not in parsed.columns


def test_float_case_counts_parse(tmp_path: Path) -> None:
    messages: list[str] = []
    sink = logger.add(messages.append, format='{message}')
    try:
        rows = _parse(
            tmp_path,
            _line(targetFromSourceId='A', studyCases='13885.0', studyCasesWithQualifyingVariants='12.0'),
            _line(targetFromSourceId='B', studyCases='1.5', studyCasesWithQualifyingVariants='x'),
            _line(targetFromSourceId='C', studyCases='42', studyCasesWithQualifyingVariants='7.00'),
            _line(targetFromSourceId='D'),
        ).sort('targetFromSourceId')
    finally:
        logger.remove(sink)
    assert rows.select('studyCases', 'studyCasesWithQualifyingVariants').rows() == [
        (13885, 12),
        (None, None),
        (42, 7),
        (None, None),
    ]
    log = ''.join(messages)
    assert '1 curated evidence have studyCases written as N.0, read as N' in log
    assert '2 curated evidence have studyCasesWithQualifyingVariants written as N.0, read as N' in log


def test_mantissa_and_exponent_are_taken_verbatim(tmp_path: Path) -> None:
    row = _parse(tmp_path, _line(resourceScore='1e-12', pValueMantissa='10.0', pValueExponent='-13')).row(0, named=True)
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
    row = _parse(tmp_path, _line(ConfidenceIntervalLower='1.0', ConfidenceIntervalUpper='3.0', **effects)).row(
        0, named=True
    )
    assert (row['oddsRatioConfidenceIntervalLower'], row['oddsRatioConfidenceIntervalUpper']) == odds_ratio_ci
    assert (row['betaConfidenceIntervalLower'], row['betaConfidenceIntervalUpper']) == beta_ci


def test_lists(tmp_path: Path) -> None:
    rows = _parse(
        tmp_path,
        _line(targetFromSourceId='A', sex='male, female', allelicRequirements='dominant'),
        _line(targetFromSourceId='B', sex='both,male', literature=''),
    ).sort('targetFromSourceId')
    assert rows['sex'].to_list() == [['male', 'female'], ['both,male']]
    assert rows['allelicRequirements'].to_list() == [['dominant'], None]
    # spark's array of a null literature is a list holding null
    assert rows['literature'].to_list() == [['12345'], [None]]


def test_rows_are_distinct(tmp_path: Path) -> None:
    # differing only in columns the parser drops
    rows = _parse(tmp_path, _line(), _line(), _line(targetFromSource='OTHER', studyId='S2', url='https://x'))
    assert rows.height == 1


def test_matches_spark(tmp_path: Path, spark: SparkSession) -> None:
    path = _write(
        tmp_path / 'curated_evidence.tsv',
        _line(),
        _line(),
        _line(projectId='', targetFromSourceId='ENSG2', sex='male, female', allelicRequirements='recessive'),
        _line(
            targetFromSourceId='ENSG3', oddsRatio=' 2.5', ConfidenceIntervalLower='1e-3', ConfidenceIntervalUpper='5.'
        ),
        _line(targetFromSourceId='ENSG4', beta='-0.0', ConfidenceIntervalLower='.5', ConfidenceIntervalUpper='2.5d'),
        _line(
            targetFromSourceId='ENSG5', oddsRatio='NaN', ConfidenceIntervalLower='Inf', ConfidenceIntervalUpper='-Inf'
        ),
        _line(targetFromSourceId='ENSG6', oddsRatio='Infinity', ConfidenceIntervalLower='-NaN', beta='nan'),
        _line(targetFromSourceId='ENSG7', oddsRatio='inf', ConfidenceIntervalLower='1,5', beta='1.0E+2F'),
        _line(targetFromSourceId='ENSG8', pValueExponent='+5', studySampleSize=' 5', resourceScore='abc'),
        _line(targetFromSourceId='ENSG9', pValueExponent='5.0', studySampleSize='2147483648', pValueMantissa='0x1p3'),
        _line(targetFromSourceId='ENSG10', pValueExponent='-2147483648', studySampleSize='1e3', literature=''),
        _line(targetFromSourceId='ENSG11', studyCases='1.5', studyCasesWithQualifyingVariants='x', sex='male,female'),
        _line(targetFromSourceId='ENSG12', studyCases='12', studyCasesWithQualifyingVariants='-3', beta='1.0f '),
        'REGENERON\tGENE\tENSG13\tShort row\n',
        _line(targetFromSourceId='ENSG14').rstrip('\n') + '\textra\tcolumns\n',
        '\n',
    )
    old = process_gene_burden_curation(spark.read.csv(path, header=True, sep='\t', schema=CURATION_SCHEMA))
    old_rows = [row.asDict(recursive=True) for row in old.collect()]
    columns = list(old_rows[0])
    new = parse_curated(path).collect()

    absent = ['releaseVersion', 'urls', 'diseaseFromSourceId']
    assert set(new.columns) - set(columns) == set(absent)
    assert new.select(absent).null_count().row(0) == (new.height,) * len(absent)
    assert multiset(new.rows(named=True), columns) == multiset(old_rows, columns)
