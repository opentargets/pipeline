"""Tests for the FinnGen gene burden parser."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from pts.schemas.gene_burden import GENE_BURDEN_PARSED
from pts.transformers.evidence.gene_burden.finngen import parse_finngen
from pts.transformers.utils.spark_math import strict_pow
from tests.evidence.gene_burden.fixtures import MANIFEST, write_finngen
from tests.evidence.gene_burden.logs import logged
from tests.evidence.gene_burden.spark_rows import expand, multiset


def _line(**fields: str) -> dict[str, str]:
    row = {
        'PHENO': 'T2D',
        'CHROM': '1',
        'GENPOS': '1000',
        'ID': 'GENE1.Mask1.0.01',
        'ALLELE0': 'ref',
        'ALLELE1': 'Mask1.0.01',
        'A1FREQ': '0.001',
        'N': '500000',
        'TEST': 'ADD',
        'BETA': '0.5',
        'SE': '0.25',
        'CHISQ': '30.1',
        'LOG10P': '8.5',
        'EXTRA': 'NA',
    }
    row.update(fields)
    return row


def _parse(tmp: Path, lines: list[dict[str, str]], manifest: list[dict[str, Any]] | None = None) -> pl.DataFrame:
    return parse_finngen(*write_finngen(tmp, lines, manifest), release='R12').collect()


def test_output_has_the_parsed_schema(tmp_path: Path) -> None:
    assert _parse(tmp_path, [_line()]).schema == GENE_BURDEN_PARSED


def test_columns(tmp_path: Path) -> None:
    row = _parse(tmp_path, [_line(LOG10P='8.30103')]).row(0, named=True)
    score = strict_pow(10.0, -8.30103)
    assert row == {
        'datasourceId': 'gene_burden',
        'datatypeId': 'genetic_association',
        'allelicRequirements': None,
        'targetFromSourceId': 'GENE1',
        'diseaseFromSource': 'Type 2 diabetes',
        'pValueMantissa': 5.0,
        'pValueExponent': -9,
        'beta': 0.5,
        'betaConfidenceIntervalLower': 0.25,
        'betaConfidenceIntervalUpper': 0.75,
        'oddsRatio': None,
        'oddsRatioConfidenceIntervalLower': None,
        'oddsRatioConfidenceIntervalUpper': None,
        'resourceScore': score,
        'ancestry': 'finnish',
        'ancestryId': 'HANCESTRO_0321',
        'literature': ['36653562'],
        'projectId': 'FinnGen',
        'cohortId': 'FinnGen R12',
        'releaseVersion': 'R12',
        'studySampleSize': 500348,
        'studyCases': None,
        'studyCasesWithQualifyingVariants': None,
        'statisticalMethod': 'LoF burden',
        'statisticalMethodOverview': 'Burden test carried out with LoF variants with MAF smaller than 1%.',
        'urls': None,
        'sex': None,
        'diseaseFromSourceId': 'T2D',
        'curatedDiseaseFromSourceMappedId': None,
    }


def test_cohort_is_r12_whatever_the_release(tmp_path: Path) -> None:
    row = parse_finngen(*write_finngen(tmp_path, [_line()]), release='R13').collect().row(0, named=True)
    assert (row['cohortId'], row['releaseVersion']) == ('FinnGen R12', 'R13')


def test_score_is_spark_pow_not_numpy(tmp_path: Path) -> None:
    # numpy's power is one ulp above fdlibm's here
    row = _parse(tmp_path, [_line(LOG10P='23.6442024')]).row(0, named=True)
    assert row['resourceScore'] == 2.268807243217692e-24


def test_unknown_phenotype_keeps_the_row(tmp_path: Path) -> None:
    rows = _parse(tmp_path, [_line(PHENO='UNKNOWN')])
    assert rows.select('diseaseFromSourceId', 'diseaseFromSource').rows() == [('UNKNOWN', None)]


def test_duplicated_phenocode_multiplies_rows(tmp_path: Path) -> None:
    manifest = [*MANIFEST, {'phenocode': 'T2D', 'phenostring': 'Diabetes, type 2'}]
    rows = _parse(tmp_path, [_line()], manifest)
    assert sorted(rows['diseaseFromSource']) == ['Diabetes, type 2', 'Type 2 diabetes']


def test_underflow_gives_a_zero_score_and_null_exponent(tmp_path: Path) -> None:
    rows, log = logged(lambda: _parse(tmp_path, [_line(LOG10P='400'), _line(ID='GENE2.m', LOG10P='323.31')]))
    underflow, subnormal = rows.sort('targetFromSourceId').rows(named=True)
    assert (underflow['resourceScore'], underflow['pValueMantissa'], underflow['pValueExponent']) == (0.0, None, None)
    assert subnormal['resourceScore'] == strict_pow(10.0, -323.31) > 0.0
    # 10 ** -324 is 0.0, so spark's mantissa is null too
    assert (subnormal['pValueMantissa'], subnormal['pValueExponent']) == (None, -324)
    assert '1 FinnGen evidence have a LOG10P too large for a p-value; their score is 0.0.' in log


def test_bonferroni_counts_a_null_target(tmp_path: Path) -> None:
    # 3 targets give 0.05 / 3 < 0.02; without the null target, 0.05 / 2 > 0.02
    lines = [_line(ID='A.m', LOG10P=str(-math.log10(0.02))), _line(ID='B.m'), _line(ID='')]
    assert sorted(_parse(tmp_path, lines)['targetFromSourceId'], key=str) == ['B', None]


def test_bonferroni_counts_non_significant_targets(tmp_path: Path) -> None:
    lines = [_line(ID='A.m', LOG10P='1.69'), _line(ID='B.m', LOG10P='0.1'), _line(ID='C.m', LOG10P='0.1')]
    assert _parse(tmp_path, lines).height == 0
    assert _parse(tmp_path, lines[:2]).height == 1


def test_target_is_the_id_up_to_the_first_dot(tmp_path: Path) -> None:
    lines = [_line(ID='A.Mask1.0.01'), _line(ID='A.Mask2.0.01'), _line(ID='B'), _line(ID='.C')]
    assert sorted(_parse(tmp_path, lines)['targetFromSourceId']) == ['', 'A', 'B']


def test_effects_are_cast_as_spark(tmp_path: Path) -> None:
    row = _parse(tmp_path, [_line(BETA='1.5E-3', SE='abc')]).row(0, named=True)
    assert (row['beta'], row['betaConfidenceIntervalLower'], row['betaConfidenceIntervalUpper']) == (0.0015, None, None)


def test_rows_are_distinct(tmp_path: Path) -> None:
    # differing only in columns the parser drops
    assert _parse(tmp_path, [_line(), _line(), _line(CHISQ='1', GENPOS='2')]).height == 1


def test_header_only_gives_an_empty_frame(tmp_path: Path) -> None:
    rows = _parse(tmp_path, [])
    assert rows.height == 0
    assert rows.schema == GENE_BURDEN_PARSED


@pytest.mark.parametrize('log10p', ['NaN', '', 'x'])
def test_unreadable_log10p_is_dropped(tmp_path: Path, log10p: str) -> None:
    assert _parse(tmp_path, [_line(LOG10P=log10p)]).height == 0


def test_matches_spark_baseline(tmp_path: Path) -> None:
    lines = [
        _line(),
        _line(),
        _line(ID='GENE2.Mask1.0.01', LOG10P='8', BETA='-1.25E-2'),
        _line(ID='GENE3.Mask1.0.01', LOG10P='400'),
        _line(ID='GENE4.Mask1.0.01', LOG10P='307.65', PHENO='UNKNOWN'),
        _line(ID='GENE5.Mask1.0.01', LOG10P='23.6442024', PHENO='ASTHMA'),
        _line(ID='GENE6.Mask1.0.01', LOG10P='1.5'),
        _line(ID='GENE7', LOG10P='9.36597e-08'),
        _line(ID='GENE8.Mask1.0.01', LOG10P='90.0033', SE='0.000001'),
        _line(ID='GENE9.Mask1.0.01', LOG10P='4.123456789', PHENO='DUP'),
    ]
    manifest = [*MANIFEST, {'phenocode': 'DUP', 'phenostring': 'One'}, {'phenocode': 'DUP', 'phenostring': 'Two'}]
    results, phenos = write_finngen(tmp_path, lines, manifest)

    new = parse_finngen(results, phenos, release='R12').collect()

    # expected rows captured from the removed pyspark implementation
    constant = {
        'datasourceId': 'gene_burden',
        'ancestry': 'finnish',
        'ancestryId': 'HANCESTRO_0321',
        'cohortId': 'FinnGen R12',
        'datatypeId': 'genetic_association',
        'projectId': 'FinnGen',
        'literature': ['36653562'],
        'releaseVersion': 'R12',
        'statisticalMethod': 'LoF burden',
        'statisticalMethodOverview': 'Burden test carried out with LoF variants with MAF smaller than 1%.',
        'studySampleSize': 500348,
    }
    varying = [
        'beta',
        'betaConfidenceIntervalLower',
        'betaConfidenceIntervalUpper',
        'diseaseFromSource',
        'diseaseFromSourceId',
        'resourceScore',
        'pValueExponent',
        'pValueMantissa',
        'targetFromSourceId',
    ]
    rows = [
        (0.5, 0.25, 0.75, 'Type 2 diabetes', 'T2D', 3.1622776601683795e-09, -9, 3.162, 'GENE1'),
        (-0.0125, -0.2625, 0.2375, 'Type 2 diabetes', 'T2D', 1e-08, -9, 10.0, 'GENE2'),
        (0.5, 0.25, 0.75, 'Type 2 diabetes', 'T2D', 0.0, None, None, 'GENE3'),
        (0.5, 0.25, 0.75, None, 'UNKNOWN', 2.2387211385684567e-308, -308, 2.239, 'GENE4'),
        (0.5, 0.25, 0.75, 'Asthma', 'ASTHMA', 2.268807243217692e-24, -24, 2.269, 'GENE5'),
        (0.5, 0.499999, 0.500001, 'Type 2 diabetes', 'T2D', 9.924302650467869e-91, -91, 9.924, 'GENE8'),
        (0.5, 0.25, 0.75, 'Two', 'DUP', 7.525636042729348e-05, -5, 7.526, 'GENE9'),
        (0.5, 0.25, 0.75, 'One', 'DUP', 7.525636042729348e-05, -5, 7.526, 'GENE9'),
    ]
    old_rows = expand(constant, varying, rows)
    columns = list(old_rows[0])

    absent = [
        'allelicRequirements',
        'oddsRatio',
        'oddsRatioConfidenceIntervalLower',
        'oddsRatioConfidenceIntervalUpper',
        'studyCases',
        'studyCasesWithQualifyingVariants',
        'urls',
        'sex',
        'curatedDiseaseFromSourceMappedId',
    ]
    assert set(new.columns) - set(columns) == set(absent)
    assert new.select(absent).null_count().row(0) == (new.height,) * len(absent)
    assert len(old_rows) == 8
    assert multiset(new.rows(named=True), columns) == multiset(old_rows, columns)
