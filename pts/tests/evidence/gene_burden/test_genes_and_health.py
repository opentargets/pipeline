"""Tests for the Genes & Health gene burden parser."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from pts.schemas.gene_burden import GENE_BURDEN_PARSED
from pts.transformers.evidence.gene_burden.genes_and_health import parse_genes_and_health
from pts.transformers.utils.spark_math import spark_exp, strict_pow
from tests.evidence.gene_burden.fixtures import Rows, write_genes_and_health
from tests.evidence.gene_burden.logs import logged
from tests.evidence.gene_burden.spark_rows import expand, multiset


def _st9(**fields: Any) -> dict[str, Any]:
    row = {
        'Gene': 'G1',
        'Gene ID': 'ENSG1',
        'Phenotype': 'Asthma',
        'EFO ID': 'MONDO_0004979',
        'QT': 0.0,
        'TEST': 'ADD',
        'Mask': 'M1.0.01',
        'Variant Mask': 'A',
        'Freq': 0.01,
        'N': 44000.0,
        'BETA': 0.5,
        'SE': 0.1,
        'LOG10P': 9.5,
    }
    return row | fields


def _st13(**fields: Any) -> dict[str, Any]:
    row = {
        'Gene': 'G2',
        'Gene ID': 'ENSG2',
        'Phenotype': 'Height',
        'EFO ID': 'EFO_0004339',
        'QT': True,
        'Mask': 'M3.singleton',
        'Variant Mask': 'C',
        'Freq': 'singleton',
        'N G&H': 40000,
        'N UKB': 400000,
        'BETA': -0.25,
        'SE': 0.125,
        'LOG10P': 12.25,
    }
    return row | fields


def _st15(**fields: Any) -> dict[str, Any]:
    row = {
        'Gene': 'G3',
        'Gene ID': 'ENSG3',
        'Phenotype': 'Type 2 diabetes',
        'EFO ID': 'MONDO_0005148',
        'QT': False,
        'Variant consequence': 'pLoF_pDM',
        'Number for ExWAS': 44000,
        'Number of Cases': 9000,
        'Biallelic Carriers': 12,
        'Recessive log(OR)': 1.5,
        'Recessive p-value': 2e-9,
        'Significant REC P-value': 'Significant',
    }
    return row | fields


def _parse(tmp: Path, st9: Rows, st13: Rows, st15: Rows) -> pl.DataFrame:
    return parse_genes_and_health(write_genes_and_health(tmp, st9, st13, st15)).collect()


def _exp(value: float) -> float:
    return pl.select(spark_exp(pl.lit(value, dtype=pl.Float64))).item()


def test_output_has_the_parsed_schema(tmp_path: Path) -> None:
    assert _parse(tmp_path, [_st9()], [_st13()], [_st15()]).schema == GENE_BURDEN_PARSED


def test_additive_columns(tmp_path: Path) -> None:
    row = _parse(tmp_path, [_st9()], [], []).row(0, named=True)
    p = strict_pow(10.0, -9.5)
    assert row == {
        'datasourceId': 'gene_burden',
        'datatypeId': 'genetic_association',
        'allelicRequirements': ['dominant'],
        'targetFromSourceId': 'ENSG1',
        'diseaseFromSource': 'Asthma',
        'pValueMantissa': 3.162,
        'pValueExponent': -10,
        'beta': None,
        'betaConfidenceIntervalLower': None,
        'betaConfidenceIntervalUpper': None,
        'oddsRatio': _exp(0.5),
        'oddsRatioConfidenceIntervalLower': _exp(0.5 - 0.1),
        'oddsRatioConfidenceIntervalUpper': _exp(0.5 + 0.1),
        'resourceScore': p,
        'ancestry': 'Pakistani and Bangladeshi',
        'ancestryId': 'HANCESTRO_0006',
        'literature': ['41896352'],
        'projectId': 'Genes & Health',
        'cohortId': 'Genes & Health',
        'releaseVersion': None,
        'studySampleSize': 44000,
        'studyCases': None,
        'studyCasesWithQualifyingVariants': None,
        'statisticalMethod': 'ADD_M1.0.01',
        'statisticalMethodOverview': (
            'Burden test carried out with high-confidence LoF variants with a MAF smaller than 1%.'
        ),
        'urls': None,
        'sex': None,
        'diseaseFromSourceId': 'MONDO_0004979',
        'curatedDiseaseFromSourceMappedId': 'MONDO_0004979',
    }


def test_meta_analysis_columns(tmp_path: Path) -> None:
    row = _parse(tmp_path, [], [_st13()], []).row(0, named=True)
    assert row['beta'] == -0.25
    assert row['betaConfidenceIntervalLower'] == -0.375
    assert row['betaConfidenceIntervalUpper'] == -0.125
    assert row['oddsRatio'] is None
    assert (row['ancestry'], row['ancestryId']) == (None, None)
    assert row['cohortId'] == 'Genes & Health + UK Biobank'
    assert row['studySampleSize'] == 440000
    assert row['statisticalMethod'] == 'META.M3.singleton'
    assert row['statisticalMethodOverview'] == (
        'Burden test carried out with missense variants restricted to singletons meta-analysed between '
        'Genes & Health and UK Biobank.'
    )
    assert row['curatedDiseaseFromSourceMappedId'] == 'EFO_0004339'


def test_recessive_columns_and_filters(tmp_path: Path) -> None:
    rows = _parse(
        tmp_path,
        [],
        [],
        [
            _st15(),
            _st15(**{'Gene ID': 'ENSG4', 'Significant REC P-value': 'Suggestive'}),
            _st15(**{'Gene ID': 'ENSG5', 'Variant consequence': 'synonymous'}),
            _st15(**{'Gene ID': 'ENSG6', 'Variant consequence': None}),
            _st15(**{'Gene ID': 'ENSG7', 'QT': True, 'Number of Cases': 'na', 'Recessive log(OR)': 0.75}),
        ],
    )
    by_target = {row['targetFromSourceId']: row for row in rows.iter_rows(named=True)}
    assert set(by_target) == {'ENSG3', 'ENSG7'}
    binary, quantitative = by_target['ENSG3'], by_target['ENSG7']
    assert binary['resourceScore'] == 2e-9
    assert binary['oddsRatio'] == _exp(1.5)
    assert binary['oddsRatioConfidenceIntervalLower'] is None
    assert binary['studySampleSize'] == 44000
    assert binary['studyCases'] == 9000
    assert binary['studyCasesWithQualifyingVariants'] == 12
    assert binary['allelicRequirements'] == ['recessive']
    assert binary['statisticalMethod'] == 'REC.pLoF_pDM'
    assert binary['statisticalMethodOverview'] == (
        'Recessive burden test carried out with biallelic pLoF and deleterious missense genotypes.'
    )
    assert (quantitative['beta'], quantitative['betaConfidenceIntervalLower']) == (0.75, None)
    assert quantitative['studyCases'] is None


@pytest.mark.parametrize(
    ('qt', 'quantitative'),
    [
        (True, True),
        (False, False),
        (1.0, True),
        (0, False),
        ('1.0', True),
        ('TRUE', True),
        ('yes', False),
        (None, False),
    ],
)
def test_qt_truthiness(tmp_path: Path, qt: Any, quantitative: bool) -> None:
    row = _parse(tmp_path, [_st9(QT=qt)], [], []).row(0, named=True)
    assert (row['beta'] is not None, row['oddsRatio'] is not None) == (quantitative, not quantitative)


def test_infinite_log10p_takes_the_minimum_over_every_sheet(tmp_path: Path) -> None:
    rows, log = logged(
        lambda: _parse(
            tmp_path,
            [_st9(LOG10P=9.5)],
            [_st13(LOG10P='inf'), _st13(**{'Gene ID': 'ENSG9', 'LOG10P': 'Infinity'})],
            [_st15(**{'Recessive p-value': 1e-30})],
        )
    )
    by_target = {row['targetFromSourceId']: row['resourceScore'] for row in rows.iter_rows(named=True)}
    assert by_target == {'ENSG1': strict_pow(10.0, -9.5), 'ENSG2': 1e-30, 'ENSG9': 1e-30, 'ENSG3': 1e-30}
    assert 'There are 2 Genes & Health evidence with a p-value of 0.0' in log


def test_pvalue_is_fdlibm_pow(tmp_path: Path) -> None:
    # 10 ** -23.6442024: numpy's power is one ulp above fdlibm's
    row = _parse(tmp_path, [_st9(LOG10P=23.6442024)], [], []).row(0, named=True)
    assert row['resourceScore'] == 2.268807243217692e-24


def test_odds_ratio_is_fdlibm_exp(tmp_path: Path) -> None:
    # exp(2.1): numpy's is one ulp above fdlibm's
    row = _parse(tmp_path, [_st9(BETA=2.1)], [], []).row(0, named=True)
    assert row['oddsRatio'] == 8.16616991256765


def test_method_concatenations(tmp_path: Path) -> None:
    rows = _parse(
        tmp_path,
        [
            _st9(TEST=None, Mask='M1.0.01'),
            _st9(**{'Gene ID': 'ENSG8', 'TEST': None, 'Mask': None}),
        ],
        [_st13(Mask=None)],
        [],
    )
    by_target = {row['targetFromSourceId']: row for row in rows.iter_rows(named=True)}
    assert by_target['ENSG1']['statisticalMethod'] == 'M1.0.01'
    assert by_target['ENSG8']['statisticalMethod'] == ''
    assert by_target['ENSG2']['statisticalMethod'] is None
    assert by_target['ENSG1']['statisticalMethodOverview'] is None


@pytest.mark.parametrize(
    ('fields', 'overview'),
    [
        (
            {'TEST': 'ADD-SKATO', 'Variant Mask': 'B', 'Freq': 0.001},
            ('SKAT-O test carried out with deleterious missense variants with a MAF smaller than 0.1%.'),
        ),
        (
            {'TEST': 'ADD-SKAT', 'Variant Mask': 'D', 'Freq': 0.0001},
            ('SKAT test carried out with synonymous variants with a MAF smaller than 0.01%.'),
        ),
        ({'Freq': 'Singleton'}, 'Burden test carried out with high-confidence LoF variants restricted to singletons.'),
        ({'TEST': 'ADD-ACATV'}, None),
        ({'Variant Mask': 'E'}, None),
        ({'Freq': 0.05}, None),
        ({'Freq': None}, None),
    ],
)
def test_method_overview_lookups(tmp_path: Path, fields: dict[str, Any], overview: str | None) -> None:
    row = _parse(tmp_path, [_st9(**fields)], [], []).row(0, named=True)
    assert row['statisticalMethodOverview'] == overview


def test_nan_effect_is_null(tmp_path: Path) -> None:
    row = _parse(tmp_path, [_st9(BETA='NAN')], [], []).row(0, named=True)
    assert (row['oddsRatio'], row['oddsRatioConfidenceIntervalLower']) == (None, None)


def test_sample_sizes_cast_as_spark(tmp_path: Path) -> None:
    rows = _parse(
        tmp_path,
        [_st9(N='44000.0'), _st9(**{'Gene ID': 'ENSG8', 'N': 'n/a'})],
        [_st13(**{'N UKB': None})],
        [],
    )
    by_target = {row['targetFromSourceId']: row['studySampleSize'] for row in rows.iter_rows(named=True)}
    assert by_target == {'ENSG1': 44000, 'ENSG8': None, 'ENSG2': None}


def test_sheets_with_only_a_header_give_an_empty_frame(tmp_path: Path) -> None:
    rows = _parse(tmp_path, [], [], [])
    assert rows.height == 0
    assert rows.schema == GENE_BURDEN_PARSED


def test_duplicates_are_dropped(tmp_path: Path) -> None:
    assert _parse(tmp_path, [_st9(), _st9()], [], []).height == 1


def test_matches_spark_baseline(tmp_path: Path) -> None:
    path = write_genes_and_health(
        tmp_path,
        [
            _st9(),
            _st9(),
            {},
            _st9(**{'Gene ID': 'ENSG10', 'BETA': 2.1, 'QT': 1.0, 'LOG10P': 23.6442024}),
            _st9(**{'Gene ID': 'ENSG11', 'TEST': 'ADD-SKATO', 'Variant Mask': 'B', 'Freq': 0.001, 'BETA': 'NAN'}),
            _st9(**{'Gene ID': 'ENSG12', 'TEST': None, 'Freq': 'Singleton', 'N': '44000.0', 'QT': 'TRUE'}),
            _st9(**{'Gene ID': 'ENSG13', 'Variant Mask': 'E', 'LOG10P': 8, 'SE': None, 'BETA': -0.0}),
            _st9(**{'Gene ID': 'ENSG14', 'LOG10P': 'inf', 'BETA': -3.7, 'Freq': 0.0001}),
        ],
        [
            _st13(),
            {},
            _st13(**{'Gene ID': 'ENSG20', 'QT': False, 'LOG10P': 'Infinity', 'Mask': None, 'BETA': 1.25}),
            _st13(**{'Gene ID': 'ENSG21', 'N UKB': '400000.0', 'Freq': 0.01, 'Variant Mask': 'D', 'LOG10P': 7.3}),
        ],
        [
            _st15(),
            {},
            _st15(**{'Gene ID': 'ENSG30', 'Significant REC P-value': 'Suggestive'}),
            _st15(**{'Gene ID': 'ENSG31', 'Variant consequence': 'synonymous'}),
            _st15(**{'Gene ID': 'ENSG32', 'QT': True, 'Number of Cases': 'na', 'Recessive log(OR)': 0.75}),
            _st15(**{'Gene ID': 'ENSG33', 'Recessive p-value': 0.0, 'Recessive log(OR)': -19.8}),
        ],
    )
    new = parse_genes_and_health(path).collect()

    # expected rows captured from the removed pyspark implementation
    lof_1 = 'Burden test carried out with high-confidence LoF variants with a MAF smaller than 1%.'
    m3_singleton = (
        'Burden test carried out with missense variants restricted to singletons meta-analysed '
        'between Genes & Health and UK Biobank.'
    )
    synonymous_1 = (
        'Burden test carried out with synonymous variants with a MAF smaller than 1% meta-analysed '
        'between Genes & Health and UK Biobank.'
    )
    recessive = 'Recessive burden test carried out with biallelic pLoF and deleterious missense genotypes.'
    constant = {
        'datasourceId': 'gene_burden',
        'datatypeId': 'genetic_association',
        'projectId': 'Genes & Health',
        'literature': ['41896352'],
    }
    varying = [
        'targetFromSourceId',
        'diseaseFromSource',
        'diseaseFromSourceId',
        'curatedDiseaseFromSourceMappedId',
        'resourceScore',
        'pValueExponent',
        'pValueMantissa',
        'beta',
        'betaConfidenceIntervalLower',
        'betaConfidenceIntervalUpper',
        'oddsRatio',
        'oddsRatioConfidenceIntervalLower',
        'oddsRatioConfidenceIntervalUpper',
        'ancestry',
        'ancestryId',
        'allelicRequirements',
        'cohortId',
        'studySampleSize',
        'studyCases',
        'studyCasesWithQualifyingVariants',
        'statisticalMethod',
        'statisticalMethodOverview',
    ]
    rows = [
        (
            'ENSG1',
            'Asthma',
            'MONDO_0004979',
            'MONDO_0004979',
            3.1622776601683795e-10,
            -10,
            3.162,
            None,
            None,
            None,
            1.6487212707001282,
            1.4918246976412703,
            1.8221188003905089,
            'Pakistani and Bangladeshi',
            'HANCESTRO_0006',
            ['dominant'],
            'Genes & Health',
            44000,
            None,
            None,
            'ADD_M1.0.01',
            lof_1,
        ),
        (
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
            'Pakistani and Bangladeshi',
            'HANCESTRO_0006',
            ['dominant'],
            'Genes & Health',
            None,
            None,
            None,
            '',
            None,
        ),
        (
            'ENSG10',
            'Asthma',
            'MONDO_0004979',
            'MONDO_0004979',
            2.268807243217692e-24,
            -24,
            2.269,
            2.1,
            2.0,
            2.2,
            None,
            None,
            None,
            'Pakistani and Bangladeshi',
            'HANCESTRO_0006',
            ['dominant'],
            'Genes & Health',
            44000,
            None,
            None,
            'ADD_M1.0.01',
            lof_1,
        ),
        (
            'ENSG11',
            'Asthma',
            'MONDO_0004979',
            'MONDO_0004979',
            3.1622776601683795e-10,
            -10,
            3.162,
            None,
            None,
            None,
            None,
            None,
            None,
            'Pakistani and Bangladeshi',
            'HANCESTRO_0006',
            ['dominant'],
            'Genes & Health',
            44000,
            None,
            None,
            'ADD-SKATO_M1.0.01',
            'SKAT-O test carried out with deleterious missense variants with a MAF smaller than 0.1%.',
        ),
        (
            'ENSG12',
            'Asthma',
            'MONDO_0004979',
            'MONDO_0004979',
            3.1622776601683795e-10,
            -10,
            3.162,
            0.5,
            0.4,
            0.6,
            None,
            None,
            None,
            'Pakistani and Bangladeshi',
            'HANCESTRO_0006',
            ['dominant'],
            'Genes & Health',
            44000,
            None,
            None,
            'M1.0.01',
            None,
        ),
        (
            'ENSG13',
            'Asthma',
            'MONDO_0004979',
            'MONDO_0004979',
            1e-08,
            -9,
            10.0,
            None,
            None,
            None,
            1.0,
            None,
            None,
            'Pakistani and Bangladeshi',
            'HANCESTRO_0006',
            ['dominant'],
            'Genes & Health',
            44000,
            None,
            None,
            'ADD_M1.0.01',
            None,
        ),
        (
            'ENSG14',
            'Asthma',
            'MONDO_0004979',
            'MONDO_0004979',
            2.268807243217692e-24,
            -24,
            2.269,
            None,
            None,
            None,
            0.024723526470339388,
            0.02237077185616559,
            0.02732372244729256,
            'Pakistani and Bangladeshi',
            'HANCESTRO_0006',
            ['dominant'],
            'Genes & Health',
            44000,
            None,
            None,
            'ADD_M1.0.01',
            'Burden test carried out with high-confidence LoF variants with a MAF smaller than 0.01%.',
        ),
        (
            'ENSG2',
            'Height',
            'EFO_0004339',
            'EFO_0004339',
            5.62341325190349e-13,
            -13,
            5.623,
            -0.25,
            -0.375,
            -0.125,
            None,
            None,
            None,
            None,
            None,
            ['dominant'],
            'Genes & Health + UK Biobank',
            440000,
            None,
            None,
            'META.M3.singleton',
            m3_singleton,
        ),
        (
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
            ['dominant'],
            'Genes & Health + UK Biobank',
            None,
            None,
            None,
            None,
            None,
        ),
        (
            'ENSG20',
            'Height',
            'EFO_0004339',
            'EFO_0004339',
            2.268807243217692e-24,
            -24,
            2.269,
            None,
            None,
            None,
            3.4903429574618414,
            3.080216848918031,
            3.955076722920577,
            None,
            None,
            ['dominant'],
            'Genes & Health + UK Biobank',
            440000,
            None,
            None,
            None,
            m3_singleton,
        ),
        (
            'ENSG21',
            'Height',
            'EFO_0004339',
            'EFO_0004339',
            5.011872336272725e-08,
            -8,
            5.012,
            -0.25,
            -0.375,
            -0.125,
            None,
            None,
            None,
            None,
            None,
            ['dominant'],
            'Genes & Health + UK Biobank',
            440000,
            None,
            None,
            'META.M3.singleton',
            synonymous_1,
        ),
        (
            'ENSG3',
            'Type 2 diabetes',
            'MONDO_0005148',
            'MONDO_0005148',
            2e-09,
            -9,
            2.0,
            None,
            None,
            None,
            4.4816890703380645,
            None,
            None,
            'Pakistani and Bangladeshi',
            'HANCESTRO_0006',
            ['recessive'],
            'Genes & Health',
            44000,
            9000,
            12,
            'REC.pLoF_pDM',
            recessive,
        ),
        (
            'ENSG32',
            'Type 2 diabetes',
            'MONDO_0005148',
            'MONDO_0005148',
            2e-09,
            -9,
            2.0,
            0.75,
            None,
            None,
            None,
            None,
            None,
            'Pakistani and Bangladeshi',
            'HANCESTRO_0006',
            ['recessive'],
            'Genes & Health',
            44000,
            None,
            12,
            'REC.pLoF_pDM',
            recessive,
        ),
        (
            'ENSG33',
            'Type 2 diabetes',
            'MONDO_0005148',
            'MONDO_0005148',
            2.268807243217692e-24,
            -24,
            2.269,
            None,
            None,
            None,
            2.5174987194382783e-09,
            None,
            None,
            'Pakistani and Bangladeshi',
            'HANCESTRO_0006',
            ['recessive'],
            'Genes & Health',
            44000,
            9000,
            12,
            'REC.pLoF_pDM',
            recessive,
        ),
    ]
    old_rows = expand(constant, varying, rows)
    columns = list(old_rows[0])

    absent = ['releaseVersion', 'urls', 'sex']
    assert set(new.columns) - set(columns) == set(absent)
    assert new.select(absent).null_count().row(0) == (new.height,) * len(absent)
    assert len(old_rows) == 14
    assert multiset(new.rows(named=True), columns) == multiset(old_rows, columns)
