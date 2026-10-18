"""Tests for the BRaVa gene burden parser."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
from pyspark.sql import SparkSession

from pts.pyspark.gene_burden import _excel_sheet_to_spark, process_brava_gene_burden
from pts.schemas.gene_burden import GENE_BURDEN_PARSED
from pts.transformers.evidence.gene_burden.brava import parse_brava
from tests.evidence.gene_burden.logs import logged
from tests.evidence.gene_burden.spark_rows import multiset
from tests.evidence.gene_burden.workbook import write_workbook

S4_COLUMNS = ['Phenotype ID', 'Ancestry', 'Biobank', 'N cases', 'N controls']
S5_COLUMNS = ['Phenotype ID', 'Ancestry', 'Biobank', 'N']
GRANULAR_COLUMNS = [
    'Gene', 'Gene ID', 'Description', 'Phenotype ID', 'meta analyzed', 'class', 'Mask', 'max MAF', 'Pvalue',
    'BETA Burden', 'SE Burden',
]  # fmt: skip
GROUP = 'Cauchy combination with minimum P-value'
PVALUE = 'Min P-value of significant Cauchy associations'
CAUCHY_COLUMNS = ['Gene', 'Gene ID', 'Description', 'Phenotype ID', GROUP, PVALUE]

Sheet = Sequence[dict[str, Any]]

# Binary contributions: two biobanks for EUR, one each for AFR and MID.
S4_ROWS = [
    {'Phenotype ID': 'T2Diab', 'Ancestry': 'EUR', 'Biobank': 'uk-biobank', 'N cases': 100, 'N controls': 900},
    {'Phenotype ID': 'T2Diab', 'Ancestry': 'EUR', 'Biobank': 'all-of-us', 'N cases': 200, 'N controls': 800},
    {'Phenotype ID': 'T2Diab', 'Ancestry': 'AFR', 'Biobank': 'all-of-us', 'N cases': 30, 'N controls': 70},
    {'Phenotype ID': 'T2Diab', 'Ancestry': 'MID', 'Biobank': 'ccpm', 'N cases': 5, 'N controls': 15},
]
# Quantitative contributions: 'all-of-us' is listed twice, identically, as Table S5 does.
S5_ROWS = [
    {'Phenotype ID': 'Height', 'Ancestry': 'EUR', 'Biobank': 'uk-biobank', 'N': 5000},
    {'Phenotype ID': 'Height', 'Ancestry': 'EUR', 'Biobank': 'all-of-us', 'N': 1000},
    {'Phenotype ID': 'Height', 'Ancestry': 'EUR', 'Biobank': 'all-of-us', 'N': 1000},
    {'Phenotype ID': 'Height', 'Ancestry': 'AFR', 'Biobank': 'all-of-us', 'N': 400},
]


def _granular(**fields: Any) -> dict[str, Any]:
    row = {
        'Gene': 'G1',
        'Gene ID': 'ENSG1',
        'Description': 'Type 2 diabetes',
        'Phenotype ID': 'T2Diab',
        'meta analyzed': 'EUR',
        'class': 'Burden',
        'Mask': 'pLoF',
        'max MAF': 0.001,
        'Pvalue': 2.5e-9,
        'BETA Burden': 0.5,
        'SE Burden': 0.125,
    }
    return row | fields


def _cauchy(**fields: Any) -> dict[str, Any]:
    row = {
        'Gene': 'G2',
        'Gene ID': 'ENSG2',
        'Description': 'Height',
        'Phenotype ID': 'Height',
        GROUP: 'ALL',
        PVALUE: 4e-12,
    }
    return row | fields


def _write(
    tmp: Path,
    *,
    s14: Sheet = (),
    s15: Sheet = (),
    s12: Sheet = (),
    s13: Sheet = (),
    s4: Sheet = S4_ROWS,
    s5: Sheet = S5_ROWS,
) -> str:
    return write_workbook(
        tmp / 'brava.xlsx',
        {
            'Table S4': (S4_COLUMNS, s4),
            'Table S5': (S5_COLUMNS, s5),
            'Table S12': (CAUCHY_COLUMNS, list(s12)),
            'Table S13': (CAUCHY_COLUMNS, list(s13)),
            'Table S14': (GRANULAR_COLUMNS, list(s14)),
            'Table S15': (GRANULAR_COLUMNS, list(s15)),
        },
    )


def _parse(tmp: Path, **sheets: Sheet) -> pl.DataFrame:
    return parse_brava(_write(tmp, **sheets)).collect()


def _sizes(rows: pl.DataFrame) -> dict[tuple[str, str], tuple[int | None, int | None]]:
    """(target, statisticalMethod) to (studySampleSize, studyCases)."""
    return {
        (r['targetFromSourceId'], r['statisticalMethod']): (r['studySampleSize'], r['studyCases'])
        for r in rows.iter_rows(named=True)
    }


def test_output_has_the_parsed_schema(tmp_path: Path) -> None:
    assert _parse(tmp_path, s15=[_granular()], s12=[_cauchy()]).schema == GENE_BURDEN_PARSED


def test_granular_columns(tmp_path: Path) -> None:
    row = _parse(tmp_path, s15=[_granular()]).row(0, named=True)
    assert row == {
        'datasourceId': 'gene_burden',
        'datatypeId': 'genetic_association',
        'allelicRequirements': None,
        'targetFromSourceId': 'ENSG1',
        'diseaseFromSource': 'Type 2 diabetes',
        'pValueMantissa': 2.5,
        'pValueExponent': -9,
        'beta': 0.5,
        'betaConfidenceIntervalLower': 0.375,
        'betaConfidenceIntervalUpper': 0.625,
        'oddsRatio': None,
        'oddsRatioConfidenceIntervalLower': None,
        'oddsRatioConfidenceIntervalUpper': None,
        'resourceScore': 2.5e-9,
        'ancestry': 'European',
        'ancestryId': 'HANCESTRO_0005',
        'literature': ['42238450'],
        'projectId': 'BRaVa Consortium',
        'cohortId': 'BRaVa Consortium',
        'releaseVersion': None,
        'studySampleSize': 2000,
        'studyCases': 300,
        'studyCasesWithQualifyingVariants': None,
        'statisticalMethod': 'Burden.pLoF.0.001.EUR',
        'statisticalMethodOverview': (
            'Burden test carried out with pLoF variants with a MAF smaller than 0.1%, meta-analyzed across '
            'BRaVa biobanks.'
        ),
        'urls': None,
        'sex': None,
        'diseaseFromSourceId': None,
        'curatedDiseaseFromSourceMappedId': None,
    }


def test_cauchy_columns(tmp_path: Path) -> None:
    row = _parse(tmp_path, s12=[_cauchy()]).row(0, named=True)
    assert row['resourceScore'] == 4e-12
    assert (row['pValueMantissa'], row['pValueExponent']) == (4.0, -12)
    assert (row['beta'], row['betaConfidenceIntervalLower']) == (None, None)
    assert (row['ancestry'], row['ancestryId']) == (None, None)
    assert row['statisticalMethod'] == 'Cauchy.ALL'
    assert row['statisticalMethodOverview'] == (
        'Cauchy combination test aggregating Burden and SKAT/SKAT-O results across variant masks and MAF cutoffs.'
    )
    assert (row['studySampleSize'], row['studyCases']) == (6400, None)


def test_identical_quantitative_contributions_are_counted_once(tmp_path: Path) -> None:
    rows = _parse(tmp_path, s12=[_cauchy(), _cauchy(**{GROUP: 'EUR'})])
    # 5000 + 1000, NOT 5000 + 1000 + 1000.
    assert _sizes(rows) == {('ENSG2', 'Cauchy.EUR'): (6000, None), ('ENSG2', 'Cauchy.ALL'): (6400, None)}


def test_ancestry_groups_of_the_sample_sizes(tmp_path: Path) -> None:
    groups = ['EUR', 'AFR', 'MID', 'ALL', 'non_EUR', 'SAS']
    rows = _parse(tmp_path, s13=[_cauchy(**{'Gene ID': 'ENSG1', 'Phenotype ID': 'T2Diab', GROUP: g}) for g in groups])
    assert _sizes(rows) == {
        ('ENSG1', 'Cauchy.EUR'): (2000, 300),
        ('ENSG1', 'Cauchy.AFR'): (100, 30),
        ('ENSG1', 'Cauchy.MID'): (20, 5),
        ('ENSG1', 'Cauchy.ALL'): (2120, 335),
        ('ENSG1', 'Cauchy.non_EUR'): (120, 35),
        ('ENSG1', 'Cauchy.SAS'): (None, None),
    }


def test_sums_over_missing_values_are_null(tmp_path: Path) -> None:
    rows = _parse(
        tmp_path,
        s13=[_cauchy(**{'Phenotype ID': 'Asthma', GROUP: g}) for g in ('EUR', 'ALL')],
        s4=[
            *S4_ROWS,
            {'Phenotype ID': 'Asthma', 'Ancestry': 'EUR', 'Biobank': 'a', 'N cases': 'na', 'N controls': 10},
            {'Phenotype ID': 'Asthma', 'Ancestry': 'EUR', 'Biobank': 'b', 'N cases': None, 'N controls': 20},
            {'Phenotype ID': 'Asthma', 'Ancestry': 'AFR', 'Biobank': 'a', 'N cases': 3, 'N controls': 30},
        ],
    )
    assert _sizes(rows) == {('ENSG2', 'Cauchy.EUR'): (None, None), ('ENSG2', 'Cauchy.ALL'): (33, 3)}


def test_sample_sizes_cast_as_spark(tmp_path: Path) -> None:
    rows = _parse(
        tmp_path,
        s12=[_cauchy(**{GROUP: 'EUR'})],
        s5=[{'Phenotype ID': 'Height', 'Ancestry': 'EUR', 'Biobank': 'a', 'N': '6000.0'}],
    )
    assert _sizes(rows) == {('ENSG2', 'Cauchy.EUR'): (6000, None)}


def test_a_phenotype_both_binary_and_quantitative_fails(tmp_path: Path) -> None:
    s5 = [*S5_ROWS, {'Phenotype ID': 'T2Diab', 'Ancestry': 'EUR', 'Biobank': 'uk-biobank', 'N': 1000}]
    with pytest.raises(pl.exceptions.ComputeError):
        _parse(tmp_path, s15=[_granular()], s5=s5)


def test_rows_without_a_phenotype_never_join(tmp_path: Path) -> None:
    s4 = [*S4_ROWS, {'Phenotype ID': None, 'Ancestry': 'EUR', 'Biobank': 'a', 'N cases': 1, 'N controls': 1}]
    s5 = [*S5_ROWS, {'Phenotype ID': None, 'Ancestry': 'EUR', 'Biobank': 'a', 'N': 1}]
    rows = _parse(tmp_path, s15=[_granular(), _granular(**{'Gene ID': 'ENSG3', 'Phenotype ID': None})], s4=s4, s5=s5)
    assert _sizes(rows) == {
        ('ENSG1', 'Burden.pLoF.0.001.EUR'): (2000, 300),
        ('ENSG3', 'Burden.pLoF.0.001.EUR'): (None, None),
    }


def test_zero_pvalues_take_the_minimum_of_their_own_sheet(tmp_path: Path) -> None:
    rows, log = logged(
        lambda: _parse(
            tmp_path,
            s14=[_granular(Pvalue=0.0), _granular(**{'Gene ID': 'ENSG4', 'Pvalue': 3e-9})],
            s15=[_granular(**{'Gene ID': 'ENSG5', 'Pvalue': 1e-20})],
            s12=[_cauchy(**{PVALUE: 0.0})],
            s13=[_cauchy(**{'Gene ID': 'ENSG6', PVALUE: 1e-14})],
        )
    )
    by_target = {r['targetFromSourceId']: r['resourceScore'] for r in rows.iter_rows(named=True)}
    assert by_target == {'ENSG1': 3e-9, 'ENSG4': 3e-9, 'ENSG5': 1e-20, 'ENSG2': None, 'ENSG6': 1e-14}
    assert 'There are 1 BRaVa evidence with a p-value of 0.0' in log
    assert 'There are 1 BRaVa Cauchy evidence with a p-value of 0.0' in log


def test_method_concatenations_skip_nulls(tmp_path: Path) -> None:
    rows = _parse(
        tmp_path,
        s15=[
            _granular(**{'class': None, 'Mask': 'damaging_missense_or_protein_altering', 'max MAF': 0.0001}),
            _granular(**{'Gene ID': 'ENSG7', 'Mask': 'synonymous', 'class': 'SKAT'}),
            _granular(**{'Gene ID': 'ENSG8', 'Mask': 'pLoF;damaging_missense_or_protein_altering', 'max MAF': 1e-05}),
        ],
        s12=[_cauchy(**{GROUP: None})],
    )
    by_target = {r['targetFromSourceId']: r for r in rows.iter_rows(named=True)}
    assert by_target['ENSG1']['statisticalMethod'] == 'DM/PA.0.0001.EUR'
    assert by_target['ENSG1']['statisticalMethodOverview'] is None
    assert by_target['ENSG7']['statisticalMethod'] == 'SKAT.0.001.EUR'
    assert by_target['ENSG7']['statisticalMethodOverview'] is None
    assert by_target['ENSG8']['statisticalMethod'] == 'Burden.pLoF;DM/PA.1e-05.EUR'
    assert by_target['ENSG8']['statisticalMethodOverview'] is None
    assert by_target['ENSG2']['statisticalMethod'] is None


@pytest.mark.parametrize(
    ('group', 'ancestry', 'ancestry_id'),
    [
        ('AFR', 'African', 'HANCESTRO_0010'),
        ('EAS', 'East Asian', 'HANCESTRO_0009'),
        ('AMR', 'Admixed American', 'HANCESTRO_0014'),
        ('MID', 'Middle Eastern', 'HANCESTRO_0852'),
        ('SAS', 'Central and South Asian', None),
        ('non_EUR', 'non-European', None),
        ('ALL', None, None),
        ('XYZ', None, None),
    ],
)
def test_ancestry(tmp_path: Path, group: str, ancestry: str | None, ancestry_id: str | None) -> None:
    row = _parse(tmp_path, s14=[_granular(**{'meta analyzed': group})]).row(0, named=True)
    assert (row['ancestry'], row['ancestryId']) == (ancestry, ancestry_id)


def test_nan_beta_is_null(tmp_path: Path) -> None:
    row = _parse(tmp_path, s14=[_granular(**{'class': 'SKAT', 'BETA Burden': 'NAN'})]).row(0, named=True)
    assert (row['beta'], row['betaConfidenceIntervalLower'], row['betaConfidenceIntervalUpper']) == (None,) * 3


def test_sheets_with_only_a_header_give_an_empty_frame(tmp_path: Path) -> None:
    rows = _parse(tmp_path)
    assert rows.height == 0
    assert rows.schema == GENE_BURDEN_PARSED


def test_duplicates_are_dropped(tmp_path: Path) -> None:
    assert _parse(tmp_path, s14=[_granular(), _granular(Gene='other symbol')]).height == 1


def test_matches_spark(tmp_path: Path, spark: SparkSession) -> None:
    path = _write(
        tmp_path,
        s14=[
            _granular(**{'Phenotype ID': 'Height', 'Description': 'Height', 'Pvalue': 0.0}),
            _granular(**{'Phenotype ID': 'Height', 'Gene ID': 'ENSG3', 'Pvalue': 1e-8, 'meta analyzed': 'ALL'}),
            {},
            _granular(**{
                'Phenotype ID': 'Height',
                'Gene ID': 'ENSG4',
                'class': 'SKAT',
                'BETA Burden': 'NaN',
                'SE Burden': 'NaN',
                'meta analyzed': 'non_EUR',
                'max MAF': 0.0001,
            }),
        ],
        s15=[
            _granular(),
            _granular(Gene='other symbol'),
            _granular(**{'Gene ID': 'ENSG5', 'Pvalue': 5e-320, 'meta analyzed': 'SAS', 'BETA Burden': -0.0}),
            _granular(**{'Gene ID': 'ENSG6', 'Mask': 'synonymous', 'class': None, 'max MAF': 1e-05}),
            _granular(**{'Gene ID': 'ENSG7', 'BETA Burden': 'NAN', 'meta analyzed': 'MID', 'Pvalue': 0.0}),
            _granular(**{'Gene ID': 'ENSG8', 'Phenotype ID': 'Unknown', 'meta analyzed': 'AFR', 'SE Burden': None}),
        ],
        s12=[
            _cauchy(),
            {},
            _cauchy(**{'Gene ID': 'ENSG9', GROUP: 'EUR', PVALUE: 0.0}),
        ],
        s13=[
            _cauchy(**{'Gene ID': 'ENSG10', 'Phenotype ID': 'T2Diab', GROUP: 'non_EUR'}),
            _cauchy(**{'Gene ID': 'ENSG11', 'Phenotype ID': 'T2Diab', GROUP: None, PVALUE: 1e-8}),
        ],
        s4=[
            *S4_ROWS,
            S4_ROWS[0],
            {},
            {'Phenotype ID': 'T2Diab', 'Ancestry': 'SAS', 'Biobank': 'x', 'N cases': 'na', 'N controls': '7.0'},
        ],
    )
    session = SimpleNamespace(spark=spark)

    sheets = ('Table S4', 'Table S5', 'Table S12', 'Table S13', 'Table S14', 'Table S15')
    old = process_brava_gene_burden(*(_excel_sheet_to_spark(session, path, sheet) for sheet in sheets))
    columns = old.columns
    old_rows = [row.asDict(recursive=True) for row in old.collect()]
    new = parse_brava(path).collect()

    absent = [
        'allelicRequirements',
        'oddsRatio',
        'oddsRatioConfidenceIntervalLower',
        'oddsRatioConfidenceIntervalUpper',
        'releaseVersion',
        'studyCasesWithQualifyingVariants',
        'urls',
        'sex',
        'diseaseFromSourceId',
        'curatedDiseaseFromSourceMappedId',
    ]
    assert set(new.columns) - set(columns) == set(absent)
    assert new.select(absent).null_count().row(0) == (new.height,) * len(absent)
    assert len(old_rows) == 14
    assert multiset(new.rows(named=True), columns) == multiset(old_rows, columns)
