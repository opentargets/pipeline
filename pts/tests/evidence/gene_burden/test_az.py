"""Tests for the AstraZeneca PheWAS gene burden parser."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest
from pyspark.sql import SparkSession

from pts.pyspark.gene_burden import process_az_gene_burden
from pts.schemas.gene_burden import GENE_BURDEN_PARSED, GENE_BURDEN_UNION
from pts.transformers.evidence.gene_burden.az import AZ_METHOD_DESCRIPTIONS, parse_az
from tests.evidence.gene_burden.logs import logged
from tests.evidence.gene_burden.spark_rows import multiset

BINARY_COLUMNS = [
    'Gene', 'Phenotype', 'CollapsingModel', 'Type', 'pValue', 'Category', 'nSamples', 'BinNcases', 'BinQVcases',
    'BinNcontrols', 'BinQVcontrols', 'BinCaseFreq', 'BinCtrlFreq', 'BinOddsRatio', 'BinOddsRatioLCI',
    'BinOddsRatioUCI',
]  # fmt: skip
QUANTITATIVE_COLUMNS = [
    'Gene', 'Phenotype', 'CollapsingModel', 'Type', 'pValue', 'Category', 'nSamples', 'YesQV', 'NoQV',
    'ConCaseMedian', 'ConCtrMedian', 'beta', 'ConBetaSe', 'LCI', 'UCI',
]  # fmt: skip
GENE_LINK = 'https://azphewas.com/geneView/{}/GENE/glr/binary'
NICE_NAME = 'AstraZeneca PheWAS Portal'


def _binary(**fields: str) -> dict[str, str | None]:
    row: dict[str, str | None] = {
        'Gene': 'ENSG1',
        'Phenotype': 'Disease A',
        'CollapsingModel': 'ptv',
        'Type': 'Binary',
        'pValue': '1e-9',
        'nSamples': '1000',
        'BinNcases': '100',
        'BinQVcases': '10',
        'BinNcontrols': '900',
        'BinOddsRatio': '2.5',
        'BinOddsRatioLCI': '1.5',
        'BinOddsRatioUCI': '3.5',
    }
    row.update(fields)
    return {c: row.get(c) for c in BINARY_COLUMNS}


def _quantitative(**fields: str) -> dict[str, str | None]:
    row: dict[str, str | None] = {
        'Gene': 'ENSG2',
        'Phenotype': 'Trait B',
        'CollapsingModel': 'flexdmg',
        'Type': 'Quantitative',
        'pValue': '2e-8',
        'nSamples': '5000',
        'YesQV': '50',
        'NoQV': '4950',
        'beta': '0.25',
        'LCI': '0.125',
        'UCI': '0.375',
    }
    row.update(fields)
    return {c: row.get(c) for c in QUANTITATIVE_COLUMNS}


def _write_parquet(path: Path, rows: list[dict[str, str | None]], columns: list[str]) -> str:
    path.mkdir(parents=True)
    pl.DataFrame(rows, schema=dict.fromkeys(columns, pl.String)).write_parquet(path / 'part-00000.parquet')
    return str(path)


def _write_lines(path: Path, lines: list[str]) -> str:
    path.write_text(''.join(f'{line}\n' for line in lines))
    return str(path)


def _inputs(
    tmp: Path,
    binary: list[dict[str, str | None]],
    quantitative: list[dict[str, str | None]],
    genes: list[str] | None = None,
    phenotypes: list[str] | None = None,
) -> tuple[str, str, str, str]:
    genes = [f'ENSG1,{GENE_LINK.format("release-1")}'] if genes is None else genes
    phenotypes = (
        ['Disease A,https://azphewas.com/phenotypeView/a', 'Trait B,https://azphewas.com/phenotypeView/b']
        if (phenotypes is None)
        else phenotypes
    )
    return (
        _write_parquet(tmp / 'binary', binary, BINARY_COLUMNS),
        _write_parquet(tmp / 'quantitative', quantitative, QUANTITATIVE_COLUMNS),
        _write_lines(tmp / 'genes.csv', genes),
        _write_lines(tmp / 'phenotypes.csv', phenotypes),
    )


def _parse(tmp: Path, *args: Any, **kwargs: Any) -> pl.DataFrame:
    return parse_az(*_inputs(tmp, *args, **kwargs)).collect()


def test_parsed_schema_is_the_union_with_the_curated_id_in_place_of_the_mapped_id() -> None:
    assert list(GENE_BURDEN_PARSED.dtypes()) == list(GENE_BURDEN_UNION.dtypes())
    assert [n for n in GENE_BURDEN_PARSED if n not in GENE_BURDEN_UNION] == ['curatedDiseaseFromSourceMappedId']
    assert [n for n in GENE_BURDEN_UNION if n not in GENE_BURDEN_PARSED] == ['diseaseFromSourceMappedId']
    assert list(GENE_BURDEN_PARSED).index('curatedDiseaseFromSourceMappedId') == list(GENE_BURDEN_UNION).index(
        'diseaseFromSourceMappedId'
    )


def test_output_has_the_parsed_schema(tmp_path: Path) -> None:
    assert _parse(tmp_path, [_binary()], [_quantitative()]).schema == GENE_BURDEN_PARSED


def test_binary_columns_are_renamed(tmp_path: Path) -> None:
    row = _parse(tmp_path, [_binary()], []).row(0, named=True)
    assert row['targetFromSourceId'] == 'ENSG1'
    assert row['diseaseFromSource'] == 'Disease A'
    assert row['statisticalMethod'] == 'ptv'
    assert (row['oddsRatio'], row['oddsRatioConfidenceIntervalLower'], row['oddsRatioConfidenceIntervalUpper']) == (
        2.5,
        1.5,
        3.5,
    )
    assert (row['beta'], row['betaConfidenceIntervalLower'], row['betaConfidenceIntervalUpper']) == (None, None, None)
    assert (row['studySampleSize'], row['studyCases'], row['studyCasesWithQualifyingVariants']) == (1000, 100, 10)
    assert row['datasourceId'] == 'gene_burden'
    assert row['datatypeId'] == 'genetic_association'
    assert row['literature'] == ['34375979']
    assert row['projectId'] == NICE_NAME
    assert row['cohortId'] == 'UK Biobank 470k'
    assert row['ancestry'] == 'EUR'
    assert row['ancestryId'] == 'HANCESTRO_0005'
    assert row['allelicRequirements'] == ['dominant']
    assert row['diseaseFromSourceId'] is None
    assert row['curatedDiseaseFromSourceMappedId'] is None


def test_quantitative_columns_are_renamed(tmp_path: Path) -> None:
    row = _parse(tmp_path, [], [_quantitative()]).row(0, named=True)
    assert (row['beta'], row['betaConfidenceIntervalLower'], row['betaConfidenceIntervalUpper']) == (
        0.25,
        0.125,
        0.375,
    )
    assert row['oddsRatio'] is None
    # quantitative cases are the whole sample
    assert (row['studySampleSize'], row['studyCases'], row['studyCasesWithQualifyingVariants']) == (5000, 5000, 50)


def test_effects_by_type_only(tmp_path: Path) -> None:
    rows = _parse(tmp_path, [_binary(Type='binary')], [_quantitative(Type='Other')])
    assert rows.select('oddsRatio', 'beta').rows() == [(None, None), (None, None)]


@pytest.mark.parametrize(
    ('pvalue', 'kept'),
    [('1e-7', True), ('1.0000001e-7', False), ('0.5', False), ('NaN', False), ('', False), ('abc', False)],
)
def test_pvalue_filter(tmp_path: Path, pvalue: str, kept: bool) -> None:
    assert _parse(tmp_path, [_binary(pValue=pvalue)], []).height == int(kept)


def test_pvalue_cast_as_spark(tmp_path: Path) -> None:
    row = _parse(tmp_path, [_binary(pValue=' 1.5e-9d ')], []).row(0, named=True)
    assert (row['resourceScore'], row['pValueMantissa'], row['pValueExponent']) == (1.5e-9, 1.5, -9)


def test_raw_mantissa_at_a_power_of_ten(tmp_path: Path) -> None:
    row = _parse(tmp_path, [_binary(pValue='1e-8')], []).row(0, named=True)
    assert (row['pValueMantissa'], row['pValueExponent']) == (10.0, -9)


def test_zero_pvalue_takes_the_smallest_positive_pvalue(tmp_path: Path) -> None:
    rows, log = logged(
        lambda: _parse(
            tmp_path,
            [_binary(pValue='0'), _binary(Gene='ENSG3', pValue='3e-9'), _binary(Gene='ENSG4', pValue='1e-12')],
            [_quantitative(pValue='0.5')],
        )
    )
    assert sorted(rows['resourceScore'].to_list()) == [1e-12, 1e-12, 3e-9]
    assert 'There are 1 AZ evidence with a p-value of 0.0' in log


def test_zero_pvalue_minimum_is_taken_after_the_filter(tmp_path: Path) -> None:
    # no significant row is above zero; the filtered-out 0.01 is not a candidate
    rows = _parse(tmp_path, [_binary(pValue='0')], [_quantitative(pValue='0.01')])
    assert rows.select('resourceScore', 'pValueMantissa', 'pValueExponent').rows() == [(None, None, None)]


def test_effects_carry_float32_noise(tmp_path: Path) -> None:
    rows = _parse(tmp_path, [_binary(BinOddsRatio='12.5085')], [_quantitative(beta='-0.3579588234424591')]).sort(
        'targetFromSourceId'
    )
    assert rows['oddsRatio'].to_list() == [12.508500099182129, None]
    assert rows['beta'].to_list() == [None, -0.3579588234424591]


def test_method_overview_replaces_whole_values(tmp_path: Path) -> None:
    methods = ['ptv', 'ptv5pcnt', 'rec', 'newModel', 'ptvx']
    rows = _parse(tmp_path, [_binary(Gene=m, CollapsingModel=m) for m in methods], []).sort('targetFromSourceId')
    by_method = dict(zip(rows['statisticalMethod'], rows['statisticalMethodOverview'], strict=True))
    assert by_method == {
        'newModel': 'newModel',
        'ptv': AZ_METHOD_DESCRIPTIONS['ptv'],
        'ptv5pcnt': AZ_METHOD_DESCRIPTIONS['ptv5pcnt'],
        'ptvx': 'ptvx',
        'rec': AZ_METHOD_DESCRIPTIONS['rec'],
    }
    allelic = dict(zip(rows['statisticalMethod'], rows['allelicRequirements'].to_list(), strict=True))
    assert allelic['rec'] == ['recessive']
    assert allelic['ptv'] == ['dominant']


def test_release_version_from_the_first_genes_row(tmp_path: Path) -> None:
    genes = [f'ENSG9,{GENE_LINK.format("first-hash")}', f'ENSG1,{GENE_LINK.format("second-hash")}']
    rows = _parse(tmp_path, [_binary()], [], genes=genes)
    assert rows['releaseVersion'].to_list() == ['first-hash']


def test_release_version_empty_without_a_match(tmp_path: Path) -> None:
    rows = _parse(tmp_path, [_binary()], [], genes=['ENSG1,https://example.org/ENSG1'])
    assert rows['releaseVersion'].to_list() == ['']


def test_missing_phenotype_link_gives_null_urls(tmp_path: Path) -> None:
    rows, log = logged(
        lambda: _parse(
            tmp_path,
            [_binary(), _binary(Phenotype='Unlinked'), _binary(Phenotype='Empty link')],
            [],
            phenotypes=['Disease A,https://azphewas.com/phenotypeView/a', 'Empty link,'],
        )
    )
    urls = dict(zip(rows['diseaseFromSource'], rows['urls'].to_list(), strict=True))
    assert urls == {
        'Disease A': [{'url': 'https://azphewas.com/phenotypeView/a', 'niceName': NICE_NAME}],
        'Unlinked': None,
        'Empty link': None,
    }
    assert '2 AZ evidence have no phenotype link' in log


def test_duplicated_link_name_multiplies_rows(tmp_path: Path) -> None:
    rows = _parse(
        tmp_path,
        [_binary()],
        [],
        phenotypes=['Disease A,https://a/1', 'Disease A,https://a/2', 'Disease A,https://a/1'],
    )
    assert sorted(u[0]['url'] for u in rows['urls'].to_list()) == ['https://a/1', 'https://a/2']


def test_rows_are_distinct(tmp_path: Path) -> None:
    # differing only in a column the parser drops
    rows = _parse(tmp_path, [_binary(), _binary(), _binary(Category='other')], [])
    assert rows.height == 1


def test_empty_inputs_give_an_empty_union(tmp_path: Path) -> None:
    rows = _parse(tmp_path, [_binary(pValue='0.5')], [])
    assert rows.height == 0
    assert rows.schema == GENE_BURDEN_PARSED


def test_matches_spark(tmp_path: Path, spark: SparkSession) -> None:
    binary = [
        _binary(),
        _binary(),
        _binary(Gene='ENSG3', pValue='0', BinOddsRatio='12.5085', BinOddsRatioLCI='NaN', BinOddsRatioUCI='abc'),
        _binary(Gene='ENSG4', pValue='1e-8', CollapsingModel='rec', BinOddsRatio='1.00000017881393432617187499'),
        _binary(Gene='ENSG5', pValue=' 3.0000005e-9 ', nSamples='12.7', BinNcases='1e3', BinQVcases=' 7 '),
        _binary(Gene='ENSG6', pValue='5e-320', Phenotype='Unlinked', BinOddsRatio='3.4028236e38'),
        _binary(Gene='ENSG7', pValue='1e-3'),
        _binary(Gene='ENSG8', pValue='NaN'),
        _binary(Gene='ENSG9', pValue='2.5e-8', CollapsingModel='unknownModel', BinOddsRatio='1e-46'),
        _binary(Gene='ENSG10', Phenotype='Twice', BinOddsRatio='+Infinity', BinOddsRatioLCI='-0.0'),
    ]
    quantitative = [
        _quantitative(),
        _quantitative(Gene='ENSG11', beta='-0.3579588234424591', LCI='1.5f', UCI=''),
        _quantitative(Gene='ENSG12', pValue='9.9995e-9', nSamples='2147483648', YesQV='-5'),
        _quantitative(Gene='ENSG13', Phenotype=None),
    ]
    genes = [f'ENSG1,{GENE_LINK.format("release-1")}', f'ENSG2,{GENE_LINK.format("release-2")}']
    phenotypes = [
        'Disease A,https://azphewas.com/phenotypeView/a',
        'Trait B,https://azphewas.com/phenotypeView/b',
        'Twice,https://a/1',
        'Twice,https://a/2',
        ',https://null-name',
    ]
    paths = _inputs(tmp_path, binary, quantitative, genes, phenotypes)

    old = process_az_gene_burden(
        spark.read.option('mergeSchema', True).parquet(paths[0]),
        spark.read.option('mergeSchema', True).parquet(paths[1]),
        spark.read.csv(paths[2], header=False, schema='gene STRING, link STRING'),
        spark.read.csv(paths[3], header=False, schema='diseaseFromSource STRING, url STRING'),
    )
    columns = old.columns
    old_rows = [row.asDict(recursive=True) for row in old.collect()]
    for row in old_rows:
        # ruling 3: no link gives null urls, not a struct with a null url
        if row['urls'] == [{'url': None, 'niceName': NICE_NAME}]:
            row['urls'] = None
    new = parse_az(*paths).collect()

    absent = ['sex', 'diseaseFromSourceId', 'curatedDiseaseFromSourceMappedId']
    assert set(new.columns) - set(columns) == set(absent)
    assert new.select(absent).null_count().row(0) == (new.height,) * len(absent)
    assert multiset(new.rows(named=True), columns) == multiset(old_rows, columns)
