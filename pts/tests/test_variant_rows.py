"""Variant row collection tests."""

import gzip
import json
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.variant.rows import as_vcf_df, collect_variants
from tests.credible_set_fixtures import published_credible_set, published_credible_set_variant


def test_as_vcf_df_expands_locus_and_preserves_rsids():
    frame = pl.DataFrame({
        'variantId': ['1_20_A_G', None, '1_20_A_G'],
        'variantRsId': ['rs1', 'rs2', 'rs1'],
        'locus': [
            [{'variantId': '1_20_A_G'}, {'variantId': '1_10_C_T'}, {'variantId': None}],
            [{'variantId': '1_30_G_A'}],
            None,
        ],
    })
    result = as_vcf_df(frame.lazy()).collect()
    assert result.rows() == [('1', 10, 'rs1', 'C', 'T', '.', '.', '.'), ('1', 20, 'rs1', 'A', 'G', '.', '.', '.')]
    assert result.schema['POS'] == pl.Int32


@pytest.mark.parametrize('variant', ['1_bad_A_G', '1_2_N_A', '1_2_A_n', '1_2_<DEL>_A', '._2_A_G', 'bad'])
def test_as_vcf_df_discards_invalid_variants(variant):
    assert as_vcf_df(pl.DataFrame({'variantId': [variant]}).lazy()).collect().is_empty()


def test_as_vcf_df_missing_alleles_and_lexicographic_chromosomes():
    result = as_vcf_df(pl.DataFrame({'variantId': ['2_3', '10_20_A_G', '10_2_T_C', '2_3']}).lazy()).collect()
    assert result.select('CHROM', 'POS', 'REF', 'ALT', 'ID').rows() == [
        ('10', 2, 'T', 'C', '.'),
        ('10', 20, 'A', 'G', '.'),
        ('2', 3, '.', '.', '.'),
    ]


def test_as_vcf_df_missing_variant_column():
    assert as_vcf_df(pl.DataFrame({'other': [1]}).lazy()).collect().is_empty()


def test_as_vcf_df_null_locus_column():
    result = as_vcf_df(pl.DataFrame({'variantId': ['1_1_A_G'], 'locus': [None]}).lazy()).collect()
    assert result.height == 1


def test_as_vcf_df_retains_distinct_rsids_before_combining_sources():
    frame = pl.DataFrame({'variantId': ['1_1_A_G', '1_1_A_G'], 'variantRsId': ['rs1', 'rs2']})
    assert sorted(as_vcf_df(frame.lazy()).collect()['ID'].to_list()) == ['rs1', 'rs2']


def _inputs(tmp_path: Path) -> dict[str, str]:
    source = {}
    for name, variants in [('uniprot_variants', ['1_3_A_G', '1_1_C_T']), ('eva', ['1_3_A_G', '2_1_T_C'])]:
        path = tmp_path / f'{name}.json.gz'
        with gzip.open(path, 'wt') as stream:
            for variant in variants:
                stream.write(json.dumps({'variantId': variant}) + '\n')
        source[name] = str(path)
    for name, frame in [
        ('pharmacogenomics', pl.DataFrame({'variantId': ['1_2_G_C']})),
        ('credible_set_variant', published_credible_set_variant({'credibleSetId': 'a', 'variantId': '1_2_G_C'})),
    ]:
        path = tmp_path / name
        path.mkdir()
        frame.write_parquet(path / 'part.parquet')
        source[name] = str(path)
    return source


def _collect(source: dict[str, str]) -> pl.DataFrame:
    return collect_variants(**source).collect()


def test_collect_variants_columns_and_order(tmp_path):
    result = _collect(_inputs(tmp_path))
    assert result.columns == ['chromosome', 'position', 'rsId', 'referenceAllele', 'alternateAllele']
    assert result.schema['position'] == pl.Int32
    assert result.select('chromosome', 'position').rows() == [('1', 1), ('1', 2), ('1', 3), ('2', 1)]
    assert result['rsId'].to_list() == ['.'] * 4


def test_rows_match_the_partition_count(tmp_path):
    assert _collect(_inputs(tmp_path)).height == 4


def test_collect_variants_deduplicates_coordinates_across_source_rsids(tmp_path):
    source = _inputs(tmp_path)
    pl.DataFrame({'variantId': ['1_3_A_G'], 'variantRsId': ['rs3']}).write_parquet(
        Path(source['pharmacogenomics']) / 'part.parquet'
    )
    result = _collect(source)
    assert result.filter((pl.col('chromosome') == '1') & (pl.col('position') == 3))['rsId'].to_list() == ['rs3']


def test_collect_variants_keeps_the_same_rsid_whatever_the_input_order(tmp_path):
    kept = []
    for order, rsids in enumerate([['rs9', 'rs2'], ['rs2', 'rs9']]):
        (tmp_path / str(order)).mkdir()
        source = _inputs(tmp_path / str(order))
        pl.DataFrame({'variantId': ['1_3_A_G', '1_3_A_G'], 'variantRsId': rsids}).write_parquet(
            Path(source['pharmacogenomics']) / 'part.parquet'
        )
        result = _collect(source)
        kept.append(result.filter((pl.col('chromosome') == '1') & (pl.col('position') == 3))['rsId'].to_list())
    assert kept == [['rs2'], ['rs2']]


def test_collect_variants_empty_inputs(tmp_path):
    source = _inputs(tmp_path)
    for name in ['uniprot_variants', 'eva']:
        with gzip.open(source[name], 'wt') as stream:
            stream.write(json.dumps({'variantId': None}) + '\n')
    for name in ['pharmacogenomics', 'credible_set_variant']:
        pl.DataFrame({'variantId': [None]}, schema={'variantId': pl.String}).write_parquet(
            Path(source[name]) / 'part.parquet'
        )
    result = _collect(source)
    assert result.is_empty()
    assert result.columns == ['chromosome', 'position', 'rsId', 'referenceAllele', 'alternateAllele']


def test_a_variant_shared_by_two_credible_sets_is_one_row_and_every_lead_is_kept(tmp_path):
    source = _inputs(tmp_path)
    published_credible_set_variant(
        *(
            {'credibleSetId': set_id, 'variantId': variant_id}
            for set_id, variant_id in [('a', '1_2_G_C'), ('a', '1_5_A_T'), ('b', '1_2_G_C'), ('b', '1_7_C_G')]
        )
    ).write_parquet(Path(source['credible_set_variant']) / 'part.parquet')
    positions = _collect(source).filter(pl.col('chromosome') == '1')['position'].to_list()
    assert positions == [1, 2, 3, 5, 7]


def test_a_credible_set_dataset_in_place_of_the_variants_is_an_error(tmp_path):
    source = _inputs(tmp_path)
    published_credible_set({'id': 'a', 'leadVariantId': '1_2_G_C'}).write_parquet(
        Path(source['credible_set_variant']) / 'part.parquet'
    )
    with pytest.raises(ValueError, match='variantId'):
        _collect(source)


def test_collect_variants_reads_additional_datasets(tmp_path):
    source = _inputs(tmp_path)
    extra = tmp_path / 'az_exwas'
    extra.mkdir()
    pl.DataFrame({'variantId': ['3_7_G_A', '1_3_A_G'], 'pValue': [1e-9, 1e-9]}).write_parquet(extra / 'part.parquet')
    result = collect_variants(**source, additional_variants=[str(extra)]).collect()
    assert result.select('chromosome', 'position').rows() == [('1', 1), ('1', 2), ('1', 3), ('2', 1), ('3', 7)]
