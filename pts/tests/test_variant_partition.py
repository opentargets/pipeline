"""Variant extraction and actual partitioned VCF output tests."""

import gzip
import json
from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest
from pydantic import ValidationError

from pts.transformers.variant_partition import (
    VariantPartitionDestination,
    VariantPartitionSettings,
    VariantPartitionSource,
    as_vcf_df,
    variant_partition,
)


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


@pytest.mark.parametrize('value', [0, -1])
def test_partition_size_must_be_positive(value):
    with pytest.raises(ValidationError):
        VariantPartitionSettings(partition_size=value)


def test_models_reject_unknown_fields():
    with pytest.raises(ValidationError):
        VariantPartitionSettings.model_validate({'extra': 1})
    with pytest.raises(ValidationError):
        VariantPartitionDestination.model_validate({'output': 'x', 'extra': 1})
    with pytest.raises(ValidationError):
        VariantPartitionSource.model_validate({
            'uniprot_variants': 'a',
            'eva': 'b',
            'pharmacogenomics': 'c',
            'credible_set': 'd',
            'extra': 1,
        })


def _inputs(tmp_path: Path) -> dict[str, str]:
    source = {}
    for name, variants in [('uniprot_variants', ['1_3_A_G', '1_1_C_T']), ('eva', ['1_3_A_G', '2_1_T_C'])]:
        path = tmp_path / f'{name}.json.gz'
        with gzip.open(path, 'wt') as stream:
            for variant in variants:
                stream.write(json.dumps({'variantId': variant}) + '\n')
        source[name] = str(path)
    for name in ['pharmacogenomics', 'credible_set']:
        path = tmp_path / name
        path.mkdir()
        pl.DataFrame({'variantId': ['1_2_G_C']}).write_parquet(path / 'part.parquet')
        source[name] = str(path)
    return source


def test_variant_partition_writes_real_vcf_parts(tmp_path):
    source = _inputs(tmp_path)
    output = tmp_path / 'output'
    variant_partition(source, {'output': str(output)}, {'partition_size': 2}, Mock())
    files = sorted(output.glob('*.csv'))
    assert len(files) == 2
    for path in files:
        assert path.read_text().splitlines()[0] == 'CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO'
        assert len(path.read_text().splitlines()) == 3
    result = pl.concat([pl.read_csv(path, separator='\t') for path in files])
    assert result.select('CHROM', 'POS').rows() == [(1, 1), (1, 2), (1, 3), (2, 1)]
    assert result.select('ID', 'QUAL', 'FILTER', 'INFO').rows() == [('.', '.', '.', '.')] * 4
    with pytest.raises(ValueError, match='already exists'):
        variant_partition(source, {'output': str(output)}, {'partition_size': 2}, Mock())


def test_variant_partition_writes_partial_final_partition(tmp_path):
    source = _inputs(tmp_path)
    output = tmp_path / 'output'
    variant_partition(source, {'output': str(output)}, {'partition_size': 3}, Mock())
    assert sorted(pl.read_csv(path, separator='\t').height for path in output.glob('*.csv')) == [1, 3]


def test_variant_partition_deduplicates_coordinates_across_source_rsids(tmp_path):
    source = _inputs(tmp_path)
    pl.DataFrame({'variantId': ['1_3_A_G'], 'variantRsId': ['rs3']}).write_parquet(
        Path(source['pharmacogenomics']) / 'part.parquet'
    )
    output = tmp_path / 'output'
    variant_partition(source, {'output': str(output)}, {}, Mock())
    result = pl.concat([pl.read_csv(path, separator='\t') for path in output.glob('*.csv')])
    assert result.filter((pl.col('CHROM') == 1) & (pl.col('POS') == 3))['ID'].to_list() == ['rs3']


def test_variant_partition_keeps_the_same_rsid_whatever_the_input_order(tmp_path):
    """Conflicting rsIDs for one variant resolve to the smallest, independent of source order."""
    kept = []
    for order, rsids in enumerate([['rs9', 'rs2'], ['rs2', 'rs9']]):
        (tmp_path / str(order)).mkdir()
        source = _inputs(tmp_path / str(order))
        pl.DataFrame({'variantId': ['1_3_A_G', '1_3_A_G'], 'variantRsId': rsids}).write_parquet(
            Path(source['pharmacogenomics']) / 'part.parquet'
        )
        output = tmp_path / str(order) / 'output'
        variant_partition(source, {'output': str(output)}, {}, Mock())
        result = pl.concat([pl.read_csv(path, separator='\t') for path in output.glob('*.csv')])
        kept.append(result.filter((pl.col('CHROM') == 1) & (pl.col('POS') == 3))['ID'].to_list())
    assert kept == [['rs2'], ['rs2']]


def test_variant_partition_empty_inputs_write_header(tmp_path):
    source = _inputs(tmp_path)
    for name in ['uniprot_variants', 'eva']:
        with gzip.open(source[name], 'wt') as stream:
            stream.write(json.dumps({'variantId': None}) + '\n')
    for name in ['pharmacogenomics', 'credible_set']:
        pl.DataFrame({'variantId': [None]}, schema={'variantId': pl.String}).write_parquet(
            Path(source[name]) / 'part.parquet'
        )
    output = tmp_path / 'output'
    variant_partition(source, {'output': str(output)}, {}, Mock())
    files = list(output.glob('*.csv'))
    assert len(files) == 1
    assert files[0].read_text() == 'CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n'
