"""Tests for the colocated variants matched on the indexed VEP cache."""

from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path

import polars as pl
import pytest

from pts.schemas.vep import vep_schema
from pts.transformers.variant.colocated import colocated_variants, extract_all_vars
from pts.transformers.variant.vep import parse_vep

CACHE_COLUMNS = (
    'chr',
    'variation_name',
    'failed',
    'somatic',
    'start',
    'end',
    'allele_string',
    'strand',
    'minor_allele',
    'minor_allele_freq',
    'clin_sig',
    'phenotype_or_disease',
    'clin_sig_allele',
    'pubmed',
    'var_synonyms',
    'AFR',
    'AMR',
    'EAS',
    'EUR',
    'SAS',
)
HOMOPOLYMER = '/'.join('T' * n for n in (17, 13, 14, 15, 16, 18, 19))


def cache_line(name: str, start: int, end: int | str, alleles: str, **values: str) -> str:
    """One `all_vars` line in the cache's `variation_cols` order."""
    fields = {'chr': '1', 'variation_name': name, 'start': str(start), 'end': str(end), 'allele_string': alleles}
    fields |= values
    return '\t'.join(fields.get(column, '.') for column in CACHE_COLUMNS)


def all_vars(path: Path, *lines: str) -> Path:
    """An `all_vars.gz`, written as bgzip writes it: several gzip members."""
    path.write_bytes(b''.join(gzip.compress(f'{line}\n'.encode()) for line in lines))
    return path


def rows(*variant_ids: str) -> pl.DataFrame:
    parts = [variant_id.split('_') for variant_id in variant_ids]
    return pl.DataFrame(
        {
            'chromosome': [part[0] for part in parts],
            'position': [int(part[1]) for part in parts],
            'rsId': ['.'] * len(parts),
            'referenceAllele': [part[2] for part in parts],
            'alternateAllele': [part[3] for part in parts],
        },
        schema_overrides={'position': pl.Int32},
    )


def ids(result: pl.DataFrame) -> dict[str, list[str]]:
    """The colocated ids of each row, by variant id."""
    return {
        '_'.join(str(row[key]) for key in ('chromosome', 'position', 'referenceAllele', 'alternateAllele')): [
            variant['id'] for variant in row['colocated_variants']
        ]
        for row in result.to_dicts()
    }


def test_matches_vep_on_padded_repeat_indels(tmp_path: Path) -> None:
    cache = all_vars(
        tmp_path / 'all_vars.gz',
        cache_line('rs1557488101', 18611280, 18611283, 'AGGA/A'),
        cache_line('rs36055856', 170333597, '.', 'T/TAT'),
    )
    result = colocated_variants(cache, rows('1_18611279_AAGGAAGGAAGG_AAAGGAAGG', '1_170333597_T_TAT'))
    assert ids(result) == {
        '1_18611279_AAGGAAGGAAGG_AAAGGAAGG': ['rs1557488101'],
        '1_170333597_T_TAT': ['rs36055856'],
    }


def test_does_not_match_shifted_homopolymer(tmp_path: Path) -> None:
    cache = all_vars(tmp_path / 'all_vars.gz', cache_line('rs34289951', 7741945, 7741961, HOMOPOLYMER))
    assert colocated_variants(cache, rows('1_7741944_GTT_GT')).is_empty()


def test_insertion_window(tmp_path: Path) -> None:
    """An insertion of C after 1000 has start 1001 and end 1000, so the window is 1000..1001."""
    cache = all_vars(
        tmp_path / 'all_vars.gz',
        cache_line('rs_before', 999, 1000, 'AA/AAC'),
        cache_line('rs_anchor', 1000, 1000, 'A/AC'),
        cache_line('rs_start', 1001, 1000, '-/C'),
    )
    assert ids(colocated_variants(cache, rows('1_1000_A_AC'))) == {'1_1000_A_AC': ['rs_anchor', 'rs_start']}


def test_failed_and_null_alleles_dropped(tmp_path: Path) -> None:
    cache = all_vars(
        tmp_path / 'all_vars.gz',
        cache_line('rs_failed', 100, 100, 'A/G', failed='1'),
        cache_line('HGMD_NULL', 100, 100, 'HGMD_MUTATION'),
        cache_line('rs_kept', 100, 100, 'A/G', failed='0'),
    )
    assert ids(colocated_variants(cache, rows('1_100_A_G'))) == {'1_100_A_G': ['rs_kept']}


def test_repeated_cache_lines_once(tmp_path: Path) -> None:
    line = cache_line('rs201833382', 100, 100, 'A/G/T')
    cache = all_vars(tmp_path / 'all_vars.gz', line, line)
    assert ids(colocated_variants(cache, rows('1_100_A_G', '1_100_A_T'))) == {
        '1_100_A_G': ['rs201833382'],
        '1_100_A_T': ['rs201833382'],
    }


def test_ids_in_cache_line_order(tmp_path: Path) -> None:
    cache = all_vars(
        tmp_path / 'all_vars.gz',
        cache_line('rs9', 100, 102, 'ACG/A'),
        cache_line('rs3', 101, 102, 'CG/-'),
        cache_line('rs5', 101, 102, 'CG/-'),
        cache_line('rs1', 103, 103, 'T/C'),
    )
    assert ids(colocated_variants(cache, rows('1_100_ACG_A', '1_103_T_C'))) == {
        '1_100_ACG_A': ['rs9', 'rs3', 'rs5'],
        '1_103_T_C': ['rs1'],
    }


def test_cached_alleles_also_trimmed_from_the_end(tmp_path: Path) -> None:
    """`CGC/C` at 101 is `GC/-` at 102 trimmed from the start, and `CG/-` at 101 from the end."""
    cache = all_vars(tmp_path / 'all_vars.gz', cache_line('rs1', 101, 103, 'CGC/C'))
    assert ids(colocated_variants(cache, rows('1_100_ACG_A'))) == {'1_100_ACG_A': ['rs1']}


def test_reverse_strand_cached_variant(tmp_path: Path) -> None:
    cache = all_vars(
        tmp_path / 'all_vars.gz',
        cache_line('rs_reverse', 100, 100, 'T/C', strand='-1'),
        cache_line('rs_forward', 100, 100, 'T/C'),
    )
    assert ids(colocated_variants(cache, rows('1_100_A_G'))) == {'1_100_A_G': ['rs_reverse']}


def test_rows_without_colocated_variants_absent(tmp_path: Path) -> None:
    cache = all_vars(tmp_path / 'all_vars.gz', cache_line('rs1', 100, 100, 'A/G'))
    result = colocated_variants(cache, rows('1_100_A_C', '1_100_A_G', '2_100_A_G'))
    assert ids(result) == {'1_100_A_G': ['rs1']}


def test_colocated_variants_in_vep_shape(tmp_path: Path) -> None:
    cache = all_vars(
        tmp_path / 'all_vars.gz',
        cache_line('rs1', 100, 100, 'A/G', var_synonyms='ClinVar::RCV000001,VCV000002--OMIM::171834.0000'),
        cache_line('rs2', 100, 100, 'A/G', var_synonyms='UniProt::VAR_000001'),
        cache_line('rs3', 100, 100, 'A/G'),
    )
    result = colocated_variants(cache, rows('1_100_A_G'))
    assert result.schema['colocated_variants'] == vep_schema['colocated_variants']
    assert result['colocated_variants'].to_list() == [
        [
            {
                'id': 'rs1',
                'allele_string': 'A/G',
                'start': 100,
                'end': 100,
                'var_synonyms': {'ClinVar': ['RCV000001', 'VCV000002'], 'OMIM': ['171834.0000']},
            },
            {
                'id': 'rs2',
                'allele_string': 'A/G',
                'start': 100,
                'end': 100,
                'var_synonyms': {'ClinVar': None, 'OMIM': None},
            },
            {'id': 'rs3', 'allele_string': 'A/G', 'start': 100, 'end': 100, 'var_synonyms': None},
        ]
    ]


def test_clinvar_ids_deduplicated(tmp_path: Path) -> None:
    cache = all_vars(
        tmp_path / 'all_vars.gz',
        cache_line('rs1', 100, 100, 'A/G', var_synonyms='ClinVar::RCV000001,VCV000009--OMIM::171834.0000'),
        cache_line('COSV1', 100, 100, 'A/G', var_synonyms='ClinVar::RCV000002,VCV000009,VCV000003'),
    )
    colocated = colocated_variants(cache, rows('1_100_A_G'))['colocated_variants'].to_list()[0]
    record = {'input': '1\t100\t.\tA\tG', 'most_severe_consequence': 'intergenic_variant'}
    frame = pl.read_ndjson(io.BytesIO(json.dumps(record).encode()), schema=vep_schema).with_columns(
        pl.Series('colocated_variants', [colocated], dtype=vep_schema['colocated_variants'])
    )
    [row] = parse_vep(frame.lazy()).collect().to_dicts()
    assert [(x['id'], x['source']) for x in row['dbXrefs']] == [
        ('rs1', 'ensembl_variation'),
        ('COSV1', 'ensembl_variation'),
        ('171834#0000', 'omim'),
        ('VCV000009', 'clinvar'),
        ('VCV000003', 'clinvar'),
    ]


def tarball(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, 'w:gz') as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def test_extract_all_vars(tmp_path: Path) -> None:
    packed = tarball(
        tmp_path / 'homo_sapiens_vep_116_GRCh38.tar.gz',
        {
            'homo_sapiens/116_GRCh38/1/1-1000000.gz': b'transcripts',
            'homo_sapiens/116_GRCh38/1/all_vars.gz': b'chromosome 1',
            'homo_sapiens/116_GRCh38/1/all_vars.gz.csi': b'index',
            'homo_sapiens/116_GRCh38/MT/all_vars.gz': b'chromosome MT',
            'homo_sapiens/116_GRCh38/info.txt': b'variation_cols\tchr',
        },
    )
    directory = tmp_path / 'variation'
    extracted = extract_all_vars(str(packed), directory)
    assert extracted == {'1': directory / '1.all_vars.gz', 'MT': directory / 'MT.all_vars.gz'}
    assert extracted['1'].read_bytes() == b'chromosome 1'
    assert extracted['MT'].read_bytes() == b'chromosome MT'
    assert sorted(path.name for path in directory.iterdir()) == ['1.all_vars.gz', 'MT.all_vars.gz']


def test_extract_all_vars_refuses_a_cache_without_variation(tmp_path: Path) -> None:
    packed = tarball(tmp_path / 'cache.tar.gz', {'homo_sapiens/116_GRCh38/info.txt': b''})
    with pytest.raises(ValueError, match=r'no all_vars\.gz'):
        extract_all_vars(str(packed), tmp_path / 'variation')
