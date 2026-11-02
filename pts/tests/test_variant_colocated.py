"""Tests for the colocated variants matched on the indexed VEP cache."""

from __future__ import annotations

import gzip
import io
import random
import tarfile
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.variant.index.rows import index_rows
from pts.transformers.variant.vep.alleles import shared_prefix_length
from pts.transformers.variant.vep.colocated import KEY, _keys, add_colocated, colocated_variants, extract_all_vars
from pts.transformers.variant.vep.frames import ID_LISTS
from tests.variant_fixtures import consequences, variants, with_effects

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
            *row['rsIds']
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


def test_trailing_empty_cached_allele_dropped(tmp_path: Path) -> None:
    """`A/G/` at 101 is `A/G`, as Perl's `split` drops trailing empty fields: no deletion of the `A`."""
    cache = all_vars(
        tmp_path / 'all_vars.gz', cache_line('rs_trailing', 101, 101, 'A/G/'), cache_line('rs1', 101, 101, 'A/-')
    )
    assert ids(colocated_variants(cache, rows('1_100_CA_C'))) == {'1_100_CA_C': ['rs1']}


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


def test_colocated_variants_as_id_lists(tmp_path: Path) -> None:
    cache = all_vars(
        tmp_path / 'all_vars.gz',
        cache_line('rs1', 100, 100, 'A/G', var_synonyms='ClinVar::RCV000001,VCV000002--OMIM::171834.0000'),
        cache_line('rs2', 100, 100, 'A/G', var_synonyms='UniProt::VAR_000001'),
        cache_line('rs3', 100, 100, 'A/G'),
    )
    result = colocated_variants(cache, rows('1_100_A_G'))
    assert result.schema == pl.Schema({**rows('1_100_A_G').select(KEY).schema, **ID_LISTS})
    assert result.select(*ID_LISTS).rows() == [(['rs1', 'rs2', 'rs3'], ['VCV000002'], ['171834#0000'])]


def index_of(variant_id: str, colocated: pl.DataFrame | None) -> dict:
    """The index row of an intergenic variant with its colocated ids from `colocated`."""
    chromosome, position, reference, alternate = variant_id.split('_')
    frame = variants(
        row=[0],
        chromosome=[chromosome],
        position=[int(position)],
        referenceAllele=[reference],
        alternateAllele=[alternate],
        most_severe_consequence=['intergenic_variant'],
        reported_transcripts=[False],
    ).drop(*ID_LISTS)
    [row] = index_rows(add_colocated(frame, colocated), with_effects(consequences())).to_dicts()
    return row


def test_clinvar_ids_deduplicated(tmp_path: Path) -> None:
    cache = all_vars(
        tmp_path / 'all_vars.gz',
        cache_line('rs1', 100, 100, 'A/G', var_synonyms='ClinVar::RCV000001,VCV000009--OMIM::171834.0000'),
        cache_line('COSV1', 100, 100, 'A/G', var_synonyms='ClinVar::RCV000002,VCV000009,VCV000003'),
    )
    colocated = colocated_variants(cache, rows('1_100_A_G'))
    assert colocated['clinvarIds'].to_list() == [['VCV000009', 'VCV000003']]
    row = index_of('1_100_A_G', colocated)
    assert [(x['id'], x['source']) for x in row['dbXrefs']] == [
        ('rs1', 'ensembl_variation'),
        ('COSV1', 'ensembl_variation'),
        ('171834#0000', 'omim'),
        ('VCV000009', 'clinvar'),
        ('VCV000003', 'clinvar'),
    ]


@pytest.mark.parametrize('located', [False, True])
def test_variants_without_colocated_have_empty_id_lists(tmp_path: Path, located: bool) -> None:
    """No colocated file for the chromosome, or no colocated variant of the row: `[]`, never null."""
    cache = all_vars(tmp_path / 'all_vars.gz', cache_line('rs1', 100, 100, 'A/G'))
    colocated = colocated_variants(cache, rows('1_100_A_G')) if located else None
    row = index_of('1_200_A_G', colocated)
    assert (row['rsIds'], row['dbXrefs']) == ([], [])
    key = rows('1_200_A_G').select(KEY)
    found = add_colocated(variants(**key.to_dict()).drop(*ID_LISTS), colocated).select(*ID_LISTS)
    assert found.schema == ID_LISTS
    assert found.rows() == [([], [], [])]


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


def _keys_per_row(alleles: pl.DataFrame) -> pl.DataFrame:
    """The previous `_keys`: every row trimmed on its own."""

    def trimmed(frame: pl.DataFrame, end_first: bool) -> pl.DataFrame:
        ref, alt = pl.col('_ref'), pl.col('_alt')
        frame = frame.with_columns(pl.col('ref').alias('_ref'), pl.col('alt').alias('_alt'), pl.lit(0).alias('_prefix'))
        for step in ('suffix', 'prefix') if end_first else ('prefix', 'suffix'):
            if step == 'prefix':
                frame = frame.with_columns(shared_prefix_length(ref, alt).alias('_prefix')).with_columns(
                    ref.str.slice(pl.col('_prefix')), alt.str.slice(pl.col('_prefix'))
                )
            else:
                frame = frame.with_columns(
                    shared_prefix_length(ref.str.reverse(), alt.str.reverse()).alias('_suffix')
                ).with_columns(
                    ref.str.head(ref.str.len_chars() - pl.col('_suffix')),
                    alt.str.head(alt.str.len_chars() - pl.col('_suffix')),
                )
        return frame.select(
            'pair',
            pl.concat_str(
                pl.when(ref == '').then(pl.lit('-')).otherwise(ref),
                pl.when(alt == '').then(pl.lit('-')).otherwise(alt),
                (pl.col('pos') + pl.col('_prefix')).cast(pl.String),
                separator='_',
            ).alias('key'),
        )

    longer = alleles.filter((pl.col('ref').str.len_chars() > 1) | (pl.col('alt').str.len_chars() > 1))
    return pl.concat([trimmed(alleles, end_first=False), trimmed(longer, end_first=True)]).unique()


@pytest.mark.parametrize('seed', range(5))
def test_keys_trimmed_once_per_allele_pair_equal_keys_trimmed_per_row(seed: int):
    rng = random.Random(seed)

    def allele() -> str:
        return rng.choice(['-', 'A', 'C', 'G', 'T', 'AC', 'CA', 'ACA', 'ATTA', 'GGGG', 'AAAAAAAAAT', 'TACGTACGTA'])

    size = 3_000
    alleles = pl.DataFrame(
        {
            'pair': range(size),
            'ref': [allele() for _ in range(size)],
            'alt': [allele() for _ in range(size)],
            'pos': [rng.randint(1, 50) for _ in range(size)],
        },
        schema_overrides={'pair': pl.UInt32, 'pos': pl.Int64},
    )
    found, expected = _keys(alleles), _keys_per_row(alleles)
    assert found.sort('pair', 'key').equals(expected.sort('pair', 'key'))
