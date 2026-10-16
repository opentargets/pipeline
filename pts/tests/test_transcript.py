"""Tests for the transcript transformer."""

import gzip
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from pts.transformers.transcript import (
    build_flags,
    build_uniprot_lut,
    join_and_finalise,
    parse_exons,
    parse_gff3,
    to_long,
    transcript,
)

GFF3_COLUMNS = [f'_c{i}' for i in range(9)]

TX = pl.Struct({
    'id': pl.String,
    'uniprot_swissprot': pl.List(pl.String),
    'uniprot_trembl': pl.List(pl.String),
    'uniprot_isoform': pl.List(pl.String),
    'alphafold': pl.List(pl.String),
})
ENSEMBL_SCHEMA = {'id': pl.String, 'transcripts': pl.List(TX)}

FLAG = pl.Struct({'label': pl.String, 'value': pl.String})
EXON = pl.Struct({
    'exonId': pl.String,
    'chromosome': pl.String,
    'start': pl.Int64,
    'end': pl.Int64,
    'strand': pl.Int32,
})
OUTPUT_SCHEMA = {
    'targetId': pl.String,
    'transcriptId': pl.String,
    'biotype': pl.String,
    'proteinId': pl.String,
    'uniprotSwissprotIds': pl.List(pl.String),
    'uniprotTremblIds': pl.List(pl.String),
    'uniprotIsoformIds': pl.List(pl.String),
    'alphafoldIds': pl.List(pl.String),
    'isEnsemblCanonical': pl.Boolean,
    'chromosome': pl.String,
    'start': pl.Int64,
    'end': pl.Int64,
    'strand': pl.Int32,
    'transcriptionStartSite': pl.Int32,
    'flags': pl.List(FLAG),
    'exons': pl.List(EXON),
}


def _gff(chrom, feature, start, end, strand, attrs) -> tuple:
    return (chrom, 'HAVANA', feature, str(start), str(end), '.', strand, '.', attrs)


def gff(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=dict.fromkeys(GFF3_COLUMNS, pl.String), orient='row')


def _ensembl_tx(id, uniprot_swissprot=None, uniprot_trembl=None, uniprot_isoform=None, alphafold=None) -> dict:
    return {
        'id': id,
        'uniprot_swissprot': uniprot_swissprot,
        'uniprot_trembl': uniprot_trembl,
        'uniprot_isoform': uniprot_isoform,
        'alphafold': alphafold,
    }


def ensembl(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=ENSEMBL_SCHEMA, orient='row')


def first(df: pl.DataFrame) -> dict:
    assert df.height >= 1
    return df.row(0, named=True)


_TX_ATTRS = (
    'gene_id=ENSG00000186092.7;transcript_id=ENST00000641515.2;'
    'transcript_type=protein_coding;protein_id=ENSP00000493376.2;'
    'tag=Ensembl_canonical,GENCODE_Primary,MANE_Select,appris_principal_1'
)
_NC_ATTRS = 'gene_id=ENSG00000290825.3;transcript_id=ENST00000832824.1;transcript_type=lncRNA;tag=basic'
_EXON_ATTRS_A = 'transcript_id=ENST00000641515.2;exon_id=ENSE00003899065.1'
_EXON_ATTRS_B = 'transcript_id=ENST00000641515.2;exon_id=ENSE00001234567.2'


# parse_gff3


def test_parse_gff3_excludes_non_transcript_features() -> None:
    result = parse_gff3(
        gff([
            _gff('chr1', 'gene', 65419, 71585, '+', _TX_ATTRS),
            _gff('chr1', 'transcript', 65419, 71585, '+', _TX_ATTRS),
            _gff('chr1', 'exon', 65419, 65433, '+', _TX_ATTRS),
        ])
    )
    assert result.height == 1
    assert first(result)['transcriptId'] == 'ENST00000641515'


def test_parse_gff3_strips_version_suffix() -> None:
    row = first(parse_gff3(gff([_gff('chr1', 'transcript', 65419, 71585, '+', _TX_ATTRS)])))
    assert row['targetId'] == 'ENSG00000186092'
    assert row['transcriptId'] == 'ENST00000641515'
    assert row['proteinId'] == 'ENSP00000493376'


def test_parse_gff3_normalises_chrm_to_mt() -> None:
    attrs = _NC_ATTRS.replace('ENSG00000290825', 'ENSG00000000001')
    row = first(parse_gff3(gff([_gff('chrM', 'transcript', 1000, 2000, '+', attrs)])))
    assert row['chromosome'] == 'MT'


def test_parse_gff3_filters_noncanonical_chromosomes() -> None:
    result = parse_gff3(
        gff([
            _gff('chr1', 'transcript', 65419, 71585, '+', _TX_ATTRS),
            _gff('chr1_KI270706v1_random', 'transcript', 65419, 71585, '+', _TX_ATTRS),
        ])
    )
    assert result.height == 1
    assert first(result)['chromosome'] == '1'


# strand: GFF3 +/- becomes Ensembl's 1/-1


def test_parse_gff3_strand_is_signed_integer() -> None:
    result = parse_gff3(
        gff([
            _gff('chr1', 'transcript', 65419, 71585, '+', _TX_ATTRS),
            _gff('chr1', 'transcript', 450740, 451678, '-', _NC_ATTRS),
        ])
    )
    assert result.schema['strand'] == pl.Int32
    assert dict(result.select('transcriptId', 'strand').rows()) == {'ENST00000641515': 1, 'ENST00000832824': -1}


@pytest.mark.parametrize('raw_strand', ['.', '?'])
def test_parse_gff3_unstranded_feature_is_null_not_zero(raw_strand: str) -> None:
    row = first(parse_gff3(gff([_gff('chr1', 'transcript', 65419, 71585, raw_strand, _TX_ATTRS)])))
    assert row['strand'] is None


@pytest.mark.parametrize('raw_strand', ['.', '?'])
def test_parse_gff3_unstranded_feature_tss_falls_back_to_end(raw_strand: str) -> None:
    row = first(parse_gff3(gff([_gff('chr1', 'transcript', 65419, 71585, raw_strand, _TX_ATTRS)])))
    assert row['strand'] is None
    assert row['transcriptionStartSite'] == 71585


def test_parse_exons_unstranded_feature_is_null() -> None:
    row = first(parse_exons(gff([_gff('chr1', 'exon', 65419, 65433, '.', _EXON_ATTRS_A)])))
    assert row['exons'][0]['strand'] is None


def test_parse_exons_strand_is_signed_integer() -> None:
    row = first(parse_exons(gff([_gff('chr1', 'exon', 450740, 451678, '-', _EXON_ATTRS_A)])))
    assert row['exons'][0]['strand'] == -1


def test_parse_gff3_tss_positive_strand() -> None:
    row = first(parse_gff3(gff([_gff('chr1', 'transcript', 65419, 71585, '+', _TX_ATTRS)])))
    assert row['transcriptionStartSite'] == 65419


def test_parse_gff3_tss_negative_strand() -> None:
    result = parse_gff3(gff([_gff('chr1', 'transcript', 450740, 451678, '-', _TX_ATTRS)]))
    assert result.schema['transcriptionStartSite'] == pl.Int32
    assert first(result)['transcriptionStartSite'] == 451678


def test_parse_gff3_null_protein_id_for_noncoding() -> None:
    row = first(parse_gff3(gff([_gff('chr1', 'transcript', 11121, 14413, '+', _NC_ATTRS)])))
    assert row['proteinId'] is None


def test_parse_gff3_is_ensembl_canonical_true_when_tagged() -> None:
    row = first(parse_gff3(gff([_gff('chr1', 'transcript', 65419, 71585, '+', _TX_ATTRS)])))
    assert row['isEnsemblCanonical'] is True


def test_parse_gff3_is_ensembl_canonical_false_when_not_tagged() -> None:
    row = first(parse_gff3(gff([_gff('chr1', 'transcript', 11121, 14413, '+', _NC_ATTRS)])))
    assert row['isEnsemblCanonical'] is False


def test_regexp_extract_no_match_is_empty_string() -> None:
    row = first(parse_gff3(gff([_gff('chr1', 'transcript', 1, 100, '+', 'gene_id=ENSG2.1;transcript_id=ENST2.1')])))
    assert row['biotype'] == ''
    assert row['proteinId'] is None
    assert row['isEnsemblCanonical'] is False
    assert row['flags'] == []


def test_parse_gff3_text_fields_follow_spark() -> None:
    result = parse_gff3(
        gff([
            _gff('chr1', 'transcript', ' 5', '9 ', '+', 'gene_id=ENSG6.1;transcript_id=ENST6.1'),
            _gff('chr1', 'transcript', '+5', 'x', '+', 'gene_id=ENSG7.1;transcript_id=ENST7.1'),
            ('chr1', 'HAVANA', 'transcript', None, '10', '.', ' +', '.', 'gene_id=ENSG8.1;transcript_id=ENST8.1'),
            _gff('chr1', 'transcript', 1, 10, '+', 'gene_id=ENSG9.1;transcript_id="ENST9.1";transcript_type="a;b"'),
            _gff(' chr1', 'transcript', 1, 10, '+', 'gene_id=ENSG10.1;transcript_id=ENST10.1'),
        ])
    )
    rows = {r['transcriptId']: r for r in result.iter_rows(named=True)}
    assert sorted(rows) == ['ENST10', 'ENST6', 'ENST7', 'ENST8']
    assert (rows['ENST6']['start'], rows['ENST6']['end']) == (5, 9)
    assert (rows['ENST7']['start'], rows['ENST7']['end'], rows['ENST7']['transcriptionStartSite']) == (5, None, 5)
    assert (rows['ENST8']['start'], rows['ENST8']['strand']) == (None, None)
    assert rows['ENST8']['transcriptionStartSite'] == 10
    assert rows['ENST10']['chromosome'] == '1'


CASTS = [
    (' 5', 5), ('5 ', 5), ('+5', 5), ('\x0b5', 5), ('5\r', 5), ('\x015', 5), ('\x1f5', 5), ('\x7f5', 5),
    ('\xa05', None), ('\u20035', None), ('\x805', None), ('5.0', 5), ('5.7', 5), ('5.', 5), ('.5', 0),
    ('-5.9', -5), ('+.5', 0), ('.', 0), ('05', 5), ('-5', -5), (' - 5', None), ('+ 5', None), ('++5', None),
    ('5.x', None), ('5..', None), ('-', None), ('+', None), ('', None), ('x', None), ('1e3', None),
    ('9223372036854775807', 9223372036854775807), ('9223372036854775808', None), (None, None),
]  # fmt: skip


def test_to_long_matches_spark_cast() -> None:
    got = pl.DataFrame({'v': [v for v, _ in CASTS]}, schema={'v': pl.String}).select(to_long(pl.col('v')))
    assert got.to_series().to_list() == [e for _, e in CASTS]


# build_flags


def _flags_for(tags: list[str] | None) -> Any:
    df = pl.DataFrame({'tags': [tags]}, schema={'tags': pl.List(pl.String)})
    return df.select(build_flags(pl.col('tags'))).to_series().to_list()[0]


def test_build_flags_mane_select() -> None:
    assert {'label': 'mane', 'value': 'select'} in _flags_for(['MANE_Select'])


def test_build_flags_mane_plus_clinical() -> None:
    assert {'label': 'mane', 'value': 'plus_clinical'} in _flags_for(['MANE_Plus_Clinical'])


def test_build_flags_gencode_primary() -> None:
    assert {'label': 'gencode_primary', 'value': 'true'} in _flags_for(['GENCODE_Primary'])


def test_build_flags_ensembl_canonical() -> None:
    assert {'label': 'ensembl_canonical', 'value': 'true'} in _flags_for(['Ensembl_canonical'])


def test_build_flags_appris_principal() -> None:
    assert {'label': 'appris', 'value': 'principal_1'} in _flags_for(['appris_principal_1'])


def test_build_flags_appris_alternative() -> None:
    assert {'label': 'appris', 'value': 'alternative_2'} in _flags_for(['appris_alternative_2'])


def test_build_flags_empty_for_irrelevant_tags() -> None:
    assert _flags_for(['basic', 'CCDS', 'RNA_Seq_supported_partial']) == []


def test_build_flags_multiple_flags() -> None:
    flags = _flags_for(['MANE_Select', 'GENCODE_Primary', 'Ensembl_canonical', 'appris_principal_1'])
    assert {f['label'] for f in flags} == {'mane', 'gencode_primary', 'ensembl_canonical', 'appris'}


def test_build_flags_static_first_then_appris_in_tag_order() -> None:
    assert _flags_for(['appris_principal_1', 'MANE_Select', 'appris_x']) == [
        {'label': 'mane', 'value': 'select'},
        {'label': 'appris', 'value': 'principal_1'},
        {'label': 'appris', 'value': 'x'},
    ]


# parse_exons


def test_parse_exons_groups_by_transcript() -> None:
    result = parse_exons(
        gff([
            _gff('chr1', 'exon', 65419, 65433, '+', _EXON_ATTRS_A),
            _gff('chr1', 'exon', 65500, 65600, '+', _EXON_ATTRS_B),
            _gff('chr1', 'transcript', 65419, 71585, '+', _TX_ATTRS),
        ])
    )
    assert result.height == 1
    row = first(result)
    assert row['transcriptId'] == 'ENST00000641515'
    assert len(row['exons']) == 2


def test_parse_exons_strips_version_suffix() -> None:
    row = first(parse_exons(gff([_gff('chr1', 'exon', 65419, 65433, '+', _EXON_ATTRS_A)])))
    assert row['transcriptId'] == 'ENST00000641515'
    assert row['exons'][0]['exonId'] == 'ENSE00003899065'


def test_parse_exons_exon_struct_fields() -> None:
    result = parse_exons(gff([_gff('chr1', 'exon', 65419, 65433, '+', _EXON_ATTRS_A)]))
    assert result.schema['exons'] == pl.List(EXON)
    assert first(result)['exons'][0] == {
        'exonId': 'ENSE00003899065',
        'chromosome': '1',
        'start': 65419,
        'end': 65433,
        'strand': 1,
    }


def test_parse_exons_drops_exons_without_id() -> None:
    result = parse_exons(
        gff([
            _gff('chr1', 'exon', 1, 2, '+', 'transcript_id=ENST1.1'),
            _gff('chr1', 'exon', 1, 2, '+', 'exon_id=E1.1'),
            _gff('chr1_KI270706v1_random', 'exon', 1, 2, '+', 'transcript_id=ENST1.1;exon_id=E2.1'),
        ])
    )
    assert result.height == 0


# build_uniprot_lut


def test_build_uniprot_lut_keeps_swissprot_and_trembl_separate() -> None:
    lut = build_uniprot_lut(
        ensembl([
            ('ENSG00000001', [_ensembl_tx('ENST00000001', uniprot_swissprot=['P12345'], uniprot_trembl=['A0A001'])]),
        ])
    )
    row = first(lut.filter(pl.col('transcriptId') == 'ENST00000001'))
    assert row['uniprotSwissprotIds'] == ['P12345']
    assert row['uniprotTremblIds'] == ['A0A001']


def test_build_uniprot_lut_handles_null_trembl() -> None:
    lut = build_uniprot_lut(ensembl([('ENSG00000002', [_ensembl_tx('ENST00000002', uniprot_swissprot=['P99999'])])]))
    row = first(lut.filter(pl.col('transcriptId') == 'ENST00000002'))
    assert row['uniprotSwissprotIds'] == ['P99999']
    assert row['uniprotTremblIds'] is None


def test_build_uniprot_lut_propagates_isoform_and_alphafold() -> None:
    lut = build_uniprot_lut(
        ensembl([
            ('ENSG00000005', [_ensembl_tx('ENST00000005', uniprot_isoform=['P12345-2'], alphafold=['AF-P12345-F1'])]),
        ])
    )
    row = first(lut.filter(pl.col('transcriptId') == 'ENST00000005'))
    assert row['uniprotIsoformIds'] == ['P12345-2']
    assert row['alphafoldIds'] == ['AF-P12345-F1']


# join_and_finalise


def _joined(uniprot_tx: dict) -> pl.DataFrame:
    return join_and_finalise(
        parse_gff3(gff([_gff('chr1', 'transcript', 65419, 71585, '+', _TX_ATTRS)])),
        parse_exons(gff([_gff('chr1', 'exon', 65419, 65433, '+', _EXON_ATTRS_A)])),
        build_uniprot_lut(ensembl([('ENSG00000186092', [uniprot_tx])])),
    )


def test_join_and_finalise_output_schema() -> None:
    result = _joined(_ensembl_tx('ENST00000641515', uniprot_swissprot=['A0A2U3U0J3']))
    assert dict(result.schema) == OUTPUT_SCHEMA


def test_join_and_finalise_publishes_strand_as_integer() -> None:
    result = _joined(_ensembl_tx('ENST00000641515', uniprot_swissprot=['A0A2U3U0J3']))
    assert result.schema['strand'] == pl.Int32
    assert result.schema['exons'] == pl.List(EXON)
    row = first(result)
    assert row['strand'] == 1
    assert row['exons'][0]['strand'] == 1


def test_join_and_finalise_propagates_uniprot_ids() -> None:
    row = first(
        _joined(
            _ensembl_tx(
                'ENST00000641515',
                uniprot_swissprot=['A0A2U3U0J3'],
                uniprot_isoform=['A0A2U3U0J3-2'],
                alphafold=['AF-A0A2U3U0J3-F1'],
            )
        )
    )
    assert row['uniprotSwissprotIds'] == ['A0A2U3U0J3']
    assert row['uniprotTremblIds'] is None
    assert row['uniprotIsoformIds'] == ['A0A2U3U0J3-2']
    assert row['alphafoldIds'] == ['AF-A0A2U3U0J3-F1']
    assert len(row['exons']) == 1


# traps

ATTRS = 'gene_id=ENSG1.3;transcript_id={tx};transcript_type=protein_coding;tag=Ensembl_canonical'
PAR_LINES = [
    _gff('chrM', 'transcript', 1, 100, '+', ATTRS.format(tx='ENST1.2')),
    _gff('chrM', 'exon', 70, 100, '+', 'transcript_id=ENST1.2;exon_id=E3.1'),
    _gff('chrM', 'exon', 1, 10, '+', 'transcript_id=ENST1.2;exon_id=E1.1'),
    _gff('chrM', 'exon', 30, 50, '+', 'transcript_id=ENST1.2;exon_id=E2.1'),
    _gff('chrY', 'transcript', 500, 600, '-', ATTRS.format(tx='ENST1.2_PAR_Y').replace('ENSG1.3', 'ENSG1.3_PAR_Y')),
    _gff('chrY', 'exon', 500, 600, '-', 'transcript_id=ENST1.2_PAR_Y;exon_id=E9.1'),
]


def test_exons_keep_file_order_and_par_y_merges() -> None:
    raw = gff(PAR_LINES)
    result = join_and_finalise(parse_gff3(raw), parse_exons(raw), build_uniprot_lut(ensembl([]))).sort('chromosome')
    assert result['transcriptId'].to_list() == ['ENST1', 'ENST1']
    assert result['targetId'].to_list() == ['ENSG1', 'ENSG1']
    assert result['chromosome'].to_list() == ['MT', 'Y']
    for exons in result['exons'].to_list():
        assert [(e['exonId'], e['chromosome']) for e in exons] == [
            ('E3', 'MT'),
            ('E1', 'MT'),
            ('E2', 'MT'),
            ('E9', 'Y'),
        ]


def test_transcript_null_and_empty_lists() -> None:
    raw = gff([
        _gff('chr1', 'transcript', 1, 100, '+', 'gene_id=ENSG3.1;transcript_id=ENST3.1;tag=basic'),
        _gff('chr1', 'transcript', 1, 100, '+', 'gene_id=ENSG4.1;transcript_id=ENST4.1;tag=appris_principal_1'),
    ])
    lut = build_uniprot_lut(
        ensembl([
            ('ENSG3', None),
            ('ENSG4', []),
            ('ENSG5', [_ensembl_tx('ENST5', uniprot_trembl=[], uniprot_isoform=['I1'])]),
        ])
    )
    assert lut.rows() == [('ENSG5', 'ENST5', None, [], ['I1'], None)]
    result = join_and_finalise(parse_gff3(raw), parse_exons(raw), lut).sort('transcriptId')
    assert result['transcriptId'].to_list() == ['ENST3', 'ENST4']
    assert result['flags'].to_list() == [[], [{'label': 'appris', 'value': 'principal_1'}]]
    assert result['exons'].to_list() == [None, None]
    assert result['uniprotSwissprotIds'].to_list() == [None, None]
    assert _flags_for(None) is None
    assert _flags_for([]) == []


def test_multi_chunk_input_matches_single_chunk() -> None:
    raw = gff([*PAR_LINES, _gff('chr1', 'transcript', 65419, 71585, '+', _TX_ATTRS)])
    chunked = pl.concat([raw.head(3), raw.tail(-3)], rechunk=False)
    ens = ensembl([('ENSG00000186092', [_ensembl_tx('ENST00000641515', uniprot_swissprot=['P1'])])])
    ens_chunked = pl.concat([ens, ensembl([('ENSG9', [_ensembl_tx('ENST9')])])], rechunk=False)
    assert chunked.n_chunks() == 2

    def run(df: pl.DataFrame, lut: pl.DataFrame) -> pl.DataFrame:
        return join_and_finalise(parse_gff3(df), parse_exons(df), build_uniprot_lut(lut))

    assert run(chunked, ens_chunked).equals(run(raw, ens))


def test_transcript_end_to_end(tmp_path: Path) -> None:
    lines = ['##gff-version 3', '#description: test']
    lines += ['\t'.join(r) for r in [*PAR_LINES, _gff('chr1', 'transcript', 65419, 71585, '+', _TX_ATTRS)]]
    lines += ['###', '\t'.join(_gff('chr1', 'exon', 65419, 65433, '+', _EXON_ATTRS_A))]
    gene_code = tmp_path / 'gencode.gff3.gz'
    gene_code.write_bytes(gzip.compress(('\n'.join(lines) + '\n').encode()))
    ens = ensembl([('ENSG00000186092', [_ensembl_tx('ENST00000641515', uniprot_swissprot=['P1'])])])
    pl.concat([ens, ensembl([('ENSG1', None)])], rechunk=False).write_parquet(tmp_path / 'ensembl.parquet')
    source = {'gene_code': str(gene_code), 'ensembl': str(tmp_path / 'ensembl.parquet')}

    transcript(source, str(tmp_path / 'a'), {}, None)
    transcript(source, str(tmp_path / 'b'), {}, None)

    out = pl.read_parquet(tmp_path / 'a')
    assert dict(out.schema) == OUTPUT_SCHEMA
    assert out.select('transcriptId', 'chromosome').rows() == [
        ('ENST00000641515', '1'),
        ('ENST1', 'MT'),
        ('ENST1', 'Y'),
    ]
    row = first(out)
    assert row['uniprotSwissprotIds'] == ['P1']
    assert row['flags'] == [
        {'label': 'mane', 'value': 'select'},
        {'label': 'gencode_primary', 'value': 'true'},
        {'label': 'ensembl_canonical', 'value': 'true'},
        {'label': 'appris', 'value': 'principal_1'},
    ]
    assert [e['exonId'] for e in row['exons']] == ['ENSE00003899065']
    files_a = sorted(p.relative_to(tmp_path / 'a') for p in (tmp_path / 'a').rglob('*') if p.is_file())
    files_b = sorted(p.relative_to(tmp_path / 'b') for p in (tmp_path / 'b').rglob('*') if p.is_file())
    assert files_a == files_b
    assert all((tmp_path / 'a' / f).read_bytes() == (tmp_path / 'b' / f).read_bytes() for f in files_a)
