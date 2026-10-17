"""Tests for the transcript transformer."""

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from pts.schemas.transcript import transcript as transcript_schema
from pts.transformers.transcript import build_flags, build_transcripts, transcript
from tests.ensembl_fixtures import NULL, RELEASE_116_COLUMNS, release_116_ddl, write_dump

ATTRIBUTE = pl.Struct({'code': pl.String, 'value': pl.String})
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

#: `pts.ensembl.transcripts` output, as the contract states it.
TX_SCHEMA = {
    'geneId': pl.String,
    'transcriptId': pl.String,
    'biotype': pl.String,
    'chromosome': pl.String,
    'start': pl.Int64,
    'end': pl.Int64,
    'strand': pl.Int32,
    'translationId': pl.String,
    'attributes': pl.List(ATTRIBUTE),
    'exons': pl.List(EXON),
}
#: `pts.ensembl.translation_xrefs` output, as the contract states it.
XREF_SCHEMA = {
    'transcriptId': pl.String,
    'translationId': pl.String,
    'uniprot_swissprot': pl.List(pl.String),
    'uniprot_trembl': pl.List(pl.String),
    'uniprot_isoform': pl.List(pl.String),
    'alphafold': pl.List(pl.String),
    'signalp': pl.List(pl.String),
}


def exon(exon_id: str, start: int, end: int, strand: int = 1, chromosome: str = '1') -> dict[str, Any]:
    return {'exonId': exon_id, 'chromosome': chromosome, 'start': start, 'end': end, 'strand': strand}


def tx(
    transcript_id: str = 'ENST1',
    gene_id: str = 'ENSG1',
    *,
    biotype: str = 'protein_coding',
    chromosome: str = '1',
    start: int = 100,
    end: int = 200,
    strand: int = 1,
    translation_id: str | None = None,
    attributes: list[tuple[str, str]] | None = None,
    exons: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        'geneId': gene_id,
        'transcriptId': transcript_id,
        'biotype': biotype,
        'chromosome': chromosome,
        'start': start,
        'end': end,
        'strand': strand,
        'translationId': translation_id,
        'attributes': [{'code': c, 'value': v} for c, v in attributes or []],
        'exons': [exon('ENSE1', start, end, strand, chromosome)] if exons is None else exons,
    }


def xref(transcript_id: str, translation_id: str, **arrays: list[str]) -> dict[str, Any]:
    names = ['uniprot_swissprot', 'uniprot_trembl', 'uniprot_isoform', 'alphafold', 'signalp']
    return {'transcriptId': transcript_id, 'translationId': translation_id} | {n: arrays.get(n, []) for n in names}


def txs(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=TX_SCHEMA)


def xrefs(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=XREF_SCHEMA)


def build(tx_rows: list[dict[str, Any]], xref_rows: list[dict[str, Any]] | None = None) -> pl.DataFrame:
    return build_transcripts(txs(tx_rows), xrefs(xref_rows or []))


def only(df: pl.DataFrame) -> dict[str, Any]:
    assert df.height == 1
    return df.row(0, named=True)


def flags_for(attributes: list[tuple[str, str]]) -> Any:
    df = pl.DataFrame(
        {'a': [[{'code': c, 'value': v} for c, v in attributes]]},
        schema={'a': pl.List(ATTRIBUTE)},
    )
    return df.select(build_flags(pl.col('a'))).to_series().to_list()[0]


# schema


def test_output_schema_is_the_published_one() -> None:
    result = build([tx()])
    assert dict(result.schema) == OUTPUT_SCHEMA
    assert result.schema == transcript_schema


def test_signalp_is_not_published() -> None:
    result = build([tx(translation_id='ENSP1')], [xref('ENST1', 'ENSP1', signalp=['SignalP-noTM'])])
    assert not any('signal' in c.lower() for c in result.columns)


# flags


def test_static_flags_then_appris_in_output_order() -> None:
    attributes = [
        ('appris', 'principal1'),
        ('is_canonical', '1'),
        ('gencode_primary', 'GENCODE primary'),
        ('MANE_Select', 'NM_001005484.2'),
    ]
    assert flags_for(attributes) == [
        {'label': 'mane', 'value': 'select'},
        {'label': 'gencode_primary', 'value': 'true'},
        {'label': 'ensembl_canonical', 'value': 'true'},
        {'label': 'appris', 'value': 'principal_1'},
    ]


def test_mane_plus_clinical_value_is_not_published() -> None:
    assert flags_for([('MANE_Plus_Clinical', 'NM_000267.4')]) == [{'label': 'mane', 'value': 'plus_clinical'}]


@pytest.mark.parametrize(
    ('core', 'published'),
    [
        ('principal1', 'principal_1'),
        ('principal2', 'principal_2'),
        ('principal3', 'principal_3'),
        ('principal4', 'principal_4'),
        ('principal5', 'principal_5'),
        ('principalM', 'principal_M'),
        ('alternative1', 'alternative_1'),
        ('alternative2', 'alternative_2'),
        ('alternativeM', 'alternative_M'),
    ],
)
def test_appris_spelled_as_gencode(core: str, published: str) -> None:
    assert flags_for([('appris', core)]) == [{'label': 'appris', 'value': published}]


def test_unrequested_codes_give_no_flags() -> None:
    assert flags_for([('Ensembl_Select', 'ENST1.2'), ('basic', '')]) == []
    assert flags_for([]) == []


def test_is_ensembl_canonical_from_the_code() -> None:
    result = build([
        tx('ENST1', attributes=[('is_canonical', '1')]),
        tx('ENST2', attributes=[('MANE_Select', 'NM_1.1')]),
        tx('ENST3'),
    ])
    assert result.select('transcriptId', 'isEnsemblCanonical').rows() == [
        ('ENST1', True),
        ('ENST2', False),
        ('ENST3', False),
    ]


# coordinates


def test_tss_follows_the_strand() -> None:
    result = build([
        tx('ENST1', start=65419, end=71585, strand=1),
        tx('ENST2', start=450740, end=451678, strand=-1),
    ])
    assert result.schema['transcriptionStartSite'] == pl.Int32
    assert result.select('transcriptId', 'strand', 'transcriptionStartSite').rows() == [
        ('ENST1', 1, 65419),
        ('ENST2', -1, 451678),
    ]


def test_coordinates_pass_through() -> None:
    row = only(build([tx(chromosome='MT', start=648, end=1601, biotype='Mt_rRNA')]))
    assert (row['chromosome'], row['start'], row['end'], row['biotype']) == ('MT', 648, 1601, 'Mt_rRNA')


# translations and xrefs


def test_protein_id_is_the_translation() -> None:
    result = build([tx('ENST1', translation_id='ENSP1'), tx('ENST2')])
    assert result['proteinId'].to_list() == ['ENSP1', None]


def test_xref_arrays_pass_through_in_query_order() -> None:
    row = only(
        build(
            [tx('ENST1', translation_id='ENSP1')],
            [
                xref(
                    'ENST1',
                    'ENSP1',
                    uniprot_swissprot=['Q8NH21'],
                    uniprot_trembl=['A2A2V1', 'X6RKS3'],
                    uniprot_isoform=['Q8NH21-1'],
                    alphafold=['AF-Q8NH21-F1'],
                )
            ],
        )
    )
    assert row['uniprotSwissprotIds'] == ['Q8NH21']
    assert row['uniprotTremblIds'] == ['A2A2V1', 'X6RKS3']
    assert row['uniprotIsoformIds'] == ['Q8NH21-1']
    assert row['alphafoldIds'] == ['AF-Q8NH21-F1']


def test_empty_xref_arrays_are_null() -> None:
    result = build(
        [
            tx('ENST1', translation_id='ENSP1'),
            tx('ENST2', translation_id='ENSP2'),
            tx('ENST3'),
        ],
        [xref('ENST1', 'ENSP1', uniprot_trembl=['A0A001'])],
    )
    assert result.select(
        'transcriptId', 'uniprotSwissprotIds', 'uniprotTremblIds', 'uniprotIsoformIds', 'alphafoldIds'
    ).rows() == [
        ('ENST1', None, ['A0A001'], None, None),
        ('ENST2', None, None, None, None),
        ('ENST3', None, None, None, None),
    ]


# exons


def test_exons_keep_rank_order_on_the_negative_strand() -> None:
    exons = [exon('ENSE9', 451000, 451678, -1), exon('ENSE1', 450740, 450900, -1)]
    row = only(build([tx(start=450740, end=451678, strand=-1, exons=exons)]))
    assert [e['exonId'] for e in row['exons']] == ['ENSE9', 'ENSE1']
    assert row['exons'][0] == {'exonId': 'ENSE9', 'chromosome': '1', 'start': 451000, 'end': 451678, 'strand': -1}


def test_transcript_without_exons_has_null_exons() -> None:
    row = only(build([tx(exons=[])]))
    assert row['exons'] is None


# rows


def test_rows_sorted_by_target_then_transcript() -> None:
    result = build([tx('ENST2', 'ENSG2'), tx('ENST9', 'ENSG1'), tx('ENST1', 'ENSG2')])
    assert result.select('targetId', 'transcriptId').rows() == [
        ('ENSG1', 'ENST9'),
        ('ENSG2', 'ENST1'),
        ('ENSG2', 'ENST2'),
    ]


def test_multi_chunk_input_matches_single_chunk() -> None:
    rows = [
        tx('ENST1', translation_id='ENSP1', attributes=[('appris', 'principal1'), ('is_canonical', '1')]),
        tx('ENST2', strand=-1, attributes=[('MANE_Select', 'NM_1.1')]),
        tx('ENST3', 'ENSG3', translation_id='ENSP3'),
    ]
    xref_rows = [xref('ENST1', 'ENSP1', uniprot_swissprot=['P1']), xref('ENST3', 'ENSP3', alphafold=['AF-P3-F1'])]
    single = build_transcripts(txs(rows), xrefs(xref_rows))
    chunked_tx = pl.concat([txs(rows[:1]), txs(rows[1:])], rechunk=False)
    chunked_xref = pl.concat([xrefs(xref_rows[:1]), xrefs(xref_rows[1:])], rechunk=False)
    assert chunked_tx.n_chunks() == 2
    assert build_transcripts(chunked_tx, chunked_xref).equals(single)


# end to end, on a core dump

def row(table: str, **values: object) -> list[str]:
    """One dump row of `table`, in DDL order; unset columns are null."""
    assert set(values) <= set(RELEASE_116_COLUMNS[table]), set(values) - set(RELEASE_116_COLUMNS[table])
    return [NULL if values.get(c) is None else str(values[c]) for c in RELEASE_116_COLUMNS[table]]


def _gene(gene_id: int, stable_id: str, region: int, start: int, end: int, strand: int, canonical: int) -> list[str]:
    return row(
        'gene',
        gene_id=gene_id,
        biotype='protein_coding',
        analysis_id=1,
        seq_region_id=region,
        seq_region_start=start,
        seq_region_end=end,
        seq_region_strand=strand,
        source='ensembl_havana',
        is_current=1,
        canonical_transcript_id=canonical,
        stable_id=stable_id,
        version=1,
    )


def _transcript(
    transcript_id: int,
    gene_id: int,
    stable_id: str,
    region: int,
    start: int,
    end: int,
    strand: int,
    biotype: str,
    translation: int | None,
) -> list[str]:
    return row(
        'transcript',
        transcript_id=transcript_id,
        gene_id=gene_id,
        analysis_id=1,
        seq_region_id=region,
        seq_region_start=start,
        seq_region_end=end,
        seq_region_strand=strand,
        source='ensembl',
        biotype=biotype,
        is_current=1,
        canonical_translation_id=translation,
        stable_id=stable_id,
        version=1,
    )


def _exon(exon_id: int, stable_id: str, region: int, start: int, end: int, strand: int) -> list[str]:
    return row(
        'exon',
        exon_id=exon_id,
        seq_region_id=region,
        seq_region_start=start,
        seq_region_end=end,
        seq_region_strand=strand,
        phase=-1,
        end_phase=-1,
        is_current=1,
        is_constitutive=0,
        stable_id=stable_id,
        version=1,
    )


TABLES: dict[str, list[list[str]]] = {
    'coord_system': [
        row(
            'coord_system',
            coord_system_id=4,
            species_id=1,
            name='chromosome',
            version='GRCh38',
            rank=1,
            attrib='default_version',
        ),
        row(
            'coord_system',
            coord_system_id=3,
            species_id=1,
            name='scaffold',
            version='GRCh38',
            rank=2,
            attrib='default_version',
        ),
    ],
    'seq_region': [
        row('seq_region', seq_region_id=131, name='1', coord_system_id=4, length=248956422),
        row('seq_region', seq_region_id=132, name='MT', coord_system_id=4, length=16569),
        row('seq_region', seq_region_id=133, name='KI270706.1', coord_system_id=3, length=175055),
    ],
    'analysis': [
        row('analysis', analysis_id=1, logic_name='alphafold'),
        row('analysis', analysis_id=2, logic_name='signalp'),
    ],
    'attrib_type': [
        row('attrib_type', attrib_type_id=427, code='appris', name='APPRIS'),
        row('attrib_type', attrib_type_id=535, code='MANE_Select', name='MANE Select'),
        row('attrib_type', attrib_type_id=536, code='Ensembl_Select', name='Ensembl Select'),
        row('attrib_type', attrib_type_id=550, code='MANE_Plus_Clinical', name='MANE Plus Clinical'),
        row('attrib_type', attrib_type_id=554, code='is_canonical', name='Ensembl Canonical'),
        row('attrib_type', attrib_type_id=576, code='gencode_primary', name='GENCODE primary annotation'),
    ],
    'external_db': [
        row('external_db', external_db_id=2000, db_name='Uniprot/SPTREMBL', status='KNOWN', priority=5, type='MISC'),
        row('external_db', external_db_id=2200, db_name='Uniprot/SWISSPROT', status='KNOWN', priority=5, type='MISC'),
        row('external_db', external_db_id=2201, db_name='Uniprot_isoform', status='KNOWN', priority=5, type='MISC'),
    ],
    'gene': [
        _gene(1, 'ENSG00000186092', 131, 65419, 71585, 1, 11),
        _gene(2, 'ENSG00000284733', 131, 450740, 451678, -1, 12),
        _gene(3, 'ENSG00000211459', 132, 648, 1601, 1, 13),
        _gene(4, 'ENSG00000277196', 133, 1000, 2000, 1, 14),
    ],
    'transcript': [
        _transcript(11, 1, 'ENST00000641515', 131, 65419, 71585, 1, 'protein_coding', 21),
        _transcript(12, 2, 'ENST00000426406', 131, 450740, 451678, -1, 'protein_coding', 22),
        _transcript(13, 3, 'ENST00000389680', 132, 648, 1601, 1, 'Mt_rRNA', None),
        _transcript(14, 4, 'ENST00000612152', 133, 1000, 2000, 1, 'protein_coding', 24),
    ],
    'translation': [
        row(
            'translation',
            translation_id=21,
            transcript_id=11,
            seq_start=1,
            start_exon_id=31,
            seq_end=3,
            end_exon_id=33,
            stable_id='ENSP00000493376',
            version=2,
        ),
        row(
            'translation',
            translation_id=22,
            transcript_id=12,
            seq_start=1,
            start_exon_id=34,
            seq_end=3,
            end_exon_id=35,
            stable_id='ENSP00000409316',
            version=1,
        ),
        row(
            'translation',
            translation_id=24,
            transcript_id=14,
            seq_start=1,
            start_exon_id=37,
            seq_end=3,
            end_exon_id=37,
            stable_id='ENSP00000482090',
            version=1,
        ),
    ],
    'exon': [
        _exon(31, 'ENSE00003812156', 131, 65419, 65433, 1),
        _exon(32, 'ENSE00003813641', 131, 65520, 65573, 1),
        _exon(33, 'ENSE00003813949', 131, 69037, 71585, 1),
        _exon(34, 'ENSE00002319515', 131, 451000, 451678, -1),
        _exon(35, 'ENSE00001000001', 131, 450740, 450900, -1),
        _exon(36, 'ENSE00001544501', 132, 648, 1601, 1),
        _exon(37, 'ENSE00003718300', 133, 1000, 2000, 1),
    ],
    'exon_transcript': [
        row('exon_transcript', exon_id=33, transcript_id=11, rank=3),
        row('exon_transcript', exon_id=31, transcript_id=11, rank=1),
        row('exon_transcript', exon_id=32, transcript_id=11, rank=2),
        row('exon_transcript', exon_id=35, transcript_id=12, rank=2),
        row('exon_transcript', exon_id=34, transcript_id=12, rank=1),
        row('exon_transcript', exon_id=36, transcript_id=13, rank=1),
        row('exon_transcript', exon_id=37, transcript_id=14, rank=1),
    ],
    'transcript_attrib': [
        row('transcript_attrib', transcript_id=11, attrib_type_id=427, value='principal1'),
        row('transcript_attrib', transcript_id=11, attrib_type_id=554, value=1),
        row('transcript_attrib', transcript_id=11, attrib_type_id=535, value='NM_001005484.2'),
        row('transcript_attrib', transcript_id=11, attrib_type_id=576, value='GENCODE primary'),
        row('transcript_attrib', transcript_id=11, attrib_type_id=536, value='ENST00000641515.2'),
        row('transcript_attrib', transcript_id=12, attrib_type_id=550, value='NM_000000.1'),
        row('transcript_attrib', transcript_id=12, attrib_type_id=427, value='alternativeM'),
        row('transcript_attrib', transcript_id=13, attrib_type_id=554, value=1),
        row('transcript_attrib', transcript_id=14, attrib_type_id=554, value=1),
    ],
    'xref': [
        row(
            'xref',
            xref_id=41,
            external_db_id=2200,
            dbprimary_acc='Q8NH21',
            display_label='OR4F5_HUMAN',
            version=1,
            info_type='DIRECT',
        ),
        row(
            'xref',
            xref_id=42,
            external_db_id=2000,
            dbprimary_acc='X6RKS3',
            display_label='X6RKS3',
            version=1,
            info_type='SEQUENCE_MATCH',
        ),
        row(
            'xref',
            xref_id=43,
            external_db_id=2000,
            dbprimary_acc='A2A2V1',
            display_label='A2A2V1',
            version=1,
            info_type='SEQUENCE_MATCH',
        ),
        row(
            'xref',
            xref_id=44,
            external_db_id=2201,
            dbprimary_acc='Q6IEY1-2',
            display_label='Q6IEY1-2',
            version=0,
            info_type='DIRECT',
        ),
    ],
    'object_xref': [
        row('object_xref', object_xref_id=51, ensembl_id=21, ensembl_object_type='Translation', xref_id=41),
        row('object_xref', object_xref_id=52, ensembl_id=21, ensembl_object_type='Translation', xref_id=42),
        row('object_xref', object_xref_id=53, ensembl_id=21, ensembl_object_type='Translation', xref_id=43),
        row('object_xref', object_xref_id=54, ensembl_id=22, ensembl_object_type='Translation', xref_id=44),
    ],
    'protein_feature': [
        row(
            'protein_feature',
            protein_feature_id=61,
            translation_id=21,
            seq_start=1,
            seq_end=312,
            hit_start=1,
            hit_end=312,
            hit_name='AF-Q8NH21-F1',
            analysis_id=1,
        ),
        row(
            'protein_feature',
            protein_feature_id=62,
            translation_id=21,
            seq_start=1,
            seq_end=20,
            hit_start=1,
            hit_end=20,
            hit_name='SignalP-noTM',
            analysis_id=2,
        ),
    ],
}


def test_transcript_end_to_end(tmp_path: Path) -> None:
    source = {'ensembl': write_dump(tmp_path, TABLES, release_116_ddl(TABLES))}

    transcript(source, str(tmp_path / 'out' / 'a'), {}, None)
    transcript(source, str(tmp_path / 'out' / 'b'), {}, None)

    out = pl.read_parquet(tmp_path / 'out' / 'a')
    assert dict(out.schema) == OUTPUT_SCHEMA
    assert out.rows(named=True) == [
        {
            'targetId': 'ENSG00000186092',
            'transcriptId': 'ENST00000641515',
            'biotype': 'protein_coding',
            'proteinId': 'ENSP00000493376',
            'uniprotSwissprotIds': ['Q8NH21'],
            'uniprotTremblIds': ['A2A2V1', 'X6RKS3'],
            'uniprotIsoformIds': None,
            'alphafoldIds': ['AF-Q8NH21-F1'],
            'isEnsemblCanonical': True,
            'chromosome': '1',
            'start': 65419,
            'end': 71585,
            'strand': 1,
            'transcriptionStartSite': 65419,
            'flags': [
                {'label': 'mane', 'value': 'select'},
                {'label': 'gencode_primary', 'value': 'true'},
                {'label': 'ensembl_canonical', 'value': 'true'},
                {'label': 'appris', 'value': 'principal_1'},
            ],
            'exons': [
                {'exonId': 'ENSE00003812156', 'chromosome': '1', 'start': 65419, 'end': 65433, 'strand': 1},
                {'exonId': 'ENSE00003813641', 'chromosome': '1', 'start': 65520, 'end': 65573, 'strand': 1},
                {'exonId': 'ENSE00003813949', 'chromosome': '1', 'start': 69037, 'end': 71585, 'strand': 1},
            ],
        },
        {
            'targetId': 'ENSG00000211459',
            'transcriptId': 'ENST00000389680',
            'biotype': 'Mt_rRNA',
            'proteinId': None,
            'uniprotSwissprotIds': None,
            'uniprotTremblIds': None,
            'uniprotIsoformIds': None,
            'alphafoldIds': None,
            'isEnsemblCanonical': True,
            'chromosome': 'MT',
            'start': 648,
            'end': 1601,
            'strand': 1,
            'transcriptionStartSite': 648,
            'flags': [{'label': 'ensembl_canonical', 'value': 'true'}],
            'exons': [{'exonId': 'ENSE00001544501', 'chromosome': 'MT', 'start': 648, 'end': 1601, 'strand': 1}],
        },
        {
            'targetId': 'ENSG00000284733',
            'transcriptId': 'ENST00000426406',
            'biotype': 'protein_coding',
            'proteinId': 'ENSP00000409316',
            'uniprotSwissprotIds': None,
            'uniprotTremblIds': None,
            'uniprotIsoformIds': ['Q6IEY1-2'],
            'alphafoldIds': None,
            'isEnsemblCanonical': False,
            'chromosome': '1',
            'start': 450740,
            'end': 451678,
            'strand': -1,
            'transcriptionStartSite': 451678,
            'flags': [{'label': 'mane', 'value': 'plus_clinical'}, {'label': 'appris', 'value': 'alternative_M'}],
            'exons': [
                {'exonId': 'ENSE00002319515', 'chromosome': '1', 'start': 451000, 'end': 451678, 'strand': -1},
                {'exonId': 'ENSE00001000001', 'chromosome': '1', 'start': 450740, 'end': 450900, 'strand': -1},
            ],
        },
    ]
    files_a = sorted(p.relative_to(tmp_path / 'out' / 'a') for p in (tmp_path / 'out' / 'a').rglob('*') if p.is_file())
    files_b = sorted(p.relative_to(tmp_path / 'out' / 'b') for p in (tmp_path / 'out' / 'b').rglob('*') if p.is_file())
    assert files_a == files_b
    assert all((tmp_path / 'out' / 'a' / f).read_bytes() == (tmp_path / 'out' / 'b' / f).read_bytes() for f in files_a)
