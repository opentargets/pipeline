"""Tests for the transcript transformer."""

from typing import Any

import polars as pl
import pytest

from pts.schemas.transcript import transcript as transcript_schema
from pts.transformers.transcript import build_flags, build_transcripts

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

