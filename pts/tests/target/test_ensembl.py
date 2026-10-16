"""The ensembl builder, from the `pts.ensembl` query frames."""

from __future__ import annotations

import polars as pl

from pts.target.ensembl import ensembl_genes
from tests.target.fixtures import GENES, TRANSCRIPTS, XREFS
from tests.target.rows import rows


def _built() -> pl.DataFrame:
    return ensembl_genes(GENES, TRANSCRIPTS, XREFS)


def _by_id() -> dict[str, dict]:
    return {r['id']: r for r in _built().to_dicts()}


def test_non_ensg_ids_are_excluded() -> None:
    assert set(_by_id()) == {'ENSG00000141510', 'ENSG00000000002', 'ENSG00000000003'}


def test_genes_on_canonical_chromosomes_are_kept() -> None:
    genes = GENES.with_columns(pl.col('chromosome').replace({'X': 'CHR_HSCHR6_MHC_APD_CTG1'}))
    ids = set(ensembl_genes(genes, TRANSCRIPTS, XREFS)['id'])
    assert ids == {'ENSG00000141510', 'ENSG00000000003'}


def test_tec_is_stripped_from_biotype_and_name() -> None:
    gene = _by_id()['ENSG00000000002']
    assert gene['biotype'] == ''
    assert gene['approvedName'] == ' protein detion '


def test_approved_name_is_the_description_before_the_source() -> None:
    assert _by_id()['ENSG00000141510']['approvedName'] == 'tumor protein p53 '
    assert _by_id()['ENSG00000000003']['approvedName'] is None


def test_protein_ids_union_every_translation() -> None:
    ids = {(p['id'], p['source']) for p in _by_id()['ENSG00000141510']['proteinIds']}
    assert ids == {
        ('P04637', 'uniprot_swissprot'),
        ('H2EHT1', 'uniprot_trembl'),
        ('K7PPA8', 'uniprot_trembl'),
        ('ENSP00000269305', 'ensembl_PRO'),
        ('ENSP00000410739', 'ensembl_PRO'),
    }


def test_a_gene_without_translations_has_no_protein_ids() -> None:
    assert _by_id()['ENSG00000000002']['proteinIds'] == []


def test_signalp_is_tagged() -> None:
    assert _by_id()['ENSG00000141510']['signalP'] == [{'id': 'SignalP-noTM', 'source': 'signalP'}]


def test_canonical_transcript_carries_its_own_coordinates() -> None:
    assert _by_id()['ENSG00000141510']['canonicalTranscript'] == {
        'id': 'ENST00000269305',
        'chromosome': '17',
        'start': 7661779,
        'end': 7687538,
        'strand': -1,
    }


def test_canonical_transcript_on_another_chromosome_is_dropped() -> None:
    transcripts = TRANSCRIPTS.with_columns(
        pl
        .when(pl.col('transcriptId') == 'ENST00000000021')
        .then(pl.lit('Y'))
        .otherwise('chromosome')
        .alias('chromosome')
    )
    gene = {r['id']: r for r in ensembl_genes(GENES, transcripts, XREFS).to_dicts()}['ENSG00000000002']
    assert gene['canonicalTranscript'] is None


def test_genomic_location_dtype() -> None:
    assert _built().schema['genomicLocation'] == pl.Struct({
        'chromosome': pl.String,
        'start': pl.Int64,
        'end': pl.Int64,
        'strand': pl.Int32,
    })


def test_matches_the_spark_output() -> None:
    assert rows(_built(), 'id', ('proteinIds', 'signalP'), empty_as_null=('signalP',)) == SPARK_ROWS


#: The spark output for this fixture, as `rows` normalises it.
SPARK_ROWS = [
    {
        'id': 'ENSG00000000002',
        'biotype': '',
        'approvedName': ' protein detion ',
        'genomicLocation': {'chromosome': 'X', 'start': 1000, 'end': 2000, 'strand': 1},
        'approvedSymbol': None,
        'proteinIds': [],
        'signalP': None,
        'canonicalTranscript': {'id': 'ENST00000000021', 'chromosome': 'X', 'start': 1000, 'end': 2000, 'strand': 1},
    },
    {
        'id': 'ENSG00000000003',
        'biotype': 'Mt_tRNA',
        'approvedName': None,
        'genomicLocation': {'chromosome': 'MT', 'start': 577, 'end': 647, 'strand': 1},
        'approvedSymbol': 'MT-TF',
        'proteinIds': [],
        'signalP': None,
        'canonicalTranscript': {'id': 'ENST00000000031', 'chromosome': 'MT', 'start': 577, 'end': 647, 'strand': 1},
    },
    {
        'id': 'ENSG00000141510',
        'biotype': 'protein_coding',
        'approvedName': 'tumor protein p53 ',
        'genomicLocation': {'chromosome': '17', 'start': 7661779, 'end': 7687546, 'strand': -1},
        'approvedSymbol': 'TP53',
        'proteinIds': [
            {'id': 'ENSP00000269305', 'source': 'ensembl_PRO'},
            {'id': 'ENSP00000410739', 'source': 'ensembl_PRO'},
            {'id': 'H2EHT1', 'source': 'uniprot_trembl'},
            {'id': 'K7PPA8', 'source': 'uniprot_trembl'},
            {'id': 'P04637', 'source': 'uniprot_swissprot'},
        ],
        'signalP': [{'id': 'SignalP-noTM', 'source': 'signalP'}],
        'canonicalTranscript': {
            'id': 'ENST00000269305',
            'chromosome': '17',
            'start': 7661779,
            'end': 7687538,
            'strand': -1,
        },
    },
]
