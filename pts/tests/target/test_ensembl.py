"""The ensembl builder, from the `pts.ensembl` query frames."""

from __future__ import annotations

import polars as pl
from pyspark.sql import Row

from pts.pyspark.target import _build_ensembl, _build_gene_code
from pts.target.ensembl import ensembl_genes
from tests.target.fixtures import GENES, TRANSCRIPTS, XREFS
from tests.target.parity import rows

# The same genes as the species JSON and the GENCODE GFF3 spelled them, for the spark builder.
JSON_SCHEMA = (
    'id string, approvedSymbol string, biotype string, description string, chromosome string, '
    'strand string, start string, end string, SignalP array<string>, uniprot_trembl array<string>, '
    'uniprot_swissprot array<string>, transcripts array<struct<id:string, translations:array<struct<id:string>>>>'
)
JSON_ROWS = [
    (
        'ENSG00000141510',
        'TP53',
        'protein_coding',
        'tumor protein p53 [Source:HGNC Symbol;Acc:HGNC:11998]',
        '17',
        '-1',
        '7661779',
        '7687546',
        ['SignalP-noTM'],
        ['K7PPA8', 'H2EHT1'],
        ['P04637'],
        [('ENST00000269305', [('ENSP00000269305',)]), ('ENST00000413465', [('ENSP00000410739',)])],
    ),
    (
        'ENSG00000000002',
        None,
        'TEC',
        'TEC protein detection [Source:x]',
        'X',
        '1',
        '1000',
        '2000',
        None,
        None,
        None,
        [('ENST00000000021', [])],
    ),
    ('ENSG00000000003', 'MT-TF', 'Mt_tRNA', None, 'MT', '1', '577', '647', None, None, None, [('ENST00000000031', [])]),
    (
        'LRG_1',
        'LRG',
        'protein_coding',
        'lrg',
        '1',
        '1',
        '10',
        '20',
        None,
        None,
        ['P00001'],
        [('LRG_1t1', [('LRG_1p1',)])],
    ),
]
GFF3_ROWS = [
    Row(
        _c0='chr17',
        _c1='HAVANA',
        _c2='transcript',
        _c3='7661779',
        _c4='7687538',
        _c5='.',
        _c6='-',
        _c7='.',
        _c8='gene_id=ENSG00000141510.19;transcript_id=ENST00000269305.9;tag=Ensembl_canonical',
    ),
    Row(
        _c0='chrX',
        _c1='HAVANA',
        _c2='transcript',
        _c3='1000',
        _c4='2000',
        _c5='.',
        _c6='+',
        _c7='.',
        _c8='gene_id=ENSG00000000002.1;transcript_id=ENST00000000021.1;tag=Ensembl_canonical',
    ),
    Row(
        _c0='chrM',
        _c1='ENSEMBL',
        _c2='transcript',
        _c3='577',
        _c4='647',
        _c5='.',
        _c6='+',
        _c7='.',
        _c8='gene_id=ENSG00000000003.1;transcript_id=ENST00000000031.1;tag=Ensembl_canonical',
    ),
]
GFF3_SCHEMA = ', '.join(f'_c{i} string' for i in range(9))


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


def test_matches_spark(spark) -> None:
    gene_code = _build_gene_code(spark.createDataFrame(GFF3_ROWS, GFF3_SCHEMA))
    expected = _build_ensembl(spark.createDataFrame(JSON_ROWS, JSON_SCHEMA), gene_code)
    unordered = ('proteinIds', 'signalP')
    assert rows(_built(), 'id', unordered, empty_as_null=('signalP',)) == rows(
        expected, 'id', unordered, empty_as_null=('signalP',)
    )
