"""The Gene Ontology builder."""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl

from pts.pyspark.target import _build_gene_ontology
from pts.schemas.target import ID_SOURCE
from pts.target.go import build_go, read_tsv
from tests.target.parity import rows


def _gaf(db: str, object_id: str, go_id: str, reference: str, evidence: str, aspect: str) -> str:
    return '\t'.join([
        db,
        object_id,
        'SYM',
        'enables',
        go_id,
        reference,
        evidence,
        '',
        aspect,
        'name',
        '',
        'protein',
        'taxon:9606',
        '20230101',
        'UniProt',
        '',
        '',
    ])


HUMAN = [
    '!gaf-version: 2.2',
    _gaf('UniProtKB', 'P04637', 'GO:0003677', 'PMID:1', 'IDA', 'F'),
    _gaf('UniProtKB', 'P04637', 'GO:0008150', 'PMID:2', 'TAS', 'P'),
    _gaf('UniProtKB', 'P04637', 'GO:0008150', 'PMID:2', 'TAS', 'P'),
    _gaf('UniProtKB', 'TP53', 'GO:0005634', 'PMID:3', 'IDA', 'C'),
    _gaf('UniProtKB', 'Q00000', 'GO:0005634', 'PMID:3', 'IDA', 'C'),
]
RNA = [
    '!gaf-version: 2.2',
    _gaf('RNAcentral', 'URS0000001346_9606', 'GO:0030533', 'GO_REF:0000115', 'IEA', 'F'),
    _gaf('RNAcentral', 'URS0000009999_9606', 'GO:0030533', 'GO_REF:0000115', 'IEA', 'F'),
]
RNA_LOOKUP = [
    'URS0000001346\tENSEMBL\tENST00000585414\t9606\tlncRNA\tENSG00000226803.10',
    'URS0000001346\tENSEMBL\tENST00000585415\t9606\tlncRNA\tENSG00000226803.10',
    'URS0000001346\tENSEMBL\tENST00000999999\t9606\tlncRNA\tENSG00000141510.1',
    'URS0000009999\tGENCODE\tX\t9606\tlncRNA\tOTTHUMG1',
]
ECO = [
    '!gpa-version: 1.1',
    'UniProtKB\tP04637\tenables\tGO:0003677\tPMID:1\tECO:0000314\t\t\t20230101\tUniProt\t\tgo_evidence=IDA',
    'UniProtKB\tP04637\tenables\tGO:0003677\tPMID:9\tECO:0000314\t\t\t20230101\tUniProt\t\tgo_evidence=IDA',
    'UniProtKB\tP04637\tenables\tGO:0003677\tPMID:9\tECO:0000269\t\t\t20230101\tUniProt\t\tgo_evidence=IDA',
]

ENSEMBL = pl.DataFrame(
    [
        ('ENSG00000141510', 'TP53', [{'id': 'P04637', 'source': 'uniprot_swissprot'}]),
        ('ENSG00000000002', None, []),
    ],
    schema={'id': pl.String, 'approvedSymbol': pl.String, 'proteinIds': pl.List(ID_SOURCE)},
    orient='row',
)


def _write(tmp_path: Path, name: str, lines: list[str]) -> str:
    path = tmp_path / name
    path.write_bytes(gzip.compress(('\n'.join(lines) + '\n').encode()))
    return str(path)


def _paths(tmp_path: Path) -> dict[str, str]:
    return {
        'human': _write(tmp_path, 'goa_human.gaf.gz', HUMAN),
        'rna': _write(tmp_path, 'goa_human_rna.gaf.gz', RNA),
        'rna_lookup': _write(tmp_path, 'ensembl.tsv.gz', RNA_LOOKUP),
        'eco': _write(tmp_path, 'goa_human_eco.gpa.gz', ECO),
    }


def _go(tmp_path: Path) -> pl.DataFrame:
    paths = _paths(tmp_path)
    return build_go(
        read_tsv(paths['human'], comment_prefix='!'),
        read_tsv(paths['rna'], comment_prefix='!'),
        read_tsv(paths['rna_lookup']),
        read_tsv(paths['eco'], comment_prefix='!'),
        ENSEMBL,
    )


def _by_gene(tmp_path: Path) -> dict:
    return {r['ensemblId']: r['go'] for r in _go(tmp_path).to_dicts()}


def test_read_tsv_names_columns_as_spark_does(tmp_path: Path) -> None:
    assert read_tsv(_paths(tmp_path)['rna_lookup']).columns == [f'_c{i}' for i in range(6)]


def test_annotations_by_accession_and_symbol_with_every_aspect(tmp_path: Path) -> None:
    go = _by_gene(tmp_path)['ENSG00000141510']
    assert {g['aspect'] for g in go} == {'F', 'P', 'C'}
    assert {g['geneProduct'] for g in go} == {'P04637', 'TP53', 'URS0000001346_9606'}


def test_duplicates_collapse_and_each_eco_id_is_its_own_entry(tmp_path: Path) -> None:
    go = _by_gene(tmp_path)['ENSG00000141510']
    assert [g for g in go if g['id'] == 'GO:0008150'] == [
        {
            'id': 'GO:0008150',
            'source': 'PMID:2',
            'evidence': 'TAS',
            'aspect': 'P',
            'geneProduct': 'P04637',
            'ecoId': None,
        }
    ]
    assert sorted(g['ecoId'] for g in go if g['id'] == 'GO:0003677') == ['ECO:0000269', 'ECO:0000314']


def test_rna_annotations_map_through_rnacentral(tmp_path: Path) -> None:
    assert _by_gene(tmp_path)['ENSG00000226803'] == [
        {
            'id': 'GO:0030533',
            'source': 'GO_REF:0000115',
            'evidence': 'IEA',
            'aspect': 'F',
            'geneProduct': 'URS0000001346_9606',
            'ecoId': None,
        }
    ]


def test_unmapped_annotations_are_dropped(tmp_path: Path) -> None:
    assert set(_by_gene(tmp_path)) == {'ENSG00000141510', 'ENSG00000226803'}


def test_matches_spark(spark, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    csv = spark.read.option('sep', '\t')
    expected = _build_gene_ontology(
        csv.option('comment', '!').csv(paths['human']),
        csv.option('comment', '!').csv(paths['rna']),
        spark.read.option('sep', '\t').csv(paths['rna_lookup']),
        csv.option('comment', '!').csv(paths['eco']),
        spark.createDataFrame(
            ENSEMBL.to_dicts(), 'id string, approvedSymbol string, proteinIds array<struct<id:string, source:string>>'
        ),
    )
    assert rows(_go(tmp_path), 'ensemblId', ('go',)) == rows(expected, 'ensemblId', ('go',))
