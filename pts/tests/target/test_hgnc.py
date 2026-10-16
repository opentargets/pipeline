"""The HGNC builder."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from pts.schemas.target import HGNC_DOC
from pts.target.hgnc import build_hgnc, read_hgnc
from tests.target.rows import rows

DOCS = [
    {
        'ensembl_gene_id': 'ENSG00000141510',
        'hgnc_id': 'HGNC:11998',
        'symbol': 'TP53',
        'name': 'tumor protein p53',
        'uniprot_ids': ['P04637'],
        'alias_symbol': ['p53', 'LFS1'],
        'alias_name': None,
        'prev_symbol': None,
        'prev_name': ['Li-Fraumeni syndrome'],
        'locus_group': 'protein-coding gene',
    },
    {'ensembl_gene_id': None, 'hgnc_id': 'HGNC:1', 'symbol': 'NOENS', 'name': 'no ensembl id'},
    {
        'ensembl_gene_id': 'ENSG00000250413',
        'hgnc_id': 'HGNC:40636',
        'symbol': 'SLC2A9-AS1',
        'name': 'SLC2A9 antisense RNA 1',
        'alias_symbol': ['A1', 'A1'],
    },
    {
        'ensembl_gene_id': 'ENSG00000250413',
        'hgnc_id': 'HGNC:59109',
        'symbol': 'SLC2A9-AS3',
        'name': 'SLC2A9 antisense RNA 3',
    },
]


def _docs() -> pl.DataFrame:
    return pl.DataFrame([{k: d.get(k) for k in HGNC_DOC} for d in DOCS], schema=HGNC_DOC)


def _by_id() -> dict:
    return {r['ensemblId']: r for r in build_hgnc(_docs()).to_dicts()}


def test_read_hgnc_keeps_file_order_and_the_used_keys(tmp_path: Path) -> None:
    path = tmp_path / 'hgnc_complete_set.json'
    path.write_text(json.dumps({'responseHeader': {}, 'response': {'numFound': 4, 'docs': DOCS}}))
    docs = read_hgnc(str(path))
    assert docs.columns == list(HGNC_DOC)
    assert docs['symbol'].to_list() == ['TP53', 'NOENS', 'SLC2A9-AS1', 'SLC2A9-AS3']


def test_symbol_and_name() -> None:
    gene = _by_id()['ENSG00000141510']
    assert gene['approvedSymbol'] == 'TP53'
    assert gene['approvedName'] == 'tumor protein p53'
    assert gene['uniprotIds'] == ['P04637']


def test_hgnc_id_is_split_into_id_and_source() -> None:
    assert _by_id()['ENSG00000141510']['hgncId'] == [{'id': '11998', 'source': 'HGNC'}]


def test_synonym_lists_are_tagged_and_null_becomes_empty() -> None:
    gene = _by_id()['ENSG00000141510']
    assert gene['hgncSymbolSynonyms'] == [{'label': 'p53', 'source': 'HGNC'}, {'label': 'LFS1', 'source': 'HGNC'}]
    assert gene['hgncNameSynonyms'] == []
    assert gene['hgncObsoleteSymbols'] == []
    assert gene['hgncObsoleteNames'] == [{'label': 'Li-Fraumeni syndrome', 'source': 'HGNC'}]


def test_the_first_doc_naming_a_gene_wins() -> None:
    assert _by_id()['ENSG00000250413']['approvedSymbol'] == 'SLC2A9-AS1'


def test_one_row_per_ensembl_id_including_null() -> None:
    assert build_hgnc(_docs())['ensemblId'].to_list() == ['ENSG00000141510', None, 'ENSG00000250413']


def test_matches_the_spark_output() -> None:
    assert rows(build_hgnc(_docs()), 'ensemblId') == SPARK_ROWS


#: The spark output for this fixture, as `rows` normalises it.
SPARK_ROWS = [
    {
        'ensemblId': 'ENSG00000141510',
        'approvedSymbol': 'TP53',
        'approvedName': 'tumor protein p53',
        'uniprotIds': ['P04637'],
        'hgncId': [{'id': '11998', 'source': 'HGNC'}],
        'hgncSymbolSynonyms': [{'label': 'p53', 'source': 'HGNC'}, {'label': 'LFS1', 'source': 'HGNC'}],
        'hgncNameSynonyms': [],
        'hgncObsoleteSymbols': [],
        'hgncObsoleteNames': [{'label': 'Li-Fraumeni syndrome', 'source': 'HGNC'}],
    },
    {
        'ensemblId': 'ENSG00000250413',
        'approvedSymbol': 'SLC2A9-AS1',
        'approvedName': 'SLC2A9 antisense RNA 1',
        'uniprotIds': None,
        'hgncId': [{'id': '40636', 'source': 'HGNC'}],
        'hgncSymbolSynonyms': [{'label': 'A1', 'source': 'HGNC'}, {'label': 'A1', 'source': 'HGNC'}],
        'hgncNameSynonyms': [],
        'hgncObsoleteSymbols': [],
        'hgncObsoleteNames': [],
    },
    {
        'ensemblId': None,
        'approvedSymbol': 'NOENS',
        'approvedName': 'no ensembl id',
        'uniprotIds': None,
        'hgncId': [{'id': '1', 'source': 'HGNC'}],
        'hgncSymbolSynonyms': [],
        'hgncNameSynonyms': [],
        'hgncObsoleteSymbols': [],
        'hgncObsoleteNames': [],
    },
]
