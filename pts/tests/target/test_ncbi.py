"""The NCBI Entrez synonyms builder."""

from __future__ import annotations

import gzip
from pathlib import Path

from pts.transformers.target.ncbi import build_ncbi, read_ncbi
from tests.target.rows import rows

HEADER = '#tax_id\tGeneID\tSymbol\tLocusTag\tSynonyms\tdbXrefs\tchromosome\tOther_designations'
LINES = [
    '9606\t1\tA1BG\t-\tA1B|ABG\tMIM:138670|HGNC:HGNC:5|Ensembl:ENSG00000121410\t19\talpha-1B-glycoprotein|HEL-S-163pA',
    '9606\t2\tA1BG-X\t-\t-\tEnsembl:ENSG00000121410\t19\t-',
    '9606\t3\tNOENS\t-\tN1\tMIM:1\t1\tnone',
    '9606\t4\tTWO\t-\tT1\tEnsembl:ENSG00000000001|Ensembl:ENSG00000000002\t1\ttwo genes',
    '9606\t5\tDASH\t-\t-\t-\t1\t-',
]

SYNONYMS = ('synonyms', 'symbolSynonyms', 'nameSynonyms')


def _write(tmp_path: Path) -> str:
    path = tmp_path / 'Homo_sapiens.gene_info.gz'
    path.write_bytes(gzip.compress(('\n'.join([HEADER, *LINES]) + '\n').encode()))
    return str(path)


def _by_id(tmp_path: Path) -> dict:
    return {r['id']: r for r in build_ncbi(read_ncbi(_write(tmp_path))).to_dicts()}


def _labels(entries: list[dict]) -> list[str]:
    assert {e['source'] for e in entries} <= {'NCBI_entrez'}
    return [e['label'] for e in entries]


def test_only_genes_with_an_ensembl_xref(tmp_path: Path) -> None:
    assert set(_by_id(tmp_path)) == {'ENSG00000121410', 'ENSG00000000001', 'ENSG00000000002'}


def test_rows_for_one_gene_are_concatenated_and_dashes_dropped(tmp_path: Path) -> None:
    gene = _by_id(tmp_path)['ENSG00000121410']
    assert _labels(gene['symbolSynonyms']) == ['A1B', 'ABG', 'A1BG', 'A1BG-X']
    assert _labels(gene['nameSynonyms']) == ['alpha-1B-glycoprotein', 'HEL-S-163pA']
    assert _labels(gene['synonyms']) == ['A1B', 'ABG', 'alpha-1B-glycoprotein', 'HEL-S-163pA', 'A1BG', 'A1BG-X']


def test_every_ensembl_xref_gets_the_synonyms(tmp_path: Path) -> None:
    genes = _by_id(tmp_path)
    assert _labels(genes['ENSG00000000001']['synonyms']) == _labels(genes['ENSG00000000002']['synonyms'])


def test_matches_the_spark_output(tmp_path: Path) -> None:
    assert rows(build_ncbi(read_ncbi(_write(tmp_path))), 'id', SYNONYMS) == SPARK_ROWS


#: The spark output for this fixture, as `rows` normalises it.
SPARK_ROWS = [
    {
        'id': 'ENSG00000000001',
        'synonyms': [
            {'label': 'T1', 'source': 'NCBI_entrez'},
            {'label': 'TWO', 'source': 'NCBI_entrez'},
            {'label': 'two genes', 'source': 'NCBI_entrez'},
        ],
        'symbolSynonyms': [{'label': 'T1', 'source': 'NCBI_entrez'}, {'label': 'TWO', 'source': 'NCBI_entrez'}],
        'nameSynonyms': [{'label': 'two genes', 'source': 'NCBI_entrez'}],
    },
    {
        'id': 'ENSG00000000002',
        'synonyms': [
            {'label': 'T1', 'source': 'NCBI_entrez'},
            {'label': 'TWO', 'source': 'NCBI_entrez'},
            {'label': 'two genes', 'source': 'NCBI_entrez'},
        ],
        'symbolSynonyms': [{'label': 'T1', 'source': 'NCBI_entrez'}, {'label': 'TWO', 'source': 'NCBI_entrez'}],
        'nameSynonyms': [{'label': 'two genes', 'source': 'NCBI_entrez'}],
    },
    {
        'id': 'ENSG00000121410',
        'synonyms': [
            {'label': 'A1B', 'source': 'NCBI_entrez'},
            {'label': 'A1BG', 'source': 'NCBI_entrez'},
            {'label': 'A1BG-X', 'source': 'NCBI_entrez'},
            {'label': 'ABG', 'source': 'NCBI_entrez'},
            {'label': 'HEL-S-163pA', 'source': 'NCBI_entrez'},
            {'label': 'alpha-1B-glycoprotein', 'source': 'NCBI_entrez'},
        ],
        'symbolSynonyms': [
            {'label': 'A1B', 'source': 'NCBI_entrez'},
            {'label': 'A1BG', 'source': 'NCBI_entrez'},
            {'label': 'A1BG-X', 'source': 'NCBI_entrez'},
            {'label': 'ABG', 'source': 'NCBI_entrez'},
        ],
        'nameSynonyms': [
            {'label': 'HEL-S-163pA', 'source': 'NCBI_entrez'},
            {'label': 'alpha-1B-glycoprotein', 'source': 'NCBI_entrez'},
        ],
    },
]
