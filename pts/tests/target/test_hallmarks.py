"""The COSMIC hallmarks builder."""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl

from pts.target.hallmarks import COLUMNS, build_hallmarks, read_hallmarks
from tests.target.rows import rows

HEADER = 'GENE_SYMBOL\tCELL_TYPE\tPUBMED_PMID\tHALLMARK\tIMPACT\tDESCRIPTION\tCELL_LINE'
LINES = [
    'TP53\tlung\t12345\tangiogenesis\tpromotes\tpromotes tumour angiogenesis\tA549',
    'TP53\tlung\t67890\tapoptosis\tsuppresses\t"induces apoptosis, in vitro"\t"HeLa, A549',
    'TP53\tlung\t67890\tapoptosis\tsuppresses\t"induces apoptosis, in vitro"\tHeLa',
    'EGFR\t\t111\trole in cancer\t\t"restrains an active ""in"" conformation"\t',
    'EGFR\t\tNA\tangiogenesis\tpromotes\tno pmid\t',
]


def _write(tmp_path: Path) -> str:
    path = tmp_path / 'cosmic-hallmarks.tsv.gz'
    path.write_bytes(gzip.compress(('\r\n'.join([HEADER, *LINES]) + '\r\n').encode()))
    return str(path)


def _by_symbol(tmp_path: Path) -> dict:
    return {r['approvedSymbol']: r['hallmarks'] for r in build_hallmarks(read_hallmarks(_write(tmp_path))).to_dicts()}


def test_read_keeps_the_used_columns(tmp_path: Path) -> None:
    assert read_hallmarks(_write(tmp_path)).columns == COLUMNS


def test_quotes_are_stripped_as_spark_strips_them(tmp_path: Path) -> None:
    descriptions = read_hallmarks(_write(tmp_path))['DESCRIPTION'].to_list()
    assert descriptions[1] == 'induces apoptosis, in vitro'
    assert descriptions[3] == '"restrains an active ""in"" conformation"'


def test_cancer_hallmarks_and_attributes_are_split(tmp_path: Path) -> None:
    tp53 = _by_symbol(tmp_path)['TP53']
    assert tp53['cancerHallmarks'] == [
        {'pmid': 12345, 'description': 'promotes tumour angiogenesis', 'impact': 'promotes', 'label': 'angiogenesis'}
    ]
    assert tp53['attributes'] == [{'pmid': 67890, 'description': 'induces apoptosis, in vitro', 'name': 'apoptosis'}]


def test_a_pmid_that_is_not_a_number_is_null(tmp_path: Path) -> None:
    egfr = _by_symbol(tmp_path)['EGFR']
    assert egfr['cancerHallmarks'] == [
        {'pmid': None, 'description': 'no pmid', 'impact': 'promotes', 'label': 'angiogenesis'}
    ]


def test_a_gene_with_no_cancer_hallmark_has_null_cancer_hallmarks() -> None:
    raw = pl.DataFrame(
        {
            'GENE_SYMBOL': ['A'],
            'PUBMED_PMID': ['1'],
            'HALLMARK': ['role in cancer'],
            'IMPACT': [None],
            'DESCRIPTION': ['x'],
        },
        schema=dict.fromkeys(COLUMNS, pl.String),
    )
    assert build_hallmarks(raw).to_dicts() == [
        {
            'approvedSymbol': 'A',
            'hallmarks': {
                'attributes': [{'pmid': 1, 'description': 'x', 'name': 'role in cancer'}],
                'cancerHallmarks': None,
            },
        }
    ]


def test_matches_the_spark_output(tmp_path: Path) -> None:
    unordered = ('hallmarks.attributes', 'hallmarks.cancerHallmarks')
    assert rows(build_hallmarks(read_hallmarks(_write(tmp_path))), 'approvedSymbol', unordered) == SPARK_ROWS


#: The spark output for this fixture, as `rows` normalises it.
SPARK_ROWS = [
    {
        'approvedSymbol': 'EGFR',
        'hallmarks': {
            'attributes': [
                {'pmid': 111, 'description': '"restrains an active ""in"" conformation"', 'name': 'role in cancer'}
            ],
            'cancerHallmarks': [
                {'pmid': None, 'description': 'no pmid', 'impact': 'promotes', 'label': 'angiogenesis'}
            ],
        },
    },
    {
        'approvedSymbol': 'TP53',
        'hallmarks': {
            'attributes': [{'pmid': 67890, 'description': 'induces apoptosis, in vitro', 'name': 'apoptosis'}],
            'cancerHallmarks': [
                {
                    'pmid': 12345,
                    'description': 'promotes tumour angiogenesis',
                    'impact': 'promotes',
                    'label': 'angiogenesis',
                }
            ],
        },
    },
]
