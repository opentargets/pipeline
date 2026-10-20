"""Assembling the target dataset from the per-source frames."""

from __future__ import annotations

import polars as pl
import pytest

from pts.schemas.target import (
    CANONICAL_TRANSCRIPT,
    CONSTRAINT,
    GENOMIC_LOCATION,
    GO,
    HALLMARKS,
    ID_SOURCE,
    LABEL_SOURCE,
    LOCATION,
    PATHWAY,
    TARGET,
    TARGET_CLASS,
)
from pts.transformers.target.assemble import (
    add_ncbi_synonyms,
    add_protein_classification,
    add_tss,
    assemble,
    filter_and_sort_protein_ids,
    merge_hgnc_ensembl,
    remove_duplicated_synonyms,
    remove_redundant_xrefs,
    uniprot_by_gene,
)
from pts.transformers.target.ensembl import ensembl_genes
from tests.target.fixtures import GENES, TRANSCRIPTS, XREFS
from tests.target.rows import rows


def _ids(*pairs: tuple[str, str]) -> list[dict]:
    return [{'id': i, 'source': s} for i, s in pairs]


def _labels(source: str, *labels: str) -> list[dict]:
    return [{'label': label, 'source': source} for label in labels]


def _loc(location: str, source: str, term: str | None = None) -> dict:
    return {
        'location': location,
        'source': source,
        'termSL': term,
        'labelSL': term and 'Cellular component',
        'targetModifier': None,
    }


TP53 = 'ENSG00000141510'
NOHGNC = 'ENSG00000000002'
TWOENTRIES = 'ENSG00000000003'

ENSEMBL = pl.DataFrame(
    [
        (
            TP53,
            'protein_coding',
            'tumor protein p53 ',
            {'chromosome': '17', 'start': 7661779, 'end': 7687546, 'strand': -1},
            'TP53',
            _ids(('P04637', 'uniprot_swissprot'), ('K7PPA8', 'uniprot_trembl'), ('ENSP00000269305', 'ensembl_PRO')),
            _ids(('SignalP-noTM', 'signalP')),
            {'id': 'ENST00000269305', 'chromosome': '17', 'start': 7661779, 'end': 7687538, 'strand': -1},
        ),
        (
            NOHGNC,
            '',
            None,
            {'chromosome': 'X', 'start': 1000, 'end': 2000, 'strand': 1},
            None,
            [],
            [],
            {'id': 'ENST00000000021', 'chromosome': 'X', 'start': 1000, 'end': 2000, 'strand': 1},
        ),
        (
            TWOENTRIES,
            'protein_coding',
            'two ',
            {'chromosome': '1', 'start': 10, 'end': 20, 'strand': 1},
            'TWO',
            _ids(('O00001', 'uniprot_swissprot')),
            [],
            None,
        ),
    ],
    schema={
        'id': pl.String,
        'biotype': pl.String,
        'approvedName': pl.String,
        'genomicLocation': GENOMIC_LOCATION,
        'approvedSymbol': pl.String,
        'proteinIds': pl.List(ID_SOURCE),
        'signalP': pl.List(ID_SOURCE),
        'canonicalTranscript': CANONICAL_TRANSCRIPT,
    },
    orient='row',
)
HGNC = pl.DataFrame(
    [
        (
            TP53,
            [{'id': '11998', 'source': 'HGNC'}],
            'TP53',
            'tumor protein p53',
            ['P04637', 'Q00001'],
            _labels('HGNC', 'p53', 'LFS1'),
            [],
            [],
            _labels('HGNC', 'Li-Fraumeni syndrome', 'Li-Fraumeni syndrome'),
        ),
        (
            TWOENTRIES,
            [{'id': '2', 'source': 'HGNC'}],
            'TWO',
            'two entries',
            ['O00001', 'O00002'],
            [],
            _labels('HGNC', 'second'),
            _labels('HGNC', 'OLD'),
            [],
        ),
        (None, [{'id': '3', 'source': 'HGNC'}], 'NOENS', 'no ensembl', ['P04637'], [], [], [], []),
    ],
    schema={
        'ensemblId': pl.String,
        'hgncId': pl.List(ID_SOURCE),
        'approvedSymbol': pl.String,
        'approvedName': pl.String,
        'uniprotIds': pl.List(pl.String),
        'hgncSymbolSynonyms': pl.List(LABEL_SOURCE),
        'hgncNameSynonyms': pl.List(LABEL_SOURCE),
        'hgncObsoleteSymbols': pl.List(LABEL_SOURCE),
        'hgncObsoleteNames': pl.List(LABEL_SOURCE),
    },
    orient='row',
)
UNIPROT = pl.DataFrame(
    [
        (
            'P04637',
            ['Acts as a tumor suppressor.'],
            _ids(('1A1U', 'PDB'), ('GO:0000785', 'GO'), ('ENST00000269305', 'Ensembl')),
            [_loc('Nucleus', 'uniprot', 'SL-0191')],
            _ids(('P04637', 'uniprot_obsolete'), ('Q15086', 'uniprot_obsolete')),
            _labels('uniprot', 'Cellular tumor antigen p53', 'p53'),
            _labels('uniprot', 'p53'),
            _labels('uniprot', 'Cellular tumor antigen p53'),
        ),
        (
            'O00001',
            ['Shared function.', 'First function.'],
            _ids(('1AAA', 'PDB')),
            None,
            _ids(('O00001', 'uniprot_obsolete')),
            _labels('uniprot', 'one'),
            [],
            _labels('uniprot', 'one'),
        ),
        (
            'O00002',
            ['Shared function.'],
            _ids(('1AAA', 'PDB')),
            [_loc('Cytoplasm', 'uniprot')],
            _ids(('O00002', 'uniprot_obsolete')),
            [],
            [],
            [],
        ),
    ],
    schema={
        'uniprotId': pl.String,
        'functionDescriptions': pl.List(pl.String),
        'dbXrefs': pl.List(ID_SOURCE),
        'subcellularLocations': pl.List(LOCATION),
        'proteinIds': pl.List(ID_SOURCE),
        'synonyms': pl.List(LABEL_SOURCE),
        'symbolSynonyms': pl.List(LABEL_SOURCE),
        'nameSynonyms': pl.List(LABEL_SOURCE),
    },
    orient='row',
)
PROTEIN_CLASS = pl.DataFrame(
    [
        ('Q15086', [{'id': 6, 'label': 'Enzyme', 'level': 'l1'}]),
        ('O00002', [{'id': 7, 'label': 'Kinase', 'level': 'l1'}]),
    ],
    schema={'accession': pl.String, 'targetClass': pl.List(TARGET_CLASS)},
    orient='row',
)
GO_FRAME = pl.DataFrame(
    [
        (
            TP53,
            [
                {
                    'id': 'GO:0003677',
                    'source': 'PMID:1',
                    'evidence': 'IDA',
                    'aspect': 'F',
                    'geneProduct': 'P04637',
                    'ecoId': None,
                }
            ],
        )
    ],
    schema={'ensemblId': pl.String, 'go': pl.List(GO)},
    orient='row',
)
HALLMARKS_FRAME = pl.DataFrame(
    [('TP53', {'attributes': [{'pmid': 1, 'description': 'x', 'name': 'role in cancer'}], 'cancerHallmarks': None})],
    schema={'approvedSymbol': pl.String, 'hallmarks': HALLMARKS},
    orient='row',
)
HPA = pl.DataFrame(
    [
        (TP53, [_loc('Nucleus', 'HPA_main', 'SL-0191'), _loc('Nucleus', 'uniprot', 'SL-0191')]),
        (NOHGNC, [_loc('Cytosol', 'HPA_main')]),
    ],
    schema={'id': pl.String, 'locations': pl.List(LOCATION)},
    orient='row',
)
CONSTRAINT_FRAME = pl.DataFrame(
    [
        (
            TP53,
            [
                {
                    'constraintType': 'syn',
                    'score': 1.5,
                    'exp': 2.0,
                    'obs': 3,
                    'oe': 0.5,
                    'oeLower': 0.25,
                    'oeUpper': 0.75,
                    'upperRank': None,
                    'upperBin': None,
                    'upperBin6': None,
                }
            ],
        )
    ],
    schema={'id': pl.String, 'constraint': pl.List(CONSTRAINT)},
    orient='row',
)
NCBI = pl.DataFrame(
    [
        (
            TP53,
            _labels('NCBI_entrez', 'p53', 'BCC7', 'cellular tumor antigen p53'),
            _labels('NCBI_entrez', 'p53', 'BCC7'),
            _labels('NCBI_entrez', 'cellular tumor antigen p53'),
        )
    ],
    schema={
        'id': pl.String,
        'synonyms': pl.List(LABEL_SOURCE),
        'symbolSynonyms': pl.List(LABEL_SOURCE),
        'nameSynonyms': pl.List(LABEL_SOURCE),
    },
    orient='row',
)
REACTOME = pl.DataFrame(
    [(TP53, [{'pathwayId': 'R-HSA-69278', 'pathway': 'Cell Cycle, Mitotic', 'topLevelTerm': 'Cell Cycle'}])],
    schema={'id': pl.String, 'pathways': pl.List(PATHWAY)},
    orient='row',
)

#: Lists whose order spark leaves to collect_set.
UNORDERED = (
    'go',
    'hallmarks.attributes',
    'hallmarks.cancerHallmarks',
    'synonyms',
    'symbolSynonyms',
    'nameSynonyms',
    'functionDescriptions',
    'subcellularLocations',
    'targetClass',
    'dbXrefs',
    'pathways',
)


def _assembled() -> dict:
    result = assemble(
        ENSEMBL, HGNC, GO_FRAME, HALLMARKS_FRAME, UNIPROT, PROTEIN_CLASS, HPA, CONSTRAINT_FRAME, NCBI, REACTOME
    )
    return {r['id']: r for r in result.to_dicts()}


def test_schema_is_the_published_one() -> None:
    result = assemble(
        ENSEMBL, HGNC, GO_FRAME, HALLMARKS_FRAME, UNIPROT, PROTEIN_CLASS, HPA, CONSTRAINT_FRAME, NCBI, REACTOME
    )
    assert result.schema == TARGET
    assert result['id'].to_list() == sorted([TP53, NOHGNC, TWOENTRIES])


def test_hgnc_symbol_and_name_win_then_ensembl_then_the_id() -> None:
    merged = {r['id']: r for r in merge_hgnc_ensembl(HGNC, ENSEMBL).to_dicts()}
    assert (merged[TP53]['approvedSymbol'], merged[TP53]['approvedName']) == ('TP53', 'tumor protein p53')
    assert (merged[NOHGNC]['approvedSymbol'], merged[NOHGNC]['approvedName']) == (NOHGNC, '')


def test_protein_ids_keep_the_highest_priority_source_per_accession() -> None:
    frame = pl.DataFrame(
        {
            'id': ['G'],
            'proteinIds': [
                _ids(
                    ('X', 'uniprot_obsolete'),
                    ('B', 'ensembl_PRO'),
                    (' X ', 'uniprot_swissprot'),
                    ('A', 'uniprot_trembl'),
                    ('C', 'uniprot'),
                    (None, 'uniprot_trembl'),
                )
            ],
        },
        schema={'id': pl.String, 'proteinIds': pl.List(ID_SOURCE)},
    )
    assert filter_and_sort_protein_ids(frame)['proteinIds'].to_list() == [
        _ids(('X', 'uniprot_swissprot'), ('A', 'uniprot_trembl'), ('C', 'uniprot'), ('B', 'ensembl_PRO'))
    ]


def test_a_gene_without_protein_ids_gets_null() -> None:
    frame = pl.DataFrame({'id': ['G'], 'proteinIds': [[]]}, schema={'id': pl.String, 'proteinIds': pl.List(ID_SOURCE)})
    assert filter_and_sort_protein_ids(frame)['proteinIds'].to_list() == [None]


def test_go_and_ensembl_xrefs_are_dropped() -> None:
    frame = pl.DataFrame(
        {'dbXrefs': [_ids(('1A1U', 'PDB'), ('GO:1', 'GO'), ('ENST1', 'Ensembl'), ('11998', 'HGNC'))]},
        schema={'dbXrefs': pl.List(ID_SOURCE)},
    )
    assert remove_redundant_xrefs(frame)['dbXrefs'].to_list() == [_ids(('1A1U', 'PDB'), ('11998', 'HGNC'))]


def test_ncbi_synonyms_are_appended_and_repeats_removed() -> None:
    genes = pl.DataFrame(
        {
            'id': [TP53],
            'synonyms': [_labels('HGNC', 'p53')],
            'symbolSynonyms': [_labels('HGNC', 'p53')],
            'nameSynonyms': [None],
            'obsoleteNames': [[]],
            'obsoleteSymbols': [_labels('HGNC', 'A', 'A')],
        },
        schema={
            c: pl.List(LABEL_SOURCE)
            for c in ['synonyms', 'symbolSynonyms', 'nameSynonyms', 'obsoleteNames', 'obsoleteSymbols']
        }
        | {'id': pl.String},
    )
    result = remove_duplicated_synonyms(add_ncbi_synonyms(genes, NCBI)).to_dicts()[0]
    assert result['symbolSynonyms'] == _labels('HGNC', 'p53') + _labels('NCBI_entrez', 'p53', 'BCC7')
    assert result['nameSynonyms'] == _labels('NCBI_entrez', 'cellular tumor antigen p53')
    assert result['obsoleteSymbols'] == _labels('HGNC', 'A')


@pytest.mark.parametrize(('strand', 'expected'), [(1, 100), (-1, 200), (0, None), (None, None)])
def test_tss_follows_the_strand(strand: int | None, expected: int | None) -> None:
    frame = pl.DataFrame(
        {
            'id': ['G'],
            'canonicalTranscript': [{'id': 'T', 'chromosome': '1', 'start': 100, 'end': 200, 'strand': strand}],
        },
        schema={'id': pl.String, 'canonicalTranscript': CANONICAL_TRANSCRIPT},
    )
    result = add_tss(frame)
    assert result['tss'].to_list() == [expected]
    assert 'canonicalTranscript' not in result.columns


def test_tss_from_the_ensembl_canonical_transcript() -> None:
    tss = {r['id']: r['tss'] for r in add_tss(ensembl_genes(GENES, TRANSCRIPTS, XREFS)).to_dicts()}
    assert tss == {'ENSG00000141510': 7687538, 'ENSG00000000002': 1000, 'ENSG00000000003': 577}


def test_missing_canonical_transcript_has_no_tss() -> None:
    frame = pl.DataFrame(
        {'id': ['G'], 'canonicalTranscript': [None]},
        schema={'id': pl.String, 'canonicalTranscript': CANONICAL_TRANSCRIPT},
    )
    assert add_tss(frame)['tss'].to_list() == [None]


def test_uniprot_lists_concatenate_each_entry_once() -> None:
    gene = _assembled()[TWOENTRIES]
    assert sorted(gene['functionDescriptions']) == ['First function.', 'Shared function.', 'Shared function.']
    assert gene['targetClass'] == [{'id': 7, 'label': 'Kinase', 'level': 'l1'}]


def test_uniprot_only_reaches_genes_through_hgnc() -> None:
    gene = _assembled()[NOHGNC]
    assert gene['functionDescriptions'] is None
    assert gene['subcellularLocations'] is None
    assert gene['proteinIds'] is None
    assert gene['dbXrefs'] == []


def test_tp53_end_to_end() -> None:
    gene = _assembled()[TP53]
    assert gene['proteinIds'] == _ids(
        ('P04637', 'uniprot_swissprot'),
        ('K7PPA8', 'uniprot_trembl'),
        ('ENSP00000269305', 'ensembl_PRO'),
        ('Q15086', 'uniprot_obsolete'),
    )
    assert sorted(x['source'] for x in gene['dbXrefs']) == ['HGNC', 'PDB', 'signalP']
    assert gene['obsoleteNames'] == _labels('HGNC', 'Li-Fraumeni syndrome')
    assert gene['targetClass'] == [{'id': 6, 'label': 'Enzyme', 'level': 'l1'}]
    assert gene['tss'] == 7687538


def test_matches_the_spark_output() -> None:
    result = assemble(
        ENSEMBL, HGNC, GO_FRAME, HALLMARKS_FRAME, UNIPROT, PROTEIN_CLASS, HPA, CONSTRAINT_FRAME, NCBI, REACTOME
    )
    assert rows(result, 'id', UNORDERED) == SPARK_TARGET


def test_uniprot_by_gene_matches_the_spark_output() -> None:
    with_class = add_protein_classification(UNIPROT, PROTEIN_CLASS)
    assert rows(with_class, 'uniprotId', ('targetClass',)) == SPARK_WITH_CLASS
    lists = (
        'synonyms',
        'symbolSynonyms',
        'nameSynonyms',
        'functionDescriptions',
        'proteinIds',
        'subcellularLocations',
        'dbXrefs',
        'targetClass',
    )
    assert rows(uniprot_by_gene(HGNC, with_class), 'id', lists) == SPARK_BY_GENE


def test_genes_without_pathways_keep_null_pathways() -> None:
    genes = _assembled()
    assert genes[TP53]['pathways'] == [
        {'pathwayId': 'R-HSA-69278', 'pathway': 'Cell Cycle, Mitotic', 'topLevelTerm': 'Cell Cycle'}
    ]
    assert genes[NOHGNC]['pathways'] is None
    assert genes[TWOENTRIES]['pathways'] is None


def test_protein_id_label_does_not_depend_on_input_order() -> None:
    pairs = [('X', 'uniprot_trembl'), ('X', 'uniprot_swissprot')]
    schema = {'id': pl.String, 'proteinIds': pl.List(ID_SOURCE)}
    labels = [
        filter_and_sort_protein_ids(pl.DataFrame({'id': ['G'], 'proteinIds': [_ids(*ordered)]}, schema=schema))[
            'proteinIds'
        ].to_list()
        for ordered in (pairs, pairs[::-1])
    ]
    assert labels[0] == labels[1] == [_ids(('X', 'uniprot_swissprot'))]


#: The spark output for `assemble`, as `rows` normalises it.
SPARK_TARGET = [
    {
        'id': 'ENSG00000000002',
        'approvedSymbol': 'ENSG00000000002',
        'biotype': '',
        'genomicLocation': {'chromosome': 'X', 'start': 1000, 'end': 2000, 'strand': 1},
        'approvedName': '',
        'go': None,
        'hallmarks': None,
        'synonyms': [],
        'symbolSynonyms': [],
        'nameSynonyms': [],
        'functionDescriptions': None,
        'subcellularLocations': None,
        'targetClass': None,
        'obsoleteSymbols': [],
        'obsoleteNames': [],
        'constraint': None,
        'proteinIds': None,
        'dbXrefs': [],
        'pathways': None,
        'tss': 1000,
    },
    {
        'id': 'ENSG00000000003',
        'approvedSymbol': 'TWO',
        'biotype': 'protein_coding',
        'genomicLocation': {'chromosome': '1', 'start': 10, 'end': 20, 'strand': 1},
        'approvedName': 'two entries',
        'go': None,
        'hallmarks': None,
        'synonyms': [{'label': 'one', 'source': 'uniprot'}, {'label': 'second', 'source': 'HGNC'}],
        'symbolSynonyms': [],
        'nameSynonyms': [{'label': 'one', 'source': 'uniprot'}, {'label': 'second', 'source': 'HGNC'}],
        'functionDescriptions': ['First function.', 'Shared function.', 'Shared function.'],
        'subcellularLocations': [
            {'location': 'Cytoplasm', 'source': 'uniprot', 'termSL': None, 'labelSL': None, 'targetModifier': None}
        ],
        'targetClass': [{'id': 7, 'label': 'Kinase', 'level': 'l1'}],
        'obsoleteSymbols': [{'label': 'OLD', 'source': 'HGNC'}],
        'obsoleteNames': [],
        'constraint': None,
        'proteinIds': [{'id': 'O00001', 'source': 'uniprot_swissprot'}, {'id': 'O00002', 'source': 'uniprot_obsolete'}],
        'dbXrefs': [{'id': '1AAA', 'source': 'PDB'}, {'id': '2', 'source': 'HGNC'}],
        'pathways': None,
        'tss': None,
    },
    {
        'id': 'ENSG00000141510',
        'approvedSymbol': 'TP53',
        'biotype': 'protein_coding',
        'genomicLocation': {'chromosome': '17', 'start': 7661779, 'end': 7687546, 'strand': -1},
        'approvedName': 'tumor protein p53',
        'go': [
            {
                'id': 'GO:0003677',
                'source': 'PMID:1',
                'evidence': 'IDA',
                'aspect': 'F',
                'geneProduct': 'P04637',
                'ecoId': None,
            }
        ],
        'hallmarks': {
            'attributes': [{'pmid': 1, 'description': 'x', 'name': 'role in cancer'}],
            'cancerHallmarks': None,
        },
        'synonyms': [
            {'label': 'BCC7', 'source': 'NCBI_entrez'},
            {'label': 'Cellular tumor antigen p53', 'source': 'uniprot'},
            {'label': 'LFS1', 'source': 'HGNC'},
            {'label': 'cellular tumor antigen p53', 'source': 'NCBI_entrez'},
            {'label': 'p53', 'source': 'HGNC'},
            {'label': 'p53', 'source': 'NCBI_entrez'},
            {'label': 'p53', 'source': 'uniprot'},
        ],
        'symbolSynonyms': [
            {'label': 'BCC7', 'source': 'NCBI_entrez'},
            {'label': 'LFS1', 'source': 'HGNC'},
            {'label': 'p53', 'source': 'HGNC'},
            {'label': 'p53', 'source': 'NCBI_entrez'},
            {'label': 'p53', 'source': 'uniprot'},
        ],
        'nameSynonyms': [
            {'label': 'Cellular tumor antigen p53', 'source': 'uniprot'},
            {'label': 'cellular tumor antigen p53', 'source': 'NCBI_entrez'},
        ],
        'functionDescriptions': ['Acts as a tumor suppressor.'],
        'subcellularLocations': [
            {
                'location': 'Nucleus',
                'source': 'HPA_main',
                'termSL': 'SL-0191',
                'labelSL': 'Cellular component',
                'targetModifier': None,
            },
            {
                'location': 'Nucleus',
                'source': 'uniprot',
                'termSL': 'SL-0191',
                'labelSL': 'Cellular component',
                'targetModifier': None,
            },
        ],
        'targetClass': [{'id': 6, 'label': 'Enzyme', 'level': 'l1'}],
        'obsoleteSymbols': [],
        'obsoleteNames': [{'label': 'Li-Fraumeni syndrome', 'source': 'HGNC'}],
        'constraint': [
            {
                'constraintType': 'syn',
                'score': 1.5,
                'exp': 2.0,
                'obs': 3,
                'oe': 0.5,
                'oeLower': 0.25,
                'oeUpper': 0.75,
                'upperRank': None,
                'upperBin': None,
                'upperBin6': None,
            }
        ],
        'proteinIds': [
            {'id': 'P04637', 'source': 'uniprot_swissprot'},
            {'id': 'K7PPA8', 'source': 'uniprot_trembl'},
            {'id': 'ENSP00000269305', 'source': 'ensembl_PRO'},
            {'id': 'Q15086', 'source': 'uniprot_obsolete'},
        ],
        'dbXrefs': [
            {'id': '11998', 'source': 'HGNC'},
            {'id': '1A1U', 'source': 'PDB'},
            {'id': 'SignalP-noTM', 'source': 'signalP'},
        ],
        'pathways': [{'pathwayId': 'R-HSA-69278', 'pathway': 'Cell Cycle, Mitotic', 'topLevelTerm': 'Cell Cycle'}],
        'tss': 7687538,
    },
]

#: The spark output for `add_protein_classification`, as `rows` normalises it.
SPARK_WITH_CLASS = [
    {
        'uniprotId': 'O00001',
        'functionDescriptions': ['Shared function.', 'First function.'],
        'dbXrefs': [{'id': '1AAA', 'source': 'PDB'}],
        'subcellularLocations': None,
        'proteinIds': [{'id': 'O00001', 'source': 'uniprot_obsolete'}],
        'synonyms': [{'label': 'one', 'source': 'uniprot'}],
        'symbolSynonyms': [],
        'nameSynonyms': [{'label': 'one', 'source': 'uniprot'}],
        'targetClass': [],
    },
    {
        'uniprotId': 'O00002',
        'functionDescriptions': ['Shared function.'],
        'dbXrefs': [{'id': '1AAA', 'source': 'PDB'}],
        'subcellularLocations': [
            {'location': 'Cytoplasm', 'source': 'uniprot', 'termSL': None, 'labelSL': None, 'targetModifier': None}
        ],
        'proteinIds': [{'id': 'O00002', 'source': 'uniprot_obsolete'}],
        'synonyms': [],
        'symbolSynonyms': [],
        'nameSynonyms': [],
        'targetClass': [{'id': 7, 'label': 'Kinase', 'level': 'l1'}],
    },
    {
        'uniprotId': 'P04637',
        'functionDescriptions': ['Acts as a tumor suppressor.'],
        'dbXrefs': [
            {'id': '1A1U', 'source': 'PDB'},
            {'id': 'GO:0000785', 'source': 'GO'},
            {'id': 'ENST00000269305', 'source': 'Ensembl'},
        ],
        'subcellularLocations': [
            {
                'location': 'Nucleus',
                'source': 'uniprot',
                'termSL': 'SL-0191',
                'labelSL': 'Cellular component',
                'targetModifier': None,
            }
        ],
        'proteinIds': [{'id': 'P04637', 'source': 'uniprot_obsolete'}, {'id': 'Q15086', 'source': 'uniprot_obsolete'}],
        'synonyms': [
            {'label': 'Cellular tumor antigen p53', 'source': 'uniprot'},
            {'label': 'p53', 'source': 'uniprot'},
        ],
        'symbolSynonyms': [{'label': 'p53', 'source': 'uniprot'}],
        'nameSynonyms': [{'label': 'Cellular tumor antigen p53', 'source': 'uniprot'}],
        'targetClass': [{'id': 6, 'label': 'Enzyme', 'level': 'l1'}],
    },
]

#: The spark output for `uniprot_by_gene`, as `rows` normalises it.
SPARK_BY_GENE = [
    {
        'id': 'ENSG00000000003',
        'synonyms': [{'label': 'one', 'source': 'uniprot'}],
        'symbolSynonyms': [],
        'nameSynonyms': [{'label': 'one', 'source': 'uniprot'}],
        'functionDescriptions': ['First function.', 'Shared function.', 'Shared function.'],
        'proteinIds': [{'id': 'O00001', 'source': 'uniprot_obsolete'}, {'id': 'O00002', 'source': 'uniprot_obsolete'}],
        'subcellularLocations': [
            {'location': 'Cytoplasm', 'source': 'uniprot', 'termSL': None, 'labelSL': None, 'targetModifier': None}
        ],
        'dbXrefs': [{'id': '1AAA', 'source': 'PDB'}],
        'targetClass': [{'id': 7, 'label': 'Kinase', 'level': 'l1'}],
    },
    {
        'id': 'ENSG00000141510',
        'synonyms': [
            {'label': 'Cellular tumor antigen p53', 'source': 'uniprot'},
            {'label': 'p53', 'source': 'uniprot'},
        ],
        'symbolSynonyms': [{'label': 'p53', 'source': 'uniprot'}],
        'nameSynonyms': [{'label': 'Cellular tumor antigen p53', 'source': 'uniprot'}],
        'functionDescriptions': ['Acts as a tumor suppressor.'],
        'proteinIds': [{'id': 'P04637', 'source': 'uniprot_obsolete'}, {'id': 'Q15086', 'source': 'uniprot_obsolete'}],
        'subcellularLocations': [
            {
                'location': 'Nucleus',
                'source': 'uniprot',
                'termSL': 'SL-0191',
                'labelSL': 'Cellular component',
                'targetModifier': None,
            }
        ],
        'dbXrefs': [
            {'id': '1A1U', 'source': 'PDB'},
            {'id': 'ENST00000269305', 'source': 'Ensembl'},
            {'id': 'GO:0000785', 'source': 'GO'},
        ],
        'targetClass': [{'id': 6, 'label': 'Enzyme', 'level': 'l1'}],
    },
    {
        'id': None,
        'synonyms': [
            {'label': 'Cellular tumor antigen p53', 'source': 'uniprot'},
            {'label': 'p53', 'source': 'uniprot'},
        ],
        'symbolSynonyms': [{'label': 'p53', 'source': 'uniprot'}],
        'nameSynonyms': [{'label': 'Cellular tumor antigen p53', 'source': 'uniprot'}],
        'functionDescriptions': ['Acts as a tumor suppressor.'],
        'proteinIds': [{'id': 'P04637', 'source': 'uniprot_obsolete'}, {'id': 'Q15086', 'source': 'uniprot_obsolete'}],
        'subcellularLocations': [
            {
                'location': 'Nucleus',
                'source': 'uniprot',
                'termSL': 'SL-0191',
                'labelSL': 'Cellular component',
                'targetModifier': None,
            }
        ],
        'dbXrefs': [
            {'id': '1A1U', 'source': 'PDB'},
            {'id': 'ENST00000269305', 'source': 'Ensembl'},
            {'id': 'GO:0000785', 'source': 'GO'},
        ],
        'targetClass': [{'id': 6, 'label': 'Enzyme', 'level': 'l1'}],
    },
]
