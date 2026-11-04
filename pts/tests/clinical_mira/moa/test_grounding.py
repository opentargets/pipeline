import polars as pl
import pytest

from pts.transformers.clinical_mira.moa.grounding import build_label_index, ground_mechanisms, resolve

LABELS = pl.List(pl.Struct({'label': pl.String, 'source': pl.String}))


def _labels(*names: str) -> list[dict]:
    return [{'label': n, 'source': 's'} for n in names]


@pytest.fixture
def index() -> pl.DataFrame:
    genes = [
        (
            'ENSG_PDCD1',
            'PDCD1',
            'programmed cell death 1',
            _labels('PD1', 'XYZ'),
            _labels('programmed cell death protein 1'),
        ),
        ('ENSG_A', 'GENE_A', 'gene a', _labels('XYZ'), []),
        (
            'ENSG_EGFR',
            'EGFR',
            'epidermal growth factor receptor',
            _labels('ERBB1'),
            _labels('receptor tyrosine-protein kinase erbB-1'),
        ),
        ('ENSG_TOP1', 'TOP1', 'DNA topoisomerase I', [], []),
        ('ENSG_DRD2', 'DRD2', 'dopamine receptor D2', [], []),
        ('ENSG_KRAS', 'KRAS', 'KRAS proto-oncogene', [], []),
        ('ENSG_MS4A1', 'MS4A1', 'membrane spanning 4-domains A1', _labels('CD20'), []),
        (
            'ENSG_BCMA',
            'TNFRSF17',
            'TNF receptor superfamily member 17',
            _labels('BCMA'),
            _labels('B-cell maturation antigen'),
        ),
    ] + [(f'ENSG_K{i}', f'K{i}', f'kinase {i}', [], _labels('kinase family')) for i in range(21)]
    target_view = pl.DataFrame(
        {
            'id': [g[0] for g in genes],
            'approvedSymbol': [g[1] for g in genes],
            'approvedName': [g[2] for g in genes],
            'symbolSynonyms': [g[3] for g in genes],
            'obsoleteSymbols': [[] for _ in genes],
            'nameSynonyms': [g[4] for g in genes],
        },
        schema={
            'id': pl.String,
            'approvedSymbol': pl.String,
            'approvedName': pl.String,
            'symbolSynonyms': LABELS,
            'obsoleteSymbols': LABELS,
            'nameSynonyms': LABELS,
        },
    )
    return build_label_index(target_view)


def test_approved_symbol_resolves(index):
    assert resolve('egfr', index) == ['ENSG_EGFR']


def test_hyphenated_synonym_resolves(index):
    assert resolve('PD-1', index) == ['ENSG_PDCD1']


def test_name_synonym_resolves(index):
    assert resolve('Programmed  cell death protein 1', index) == ['ENSG_PDCD1']


def test_shared_synonym_is_unresolved(index):
    assert resolve('xyz', index) == []


def _mechanisms(*rows: dict) -> pl.DataFrame:
    base = {'kept': True, 'isHumanTarget': True, 'targetSymbolSuggestion': []}
    return pl.DataFrame(
        [base | r for r in rows],
        schema={
            'targetFromSource': pl.String,
            'targetSymbolSuggestion': pl.List(pl.String),
            'kept': pl.Boolean,
            'isHumanTarget': pl.Boolean,
        },
    )


def test_source_name_wins(index):
    row = ground_mechanisms(_mechanisms({'targetFromSource': 'PD-1', 'targetSymbolSuggestion': ['EGFR']}), index)

    assert row.select('targetIds', 'grounding').to_dicts() == [{'targetIds': ['ENSG_PDCD1'], 'grounding': 'source'}]


def test_suggestions_rescue_ambiguous_source(index):
    row = ground_mechanisms(_mechanisms({'targetFromSource': 'XYZ', 'targetSymbolSuggestion': ['GENE_A']}), index)

    assert row.select('targetIds', 'grounding').to_dicts() == [{'targetIds': ['ENSG_A'], 'grounding': 'suggestion'}]


def test_unresolvable_is_marked(index):
    row = ground_mechanisms(_mechanisms({'targetFromSource': 'nothing', 'targetSymbolSuggestion': ['NOPE']}), index)

    assert row.select('targetIds', 'grounding').to_dicts() == [{'targetIds': [], 'grounding': 'unresolved'}]


def test_non_human_and_dropped_rows_skipped(index):
    rows = ground_mechanisms(
        _mechanisms(
            {'targetFromSource': 'EGFR', 'isHumanTarget': False},
            {'targetFromSource': 'EGFR', 'kept': False},
        ),
        index,
    )

    assert rows['grounding'].to_list() == ['skipped', 'skipped']
    assert rows['targetIds'].to_list() == [[], []]


def test_class_level_flag(index):
    family = ground_mechanisms(
        _mechanisms({'targetFromSource': 'kinase family', 'targetSymbolSuggestion': [f'K{i}' for i in range(21)]}),
        index,
    )
    small = ground_mechanisms(
        _mechanisms({'targetFromSource': 'kinase family', 'targetSymbolSuggestion': [f'K{i}' for i in range(20)]}),
        index,
    )

    assert family['class_level'].to_list() == [True]
    assert small['class_level'].to_list() == [False]


def test_suggestion_must_be_a_candidate_of_an_ambiguous_name(index):
    row = ground_mechanisms(_mechanisms({'targetFromSource': 'XYZ', 'targetSymbolSuggestion': ['EGFR']}), index)

    assert row.select('targetIds', 'grounding').to_dicts() == [{'targetIds': [], 'grounding': 'suggestion_unsupported'}]


def test_suggestion_for_an_unknown_name_must_share_a_label_with_it(index):
    rows = ground_mechanisms(
        _mechanisms(
            {'targetFromSource': 'KRAS G12C', 'targetSymbolSuggestion': ['KRAS']},
            {'targetFromSource': 'kinase', 'targetSymbolSuggestion': ['EGFR']},
            {'targetFromSource': 'anti-CD20', 'targetSymbolSuggestion': ['MS4A1']},
            {'targetFromSource': 'B-cell maturation antigen protein', 'targetSymbolSuggestion': ['TNFRSF17']},
        ),
        index,
    )

    assert rows.select('targetIds', 'grounding').to_dicts() == [
        {'targetIds': ['ENSG_KRAS'], 'grounding': 'suggestion'},
        {'targetIds': [], 'grounding': 'suggestion_unsupported'},
        {'targetIds': ['ENSG_MS4A1'], 'grounding': 'suggestion'},
        {'targetIds': ['ENSG_BCMA'], 'grounding': 'suggestion'},
    ]


def test_suggestion_sharing_a_specific_word_with_the_name_is_kept(index):
    rows = ground_mechanisms(
        _mechanisms(
            {'targetFromSource': 'Topoisomerase I', 'targetSymbolSuggestion': ['TOP1']},
            {'targetFromSource': 'D2 dopamine receptor', 'targetSymbolSuggestion': ['DRD2']},
            {'targetFromSource': 'tyrosine kinase', 'targetSymbolSuggestion': ['EGFR']},
            {'targetFromSource': 'receptor kinase', 'targetSymbolSuggestion': ['EGFR']},
        ),
        index,
    )

    assert rows['grounding'].to_list() == [
        'suggestion',
        'suggestion',
        'suggestion_unsupported',
        'suggestion_unsupported',
    ]
