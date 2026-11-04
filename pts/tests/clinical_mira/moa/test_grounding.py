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
        ('ENSG_EGFR', 'EGFR', 'epidermal growth factor receptor', _labels('ERBB1'), []),
    ] + [(f'ENSG_K{i}', f'K{i}', f'kinase {i}', [], []) for i in range(21)]
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
        _mechanisms({'targetFromSource': 'kinases', 'targetSymbolSuggestion': [f'K{i}' for i in range(21)]}), index
    )
    small = ground_mechanisms(
        _mechanisms({'targetFromSource': 'kinases', 'targetSymbolSuggestion': [f'K{i}' for i in range(20)]}), index
    )

    assert family['class_level'].to_list() == [True]
    assert small['class_level'].to_list() == [False]
