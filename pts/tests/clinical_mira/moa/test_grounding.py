import polars as pl
import pytest

from pts.transformers.clinical_mira.moa.grounding import GROUNDED, build_label_index, ground_mechanisms, resolve

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
        ('ENSG_HDAC1', 'HDAC1', 'histone deacetylase 1', [], []),
        ('ENSG_HDAC9', 'HDAC9', 'histone deacetylase 9', _labels('HDAC'), []),
        ('ENSG_GNRH1', 'GNRH1', 'gonadotropin releasing hormone 1', _labels('GnRH'), []),
        ('ENSG_GNRHR', 'GNRHR', 'gonadotropin releasing hormone receptor', _labels('LHRHR'), []),
        ('ENSG_FGFR1', 'FGFR1', 'fibroblast growth factor receptor 1', [], []),
        ('ENSG_FGFR2', 'FGFR2', 'fibroblast growth factor receptor 2', [], []),
        ('ENSG_FGFR3', 'FGFR3', 'fibroblast growth factor receptor 3', [], []),
        ('ENSG_ACAT1', 'ACAT1', 'acetyl-CoA acetyltransferase 1', [], []),
        ('ENSG_MLST8', 'MLST8', 'MTOR associated protein, LST8 homolog', [], _labels('TORC subunit LST8')),
        ('ENSG_SOAT1', 'SOAT1', 'sterol O-acyltransferase 1', _labels('ACAT1', 'ACAT'), []),
        ('ENSG_M7', 'MAP2K7', 'mitogen-activated protein kinase kinase 7', _labels('MEK'), []),
        ('ENSG_M1', 'MAP2K1', 'mitogen-activated protein kinase kinase 1', _labels('MEK1'), []),
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
    base = {'kept': True, 'isHumanTarget': True, 'actsOnReceptor': False, 'targetSymbolSuggestion': []}
    return pl.DataFrame(
        [base | r for r in rows],
        schema={
            'targetFromSource': pl.String,
            'targetSymbolSuggestion': pl.List(pl.String),
            'kept': pl.Boolean,
            'isHumanTarget': pl.Boolean,
            'actsOnReceptor': pl.Boolean,
        },
    )


def test_source_synonym_agreeing_with_suggestion_grounds_by_source(index):
    row = ground_mechanisms(_mechanisms({'targetFromSource': 'PD-1', 'targetSymbolSuggestion': ['PDCD1']}), index)

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


def test_synonym_that_disagrees_with_the_suggestion_is_a_conflict(index):
    rows = ground_mechanisms(
        _mechanisms(
            {'targetFromSource': 'MEK', 'targetSymbolSuggestion': ['MAP2K1']},
            {'targetFromSource': 'MEK', 'targetSymbolSuggestion': ['MAP2K7']},
            {'targetFromSource': 'MEK', 'targetSymbolSuggestion': []},
            {'targetFromSource': 'EGFR', 'targetSymbolSuggestion': ['KRAS']},
        ),
        index,
    )

    assert rows.select('targetIds', 'grounding').to_dicts() == [
        {'targetIds': [], 'grounding': 'conflict'},
        {'targetIds': ['ENSG_M7'], 'grounding': 'source'},
        {'targetIds': ['ENSG_M7'], 'grounding': 'source'},
        {'targetIds': ['ENSG_EGFR'], 'grounding': 'source'},
    ]


def test_receptor_of_a_named_ligand_grounds_to_the_receptor(index):
    rows = ground_mechanisms(
        _mechanisms(
            {'targetFromSource': 'GnRH', 'targetSymbolSuggestion': ['GNRHR'], 'actsOnReceptor': True},
            {'targetFromSource': 'GnRH', 'targetSymbolSuggestion': ['GNRHR'], 'actsOnReceptor': False},
            {'targetFromSource': 'GnRH', 'targetSymbolSuggestion': ['EGFR'], 'actsOnReceptor': True},
        ),
        index,
    )

    assert rows.select('targetIds', 'grounding').to_dicts() == [
        {'targetIds': ['ENSG_GNRHR'], 'grounding': 'receptor'},
        {'targetIds': [], 'grounding': 'conflict'},
        {'targetIds': [], 'grounding': 'suggestion_unsupported'},
    ]


def test_family_shorthand_is_the_stem_of_its_members(index):
    rows = ground_mechanisms(
        _mechanisms(
            {'targetFromSource': 'FGFR', 'targetSymbolSuggestion': ['FGFR1', 'FGFR2']},
            {'targetFromSource': 'MEK', 'targetSymbolSuggestion': ['MAP2K1']},
            {'targetFromSource': 'mu', 'targetSymbolSuggestion': ['MAP2K1']},
        ),
        index,
    )

    assert rows.select('targetIds', 'grounding').to_dicts() == [
        {'targetIds': ['ENSG_FGFR1', 'ENSG_FGFR2'], 'grounding': 'suggestion'},
        {'targetIds': [], 'grounding': 'conflict'},
        {'targetIds': [], 'grounding': 'suggestion_unsupported'},
    ]


def test_symbol_that_is_also_a_synonym_of_the_suggested_gene_grounds_to_the_suggestion(index):
    rows = ground_mechanisms(
        _mechanisms(
            {'targetFromSource': 'ACAT1', 'targetSymbolSuggestion': ['SOAT1']},
            {'targetFromSource': 'ACAT1', 'targetSymbolSuggestion': ['ACAT1']},
            {'targetFromSource': 'ACAT1', 'targetSymbolSuggestion': []},
        ),
        index,
    )

    assert rows.select('targetIds', 'grounding').to_dicts() == [
        {'targetIds': ['ENSG_SOAT1'], 'grounding': 'suggestion'},
        {'targetIds': ['ENSG_ACAT1'], 'grounding': 'source'},
        {'targetIds': ['ENSG_ACAT1'], 'grounding': 'source'},
    ]


def test_numbered_range_covers_every_member_it_spans(index):
    rows = ground_mechanisms(
        _mechanisms(
            {'targetFromSource': 'FGFR1\u20133', 'targetSymbolSuggestion': ['FGFR1', 'FGFR2', 'FGFR3']},
            {'targetFromSource': 'FGFR1/2', 'targetSymbolSuggestion': ['FGFR1', 'FGFR2']},
            {'targetFromSource': 'FGFR1 and 3', 'targetSymbolSuggestion': ['FGFR1', 'FGFR3']},
        ),
        index,
    )

    assert rows.select('targetIds', 'grounding').to_dicts() == [
        {'targetIds': ['ENSG_FGFR1', 'ENSG_FGFR2', 'ENSG_FGFR3'], 'grounding': 'suggestion'},
        {'targetIds': ['ENSG_FGFR1', 'ENSG_FGFR2'], 'grounding': 'suggestion'},
        {'targetIds': ['ENSG_FGFR1', 'ENSG_FGFR3'], 'grounding': 'suggestion'},
    ]


def test_range_stem_does_not_match_the_bare_word(index):
    rows = ground_mechanisms(_mechanisms({'targetFromSource': 'TORC1/2', 'targetSymbolSuggestion': ['MLST8']}), index)

    assert rows.select('targetIds', 'grounding').to_dicts() == [
        {'targetIds': [], 'grounding': 'suggestion_unsupported'}
    ]


def test_single_numbered_member_does_not_cover_its_siblings(index):
    rows = ground_mechanisms(
        _mechanisms({'targetFromSource': 'FGFR1 kinase domain', 'targetSymbolSuggestion': ['FGFR1', 'FGFR2']}),
        index,
    )

    assert rows.select('targetIds', 'grounding').to_dicts() == [
        {'targetIds': ['ENSG_FGFR1'], 'grounding': 'suggestion'}
    ]


def test_grounded_genes_carry_their_symbols(index):
    rows = ground_mechanisms(_mechanisms({'targetFromSource': 'EGFR'}, {'targetFromSource': 'nothing'}), index)

    assert rows['targetSymbols'].to_list() == [['EGFR'], []]


def test_family_name_uses_the_members_not_a_single_synonym(index):
    rows = ground_mechanisms(
        _mechanisms({'targetFromSource': 'HDAC', 'targetSymbolSuggestion': ['HDAC1', 'HDAC9']}),
        index,
    )

    assert rows.select('targetIds', 'grounding').to_dicts() == [
        {'targetIds': ['ENSG_HDAC1', 'ENSG_HDAC9'], 'grounding': 'suggestion'}
    ]


def _located_index(*genes: tuple[str, str, str, str, list[str]]) -> pl.DataFrame:
    target_view = pl.DataFrame(
        {
            'id': [g[0] for g in genes],
            'approvedSymbol': [g[1] for g in genes],
            'biotype': [g[2] for g in genes],
            'genomicLocation': [{'chromosome': g[3]} for g in genes],
            'approvedName': [None for _ in genes],
            'symbolSynonyms': [_labels(*g[4]) for g in genes],
            'obsoleteSymbols': [[] for _ in genes],
            'nameSynonyms': [[] for _ in genes],
        },
        schema={
            'id': pl.String,
            'approvedSymbol': pl.String,
            'biotype': pl.String,
            'genomicLocation': pl.Struct({'chromosome': pl.String}),
            'approvedName': pl.String,
            'symbolSynonyms': LABELS,
            'obsoleteSymbols': LABELS,
            'nameSynonyms': LABELS,
        },
    )
    return build_label_index(target_view)


def test_pseudoautosomal_y_copy_gives_way_to_the_x_copy():
    index = _located_index(
        ('ENSG_X', 'IL3RA', 'protein_coding', 'X', ['CD123']),
        ('ENSG_Y', 'IL3RA', 'protein_coding', 'Y', ['CD123']),
    )

    assert resolve('IL3RA', index) == ['ENSG_X']
    assert resolve('CD123', index) == ['ENSG_X']


def test_protein_coding_copy_wins_over_other_biotypes():
    index = _located_index(
        ('ENSG_RNA', 'SIGLEC5', 'lncRNA', '19', []),
        ('ENSG_PC', 'SIGLEC5', 'protein_coding', '19', []),
    )

    assert resolve('SIGLEC5', index) == ['ENSG_PC']


def test_two_protein_coding_copies_stay_ambiguous():
    index = _located_index(
        ('ENSG_A', 'PDE4C', 'protein_coding', '19', []),
        ('ENSG_B', 'PDE4C', 'protein_coding', '19', []),
    )

    assert resolve('PDE4C', index) == []


def test_y_gene_with_its_own_symbol_is_kept():
    index = _located_index(('ENSG_SRY', 'SRY', 'protein_coding', 'Y', []))

    assert resolve('SRY', index) == ['ENSG_SRY']


def _confident(*rows: dict) -> pl.DataFrame:
    frame = _mechanisms(*[{k: v for k, v in r.items() if k != 'confidence'} for r in rows])
    return frame.with_columns(confidence=pl.Series([r['confidence'] for r in rows], dtype=pl.Float64))


def test_confident_unsupported_suggestion_falls_back_to_the_suggestion(index):
    rows = ground_mechanisms(
        _confident(
            {'targetFromSource': 'DA-2', 'targetSymbolSuggestion': ['DRD2'], 'confidence': 0.9},
            {'targetFromSource': 'DA-2', 'targetSymbolSuggestion': ['DRD2'], 'confidence': 0.75},
            {'targetFromSource': 'DA-2', 'targetSymbolSuggestion': ['NOPE'], 'confidence': 0.9},
        ),
        index,
        suggestion_min_confidence=0.8,
    )

    assert rows.select('targetIds', 'grounding').to_dicts() == [
        {'targetIds': ['ENSG_DRD2'], 'grounding': 'suggestion_fallback'},
        {'targetIds': [], 'grounding': 'suggestion_unsupported'},
        {'targetIds': [], 'grounding': 'unresolved'},
    ]


def test_confident_conflict_falls_back_to_the_suggestion(index):
    rows = ground_mechanisms(
        _confident({'targetFromSource': 'MEK', 'targetSymbolSuggestion': ['MAP2K1'], 'confidence': 0.95}),
        index,
        suggestion_min_confidence=0.8,
    )

    assert rows.select('targetIds', 'grounding').to_dicts() == [
        {'targetIds': ['ENSG_M1'], 'grounding': 'suggestion_fallback'}
    ]


def test_confident_receptor_without_a_named_ligand_falls_back_to_the_suggestion(index):
    rows = ground_mechanisms(
        _confident({
            'targetFromSource': 'its receptor',
            'targetSymbolSuggestion': ['EGFR'],
            'actsOnReceptor': True,
            'confidence': 0.8,
        }),
        index,
        suggestion_min_confidence=0.8,
    )

    assert rows.select('targetIds', 'grounding').to_dicts() == [
        {'targetIds': ['ENSG_EGFR'], 'grounding': 'suggestion_fallback'}
    ]


def test_without_a_fallback_confidence_suggestions_stay_unsupported(index):
    rows = ground_mechanisms(
        _confident({'targetFromSource': 'DA-2', 'targetSymbolSuggestion': ['DRD2'], 'confidence': 0.99}), index
    )

    assert rows['grounding'].to_list() == ['suggestion_unsupported']


def test_fallback_is_a_grounded_outcome():
    assert 'suggestion_fallback' in GROUNDED
