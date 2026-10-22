"""Tests for the gene_essentiality transformer."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pts.transformers.gene_essentiality import (
    build_screens,
    gene_essentiality,
    melt_matrix,
    read_models,
    resolve_target_ids,
)
from pts.transformers.utils.target import ensg_lookup

HEADER = ',SequencingID,ModelID,ModelConditionID,IsDefaultEntryForModel,IsDefaultEntryForMC'

FILES = {
    'models': (
        'ModelID,CellLineName,CCLEName,OncotreeLineage,OncotreePrimaryDisease,SangerModelID\n'
        'ACH-1,Line1,CCLE1,Lung,Lung Cancer,SIDM1\n'
        'ACH-2,,CCLE2,Breast,Breast Cancer,SIDM2\n'
        'ACH-3,,,Skin,Melanoma,\n'
        'ACH-4,,,Skin,,SIDM4\n'
    ),
    'essential_genes': 'Essentials\nGENEA (1)\n',
    'gene_effects': ',GENEA (1),GENEB (2)\nACH-1,-1.5,0.1\nACH-2,-0.25,\nACH-3,0.3,0.2\nACH-4,-2.0,0.5\n',
    'gene_expression': (
        ',SequencingID,ModelConditionID,ModelID,IsDefaultEntryForMC,IsDefaultEntryForModel,GENEA (1),GENEB (2)\n'
        '0,CDS-1,MC-1,ACH-1,Yes,Yes,4.5,1.25\n'
        '1,CDS-2,MC-2,ACH-2,Yes,Yes,0.0,2.5\n'
    ),
    'mutation_hotspots': (
        f'{HEADER},GENEA (1),GENEB (2)\n0,CDS-1,ACH-1,MC-1,Yes,Yes,1.0,0.0\n1,CDS-3,ACH-3,MC-3,Yes,Yes,0.0,2.0\n'
    ),
    'mutation_damaging': (
        f'{HEADER},GENEA (1),GENEB (2)\n0,CDS-1,ACH-1,MC-1,Yes,Yes,1.0,0.0\n1,CDS-4,ACH-4,MC-4,Yes,Yes,0.0,0.0\n'
    ),
    'depmap_tissue_mapping': 'tissueFromSource,tissueId,tissueName\nlung,UBERON_L,lung\nbreast,UBERON_B,breast\n',
}

TARGET_SCHEMA = {
    'id': pl.String,
    'approvedSymbol': pl.String,
    'proteinIds': pl.List(pl.Struct({'id': pl.String, 'source': pl.String})),
}
TARGET = [
    {'id': 'ENSG_A', 'approvedSymbol': 'GENEA', 'proteinIds': [{'id': 'P_A', 'source': 'uniprot'}]},
    {'id': 'ENSG_B', 'approvedSymbol': 'GENEB', 'proteinIds': None},
]

SCREEN = pl.Struct({
    'depmapId': pl.String,
    'cellLineName': pl.String,
    'diseaseFromSource': pl.String,
    'diseaseCellLineId': pl.String,
    'mutation': pl.String,
    'geneEffect': pl.Float32,
    'expression': pl.Float32,
})
OUTPUT_SCHEMA = pl.Schema({
    'targetId': pl.String,
    'isEssential': pl.Boolean,
    'depMapEssentiality': pl.List(
        pl.Struct({'tissueId': pl.String, 'tissueName': pl.String, 'screens': pl.List(SCREEN)})
    ),
})


def f32(x: float) -> float:
    return float(np.float32(x))


def screen(depmap_id, line, disease, model, mutation, effect, expression):
    return {
        'depmapId': depmap_id,
        'cellLineName': line,
        'diseaseFromSource': disease,
        'diseaseCellLineId': model,
        'mutation': mutation,
        'geneEffect': f32(effect),
        'expression': None if expression is None else f32(expression),
    }


GENEA_TISSUES = [
    {
        'tissueId': None,
        'tissueName': 'other',
        'screens': [
            screen('ACH-3', 'Melanoma cells', 'Melanoma', None, None, 0.3, None),
            screen('ACH-4', None, None, 'SIDM4', None, -2.0, None),
        ],
    },
    {
        'tissueId': 'UBERON_B',
        'tissueName': 'breast',
        'screens': [screen('ACH-2', 'CCLE2', 'Breast Cancer', 'SIDM2', None, -0.25, 0.0)],
    },
    {
        'tissueId': 'UBERON_L',
        'tissueName': 'lung',
        'screens': [screen('ACH-1', 'Line1', 'Lung Cancer', 'SIDM1', 'damaging', -1.5, 4.5)],
    },
]
GENEB_TISSUES = [
    {
        'tissueId': None,
        'tissueName': 'other',
        'screens': [
            screen('ACH-3', 'Melanoma cells', 'Melanoma', None, 'hotspot', 0.2, None),
            screen('ACH-4', None, None, 'SIDM4', None, 0.5, None),
        ],
    },
    {
        'tissueId': 'UBERON_L',
        'tissueName': 'lung',
        'screens': [screen('ACH-1', 'Line1', 'Lung Cancer', 'SIDM1', None, 0.1, 1.25)],
    },
]


def write_inputs(tmp_path: Path, files: dict[str, str] | None = None, target: list[dict] | None = None) -> dict:
    source = {}
    for name, text in (FILES | (files or {})).items():
        path = tmp_path / f'{name}.csv'
        path.write_text(text)
        source[name] = str(path)
    pl.DataFrame(target or TARGET, schema=TARGET_SCHEMA).write_parquet(tmp_path / 'target.parquet')
    source['target'] = str(tmp_path / 'target.parquet')
    return source


def run(tmp_path: Path, files=None, target=None, settings=None, out: str = 'out') -> pl.DataFrame:
    source = write_inputs(tmp_path, files, target)
    destination = tmp_path / out
    gene_essentiality(
        source, str(destination), {'keep_only_essentials': False} if settings is None else settings, None
    )
    return pl.read_parquet(sorted(destination.glob('*.parquet')))


def csv(text: str) -> pl.LazyFrame:
    return pl.read_csv(io.StringIO(text), infer_schema=False, null_values=['']).lazy()


def lookup(target: list[dict]) -> pl.LazyFrame:
    return ensg_lookup(pl.LazyFrame(target, schema=TARGET_SCHEMA))


# ---------------------------------------------------------------------------
# ensg_lookup
# ---------------------------------------------------------------------------


def test_ensg_lookup_fans_out_and_tolerates_null_proteins():
    target = pl.LazyFrame(
        {
            'id': ['ENSG1', 'ENSG2', 'ENSG3'],
            'approvedSymbol': ['A', 'B', None],
            'proteinIds': [[{'id': 'P1'}, {'id': 'P1'}], None, [{'id': 'A'}]],
        }
    )
    got = ensg_lookup(target).collect().sort('ensgId', 'name').rows()
    assert got == [('ENSG1', 'A'), ('ENSG1', 'P1'), ('ENSG2', 'B'), ('ENSG3', 'A')]


def test_ensg_lookup_includes_approved_symbol():
    lut = lookup([{'id': 'ENSG00000001', 'approvedSymbol': 'GENE1', 'proteinIds': []}]).collect()
    assert lut.rows() == [('ENSG00000001', 'GENE1')]


def test_ensg_lookup_output_columns():
    lut = lookup([{'id': 'ENSG00000001', 'approvedSymbol': 'GENE1', 'proteinIds': []}])
    assert lut.collect_schema().names() == ['ensgId', 'name']


def test_ensg_lookup_handles_null_protein_ids():
    lut = lookup([{'id': 'ENSG00000003', 'approvedSymbol': 'MIR122', 'proteinIds': None}]).collect()
    assert lut.rows() == [('ENSG00000003', 'MIR122')]


# ---------------------------------------------------------------------------
# resolve_target_ids
# ---------------------------------------------------------------------------


def essentiality(rows: list[tuple[str, bool, list[str]]]) -> pl.LazyFrame:
    return pl.LazyFrame(
        rows,
        schema={'targetSymbol': pl.String, 'isEssential': pl.Boolean, 'depMapEssentiality': pl.List(pl.String)},
        orient='row',
    )


GENE1 = [{'id': 'ENSG00000001', 'approvedSymbol': 'GENE1', 'proteinIds': []}]


def test_resolve_target_ids_output_columns():
    result = resolve_target_ids(essentiality([('GENE1', True, [])]), lookup(GENE1))
    assert result.collect_schema().names() == ['targetId', 'isEssential', 'depMapEssentiality']


def test_resolve_target_ids_resolves_symbol_to_ensg():
    result = resolve_target_ids(essentiality([('GENE1', True, [])]), lookup(GENE1)).collect()
    assert result['targetId'].to_list() == ['ENSG00000001']


def test_resolve_target_ids_drops_unresolvable_rows():
    result = resolve_target_ids(essentiality([('UNKNOWN_SYMBOL', True, [])]), lookup(GENE1)).collect()
    assert result.height == 0


def test_resolve_target_ids_resolves_symbol_for_non_coding_gene():
    target = [{'id': 'ENSG_MIR122', 'approvedSymbol': 'MIR122', 'proteinIds': None}]
    result = resolve_target_ids(essentiality([('MIR122', True, [])]), lookup(target)).collect()
    assert result['targetId'].to_list() == ['ENSG_MIR122']


def test_resolve_target_ids_merges_multiple_entries_per_target():
    rows = [('GENE1', True, ['a']), ('GENE1', False, ['b'])]
    result = resolve_target_ids(essentiality(rows), lookup(GENE1)).collect()
    assert result.height == 1
    assert result['isEssential'].to_list() == [True]
    assert sorted(result['depMapEssentiality'].to_list()[0]) == ['a', 'b']


def test_two_symbols_same_ensg_merge_distinct():
    rows = [('GENEA', True, ['a', 'b']), ('GENEB', False, ['b', 'c']), ('GENEC', False, ['d'])]
    target = [
        {'id': 'ENSG_A', 'approvedSymbol': 'GENEA', 'proteinIds': [{'id': 'GENEB', 'source': 'x'}]},
        {'id': 'ENSG_B', 'approvedSymbol': 'GENEB', 'proteinIds': None},
    ]
    result = resolve_target_ids(essentiality(rows), lookup(target)).collect().sort('targetId')
    assert result.rows() == [('ENSG_A', True, ['a', 'b', 'c']), ('ENSG_B', False, ['b', 'c'])]


def test_two_symbols_same_ensg_end_to_end(tmp_path):
    target = [
        {'id': 'ENSG_A', 'approvedSymbol': 'GENEA', 'proteinIds': [{'id': 'GENEB', 'source': 'x'}]},
        {'id': 'ENSG_B', 'approvedSymbol': 'GENEB', 'proteinIds': None},
    ]
    got = run(tmp_path, target=target)
    assert got.rows() == [
        ('ENSG_A', True, GENEA_TISSUES + GENEB_TISSUES),
        ('ENSG_B', False, GENEB_TISSUES),
    ]


def test_identical_tissues_from_two_symbols_kept_once(tmp_path):
    files = {
        'essential_genes': 'Essentials\nGENEA (1)\nGENEX (9)\n',
        'gene_effects': ',GENEA (1),GENEX (9)\nACH-1,-1.5,-1.5\nACH-2,-0.25,-0.25\nACH-3,0.3,0.3\nACH-4,-2.0,-2.0\n',
        'gene_expression': (
            ',SequencingID,ModelConditionID,ModelID,IsDefaultEntryForMC,IsDefaultEntryForModel,GENEA (1),GENEX (9)\n'
            '0,CDS-1,MC-1,ACH-1,Yes,Yes,4.5,4.5\n1,CDS-2,MC-2,ACH-2,Yes,Yes,0.0,0.0\n'
        ),
        'mutation_hotspots': f'{HEADER},GENEB (2)\n0,CDS-1,ACH-1,MC-1,Yes,Yes,0.0\n',
        'mutation_damaging': f'{HEADER},GENEA (1),GENEX (9)\n0,CDS-1,ACH-1,MC-1,Yes,Yes,1.0,1.0\n',
    }
    target = [{'id': 'ENSG_A', 'approvedSymbol': 'GENEA', 'proteinIds': [{'id': 'GENEX', 'source': 'x'}]}]
    got = run(tmp_path, files=files, target=target)
    assert got.rows() == [('ENSG_A', True, GENEA_TISSUES)]


# ---------------------------------------------------------------------------
# DepMap parsing
# ---------------------------------------------------------------------------


def test_melt_id_column_and_float32():
    effects = melt_matrix(csv(',GENEA (1),GENEB\nACH-1,-1.5,0.1\n'), 'geneEffect').collect()
    assert effects.schema == pl.Schema({'depmapId': pl.String, 'targetSymbol': pl.String, 'geneEffect': pl.Float32})
    assert effects.sort('targetSymbol').rows() == [('ACH-1', 'GENEA', -1.5), ('ACH-1', 'GENEB', f32(0.1))]

    expression = melt_matrix(csv(FILES['gene_expression']), 'expression').collect()
    assert expression.columns == ['depmapId', 'targetSymbol', 'expression']
    assert expression.sort('depmapId', 'targetSymbol').rows() == [
        ('ACH-1', 'GENEA', 4.5),
        ('ACH-1', 'GENEB', 1.25),
        ('ACH-2', 'GENEA', 0.0),
        ('ACH-2', 'GENEB', 2.5),
    ]


def test_melt_float32_rounds_through_float64():
    got = melt_matrix(csv(',GENEA (1)\nACH-1,-0.1234567890123456789\n'), 'geneEffect').collect()
    assert got['geneEffect'].to_list() == [f32(-0.1234567890123456789)]


ODD = {
    ' -1.5': -1.5,
    '-1.5 ': -1.5,
    '\t-1.5': -1.5,
    'abc': None,
    '1e-3': f32(0.001),
    'inf': float('inf'),
    'Infinity': float('inf'),
    '-Inf': float('-inf'),
    '1e400': float('inf'),
    '"-1.5"': -1.5,
    '+1.5': 1.5,
    '1.5d': 1.5,
    '0x10': None,
    '': None,
    '""': None,
}


@pytest.mark.parametrize('value', list(ODD))
def test_melt_odd_numeric_strings(tmp_path, value):
    got = run(tmp_path, files={'gene_effects': f',GENEA (1)\nACH-1,{value}\nACH-2,-0.25\n'}, settings={})
    screens = {s['depmapId']: s['geneEffect'] for t in got['depMapEssentiality'][0] for s in t['screens']}
    assert screens.get('ACH-1') == ODD[value]
    assert ('ACH-1' in screens) == (ODD[value] is not None)


@pytest.mark.parametrize('value', ['NaN', 'nan'])
def test_melt_nan_kept(tmp_path, value):
    got = run(tmp_path, files={'gene_effects': f',GENEA (1)\nACH-1,{value}\n'}, settings={})
    assert np.isnan(got['depMapEssentiality'][0][0]['screens'][0]['geneEffect'])


def test_cell_line_name_falls_back_to_disease_cells():
    models = read_models(csv(FILES['models'])).collect()
    assert models.rows() == [
        ('ACH-1', 'Line1', 'SIDM1', 'lung', 'Lung Cancer'),
        ('ACH-2', 'CCLE2', 'SIDM2', 'breast', 'Breast Cancer'),
        ('ACH-3', 'Melanoma cells', None, 'skin', 'Melanoma'),
        ('ACH-4', None, 'SIDM4', 'skin', None),
    ]
    assert models.columns == ['depmapId', 'cellLineName', 'modelId', 'tissueFromSource', 'diseaseFromSource']


def test_quoted_empty_string_is_null(tmp_path):
    models = FILES['models'].replace('ACH-3,,,Skin', 'ACH-3,"","",Skin')
    got = run(tmp_path, files={'models': models})
    assert got.rows() == [('ENSG_A', True, GENEA_TISSUES), ('ENSG_B', False, GENEB_TISSUES)]


def screens_of(**files: str) -> pl.DataFrame:
    frames = {name: csv(text) for name, text in (FILES | files).items()}
    return build_screens(**frames, keep_only_essentials=False).collect()


def test_mutation_priority_damaging_over_hotspot():
    got = screens_of().select('targetSymbol', 'depmapId', 'mutation').sort('targetSymbol', 'depmapId')
    assert got.rows() == [
        ('GENEA', 'ACH-1', 'damaging'),
        ('GENEA', 'ACH-2', None),
        ('GENEA', 'ACH-3', None),
        ('GENEA', 'ACH-4', None),
        ('GENEB', 'ACH-1', None),
        ('GENEB', 'ACH-3', 'hotspot'),
        ('GENEB', 'ACH-4', None),
    ]


def test_null_gene_effect_dropped():
    got = screens_of()
    assert got.filter(pl.col('targetSymbol') == 'GENEB')['depmapId'].sort().to_list() == ['ACH-1', 'ACH-3', 'ACH-4']
    assert got['geneEffect'].null_count() == 0


def test_tissue_name_defaults_to_other():
    got = screens_of().select('depmapId', 'tissueId', 'tissueName').unique().sort('depmapId')
    assert got.rows() == [
        ('ACH-1', 'UBERON_L', 'lung'),
        ('ACH-2', 'UBERON_B', 'breast'),
        ('ACH-3', None, 'other'),
        ('ACH-4', None, 'other'),
    ]


def test_duplicate_symbol_columns_and_model_rows_collapse_as_a_set(tmp_path):
    files = {
        'gene_effects': ',GENEA (1),GENEA (77)\nACH-1,-1.5,-1.5\n',
        'gene_expression': (
            ',SequencingID,ModelConditionID,ModelID,IsDefaultEntryForMC,IsDefaultEntryForModel,GENEA (1)\n'
            '0,CDS-1,MC-1,ACH-1,Yes,Yes,4.5\n1,CDS-9,MC-9,ACH-1,No,No,3.0\n'
        ),
        'mutation_hotspots': f'{HEADER},GENEB (2)\n0,CDS-1,ACH-1,MC-1,Yes,Yes,0.0\n',
        'mutation_damaging': f'{HEADER},GENEA (1)\n0,CDS-1,ACH-1,MC-1,Yes,Yes,2.0\n1,CDS-9,ACH-1,MC-9,No,No,1.0\n',
    }
    got = run(tmp_path, files=files)
    lung = {
        'tissueId': 'UBERON_L',
        'tissueName': 'lung',
        'screens': [
            screen('ACH-1', 'Line1', 'Lung Cancer', 'SIDM1', 'damaging', -1.5, 3.0),
            screen('ACH-1', 'Line1', 'Lung Cancer', 'SIDM1', 'damaging', -1.5, 4.5),
        ],
    }
    assert got.rows() == [('ENSG_A', True, [lung])]


def test_keep_only_essentials_setting(tmp_path):
    assert run(tmp_path, settings={}, out='default')['targetId'].to_list() == ['ENSG_A']
    assert run(tmp_path, settings={'keep_only_essentials': True}, out='true')['targetId'].to_list() == ['ENSG_A']
    assert run(tmp_path, settings={'keep_only_essentials': False}, out='false')['targetId'].to_list() == [
        'ENSG_A',
        'ENSG_B',
    ]


def test_full_output_and_schema(tmp_path):
    got = run(tmp_path)
    assert got.schema == OUTPUT_SCHEMA
    assert got.rows() == [('ENSG_A', True, GENEA_TISSUES), ('ENSG_B', False, GENEB_TISSUES)]








def test_gene_essentiality_null_and_empty_lists(tmp_path):
    target = [
        {'id': 'ENSG_A', 'approvedSymbol': 'GENEA', 'proteinIds': []},
        {'id': 'ENSG_B', 'approvedSymbol': 'GENEB', 'proteinIds': None},
        {'id': 'ENSG_N', 'approvedSymbol': None, 'proteinIds': None},
        {'id': 'ENSG_P', 'approvedSymbol': None, 'proteinIds': [{'id': None, 'source': 'x'}]},
    ]
    got = run(tmp_path, target=target)
    assert got.rows() == [('ENSG_A', True, GENEA_TISSUES), ('ENSG_B', False, GENEB_TISSUES)]


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------







