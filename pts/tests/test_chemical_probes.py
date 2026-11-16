"""Tests for the chemical_probes transformer."""

import math
from pathlib import Path

import openpyxl
import polars as pl
import pytest

from pts.transformers.chemical_probes import (
    PROBES_SETS,
    chemical_probes,
    collapse_cols_data_in_array,
    convert_stringified_array_to_array,
    generate_chemical_probes_evidence,
    process_drugs_xrefs,
    process_probes_data,
    process_probes_sets_data,
    process_probes_targets_data,
    process_scores,
    process_targets_xrefs,
    replace_missing_placeholder,
    resolve_targets,
)
from pts.transformers.utils.target import ensg_lookup

HQ = 'High-quality chemical probes'
SGC = 'SGC Probes'
CPO = 'Chemical Probes.org'

PROTEIN_IDS = pl.List(pl.Struct({'id': pl.String, 'source': pl.String}))
TARGET_SCHEMA = {'id': pl.String, 'approvedSymbol': pl.String, 'proteinIds': PROTEIN_IDS}
EVIDENCE_SCHEMA = {'targetFromSourceId': pl.String, 'id': pl.String, 'drugFromSourceId': pl.String, 'drugId': pl.String}

OUTPUT_SCHEMA = {
    'targetFromSourceId': pl.String,
    'drugFromSource': pl.String,
    'drugFromSourceId': pl.String,
    'drugId': pl.String,
    'mechanismOfAction': pl.List(pl.String),
    'origin': pl.List(pl.String),
    'control': pl.String,
    'isHighQuality': pl.Boolean,
    'probesDrugsScore': pl.Float64,
    'scoreInCells': pl.Float64,
    'scoreInOrganisms': pl.Float64,
    'urls': pl.List(pl.Struct({'niceName': pl.String, 'url': pl.String})),
    'targetId': pl.String,
}


def target(rows: list[tuple]) -> pl.LazyFrame:
    return pl.LazyFrame(
        [
            (ensg, symbol, None if proteins is None else [{'id': p, 'source': 'uniprot'} for p in proteins])
            for ensg, symbol, proteins in rows
        ],
        schema=TARGET_SCHEMA,
        orient='row',
    )


def evidence(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=EVIDENCE_SCHEMA, orient='row')


def resolve(evidence_rows: list[tuple], target_rows: list[tuple]) -> pl.DataFrame:
    return resolve_targets(evidence(evidence_rows), ensg_lookup(target(target_rows)).collect())


def strings(column: str, values: list) -> pl.DataFrame:
    return pl.DataFrame({column: values}, schema={column: pl.String})


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


def test_ensg_lookup_output_columns():
    assert ensg_lookup(target([('ENSG00000001', 'GENE1', [])])).collect().columns == ['ensgId', 'name']


def test_ensg_lookup_includes_symbol_and_protein_id():
    lut = ensg_lookup(target([('ENSG00000001', 'GENE1', ['P12345'])])).collect()
    assert sorted(lut['name']) == ['GENE1', 'P12345']


def test_ensg_lookup_handles_empty_protein_ids():
    lut = ensg_lookup(target([('ENSG00000002', 'GENE2', [])])).collect()
    assert lut.rows() == [('ENSG00000002', 'GENE2')]


def test_ensg_lookup_handles_null_protein_ids():
    """Null proteinIds (non-coding genes, e.g. microRNAs) does not wipe out the symbol."""
    lut = ensg_lookup(target([('ENSG00000003', 'MIR122', None)])).collect()
    assert lut.rows() == [('ENSG00000003', 'MIR122')]


# ---------------------------------------------------------------------------
# resolve_targets
# ---------------------------------------------------------------------------


def test_resolve_targets_resolves_symbol_to_ensg():
    result = resolve([('GENE1', 'probe1', 'CP001', None)], [('ENSG00000001', 'GENE1', [])])
    assert result.columns == ['targetFromSourceId', 'id', 'drugFromSourceId', 'drugId', 'targetId']
    assert result['targetId'].to_list() == ['ENSG00000001']


def test_resolve_targets_resolves_protein_id_to_ensg():
    result = resolve([('P12345', 'probe1', 'CP001', None)], [('ENSG00000002', 'GENE2', ['P12345'])])
    assert result['targetId'].to_list() == ['ENSG00000002']


def test_resolve_targets_drops_unresolvable_rows():
    assert resolve([('UNKNOWN', 'probe1', 'CP001', None)], [('ENSG00000001', 'GENE1', [])]).height == 0


def test_resolve_targets_resolves_symbol_for_non_coding_gene():
    """Probes for non-coding genes (null proteinIds) still resolve by symbol."""
    result = resolve([('MIR122', 'probe1', 'CP001', None)], [('ENSG_MIR122', 'MIR122', None)])
    assert result['targetId'].to_list() == ['ENSG_MIR122']


def test_resolve_targets_retains_target_from_source_id():
    result = resolve([('GENE1', 'probe1', 'CP001', None)], [('ENSG00000001', 'GENE1', [])])
    assert result['targetFromSourceId'].to_list() == ['GENE1']


def test_resolve_targets_multiple_probes_same_target():
    result = resolve(
        [('GENE1', 'probe_a', 'CP001', None), ('GENE1', 'probe_b', 'CP001', None)], [('ENSG00000001', 'GENE1', [])]
    )
    assert result['targetId'].to_list() == ['ENSG00000001', 'ENSG00000001']


def test_symbol_with_two_ensg_ids_fans_out():
    """An id shared by two targets yields one row per target, as spark's array_contains join does."""
    result = resolve(
        [('Q99999', 'probe1', 'CP001', None), ('M1', 'probe2', 'CP002', None)],
        [('ENSG_M1', 'M1', ['Q99999', 'Q99999']), ('ENSG_M2', 'M2', ['Q99999']), ('ENSG_M3', 'M1', None)],
    )
    assert sorted(result.select('drugFromSourceId', 'targetId').rows()) == [
        ('CP001', 'ENSG_M1'),
        ('CP001', 'ENSG_M2'),
        ('CP002', 'ENSG_M1'),
        ('CP002', 'ENSG_M3'),
    ]


# ---------------------------------------------------------------------------
# replace_missing_placeholder / process_scores / convert_stringified_array_to_array
# ---------------------------------------------------------------------------


def test_replace_missing_placeholder_nulls_tokens():
    """'-' and any case of 'nan', padded with spaces, become null; other values pass through unchanged."""
    values = ['-', 'NaN', 'nan', 'NAN', ' NaN ', '- ', '\tNaN', 'BI-6953', ' ', None]
    df = strings('control_name', values).select(replace_missing_placeholder(pl.col('control_name')).alias('control'))
    assert df['control'].to_list() == [None, None, None, None, None, None, '\tNaN', 'BI-6953', ' ', None]


def test_scores_bad_text_is_null():
    """Spark's try_cast to double: surrounding control chars trimmed, Java suffixes and literals accepted."""
    values = [' 1.5 ', '\t2', '2\n', '\x7f2', '\xa02', 'abc', '-', '', ' ', '1,5', '1_000', '0x10', None]
    assert strings('s', values).select(process_scores('s'))['s'].to_list() == [
        1.5, 2.0, 2.0, None, None, None, None, None, None, None, None, None, None,
    ]
    special = ['1e3', '1.5d', '1.5F', '1.5e3d', '1.5dd', '.5', '5.', '+3', 'inf', ' inf ', '-Infinity', 'Infinityd']
    assert strings('s', special).select(process_scores('s'))['s'].to_list() == [
        1000.0, 1.5, 1.5, 1500.0, None, 0.5, 5.0, 3.0, float('inf'), float('inf'), float('-inf'), None,
    ]
    signed = ['NaN', 'NAN', '\tnan', '+NaN', '-NaN', '+nan', '-nan', '-NAN', '+Nan']
    nans = strings('s', signed).select(process_scores('s'))['s'].to_list()
    assert [None if v is None else math.isnan(v) for v in nans] == [True] * 5 + [None] * 4


def test_convert_stringified_array_to_array():
    df = strings('a', ["['x', 'y']", '[]', None]).select(convert_stringified_array_to_array('a'))
    assert df['a'].to_list() == [['x', 'y'], [''], None]


def _write_xlsx(path: Path, sheets: dict[str, dict[str, list]]) -> str:
    """Write a workbook with one sheet per entry, each given as columns of cell values."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, columns in sheets.items():
        ws = wb.create_sheet(name)
        ws.append(list(columns))
        for row in zip(*columns.values(), strict=True):
            ws.append(list(row))
    wb.save(path)
    return str(path)


def _records(rows: list[dict]) -> dict[str, list]:
    return {c: [r[c] for r in rows] for c in rows[0]}

# ---------------------------------------------------------------------------
# collapse_cols_data_in_array / PROBES_SETS
# ---------------------------------------------------------------------------

# One-hot datasource columns of the PROBES sheet, read from the upstream Probes &
# Drugs spreadsheet used by the 2026.09 release
# (input/target/chemicalprobes/probes.xlsx). Kept verbatim so that a name drifting
# upstream shows up here as a failure rather than as a missing-column crash in
# production.
PROBES_SHEET_DATASOURCE_COLUMNS = [
    'Bromodomains chemical toolbox',
    'Chemical Probes for Understudied Kinases',
    'Chemical Probes.org',
    'Gray Laboratory Probes',
    'High-quality chemical probes',
    'MLP Probes',
    'Nature Chemical Biology Probes',
    'Open Science Probes',
    'opnMe Portal',
    'Protein methyltransferases chemical toolbox',
    'SGC Probes',
    'A Collection of Useful Nuisance Compounds (CONS) for Interrogation of Bioassay Integrity',
    'Concise Guide to Pharmacology 2025/26',
    'Kinase Chemogenomic Set (KCGS)',
    'Kinase Inhibitors (best-in-class)',
    'Novartis Chemogenetic Library (NIBR MoA Box)',
    'Nuisance compounds in cellular assays',
]


def _probes_sheet_df(memberships: list[str]) -> pl.DataFrame:
    """One PROBES-sheet-shaped row with the real upstream column names, as strings."""
    row = {c: ('1.0' if c in memberships else None) for c in PROBES_SHEET_DATASOURCE_COLUMNS}
    return pl.DataFrame([{'pdid': 'PD000001', **row}], schema=dict.fromkeys(['pdid', *row], pl.String))


def test_probes_sets_are_all_columns_of_the_probes_sheet():
    """Every PROBES_SETS entry resolves against the upstream PROBES sheet columns."""
    result = collapse_cols_data_in_array(_probes_sheet_df([SGC]), PROBES_SETS, 'datasourceIds')
    assert result['datasourceIds'].to_list() == [[SGC]]


def test_collapse_cols_data_in_array_collects_every_membership():
    """A probe in several sets collects all of them, and only them, in PROBES_SETS order."""
    memberships = ['Concise Guide to Pharmacology 2025/26', CPO, HQ]
    result = collapse_cols_data_in_array(_probes_sheet_df(memberships), PROBES_SETS, 'datasourceIds')
    assert result['datasourceIds'].to_list() == [[CPO, HQ, 'Concise Guide to Pharmacology 2025/26']]


def test_collapse_cols_data_in_array_skips_names_without_a_space():
    df = pl.DataFrame({'pdid': ['P'], 'single': ['1'], 'two words': ['1']})
    result = collapse_cols_data_in_array(df, ['single', 'two words'], 'sets')
    assert result['sets'].to_list() == [['two words']]


def test_the_cons_set_is_collected_as_its_own_datasource():
    """CONS is a set of its own, not a rename of 'Nuisance compounds in cellular assays'."""
    cons = 'A Collection of Useful Nuisance Compounds (CONS) for Interrogation of Bioassay Integrity'
    older = 'Nuisance compounds in cellular assays'
    assert cons in PROBES_SHEET_DATASOURCE_COLUMNS
    assert cons in PROBES_SETS
    assert older in PROBES_SETS, 'adopting CONS must not displace the set it sits alongside'

    row = collapse_cols_data_in_array(_probes_sheet_df([cons]), PROBES_SETS, 'datasourceIds')
    assert row['datasourceIds'].to_list() == [[cons]]

    both = collapse_cols_data_in_array(_probes_sheet_df([cons, older]), PROBES_SETS, 'datasourceIds')
    assert sorted(both['datasourceIds'].to_list()[0]) == sorted([cons, older])


def _probes_xlsx(path: Path, flag) -> str:
    """Write a PROBES workbook with the real upstream set columns.

    PD-1 is in every set and both origins, PD-2 in SGC Probes only and is experimental,
    PD-3 has no target and is dropped. Set cells hold `flag` or are blank.
    """
    every = [flag, None, flag]
    sgc_only = {c: [flag, flag if c == SGC else None, None] for c in PROBES_SHEET_DATASOURCE_COLUMNS}
    frame = {
        'pdid': ['PD-1', 'PD-2', 'PD-3'],
        'compound_name': ['cmpd-1', 'cmpd-2', 'cmpd-3'],
        'target': ['BRD4', 'EGFR', None],
        'control_name': ['ctrl-1', '-', None],
        'control_smiles': ['CC', 'CC', 'CC'],
        **sgc_only,
        'experimental probe': [flag, flag, flag],
        'calculated probe': every,
        # unused, and mixing numbers and text
        'mw': [1.5, '-', None],
    }
    return _write_xlsx(path, {'PROBES': frame})


@pytest.mark.parametrize('flag', [1, 1.0, True], ids=['int', 'float', 'bool'])
def test_probes_data_collects_every_set_and_origin(tmp_path, flag):
    """Every PROBES_SETS column is read from the sheet, whatever type its flags have."""
    xlsx = _probes_xlsx(tmp_path / 'probes.xlsx', flag)
    if flag is True:
        sheet = pl.read_excel(xlsx, sheet_name='PROBES', engine='openpyxl', infer_schema_length=None)
        assert sheet.schema['experimental probe'] == pl.Boolean, 'the fixture must hold real booleans'

    rows = process_probes_data(xlsx)
    assert rows.columns == ['pdid', 'drugFromSource', 'origin', 'isHighQuality', 'datasourceId', 'control']
    memberships = {
        pdid: (sorted(g['datasourceId']), g['origin'].to_list()[0], set(g['isHighQuality']), set(g['control']))
        for (pdid,), g in rows.group_by('pdid')
    }
    every_set = sorted(s for s in PROBES_SETS if s != HQ)
    assert memberships == {
        'PD-1': (every_set, ['experimental', 'calculated'], {True}, {'ctrl-1'}),
        'PD-2': ([SGC], ['experimental'], {False}, {None}),
    }


# ---------------------------------------------------------------------------
# PROBES TARGETS sheet
# ---------------------------------------------------------------------------


def _probes_targets_xlsx(path: Path, extra_columns: dict | None = None) -> str:
    """Write a minimal PROBES TARGETS workbook, shaped like the upstream export."""
    frame = {
        'pdid': ['PD-1', 'PD-2', 'PD-3'],
        'gene_name': ['BRD4', 'EGFR', '-'],
        'organism': ['Homo sapiens', 'Homo sapiens', 'Homo sapiens'],
        'target': ['BRD4', 'EGFR', 'KRAS'],
        'action': ['inhibitor', '-', '-'],
        'control_smiles': ['CC', 'CC', 'CC'],
        'P&D probe-likeness score': [1.0, 2.0, 3.0],
        'Cells score (Chemical Probes.org)': [1.0, 2.0, 3.0],
        'Organisms score (Chemical Probes.org)': [1.0, 2.0, 3.0],
        **(extra_columns or {}),
    }
    return _write_xlsx(path, {'PROBES TARGETS': frame})


def test_probes_targets_survives_a_mixed_type_column(tmp_path):
    """A sheet column mixing numbers and text must not stop the step; the score's '-' becomes null."""
    xlsx = _probes_targets_xlsx(
        tmp_path / 'probes.xlsx',
        extra_columns={'covalent': [1.0, '-', None], 'Cells score (Chemical Probes.org)': [1.5, '-', None]},
    )
    df = process_probes_targets_data(xlsx)
    rows = {r['pdid']: r for r in df.to_dicts()}

    assert df.columns == [
        'pdid', 'targetFromSource', 'mechanismOfAction', 'probesDrugsScore', 'scoreInCells', 'scoreInOrganisms',
    ]
    assert sorted(rows) == ['PD-1', 'PD-2'], 'only the row without a gene_name should be dropped'
    assert rows['PD-1']['scoreInCells'] == 1.5
    assert rows['PD-2']['scoreInCells'] is None
    assert rows['PD-1']['probesDrugsScore'] == 1.0


def test_probes_targets_nulls_missing_action(tmp_path):
    """A blank or '-' action is null, not a single-element list; ';' splits without trimming."""
    xlsx = _probes_targets_xlsx(tmp_path / 'probes.xlsx', extra_columns={'action': [None, 'inhibitor; agonist;', '-']})
    rows = {r['pdid']: r['mechanismOfAction'] for r in process_probes_targets_data(xlsx).to_dicts()}
    assert rows == {'PD-1': None, 'PD-2': ['inhibitor', ' agonist', '']}


def test_probes_targets_drops_blank_gene_name_and_other_organisms(tmp_path):
    xlsx = _probes_targets_xlsx(
        tmp_path / 'probes.xlsx',
        extra_columns={
            'gene_name': ['BRD4', None, 'KRAS'],
            'organism': ['Homo sapiens', 'Homo sapiens', 'Mus musculus'],
        },
    )
    assert process_probes_targets_data(xlsx)['pdid'].to_list() == ['PD-1']


def test_process_drugs_xrefs_keeps_chembl_ids_only(tmp_path):
    csv = tmp_path / 'links.csv'
    csv.write_text('compound_id,pdid,name,ChEMBL\n1,PD-1,"multi\nline",CHEMBL1\n2,PD-2,b,CHEMBL2\n3,PD-3,c,\n')
    got = process_drugs_xrefs(str(csv), pl.DataFrame({'id': ['CHEMBL1', 'CHEMBL3']}))
    assert got.columns == ['pdid', 'drugId']
    assert got.rows() == [('PD-1', 'CHEMBL1')]


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


PROBES_TARGETS_HEADER = [
    'pdid', 'gene_name', 'organism', 'target', 'action', 'P&D probe-likeness score',
    'Cells score (Chemical Probes.org)', 'Organisms score (Chemical Probes.org)',
]  # fmt: skip


def _sets(*names: str) -> dict:
    return {c: (1.0 if c in names else None) for c in PROBES_SETS}


def _write_inputs(src: Path) -> dict[str, str]:
    """A workbook, links csv, chembl molecules and a two-chunk target, covering the traps.

    PD-A is in SGC, Chemical Probes.org and the HQ set, with a duplicated PROBES TARGETS
    row; PD-HQ is only in the HQ set; PD-NONE is in no set; PD-B maps to an accession
    shared by two targets; PD-MISSING is not in the PROBES sheet.
    """
    src.mkdir()
    probes = _records([
        {'pdid': 'PD-A', 'compound_name': 'cmpd-a', 'target': 'x', 'control_name': 'ctrl', **_sets(SGC, HQ, CPO),
         'experimental probe': 1.0, 'calculated probe': None},
        {'pdid': 'PD-HQ', 'compound_name': 'cmpd-hq', 'target': 'x', 'control_name': '-', **_sets(HQ),
         'experimental probe': 1.0, 'calculated probe': 1.0},
        {'pdid': 'PD-NONE', 'compound_name': 'cmpd-none', 'target': 'x', 'control_name': None, **_sets(),
         'experimental probe': None, 'calculated probe': 1.0},
        {'pdid': 'PD-B', 'compound_name': 'cmpd-b', 'target': 'x', 'control_name': ' NaN ', **_sets(SGC),
         'experimental probe': None, 'calculated probe': None},
        {'pdid': 'PD-NOTGT', 'compound_name': 'cmpd-n', 'target': None, 'control_name': None, **_sets(SGC),
         'experimental probe': 1.0, 'calculated probe': None},
    ])  # fmt: skip
    probes_targets = _records([
        dict(zip(PROBES_TARGETS_HEADER, row, strict=True))
        for row in [
            ('PD-A', 'BRD4', 'Homo sapiens', 'Bromo 4', 'inhibitor;antagonist', ' 1.5 ', '\t2', 'abc'),
            ('PD-A', 'BRD4', 'Homo sapiens', 'Bromo 4', 'inhibitor;antagonist', ' 1.5 ', '\t2', 'abc'),
            ('PD-HQ', 'EGFR', 'Homo sapiens', 'EGF receptor', '-', 'NaN', '1e3', '-'),
            ('PD-NONE', 'KRAS', 'Homo sapiens', 'K-Ras', None, 'Infinity', '1.5d', ''),
            ('PD-B', 'MULTI', 'Homo sapiens', 'Multi', ' ', '3', 'inf', '-1'),
            ('PD-MISSING', 'BRD4', 'Homo sapiens', 'Bromo 4', 'agonist', '1', '2', '3'),
            ('PD-A', 'BRD4', 'Mus musculus', 'Bromo 4', 'inhibitor', '1', '1', '1'),
        ]
    ])  # fmt: skip
    compoundsets = {
        'INDEX': [1, 2, 3, 4],
        'COMPOUNDSET': [SGC, CPO, HQ, 'Bromodomains chemical toolbox'],
        'SOURCE_URL': ['http://sgc', 'https://cpo', 'http://hq', 'n/a'],
    }
    targets = {
        'target': ['Bromo 4', 'EGF receptor', 'K-Ras', 'Multi', None],
        'uniprot': ['O60885', 'P00533', 'P01116', 'Q99999', None],
    }
    sheets = {'PROBES': probes, 'PROBES TARGETS': probes_targets, 'COMPOUNDSETS': compoundsets, 'TARGETS': targets}
    _write_xlsx(src / 'probes.xlsx', sheets)
    (src / 'links.csv').write_text(
        'compound_id,pdid,ChEMBL\n1,PD-A,CHEMBL1\n2,PD-A,CHEMBL1\n3,PD-B,CHEMBL2\n4,PD-B,\n5,PD-HQ,CHEMBL3\n'
    )
    pl.DataFrame({'id': ['CHEMBL1', 'CHEMBL3']}).write_parquet(src / 'chembl_molecule.parquet')
    first = target([
        ('ENSG_BRD4', 'BRD4', ['O60885']),
        ('ENSG_EGFR', 'EGFR', ['P00533']),
        ('ENSG_KRAS', 'KRAS', ['P01116']),
    ]).collect()
    second = target([('ENSG_M1', 'M1', ['Q99999', 'Q99999']), ('ENSG_M2', 'M2', ['Q99999']), ('ENSG_X', 'X', None)])
    pl.concat([first, second.collect()], rechunk=False).write_parquet(src / 'target.parquet')
    return {
        'probes_excel': str(src / 'probes.xlsx'),
        'drugs_csv': str(src / 'links.csv'),
        'chembl_molecule': str(src / 'chembl_molecule.parquet'),
        'target': str(src / 'target.parquet'),
    }


def _run(tmp_path: Path, name: str) -> pl.DataFrame:
    sources = _write_inputs(tmp_path / f'src-{name}')
    chemical_probes(sources, str(tmp_path / name), {}, None)
    return pl.read_parquet(tmp_path / name)


def _by_probe(df: pl.DataFrame) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in df.to_dicts():
        out.setdefault(row['drugFromSourceId'], []).append(row)
    return out


def test_chemical_probes_end_to_end(tmp_path):
    df = _run(tmp_path, 'a')
    assert df.schema == pl.Schema(OUTPUT_SCHEMA)
    rows = _by_probe(df)
    assert sorted(rows) == ['PD-A', 'PD-B', 'PD-HQ', 'PD-MISSING', 'PD-NONE']
    assert rows['PD-A'] == [
        {
            'targetFromSourceId': 'O60885',
            'drugFromSource': 'cmpd-a',
            'drugFromSourceId': 'PD-A',
            'drugId': 'CHEMBL1',
            'mechanismOfAction': ['inhibitor', 'antagonist'],
            'origin': ['experimental'],
            'control': 'ctrl',
            'isHighQuality': True,
            'probesDrugsScore': 1.5,
            'scoreInCells': 2.0,
            'scoreInOrganisms': None,
            'urls': [{'niceName': CPO, 'url': 'https://cpo'}, {'niceName': SGC, 'url': 'http://sgc'}],
            'targetId': 'ENSG_BRD4',
        }
    ]
    assert rows['PD-MISSING'][0] | {'urls': None} == {
        'targetFromSourceId': 'O60885',
        'drugFromSource': None,
        'drugFromSourceId': 'PD-MISSING',
        'drugId': None,
        'mechanismOfAction': ['agonist'],
        'origin': None,
        'control': None,
        'isHighQuality': None,
        'probesDrugsScore': 1.0,
        'scoreInCells': 2.0,
        'scoreInOrganisms': 3.0,
        'urls': None,
        'targetId': 'ENSG_BRD4',
    }
    assert rows['PD-MISSING'][0]['urls'] == [{'niceName': None, 'url': None}]
    assert df.equals(df.sort(df.columns))

    _run(tmp_path, 'b')
    files_a = sorted((tmp_path / 'a').iterdir())
    files_b = sorted((tmp_path / 'b').iterdir())
    assert [p.name for p in files_a] == [p.name for p in files_b]
    assert [p.read_bytes() for p in files_a] == [p.read_bytes() for p in files_b]


def test_probe_in_no_non_hq_set_is_dropped(tmp_path):
    """Its PROBES rows are exploded away, but spark's left join keeps its target rows with null probe fields."""
    rows = _by_probe(_run(tmp_path, 'a'))
    expected = [('PD-HQ', 'CHEMBL3', (None, 1000.0, None)), ('PD-NONE', None, (float('inf'), 1.5, None))]
    for pdid, drug_id, scores in expected:
        (row,) = rows[pdid]
        assert (row['drugFromSource'], row['origin'], row['isHighQuality'], row['control']) == (None, None, None, None)
        assert row['drugId'] == drug_id
        assert row['mechanismOfAction'] is None
        assert (row['probesDrugsScore'], row['scoreInCells'], row['scoreInOrganisms']) == scores
        assert row['urls'] == [{'niceName': None, 'url': None}]


def test_urls_sorted_and_deduplicated(tmp_path):
    xlsx = _write_inputs(tmp_path / 'src')['probes_excel']
    probes = pl.DataFrame(
        {'pdid': ['P1'] * 4, 'drugFromSource': ['c'] * 4, 'origin': [['experimental']] * 4,
         'isHighQuality': [False] * 4, 'datasourceId': [SGC, CPO, SGC, 'Bromodomains chemical toolbox'],
         'control': [None] * 4}
    )  # fmt: skip
    probes_targets = pl.DataFrame(
        {'pdid': ['P1', 'P1'], 'targetFromSource': ['Bromo 4', 'Bromo 4'], 'mechanismOfAction': [['x'], ['x']],
         'probesDrugsScore': [1.0, 1.0], 'scoreInCells': [None, None], 'scoreInOrganisms': [None, None]},
        schema_overrides={'scoreInCells': pl.Float64, 'scoreInOrganisms': pl.Float64},
    )  # fmt: skip
    got = generate_chemical_probes_evidence(
        probes,
        probes_targets,
        process_probes_sets_data(xlsx),
        process_targets_xrefs(xlsx),
        pl.DataFrame({'pdid': ['P1', 'P1'], 'drugId': ['CHEMBL1', 'CHEMBL1']}),
    )
    assert got.height == 1
    assert got['urls'].to_list() == [
        [
            {'niceName': 'Bromodomains chemical toolbox', 'url': None},
            {'niceName': CPO, 'url': 'https://cpo'},
            {'niceName': SGC, 'url': 'http://sgc'},
        ]
    ]


def test_chemical_probes_null_and_empty_lists(tmp_path):
    """No-flag origin is an empty list, a probe absent from PROBES has null lists, and a ' ' action is kept."""
    rows = _by_probe(_run(tmp_path, 'a'))
    assert [(r['origin'], r['mechanismOfAction'], r['isHighQuality']) for r in rows['PD-B']] == [([], [' '], False)] * 2
    assert rows['PD-HQ'][0]['origin'] is None
    assert 'ENSG_X' not in {r['targetId'] for rs in rows.values() for r in rs}


def test_symbol_with_two_ensg_ids_fans_out_end_to_end(tmp_path):
    rows = _by_probe(_run(tmp_path, 'a'))
    assert [r['targetId'] for r in rows['PD-B']] == ['ENSG_M1', 'ENSG_M2']
    assert rows['PD-B'][0] | {'targetId': None} == rows['PD-B'][1] | {'targetId': None}
