"""Tests for the pharmacovigilance transformer."""

import json
import sys
import zipfile
from math import log
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pts.schemas.pharmacovigilance import faers_schema, pharmacovigilance_schema
from pts.transformers.pharmacovigilance import (
    contingency_table,
    critical_value,
    critical_values,
    drug_names,
    normalise_event,
    pharmacovigilance,
    read_archive,
    read_blacklist,
    report_rows,
    significant_events,
)
from pts.transformers.utils.dataset import scan_dataset, write_dataset

ROW_COLUMNS = ['safetyreportid', 'chembl_id', 'event']


def _event(value: str) -> str:
    return pl.DataFrame({'e': [value]}).select(normalise_event(pl.col('e')))['e'][0]


def _drug(name: str = 'ASPIRIN', characterization: str | None = '1', **openfda) -> dict:
    drug: dict = {'drugcharacterization': characterization, 'medicinalproduct': name}
    if openfda:
        drug['openfda'] = openfda
    return drug


def _report(**overrides) -> dict:
    report: dict = {
        'safetyreportid': 'R1',
        'primarysource': {'qualification': '1'},
        'patient': {
            'reaction': [{'reactionmeddrapt': 'HEADACHE'}],
            'drug': [_drug()],
        },
    }
    report.update(overrides)
    return report


def _reports(*dicts: dict) -> pl.DataFrame:
    return pl.DataFrame({'results': [list(dicts)]}, schema=faers_schema).explode('results').unnest('results')


def _lookup(*pairs: tuple[str, str]) -> pl.DataFrame:
    return pl.DataFrame(
        {'chembl_id': [p[0] for p in pairs], 'drug_name': [p[1] for p in pairs]},
        schema={'chembl_id': pl.String(), 'drug_name': pl.String()},
    )


ASPIRIN = _lookup(('CHEMBL1', 'aspirin'))
NO_BLACKLIST = pl.DataFrame(schema={'event': pl.String()})


def _rows(reports: pl.DataFrame, drugs: pl.DataFrame = ASPIRIN, blacklist: pl.DataFrame = NO_BLACKLIST) -> list[tuple]:
    return sorted(report_rows(reports, drugs, blacklist).select(ROW_COLUMNS).rows())


def _patient(reactions: list[str], drugs: list[dict]) -> dict:
    return {'reaction': [{'reactionmeddrapt': r} for r in reactions], 'drug': drugs}


def test_normalise_event_replaces_caret_and_strips_only_spaces():
    # non-breaking space survives: spark trim strips spaces only
    assert _event(' Crohn^S disease ') == "crohn's disease"
    assert _event('\u00a0headache ') == '\u00a0headache'


def test_drug_names_lowercases_and_covers_all_three_sources():
    label = pl.Struct({'label': pl.String(), 'source': pl.String()})
    molecules = pl.DataFrame(
        {
            'id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3'],
            'name': ['Aspirin', 'Other', None],
            'tradeNames': [
                [{'label': 'Bayer', 'source': 'x'}, {'label': 'ASPIRIN', 'source': 'x'}],
                [],
                [{'label': 'Brand', 'source': 'x'}],
            ],
            'synonyms': [[{'label': 'Acetylsalicylic Acid', 'source': 'x'}], [], []],
        },
        schema={
            'id': pl.String(),
            'name': pl.String(),
            'tradeNames': pl.List(label),
            'synonyms': pl.List(label),
        },
    )
    result = drug_names(molecules)
    assert result.columns == ['chembl_id', 'drug_name']
    assert sorted(result.rows()) == [
        ('CHEMBL1', 'acetylsalicylic acid'),
        ('CHEMBL1', 'aspirin'),
        ('CHEMBL1', 'bayer'),
        ('CHEMBL2', 'other'),
        ('CHEMBL3', 'brand'),
    ]


def test_read_blacklist_normalises_lines():
    assert read_blacklist('  DEATH \nDrug Ineffective\n')['event'].to_list() == ['death', 'drug ineffective']


def test_read_blacklist_drops_blank_lines():
    assert read_blacklist('death\n\nnausea\n')['event'].to_list() == ['death', 'nausea']
    assert read_blacklist('death\n   \n')['event'].to_list() == ['death']


def test_rows_cross_product_of_reactions_and_drugs():
    patient = _patient(['HEADACHE', 'NAUSEA'], [_drug('ASPIRIN'), _drug('IBUPROFEN')])
    drugs = _lookup(('CHEMBL1', 'aspirin'), ('CHEMBL2', 'ibuprofen'))
    assert _rows(_reports(_report(patient=patient)), drugs) == [
        ('R1', 'CHEMBL1', 'headache'),
        ('R1', 'CHEMBL1', 'nausea'),
        ('R1', 'CHEMBL2', 'headache'),
        ('R1', 'CHEMBL2', 'nausea'),
    ]


def test_null_seriousnessdeath_is_kept_and_one_is_dropped():
    reports = _reports(
        _report(safetyreportid='NULL'),
        _report(safetyreportid='ZERO', seriousnessdeath='0'),
        _report(safetyreportid='DEAD', seriousnessdeath='1'),
    )
    assert [r[0] for r in _rows(reports)] == ['NULL', 'ZERO']


def test_qualification_4_and_null_are_dropped():
    reports = _reports(
        _report(safetyreportid='Q1'),
        _report(safetyreportid='Q3', primarysource={'qualification': '3'}),
        _report(safetyreportid='Q4', primarysource={'qualification': '4'}),
        _report(safetyreportid='QN', primarysource={'qualification': None}),
        _report(safetyreportid='NOSOURCE', primarysource=None),
    )
    assert [r[0] for r in _rows(reports)] == ['Q1', 'Q3']


def test_drugcharacterization_2_is_dropped():
    patient = _patient(['HEADACHE'], [_drug('ASPIRIN', '2'), _drug('IBUPROFEN', '1'), _drug('ASPIRIN', None)])
    drugs = _lookup(('CHEMBL1', 'aspirin'), ('CHEMBL2', 'ibuprofen'))
    assert _rows(_reports(_report(patient=patient)), drugs) == [('R1', 'CHEMBL2', 'headache')]


def test_blacklisted_event_is_dropped():
    patient = _patient(['DEATH', 'Headache '], [_drug()])
    blacklist = read_blacklist('Death\n')
    assert _rows(_reports(_report(patient=patient)), blacklist=blacklist) == [('R1', 'CHEMBL1', 'headache')]


def test_name_matches_through_brand_generic_and_substance():
    drugs = _lookup(('CHEMBL1', 'bayer'), ('CHEMBL2', 'acetylsalicylic acid'), ('CHEMBL3', 'salicylate'))
    patient = _patient(
        ['HEADACHE'],
        [
            _drug('UNKNOWN A', brand_name=['BAYER']),
            _drug('UNKNOWN B', generic_name=['Acetylsalicylic Acid']),
            _drug('UNKNOWN C', substance_name=['SALICYLATE']),
            _drug('UNKNOWN D', brand_name=['nothing']),
        ],
    )
    assert _rows(_reports(_report(patient=patient)), drugs) == [
        ('R1', 'CHEMBL1', 'headache'),
        ('R1', 'CHEMBL2', 'headache'),
        ('R1', 'CHEMBL3', 'headache'),
    ]


def test_empty_medicinalproduct_does_not_match_an_empty_name():
    drugs = _lookup(('CHEMBL1', ''), ('CHEMBL2', 'aspirin'))
    patient = _patient(['HEADACHE'], [_drug(''), _drug(None)])
    assert _rows(_reports(_report(patient=patient)), drugs) == []


def test_rows_are_distinct_when_two_names_map_to_one_drug():
    drugs = _lookup(('CHEMBL1', 'aspirin'), ('CHEMBL1', 'bayer'))
    patient = _patient(['HEADACHE'], [_drug('ASPIRIN', brand_name=['Bayer']), _drug('Bayer')])
    assert _rows(_reports(_report(patient=patient)), drugs) == [('R1', 'CHEMBL1', 'headache')]


def test_drug_without_openfda_block_still_links_by_medicinalproduct():
    patient = _patient(['HEADACHE'], [_drug('aspirin')])
    assert 'openfda' not in patient['drug'][0]
    assert _rows(_reports(_report(patient=patient))) == [('R1', 'CHEMBL1', 'headache')]


def test_null_or_empty_drug_and_reaction_lists_give_no_rows():
    reports = _reports(
        _report(safetyreportid='NODRUG', patient={'reaction': [{'reactionmeddrapt': 'X'}], 'drug': None}),
        _report(safetyreportid='EMPTYDRUG', patient={'reaction': [{'reactionmeddrapt': 'X'}], 'drug': []}),
        _report(safetyreportid='NOREACTION', patient={'reaction': None, 'drug': [_drug()]}),
        _report(safetyreportid='EMPTYREACTION', patient={'reaction': [], 'drug': [_drug()]}),
        _report(safetyreportid='NOPATIENT', patient=None),
    )
    assert _rows(reports) == []


def test_drugs_and_events_join_within_a_report_not_across_reports_sharing_an_id():
    first = _report(patient=_patient(['X'], [_drug('A')]))
    second = _report(patient=_patient(['Y'], [_drug('B')]))
    drugs = _lookup(('CHEMBL_A', 'a'), ('CHEMBL_B', 'b'))
    assert _rows(_reports(first, second), drugs) == [('R1', 'CHEMBL_A', 'x'), ('R1', 'CHEMBL_B', 'y')]


def test_empty_results_gives_empty_frame_with_the_output_columns():
    result = report_rows(_reports(), ASPIRIN, NO_BLACKLIST)
    assert result.columns == ROW_COLUMNS
    assert result.height == 0


def _write_zip(path: Path, results: list[dict]) -> str:
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('events.json', json.dumps({'meta': {'last_updated': '2026-01-01'}, 'results': results}))
    return str(path)


def test_read_archive_ignores_reportduplicate_in_either_shape(tmp_path):
    as_list = _report(safetyreportid='LIST', reportduplicate=[{'duplicatenumb': '1', 'duplicatesource': 'a'}])
    as_struct = _report(safetyreportid='STRUCT', reportduplicate={'duplicatenumb': '2', 'duplicatesource': 'b'})
    first = _write_zip(tmp_path / 'first.zip', [as_list])
    second = _write_zip(tmp_path / 'second.zip', [as_struct])
    assert sorted(read_archive(first, ASPIRIN, NO_BLACKLIST).rows()) == [('LIST', 'CHEMBL1', 'headache')]
    assert sorted(read_archive(second, ASPIRIN, NO_BLACKLIST).rows()) == [('STRUCT', 'CHEMBL1', 'headache')]


def test_read_archive_holds_only_the_reports_frame_while_reducing(tmp_path, monkeypatch):
    path = _write_zip(tmp_path / 'a.zip', [_report()])
    held: list = []

    def spy(reports, drugs, blacklist):
        caller = sys._getframe(1).f_locals
        arguments = (reports, drugs, blacklist)
        frames = [v for v in caller.values() if isinstance(v, (bytes, pl.DataFrame))]
        held.extend(v for v in frames if all(v is not arg for arg in arguments))
        return report_rows(reports, drugs, blacklist)

    monkeypatch.setattr('pts.transformers.pharmacovigilance.report_rows', spy)
    assert read_archive(path, ASPIRIN, NO_BLACKLIST).rows() == [('R1', 'CHEMBL1', 'headache')]
    assert held == []


GOLDEN_1 = 11.000861078625661
GOLDEN_2 = 8.047189562170503


def _pairs(*triples: tuple[str, str, str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            'safetyreportid': [t[0] for t in triples],
            'chembl_id': [t[1] for t in triples],
            'event': [t[2] for t in triples],
        },
        schema={'safetyreportid': pl.String(), 'chembl_id': pl.String(), 'event': pl.String()},
    )


def _table_row(table: pl.DataFrame, chembl_id: str, event: str) -> dict:
    return table.filter((pl.col('chembl_id') == chembl_id) & (pl.col('event') == event)).row(0, named=True)


def test_contingency_counts_are_exact_distinct_reports():
    rows = _pairs(
        ('R1', 'D1', 'e1'),
        ('R1', 'D1', 'e2'),
        ('R2', 'D1', 'e1'),
        ('R3', 'D1', 'e2'),
        ('R4', 'D2', 'e1'),
        ('R5', 'D2', 'e3'),
    )
    table = contingency_table(rows)
    row = _table_row(table, 'D1', 'e1')
    assert (row['A'], row['B'], row['C'], row['D']) == (2, 1, 1, 1)
    assert (row['drug_reports'], row['event_reports']) == (3, 3)
    assert all(table.schema[c] == pl.Int64 for c in ['A', 'B', 'C', 'D', 'drug_reports', 'event_reports'])
    assert set(table.columns) == {'chembl_id', 'event', 'A', 'B', 'C', 'D', 'drug_reports', 'event_reports', 'llr'}


def test_llr_matches_the_formula():
    rows = _pairs(
        *[(f'a{i}', 'D1', 'e1') for i in range(30)],
        *[(f'b{i}', 'X', 'e1') for i in range(370)],
        *[(f'c{i}', 'D1', 'e2') for i in range(30)],
        *[(f'd{i}', 'X', 'e2') for i in range(570)],
    )
    row = _table_row(contingency_table(rows), 'D1', 'e1')
    assert (row['A'], row['B'], row['C'], row['D']) == (30, 370, 30, 570)
    expected = 30 * (log(30) - log(400)) + 30 * (log(30) - log(600)) - 60 * (log(60) - log(1000))
    assert row['llr'] == pytest.approx(expected, rel=1e-12)


def test_c_zero_row_is_dropped():
    rows = _pairs(('R1', 'D1', 'e1'), ('R2', 'D1', 'e1'), ('R3', 'D2', 'e2'))
    table = contingency_table(rows)
    assert table.filter(pl.col('chembl_id') == 'D1').is_empty()
    assert table['llr'].is_nan().sum() == 0
    assert table['llr'].null_count() == 0


def test_d_zero_row_is_kept():
    rows = _pairs(('R1', 'D1', 'e1'), ('R2', 'D1', 'e2'), ('R3', 'D2', 'e1'), ('R3', 'D2', 'e2'))
    row = _table_row(contingency_table(rows), 'D1', 'e1')
    assert (row['A'], row['B'], row['C'], row['D']) == (1, 1, 1, 0)
    assert row['llr'] == pytest.approx(1 * (log(1) - log(2)) + 1 * (log(1) - log(1)) - 2 * (log(2) - log(3)), rel=1e-12)


def test_total_counts_distinct_reports_across_drugs():
    rows = _pairs(('R1', 'D1', 'e1'), ('R1', 'D2', 'e1'), ('R2', 'D1', 'e2'), ('R3', 'D2', 'e2'))
    table = contingency_table(rows)
    assert table.select((pl.col('A') + pl.col('B') + pl.col('C') + pl.col('D')).alias('t'))['t'].unique().to_list() == [
        3
    ]


def test_critical_value_matches_the_spark_implementation():
    assert critical_value(200, 60, np.array([40.0, 90.0, 150.0, 400.0]), 1000, 0.95) == pytest.approx(
        GOLDEN_1, rel=1e-12
    )
    assert critical_value(500, 5, np.array([1.0, 2.0]), 10, 0.95) == pytest.approx(GOLDEN_2, rel=1e-12)


def test_critical_value_guards_return_zero():
    n_i = np.array([1.0, 2.0])
    assert critical_value(100, 0, n_i, 10, 0.95) == 0.0
    assert critical_value(100, 5, n_i, 0, 0.95) == 0.0
    assert critical_value(100, 5, np.array([]), 10, 0.95) == 0.0


def _hand_table(event_reports: list[int], drug_reports: int, total: int) -> pl.DataFrame:
    count = len(event_reports)
    return pl.DataFrame({
        'chembl_id': ['D1'] * count,
        'event': [f'e{i}' for i in range(count)],
        'A': [1] * count,
        'B': [er - 1 for er in event_reports],
        'C': [drug_reports - 1] * count,
        'D': [total - er - drug_reports + 1 for er in event_reports],
        'drug_reports': [drug_reports] * count,
        'event_reports': event_reports,
        'llr': [1.0] * count,
    })


def test_critical_values_do_not_depend_on_row_order():
    expected = critical_value(500, 900, np.array([40.0, 90.0, 150.0, 400.0]), 1000, 0.95)
    reversed_n_i = critical_value(500, 900, np.array([400.0, 150.0, 90.0, 40.0]), 1000, 0.95)
    assert expected != reversed_n_i
    for event_reports in ([400, 150, 90, 40], [40, 90, 150, 400], [150, 400, 40, 90]):
        result = critical_values(_hand_table(event_reports, 900, 1000), 500, 0.95)
        assert result.columns == ['chembl_id', 'critval']
        assert result['critval'].to_list() == [expected]


def test_n_i_uses_only_rows_that_survived_the_llr_filter():
    rows = [(f'r{i}', 'D1', 'all') for i in range(6)]
    rows += [(f'r{i}', 'D1', 'e1') for i in range(2)]
    rows += [(f'r{i}', 'D2', 'e1') for i in range(6, 11)]
    rows += [(f'r{i}', 'D1', 'e2') for i in (2, 3, 4)]
    rows += [('r11', 'D2', 'e2')]
    rows += [(f'r{i}', 'D2', 'all') for i in range(12, 32)]
    table = contingency_table(_pairs(*rows))
    d1 = table.filter(pl.col('chembl_id') == 'D1')
    assert d1['event'].sort().to_list() == ['e1', 'e2']
    assert d1['event_reports'].sort().to_list() == [4, 7]
    got = critical_values(table, 200, 0.95).filter(pl.col('chembl_id') == 'D1')['critval'][0]
    assert got == critical_value(200, 6, np.array([4.0, 7.0]), 32, 0.95)
    assert got != critical_value(200, 6, np.array([4.0, 7.0, 26.0]), 32, 0.95)


def test_significant_events_filter_and_schema():
    table = pl.DataFrame({
        'chembl_id': ['B', 'A', 'A', 'A', 'C'],
        'event': ['e1', 'e2', 'e1', 'e3', 'e1'],
        'A': [4, 3, 2, 5, 6],
        'B': [0] * 5,
        'C': [0] * 5,
        'D': [0] * 5,
        'drug_reports': [0] * 5,
        'event_reports': [0] * 5,
        'llr': [5.0, 5.0, 3.0, 2.0, 9.0],
    })
    critical = pl.DataFrame({'chembl_id': ['A', 'B', 'C'], 'critval': [3.0, 1.0, 0.0]})
    result = significant_events(table, critical)
    assert result.schema == pharmacovigilance_schema
    assert result.rows() == [('A', 'e2', 3, 5.0, 3.0, ''), ('B', 'e1', 4, 5.0, 1.0, '')]


PERMUTATIONS = 500
PERCENTILE = 0.95
SETTINGS = {'montecarlo_permutations': PERMUTATIONS, 'montecarlo_percentile': PERCENTILE, 'archive_workers': 2}


def _made(path: Path) -> Path:
    path.mkdir(parents=True)
    return path


def _single(report_id: str, drug: str, event: str, **overrides) -> dict:
    return _report(
        safetyreportid=report_id,
        patient=_patient([event], [_drug(drug)]),
        **overrides,
    )


def _fixture_reports() -> list[dict]:
    """ASPIRIN: 10 HEADACHE + 1 RASH; Advil (ibuprofen): 18 NAUSEA, 19 RASH, 2 HEADACHE; plus excluded reports."""
    valid = (
        [_single(f'A{i:02d}', 'ASPIRIN', 'HEADACHE') for i in range(10)]
        + [_single('A10', 'ASPIRIN', 'RASH')]
        + [_single(f'IN{i:02d}', 'Advil', 'NAUSEA') for i in range(18)]
        + [_single(f'IR{i:02d}', 'Advil', 'RASH') for i in range(19)]
        + [_single(f'IH{i:02d}', 'Advil', 'HEADACHE') for i in range(2)]
    )
    excluded = [
        _single('DEAD', 'ASPIRIN', 'HEADACHE', seriousnessdeath='1'),
        _single('LAY', 'ASPIRIN', 'HEADACHE', primarysource={'qualification': '4'}),
        _single('BLACK', 'ASPIRIN', 'DRUG INEFFECTIVE'),
        _single('UNKNOWN', 'MYSTERY', 'HEADACHE'),
    ]
    return valid + excluded


def _setup(root: Path, split: int | None = None, order: tuple[str, str] = ('2024q1', '2024q2')) -> dict[str, str]:
    reports = _fixture_reports()
    half = len(reports) // 2 if split is None else split
    first, second = order
    for quarter, chunk in ((first, reports[:half]), (second, reports[half:])):
        (root / 'input' / 'openfda' / quarter).mkdir(parents=True, exist_ok=True)
        _write_zip(root / 'input' / 'openfda' / quarter / 'events.zip', chunk)
    (root / 'input' / 'openfda' / 'blacklisted_events.txt').write_text(' Drug Ineffective \n\n')
    label = {'label': 'Advil', 'source': 'x'}
    labels = pl.List(pl.Struct({'label': pl.String(), 'source': pl.String()}))
    molecules = pl.DataFrame(
        {
            'id': ['CHEMBL1', 'CHEMBL2', 'CHEMBL3'],
            'name': ['Aspirin', 'Ibuprofen', 'Idle'],
            'tradeNames': [[], [label], None],
            'synonyms': [[], [], []],
            'unused': ['a', 'b', 'c'],
        },
        schema={
            'id': pl.String(),
            'name': pl.String(),
            'tradeNames': labels,
            'synonyms': labels,
            'unused': pl.String(),
        },
    )
    write_dataset(molecules, str(root / 'input' / 'drug_molecule'))
    return {
        'drug_molecule': str(root / 'input' / 'drug_molecule'),
        'fda_events': str(root / 'input' / 'openfda'),
        'blacklisted_events': str(root / 'input' / 'openfda' / 'blacklisted_events.txt'),
    }


def _files(root: Path) -> set[str]:
    return {str(p) for p in root.rglob('*')}


def _run(root: Path, source: dict[str, str], **settings) -> pl.DataFrame:
    destination = str(root / 'output' / 'pharmacovigilance')
    pharmacovigilance(source, destination, {**SETTINGS, **settings}, None)
    return scan_dataset(destination).collect().sort('chembl_id', 'event')


def test_end_to_end_writes_only_the_expected_output(tmp_path):
    source = _setup(tmp_path)
    before = _files(tmp_path)
    result = _run(tmp_path, source)

    # 50 valid reports; events: headache 12, rash 20, nausea 18.
    # ASPIRIN n_j=11: headache A=10 B=2 C=1 D=37; rash A=1 B=19 C=10 D=20.
    # Advil n_j=39: nausea A=18 B=0 C=21 D=11; rash A=19 B=1 C=20 D=10; headache A=2 B=10 C=37 D=1.
    def llr(a: int, b: int, c: int, d: int) -> float:
        return a * (log(a) - log(a + b)) + c * (log(c) - log(c + d)) - (a + c) * (log(a + c) - log(a + b + c + d))

    aspirin_critval = critical_value(PERMUTATIONS, 11, np.array([12.0, 20.0]), 50, PERCENTILE)
    advil_critval = critical_value(PERMUTATIONS, 39, np.array([12.0, 18.0, 20.0]), 50, PERCENTILE)
    significant = {('CHEMBL1', 'headache'): (10, llr(10, 2, 1, 37), aspirin_critval)}
    significant[('CHEMBL2', 'headache')] = (2, llr(2, 10, 37, 1), advil_critval)
    # the pairs left out score below their drug's critical value
    assert llr(10, 2, 1, 37) > aspirin_critval > llr(1, 19, 10, 20)
    assert llr(2, 10, 37, 1) > advil_critval > max(llr(18, 0, 21, 11), llr(19, 1, 20, 10))

    assert result.schema == pharmacovigilance_schema
    assert result.select('chembl_id', 'event').rows() == sorted(significant)
    for chembl_id, event, count, score, critval, meddra in result.rows():
        want = significant[(chembl_id, event)]
        assert count == want[0]
        assert score == pytest.approx(want[1], abs=1e-12)
        assert critval == pytest.approx(want[2], abs=1e-12)
        assert meddra == ''

    new = _files(tmp_path) - before
    output = str(tmp_path / 'output')
    assert all(p == output or p.startswith(f'{output}/pharmacovigilance') for p in new)


def test_report_in_two_archives_counts_once(tmp_path):
    expected = _run(tmp_path / 'single', _setup(_made(tmp_path / 'single')))
    root = _made(tmp_path / 'dup')
    source = _setup(root)
    _write_zip(root / 'input' / 'openfda' / '2024q2' / 'again.zip', _fixture_reports()[:12])
    assert _run(root, source).equals(expected)


def test_output_is_identical_across_archive_order_and_worker_count(tmp_path):
    runs = []
    for name, split, order, workers in [
        ('a', None, ('2024q1', '2024q2'), 1),
        ('b', None, ('2024q2', '2024q1'), 4),
        ('c', 7, ('2024q1', '2024q2'), 3),
    ]:
        root = _made(tmp_path / name)
        runs.append(_run(root, _setup(root, split, order), archive_workers=workers))
    assert runs[0].height == 2
    assert runs[1].equals(runs[0])
    assert runs[2].equals(runs[0])


def test_no_archives_raises_and_writes_nothing(tmp_path):
    source = _setup(tmp_path)
    for archive in (tmp_path / 'input' / 'openfda').rglob('*.zip'):
        archive.unlink()
    destination = tmp_path / 'output' / 'pharmacovigilance'
    with pytest.raises(ValueError, match='no FAERS archives'):
        pharmacovigilance(source, str(destination), SETTINGS, None)
    assert not destination.exists()


def test_corrupt_zip_raises_and_writes_nothing(tmp_path):
    source = _setup(tmp_path)
    (tmp_path / 'input' / 'openfda' / '2024q2' / 'events.zip').write_bytes(b'not a zip')
    destination = tmp_path / 'output' / 'pharmacovigilance'
    with pytest.raises(zipfile.BadZipFile):
        pharmacovigilance(source, str(destination), SETTINGS, None)
    assert not destination.exists()


def test_no_surviving_rows_raises_and_writes_nothing(tmp_path):
    source = _setup(tmp_path)
    for archive in (tmp_path / 'input' / 'openfda').rglob('*.zip'):
        _write_zip(archive, [_single('X', 'MYSTERY', 'HEADACHE')])
    destination = tmp_path / 'output' / 'pharmacovigilance'
    with pytest.raises(ValueError, match='no report rows'):
        pharmacovigilance(source, str(destination), SETTINGS, None)
    assert not destination.exists()


def test_populated_destination_raises_before_any_archive_is_read(tmp_path):
    source = _setup(tmp_path)
    (tmp_path / 'input' / 'openfda' / '2024q2' / 'events.zip').write_bytes(b'not a zip')
    destination = tmp_path / 'output' / 'pharmacovigilance'
    destination.mkdir(parents=True)
    with pytest.raises(ValueError, match='already exists'):
        pharmacovigilance(source, str(destination), SETTINGS, None)
